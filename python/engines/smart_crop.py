"""Smart cropping engine for video content.

Provides intelligent, subject-aware cropping with smooth position tracking,
multi-subject detection, and ffmpeg-based output rendering.

Usage::

    from python.engines.smart_crop import SmartCropper

    cropper = SmartCropper()
    result = cropper.crop_video("input.mp4", "9:16", "output.mp4")
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Crop aspect import
# ---------------------------------------------------------------------------

try:
    from python.core.config import CropAspect  # type: ignore[import-untyped]
except (ImportError, ModuleNotFoundError):

    class CropAspect(str, Enum):  # type: ignore[no-redef]
        """Supported crop aspect ratios as width:height strings."""

        PORTRAIT_9_16 = "9:16"
        SQUARE_1_1 = "1:1"
        PORTRAIT_3_4 = "3:4"
        LANDSCAPE_16_9 = "16:9"

        @classmethod
        def from_string(cls, value: str) -> "CropAspect":
            normalised = value.strip().replace(" ", "")
            for member in cls:
                if member.value == normalised:
                    return member
            raise ValueError(
                f"Unsupported aspect ratio '{value}'. "
                f"Choose from: {[m.value for m in cls]}"
            )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SubjectType(str, Enum):
    """Category of a detected visual subject."""

    FACE = "face"
    HUMAN = "human"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    PRODUCT = "product"
    TEXT = "text"


@dataclass
class Subject:
    """A detected subject inside a video frame."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    subject_type: SubjectType

    @property
    def center(self) -> Tuple[int, int]:
        """Return the centre (cx, cy) of the bounding box."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def area(self) -> int:
        """Return the area in pixels of the bounding box."""
        _, _, w, h = self.bbox
        return w * h

    def expanded(
        self, margin: int = 10, frame_w: int = 0, frame_h: int = 0
    ) -> "Subject":
        """Return a new Subject with bbox expanded by margin pixels, clamped to frame dims."""
        x, y, w, h = self.bbox
        nx = max(0, x - margin)
        ny = max(0, y - margin)
        nw = w + 2 * margin
        nh = h + 2 * margin
        if frame_w:
            nw = min(nw, frame_w - nx)
        if frame_h:
            nh = min(nh, frame_h - ny)
        return Subject(
            bbox=(nx, ny, nw, nh),
            confidence=self.confidence,
            subject_type=self.subject_type,
        )


@dataclass
class CropRegion:
    """Rectangular crop region in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class CropAnalysis:
    """Result of analysing a single frame for smart cropping."""

    subjects: List[Subject]
    recommended_region: CropRegion


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_TYPE_WEIGHTS: Dict[SubjectType, float] = {
    SubjectType.FACE: 5.0,
    SubjectType.HUMAN: 4.0,
    SubjectType.PRODUCT: 3.5,
    SubjectType.TEXT: 3.0,
    SubjectType.VEHICLE: 2.5,
    SubjectType.ANIMAL: 2.0,
}

_ASPECT_RATIOS: Dict[str, float] = {
    "9:16": 9 / 16,
    "1:1": 1.0,
    "3:4": 3 / 4,
    "16:9": 16 / 9,
}


def _resolve_aspect(aspect: "str | CropAspect") -> float:
    """Return the numeric width/height ratio for aspect."""
    value = aspect.value if isinstance(aspect, CropAspect) else str(aspect)
    if value in _ASPECT_RATIOS:
        return _ASPECT_RATIOS[value]
    raise ValueError(f"Unsupported aspect ratio '{value}'.")


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))

# ---------------------------------------------------------------------------
# SmartCropper
# ---------------------------------------------------------------------------


class SmartCropper:
    """Intelligent, subject-aware video cropper.

    Parameters:
        face_cascade_path: Optional override for the OpenCV Haar cascade XML.
        confidence_threshold: Minimum detection confidence to keep a subject.
        batch_size: Frames loaded into memory at once during processing.
        smoothing_window: Default window size (in frames) for smoothing.
    """

    def __init__(
        self,
        face_cascade_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        batch_size: int = 64,
        smoothing_window: int = 15,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.smoothing_window = smoothing_window
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._face_cascade_path = face_cascade_path
        self._hog: Optional[cv2.HOGDescriptor] = None
        logger.info(
            "SmartCropper initialised (batch_size=%d, smooth_window=%d)",
            batch_size, smoothing_window,
        )

    # ------------------------------------------------------------------
    # Lazy resource loaders
    # ------------------------------------------------------------------

    def _get_face_cascade(self) -> cv2.CascadeClassifier:
        if self._face_cascade is None:
            path = self._face_cascade_path or (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                raise RuntimeError(f"Failed to load face cascade from {path}")
            self._face_cascade = cascade
        return self._face_cascade

    def _get_hog(self) -> cv2.HOGDescriptor:
        if self._hog is None:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog = hog
        return self._hog

    # ------------------------------------------------------------------
    # Static / private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        """Intersection-over-union between two (x, y, w, h) boxes."""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _nms(
        subjects: List[Subject], iou_threshold: float = 0.5
    ) -> List[Subject]:
        """Non-maximum suppression grouped by subject type."""
        if not subjects:
            return []
        by_type: Dict[SubjectType, List[Subject]] = {}
        for s in subjects:
            by_type.setdefault(s.subject_type, []).append(s)
        kept: List[Subject] = []
        for _type, group in by_type.items():
            group.sort(key=lambda s: s.confidence, reverse=True)
            alive: List[Subject] = []
            for candidate in group:
                dominated = False
                for existing in alive:
                    if SmartCropper._compute_iou(
                        candidate.bbox, existing.bbox
                    ) > iou_threshold:
                        dominated = True
                        break
                if not dominated:
                    alive.append(candidate)
            kept.extend(alive)
        return kept

    @staticmethod
    def _merge_nearby_boxes(
        boxes: List[Tuple[int, int, int, int]],
        proximity_x: int = 15,
        proximity_y: int = 8,
    ) -> List[Tuple[int, int, int, int]]:
        """Greedy-merge boxes whose gaps are within thresholds."""
        if not boxes:
            return []
        merged = list(boxes)
        changed = True
        while changed:
            changed = False
            new_merged: List[Tuple[int, int, int, int]] = []
            used = [False] * len(merged)
            for i in range(len(merged)):
                if used[i]:
                    continue
                x1, y1, w1, h1 = merged[i]
                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    x2, y2, w2, h2 = merged[j]
                    gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
                    gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
                    if gap_x <= proximity_x and gap_y <= proximity_y:
                        nx = min(x1, x2)
                        ny = min(y1, y2)
                        nw = max(x1 + w1, x2 + w2) - nx
                        nh = max(y1 + h1, y2 + h2) - ny
                        x1, y1, w1, h1 = nx, ny, nw, nh
                        used[j] = True
                        changed = True
                new_merged.append((x1, y1, w1, h1))
            merged = new_merged
        return merged

    def _center_crop(self, h: int, w: int, aspect: float) -> CropRegion:
        """Return the largest centred crop rectangle matching aspect."""
        if w / h > aspect:
            crop_w = int(h * aspect)
            crop_h = h
        else:
            crop_w = w
            crop_h = int(w / aspect)
        cx = (w - crop_w) // 2
        cy = (h - crop_h) // 2
        return CropRegion(x=cx, y=cy, width=crop_w, height=crop_h)

    # ------------------------------------------------------------------
    # Public analysis API
    # ------------------------------------------------------------------

    def analyze_frame(self, frame: np.ndarray) -> CropAnalysis:
        """Analyse a single BGR frame and return subjects plus recommended crop.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            A CropAnalysis containing detected subjects and the recommended CropRegion.
        """
        subjects = self.detect_subjects(frame)
        region = self.calculate_crop_region(frame, subjects, target_aspect="9:16")
        return CropAnalysis(subjects=subjects, recommended_region=region)

    # ------------------------------------------------------------------
    # Subject detection
    # ------------------------------------------------------------------

    def detect_subjects(self, frame: np.ndarray) -> List[Subject]:
        """Run every detector and merge results via per-type NMS.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            Deduplicated list of Subject instances.
        """
        all_subjects: List[Subject] = []
        for detector in (
            self.detect_faces,
            self.detect_humans,
            self.detect_vehicles,
            self.detect_text_regions,
            self.detect_animals,
        ):
            try:
                all_subjects.extend(detector(frame))
            except Exception:
                logger.debug(
                    "Detector %s failed", detector.__name__, exc_info=True
                )
        return self._nms(all_subjects, iou_threshold=0.5)

    def detect_faces(self, frame: np.ndarray) -> List[Subject]:
        """Detect faces using OpenCV Haar cascades.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            List of face subjects with confidence 1.0.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = self._get_face_cascade()
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        h, w = frame.shape[:2]
        subjects: List[Subject] = []
        for fx, fy, fw, fh in faces:
            subjects.append(
                Subject(
                    bbox=(int(fx), int(fy), int(fw), int(fh)),
                    confidence=1.0,
                    subject_type=SubjectType.FACE,
                ).expanded(margin=15, frame_w=w, frame_h=h)
            )
        return subjects

    def detect_humans(self, frame: np.ndarray) -> List[Subject]:
        """Detect full-body humans via HOG + SVM.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            List of human subjects.
        """
        hog = self._get_hog()
        rects, weights = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
        )
        h, w = frame.shape[:2]
        subjects: List[Subject] = []
        for (rx, ry, rw, rh), weight in zip(rects, weights):
            conf = float(weight)
            if conf < self.confidence_threshold:
                continue
            subjects.append(
                Subject(
                    bbox=(int(rx), int(ry), int(rw), int(rh)),
                    confidence=min(conf, 1.0),
                    subject_type=SubjectType.HUMAN,
                ).expanded(margin=10, frame_w=w, frame_h=h)
            )
        return subjects

    def detect_vehicles(self, frame: np.ndarray) -> List[Subject]:
        """Detect vehicles using cascade + edge-density heuristics.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            List of vehicle subjects.
        """
        subjects: List[Subject] = []
        h, w = frame.shape[:2]

        cascade_names = ["haarcascade_frontalface_default.xml"]
        for name in cascade_names:
            path = cv2.data.haarcascades + name
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60),
            )
            for rx, ry, rw, rh in rects:
                subjects.append(
                    Subject(
                        bbox=(int(rx), int(ry), int(rw), int(rh)),
                        confidence=0.4,
                        subject_type=SubjectType.VEHICLE,
                    )
                )
            if subjects:
                break

        # Fallback: edge-density heuristics for large rectangular blobs.
        if not subjects:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            frame_area = h * w
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < frame_area * 0.02 or area > frame_area * 0.5:
                    continue
                bx, by, bw, bh = cv2.boundingRect(contour)
                ar = bw / bh if bh else 0
                if ar < 1.0 or ar > 4.0:
                    continue
                solidity = area / (bw * bh) if (bw * bh) else 0
                if solidity < 0.4:
                    continue
                subjects.append(
                    Subject(
                        bbox=(int(bx), int(by), int(bw), int(bh)),
                        confidence=round(min(solidity, 1.0), 3),
                        subject_type=SubjectType.VEHICLE,
                    )
                )
        return subjects

    def detect_animals(self, frame: np.ndarray) -> List[Subject]:
        """Detect animal-like regions via contour analysis.

        Lightweight heuristic for moderately sized blobs with circularity.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            List of animal subjects.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        h, w = frame.shape[:2]
        frame_area = h * w
        subjects: List[Subject] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < frame_area * 0.005 or area > frame_area * 0.4:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.15:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            ar = bw / bh if bh else 0
            if ar < 0.2 or ar > 5.0:
                continue

            confidence = min(circularity * 1.2, 1.0)
            if confidence < self.confidence_threshold:
                continue
            subjects.append(
                Subject(
                    bbox=(int(bx), int(by), int(bw), int(bh)),
                    confidence=round(confidence, 3),
                    subject_type=SubjectType.ANIMAL,
                )
            )
        return subjects

    def detect_text_regions(self, frame: np.ndarray) -> List[Subject]:
        """Detect text regions using MSER.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            List of text subjects.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create()
        mser.setMinArea(60)
        mser.setMaxArea(int(frame.shape[0] * frame.shape[1] * 0.1))

        regions, _ = mser.detectRegions(gray)
        h, w = frame.shape[:2]

        boxes: List[Tuple[int, int, int, int]] = []
        for region in regions:
            x, y, bw, bh = cv2.boundingRect(region.reshape(-1, 1, 2))
            boxes.append((x, y, bw, bh))

        merged = self._merge_nearby_boxes(boxes, proximity_x=15, proximity_y=8)
        subjects: List[Subject] = []
        for bx, by, bw, bh in merged:
            area_ratio = (bw * bh) / (w * h)
            if area_ratio < 0.001 or area_ratio > 0.25:
                continue
            if bh < 8 or bw < 20:
                continue
            subjects.append(
                Subject(
                    bbox=(int(bx), int(by), int(bw), int(bh)),
                    confidence=0.5,
                    subject_type=SubjectType.TEXT,
                ).expanded(margin=5, frame_w=w, frame_h=h)
            )
        return subjects

    def detect_movement_regions(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> List[Subject]:
        """Detect significant motion between two consecutive frames.

        Uses dense optical flow (Farneback), thresholds magnitude, and
        returns bounding boxes of motion blobs.

        Args:
            prev_frame: Previous uint8 BGR frame.
            curr_frame: Current uint8 BGR frame.

        Returns:
            List of movement subjects typed as SubjectType.HUMAN.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        thresh_mag = (magnitude * 255).astype(np.uint8)
        _, binary = cv2.threshold(thresh_mag, 25, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        h, w = curr_frame.shape[:2]
        frame_area = h * w
        subjects: List[Subject] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < frame_area * 0.002:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            subjects.append(
                Subject(
                    bbox=(int(bx), int(by), int(bw), int(bh)),
                    confidence=min(float(area / (frame_area * 0.05)), 1.0),
                    subject_type=SubjectType.HUMAN,
                ).expanded(margin=20, frame_w=w, frame_h=h)
            )
        return subjects

    # ------------------------------------------------------------------
    # Crop region calculation
    # ------------------------------------------------------------------

    def calculate_crop_region(
        self,
        frame: np.ndarray,
        subjects: Sequence[Subject],
        target_aspect: "str | CropAspect" = "9:16",
    ) -> CropRegion:
        """Compute the optimal crop rectangle for the given frame and subjects.

        When no subjects are detected the crop defaults to the centre of the
        frame.  The algorithm computes a weighted centroid of all subject
        bounding boxes, then expands a centred crop to match the requested
        aspect ratio while staying within frame bounds.

        Args:
            frame: (H, W, 3) source frame (used for dimensions).
            subjects: Detected subjects.
            target_aspect: Desired output aspect ratio (e.g. "9:16").

        Returns:
            A CropRegion that fits within the frame.
        """
        h, w = frame.shape[:2]
        aspect = _resolve_aspect(target_aspect)

        if not subjects:
            return self._center_crop(h, w, aspect)

        # Weighted centroid
        total_weight = 0.0
        cx_sum = 0.0
        cy_sum = 0.0
        for subj in subjects:
            weight = _TYPE_WEIGHTS.get(subj.subject_type, 1.0) * subj.confidence
            sx, sy = subj.center
            cx_sum += sx * weight
            cy_sum += sy * weight
            total_weight += weight

        if total_weight == 0:
            return self._center_crop(h, w, aspect)

        focus_x = cx_sum / total_weight
        focus_y = cy_sum / total_weight

        # Bounding box covering all subjects with padding.
        all_x1 = min(s.bbox[0] for s in subjects)
        all_y1 = min(s.bbox[1] for s in subjects)
        all_x2 = max(s.bbox[0] + s.bbox[2] for s in subjects)
        all_y2 = max(s.bbox[1] + s.bbox[3] for s in subjects)

        subject_w = all_x2 - all_x1
        subject_h = all_y2 - all_y1
        padding = max(int(subject_w * 0.15), int(subject_h * 0.15))

        # Expand from the focus point.
        if aspect < 1:
            crop_h = max(subject_h + 2 * padding, int(h * 0.4))
            crop_h = min(crop_h, h)
            crop_w = int(crop_h * aspect)
            crop_w = min(crop_w, w)
        else:
            crop_w = max(subject_w + 2 * padding, int(w * 0.4))
            crop_w = min(crop_w, w)
            crop_h = int(crop_w / aspect)
            crop_h = min(crop_h, h)

        # Centre on the focus point.
        cx = int(focus_x - crop_w / 2)
        cy = int(focus_y - crop_h / 2)

        # Clamp to frame bounds.
        cx = _clamp(cx, 0, w - crop_w)
        cy = _clamp(cy, 0, h - crop_h)

        return CropRegion(x=cx, y=cy, width=crop_w, height=crop_h)

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------

    def smooth_crop_positions(
        self,
        positions: Sequence[CropRegion],
        window_size: int = 0,
    ) -> List[CropRegion]:
        """Apply a centred moving-average filter to smooth crop positions.

        Each coordinate (x, y, width, height) is independently smoothed.
        Edge frames use a smaller asymmetric window.

        Args:
            positions: Per-frame crop regions.
            window_size: Smoothing window in frames.  0 or negative
                falls back to self.smoothing_window.

        Returns:
            Smoothed list of CropRegion with the same length as positions.
        """
        if not positions:
            return []

        win = window_size if window_size > 0 else self.smoothing_window
        if win < 2:
            return list(positions)

        n = len(positions)
        raw = np.array(
            [[p.x, p.y, p.width, p.height] for p in positions],
            dtype=np.float64,
        )

        smoothed = np.empty_like(raw)
        half = win // 2

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed[i] = raw[lo:hi].mean(axis=0)

        result: List[CropRegion] = []
        for row in smoothed:
            result.append(
                CropRegion(
                    x=int(round(row[0])),
                    y=int(round(row[1])),
                    width=int(round(row[2])),
                    height=int(round(row[3])),
                )
            )
        return result

    # ------------------------------------------------------------------
    # Subject tracking across frames
    # ------------------------------------------------------------------

    def track_subject_across_frames(
        self,
        video_path: "str | Path",
        frame_range: Optional[Tuple[int, int]] = None,
    ) -> List[CropRegion]:
        """Track subjects through a range of frames and return per-frame crop regions.

        Detects subjects in each frame, incorporates motion-based detections
        from optical flow, then applies temporal smoothing.

        Args:
            video_path: Path to the source video file.
            frame_range: (start, end) inclusive 0-based frame indices.
                When None the entire video is processed.

        Returns:
            List of smoothed CropRegion values, one per processed frame.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        start = 0
        end = total_frames - 1
        if frame_range is not None:
            start = max(0, frame_range[0])
            end = min(total_frames - 1, frame_range[1])

        logger.info(
            "Tracking subjects: frames %d-%d of %d (%.1f fps)",
            start, end, total_frames, fps,
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        positions: List[CropRegion] = []
        prev_frame: Optional[np.ndarray] = None

        for idx in range(start, end + 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Frame %d unreadable - using previous crop", idx)
                if positions:
                    positions.append(positions[-1])
                else:
                    h, w = frame.shape[:2] if frame is not None else (1080, 1920)
                    positions.append(self._center_crop(h, w, 9.0 / 16))
                continue

            subjects = self.detect_subjects(frame)

            if prev_frame is not None:
                try:
                    movement = self.detect_movement_regions(prev_frame, frame)
                    subjects.extend(movement)
                    subjects = self._nms(subjects, iou_threshold=0.4)
                except Exception:
                    logger.debug(
                        "Movement detection failed at frame %d", idx, exc_info=True,
                    )

            region = self.calculate_crop_region(
                frame, subjects, target_aspect="9:16",
            )
            positions.append(region)
            prev_frame = frame

        cap.release()

        positions = self.smooth_crop_positions(positions, self.smoothing_window)

        logger.info(
            "Tracked %d frames, produced %d crop regions",
            len(positions), len(positions),
        )
        return positions

    # ------------------------------------------------------------------
    # Full video crop pipeline
    # ------------------------------------------------------------------

    def crop_video(
        self,
        video_path: "str | Path",
        target_aspect: "str | CropAspect" = "9:16",
        output_path: "str | Path | None" = None,
    ) -> Path:
        """Crop an entire video to the target aspect ratio using smart tracking.

        Pipeline:
          1. Scan the video for subjects and compute per-frame crop regions.
          2. Temporally smooth the crop positions.
          3. Detect segment boundaries and generate an ffmpeg filter_complex.
          4. Render the cropped output.

        Args:
            video_path: Path to the source video.
            target_aspect: Desired output aspect ratio.
            output_path: Where to write the cropped video.  When None a
                path is derived from video_path with _cropped appended.

        Returns:
            Path of the cropped output file.

        Raises:
            IOError: If the video cannot be opened.
            RuntimeError: If ffmpeg fails.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise IOError(f"Video not found: {video_path}")

        aspect_label = (
            target_aspect.value
            if isinstance(target_aspect, CropAspect)
            else str(target_aspect)
        )

        if output_path is None:
            suffix = aspect_label.replace(":", "x")
            output_path = video_path.with_name(
                video_path.stem + "_cropped_" + suffix + video_path.suffix
            )
        else:
            output_path = Path(output_path)

        logger.info(
            "Cropping %s -> %s (aspect %s)", video_path, output_path, aspect_label,
        )

        regions = self.track_subject_across_frames(video_path)
        if not regions:
            raise RuntimeError("No frames could be processed from the video.")

        segments = self._segment_positions(regions)

        self._run_ffmpeg_crop(
            video_path=video_path,
            output_path=output_path,
            segments=segments,
        )

        logger.info("Cropped video written to %s", output_path)
        return output_path

    @staticmethod
    def _segment_positions(
        regions: List[CropRegion],
        threshold: float = 5.0,
    ) -> List[Tuple[int, CropRegion]]:
        """Split regions into segments of stable crop positions.

        A new segment starts whenever any coordinate changes by more than
        threshold pixels relative to the current segment representative.

        Returns:
            List of (start_frame_index, CropRegion) tuples.
        """
        if not regions:
            return []

        segments: List[Tuple[int, CropRegion]] = [(0, regions[0])]
        current = regions[0]

        for i, region in enumerate(regions[1:], start=1):
            dx = abs(region.x - current.x)
            dy = abs(region.y - current.y)
            dw = abs(region.width - current.width)
            dh = abs(region.height - current.height)
            if max(dx, dy, dw, dh) > threshold:
                segments.append((i, region))
                current = region

        return segments

    @staticmethod
    def _run_ffmpeg_crop(
        video_path: Path,
        output_path: Path,
        segments: List[Tuple[int, CropRegion]],
    ) -> None:
        """Build and execute the ffmpeg command for the given crop segments.

        For a single segment a simple crop filter is used.
        For multiple segments enable expressions switch crop parameters
        at specific frame numbers.
        """
        cmd: List[str] = ["ffmpeg", "-y", "-i", str(video_path)]

        if len(segments) == 1:
            _, region = segments[0]
            filter_str = (
                f"crop={region.width}:{region.height}:{region.x}:{region.y}"
            )
        else:
            parts: List[str] = []
            for seg_idx, (start_frame, region) in enumerate(segments):
                end_frame = (
                    segments[seg_idx + 1][0] - 1
                    if seg_idx + 1 < len(segments)
                    else 999999
                )
                part = (
                    f"crop={region.width}:{region.height}:{region.x}:{region.y}"
                    f":enable=between(n\\,{start_frame}\\,{end_frame})"
                )
                parts.append(part)
            filter_str = ",".join(parts)

        cmd.extend([
            "-filter_complex",
            f"[0:v]{filter_str}[outv]",
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path),
        ])

        logger.debug("ffmpeg command: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else "(empty)"
            logger.error("ffmpeg stderr:\\n%s", stderr_tail)
            raise RuntimeError(
                f"ffmpeg exited with code {result.returncode}. See stderr for details."
            )


# ---------------------------------------------------------------------------
# Module convenience
# ---------------------------------------------------------------------------

def create_smart_cropper(**kwargs) -> SmartCropper:
    """Factory function to create a SmartCropper with sensible defaults."""
    return SmartCropper(**kwargs)
