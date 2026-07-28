"""Utility / helper functions for the video editing engine.

Provides file hashing, directory management, temp-file cleanup, format
detection, retry logic, memory-efficient chunking, metadata extraction,
and path resolution helpers.
"""

from __future__ import annotations

import functools
import hashlib
import imghdr
import os
import shutil
import struct
import tempfile
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, BinaryIO, Callable, Generator, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PATH_LENGTH = 32767
_VIDEO_SIGNATURES: dict[str, list[bytes]] = {
    "mp4": [b"\x00\x00\x00\x18ftyp", b"\x00\x00\x00\x1cftyp", b"\x00\x00\x00\x20ftyp", b"ftypisom"],
    "mov": [b"\x00\x00\x00\x20ftypqt  "],
    "avi": [b"RIFF"],
    "mkv": [b"\x1a\x45\xdf\xa3"],
    "webm": [b"\x1a\x45\xdf\xa3"],
    "flv": [b"FLV\x01"],
    "wmv": [b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"],
    "mpg": [b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"],
}
_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv",
    ".mpg", ".mpeg", ".m4v", ".3gp", ".ts", ".mts",
}
_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".svg", ".avif",
}
_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".wma", ".m4a", ".opus",
}


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def compute_file_hash(
    path: str | Path,
    algorithm: str = "sha256",
    chunk_size: int = 65536,
) -> str:
    """Compute the hex digest hash of a file.

    Args:
        path: Path to the file.
        algorithm: Hash algorithm name (any :mod:`hashlib` supported name).
        chunk_size: Read chunk size in bytes.

    Returns:
        Hex-encoded hash string.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the algorithm is not supported.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Not a regular file: {path}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm '{algorithm}'") from exc

    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_bytes_hash(data: bytes, algorithm: str = "sha256") -> str:
    """Return the hex digest of raw *data*."""
    return hashlib.new(algorithm, data).hexdigest()


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------

def ensure_directory(path: str | Path) -> Path:
    """Create the directory (and parents) if it does not exist.

    Returns:
        The resolved :class:`Path`.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_directory_name(name: str, replacement: str = "_") -> str:
    """Sanitize a string so it is safe to use as a directory name."""
    forbidden = '<>:"/\\|?*\x00'
    result = "".join(replacement if c in forbidden else c for c in name)
    result = result.strip(". ")
    return result or "unnamed"


def clean_directory(path: str | Path, remove_root: bool = False) -> int:
    """Remove all contents of a directory.

    Args:
        path: Target directory.
        remove_root: If *True*, also remove the directory itself.

    Returns:
        Number of items removed.
    """
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")

    count = 0
    for item in p.iterdir():
        if item.is_dir():
            count += shutil.rmtree(item)
        else:
            item.unlink()
            count += 1
    if remove_root:
        p.rmdir()
        count += 1
    return count


def directory_size(path: str | Path) -> int:
    """Return the total size in bytes of all files under *path*."""
    p = Path(path)
    if not p.exists():
        return 0
    total = 0
    for entry in os.scandir(p):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat().st_size
        elif entry.is_dir(follow_symlinks=False):
            total += directory_size(entry.path)
    return total


# ---------------------------------------------------------------------------
# Temp-file management
# ---------------------------------------------------------------------------

def create_temp_dir(prefix: str = "ve_", suffix: str = "") -> Path:
    """Create and return a new temporary directory."""
    return Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))


def create_temp_file(
    suffix: str = ".tmp",
    prefix: str = "ve_",
    directory: Optional[str | Path] = None,
) -> Path:
    """Create and return the path to a new temporary file (closed)."""
    fd, path_str = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=directory)
    os.close(fd)
    return Path(path_str)


def cleanup_temp_files(directory: str | Path, max_age_hours: float = 24.0) -> int:
    """Remove temporary files older than *max_age_hours*.

    Returns:
        Number of files removed.
    """
    p = Path(directory)
    if not p.is_dir():
        return 0

    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    for item in p.rglob("*"):
        if item.is_file():
            try:
                if item.stat().st_mtime < cutoff:
                    item.unlink()
                    removed += 1
            except OSError:
                continue
    return removed


def cleanup_temp_dir() -> None:
    """Best-effort removal of the process-level default temp directory contents."""
    temp_root = Path(tempfile.gettempdir()) / "video_engine"
    if temp_root.is_dir():
        shutil.rmtree(temp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_file_type(path: str | Path) -> str:
    """Detect file type by extension and magic bytes.

    Returns:
        One of ``"video"``, ``"image"``, ``"audio"``, or ``"unknown"``.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in _VIDEO_EXTENSIONS:
        return "video"
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _AUDIO_EXTENSIONS:
        return "audio"

    # Fallback: check magic bytes
    try:
        with open(p, "rb") as fh:
            header = fh.read(32)
    except OSError:
        return "unknown"

    if any(header.startswith(sig) for sigs in _VIDEO_SIGNATURES.values() for sig in sigs):
        return "video"

    img_type = imghdr.what(h=header)
    if img_type is not None:
        return "image"

    return "unknown"


def detect_video_codec(path: str | Path) -> Optional[str]:
    """Heuristically detect the video container format from magic bytes."""
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            header = fh.read(32)
    except OSError:
        return None

    for fmt, signatures in _VIDEO_SIGNATURES.items():
        if any(header.startswith(sig) for sig in signatures):
            return fmt
    return None


def is_video_file(path: str | Path) -> bool:
    """Return *True* if *path* looks like a video file."""
    return detect_file_type(path) == "video"


def is_image_file(path: str | Path) -> bool:
    """Return *True* if *path* looks like an image file."""
    return detect_file_type(path) == "image"


def is_audio_file(path: str | Path) -> bool:
    """Return *True* if *path* looks like an audio file."""
    return detect_file_type(path) == "audio"


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry(
    max_attempts: int = 3,
    delay_sec: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    logger: Optional[Any] = None,
) -> Callable[[F], F]:
    """Decorator that retries a function call on failure.

    Args:
        max_attempts: Maximum number of attempts (must be >= 1).
        delay_sec: Initial delay between retries.
        backoff_factor: Multiplier applied to the delay after each failure.
        exceptions: Tuple of exception types that trigger a retry.
        logger: Optional logger for retry messages.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            current_delay = delay_sec
            last_exc: Optional[BaseException] = None
            while attempt < max_attempts:
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    last_exc = exc
                    if attempt < max_attempts:
                        if logger is not None:
                            logger.warning(
                                "Attempt %d/%d for %s failed: %s. Retrying in %.1fs",
                                attempt,
                                max_attempts,
                                fn.__qualname__,
                                exc,
                                current_delay,
                            )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Memory-efficient file chunking
# ---------------------------------------------------------------------------

def read_file_chunks(
    path: str | Path,
    chunk_size: int = 65536,
) -> Generator[bytes, None, None]:
    """Yield *chunk_size* byte slices from *path*.

    This is a generator to allow streaming large files without loading them
    entirely into memory.
    """
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            yield chunk


def split_file_into_chunks(
    path: str | Path,
    output_dir: str | Path,
    chunk_size: int = 10 * 1024 * 1024,
    prefix: str = "chunk_",
) -> list[Path]:
    """Split a file into fixed-size chunks on disk.

    Args:
        path: Source file.
        output_dir: Directory to write chunks into.
        chunk_size: Bytes per chunk.
        prefix: Filename prefix for chunk files.

    Returns:
        Ordered list of chunk file paths.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks: list[Path] = []
    idx = 0
    with open(src, "rb") as fh:
        while True:
            data = fh.read(chunk_size)
            if not data:
                break
            chunk_path = out / f"{prefix}{idx:06d}"
            chunk_path.write_bytes(data)
            chunks.append(chunk_path)
            idx += 1

    return chunks


def reassemble_chunks(
    chunk_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    """Write chunks back together into a single file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as out_fh:
        for cp in chunk_paths:
            p = Path(cp)
            if not p.is_file():
                raise FileNotFoundError(f"Chunk not found: {p}")
            with open(p, "rb") as chunk_fh:
                shutil.copyfileobj(chunk_fh, out_fh)
    return out


# ---------------------------------------------------------------------------
# Video metadata helpers (lightweight – no ffmpeg dependency)
# ---------------------------------------------------------------------------

def get_file_size_mb(path: str | Path) -> float:
    """Return file size in megabytes."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.stat().st_size / (1024 * 1024)


def get_file_modification_time(path: str | Path) -> float:
    """Return the last modification time as a UNIX timestamp."""
    return Path(path).stat().st_mtime


def ffprobe_available() -> bool:
    """Return *True* if ``ffprobe`` is on the system PATH."""
    return shutil.which("ffprobe") is not None


def extract_metadata_with_ffprobe(path: str | Path) -> dict[str, Any]:
    """Extract metadata via ``ffprobe`` (must be installed).

    Returns:
        A dict containing ``duration``, ``width``, ``height``, ``codec``,
        ``fps``, ``bitrate``, ``audio_codec``, ``audio_sample_rate``.
    """
    import json as _json
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe is not installed or not on PATH")

    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

    info = _json.loads(result.stdout)
    metadata: dict[str, Any] = {}

    fmt = info.get("format", {})
    metadata["duration"] = float(fmt.get("duration", 0))
    metadata["bitrate"] = int(fmt.get("bit_rate", 0))

    for stream in info.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and "width" not in metadata:
            metadata["width"] = int(stream.get("width", 0))
            metadata["height"] = int(stream.get("height", 0))
            metadata["codec"] = stream.get("codec_name", "")
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            try:
                num, den = r_frame_rate.split("/")
                metadata["fps"] = round(int(num) / int(den), 3) if int(den) else 0.0
            except (ValueError, ZeroDivisionError):
                metadata["fps"] = 0.0
        elif codec_type == "audio" and "audio_codec" not in metadata:
            metadata["audio_codec"] = stream.get("codec_name", "")
            metadata["audio_sample_rate"] = int(stream.get("sample_rate", 0))
            metadata["audio_channels"] = int(stream.get("channels", 0))

    return metadata


def get_video_duration(path: str | Path) -> float:
    """Return video duration in seconds using ffprobe (raises on failure)."""
    meta = extract_metadata_with_ffprobe(path)
    return float(meta.get("duration", 0))


def format_duration(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.ms``."""
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    hours, remainder = divmod(total_sec, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - total_sec) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def resolve_project_path(
    relative: str | Path,
    base_dir: Optional[str | Path] = None,
) -> Path:
    """Resolve *relative* against a base directory.

    If *base_dir* is not given the current working directory is used.
    The result is fully resolved (symlinks followed).
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    return (base / relative).resolve()


def safe_path(path: str | Path, max_length: int = MAX_PATH_LENGTH) -> Path:
    """Return *path* truncated / sanitized to fit within OS limits."""
    p = Path(path)
    parts = p.parts
    result = Path(parts[0]) if len(parts) > 1 else Path(".")
    for part in parts[1:]:
        safe_part = part
        if len(str(result / safe_part)) > max_length:
            safe_part = safe_part[: max_length - len(str(result)) - 1]
        result = result / safe_part
    return result


def ensure_unique_path(path: str | Path) -> Path:
    """If *path* exists, append an incrementing suffix until unique.

    Example: ``out.mp4`` -> ``out_001.mp4`` -> ``out_002.mp4``.
    """
    p = Path(path)
    if not p.exists():
        return p

    stem = p.stem
    suffix = p.suffix
    parent = p.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
        if counter > 9999:
            raise RuntimeError(f"Could not find a unique path near {path}")


def relative_to(path: str | Path, base: str | Path) -> str:
    """Return *path* as a string relative to *base*, or the absolute path."""
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(Path(path).resolve())


def make_safe_filename(name: str, replacement: str = "_") -> str:
    """Sanitize a string for use as a filename (preserves extension)."""
    p = Path(name)
    stem = p.stem
    suffix = p.suffix
    forbidden = '<>:"/\\|?*\x00'
    safe_stem = "".join(replacement if c in forbidden else c for c in stem)
    safe_stem = safe_stem.strip(". ")
    if not safe_stem:
        safe_stem = "unnamed"
    return f"{safe_stem}{suffix}"
