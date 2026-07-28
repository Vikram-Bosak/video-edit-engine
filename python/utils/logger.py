"""Structured logging module for the video editing engine.

Provides JSON-structured log output, file and console handlers, progress
tracking, memory usage tracking, and a performance timing decorator.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data  # type: ignore[attr-defined]
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class _ConsoleFormatter(logging.Formatter):
    """Human-friendly colored console output."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        prefix = f"{color}{ts} [{record.levelname:8s}]{self.RESET}"
        msg = record.getMessage()
        if record.exc_info and record.exc_info[0] is not None:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        return f"{prefix} {record.name}: {msg}"


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

_initialized = False
_root_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "video_engine",
    level: str = "INFO",
    log_dir: Optional[str | Path] = None,
    json_output: bool = False,
    console_output: bool = True,
) -> logging.Logger:
    """Initialise and return the application logger.

    Args:
        name: Root logger name.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: If provided, a file handler writes logs here.
        json_output: When *True* the file handler uses JSON format.
        console_output: When *True* a console handler is attached.

    Returns:
        Configured :class:`logging.Logger`.
    """
    global _initialized, _root_logger

    if _initialized and _root_logger is not None:
        return _root_logger.getChild(name)

    root = logging.getLogger("video_engine")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(root.level)
        console_handler.setFormatter(_ConsoleFormatter())
        root.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / "engine.log", encoding="utf-8", delay=True
        )
        file_handler.setLevel(root.level)
        file_handler.setFormatter(
            _JsonFormatter() if json_output else logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s %(module)s:%(funcName)s:%(lineno)d %(message)s"
            )
        )
        root.addHandler(file_handler)

    _initialized = True
    _root_logger = root
    return root.getChild(name)


def get_logger(name: str = "video_engine") -> logging.Logger:
    """Return a child logger, initialising the root if necessary."""
    if not _initialized:
        setup_logger()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Track and log progress of a multi-step operation.

    Example::

        tracker = ProgressTracker(total=100, logger=logger, task="encoding")
        for chunk in data:
            process(chunk)
            tracker.advance(1)
        tracker.complete()
    """

    def __init__(
        self,
        total: int,
        logger: Optional[logging.Logger] = None,
        task: str = "progress",
        log_interval: int = 10,
    ) -> None:
        if total <= 0:
            raise ValueError(f"total must be positive, got {total}")
        self._total = total
        self._current = 0
        self._start_time = time.monotonic()
        self._task = task
        self._log_interval = max(1, log_interval)
        self._logger = logger or get_logger()
        self._completed = False

    @property
    def percentage(self) -> float:
        """Return progress as a percentage (0-100)."""
        return (self._current / self._total) * 100 if self._total else 0.0

    @property
    def elapsed(self) -> float:
        """Return elapsed seconds since creation."""
        return time.monotonic() - self._start_time

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated seconds remaining, or *None* if no data yet."""
        if self._current == 0:
            return None
        rate = self._current / self.elapsed
        remaining = self._total - self._current
        return remaining / rate if rate > 0 else None

    def advance(self, count: int = 1) -> None:
        """Advance the counter by *count* units."""
        if self._completed:
            raise RuntimeError("Cannot advance a completed tracker")
        if count < 0:
            raise ValueError("count must be non-negative")
        self._current = min(self._current + count, self._total)
        if self._current % self._log_interval == 0 or self._current == self._total:
            self._log_progress()

    def _log_progress(self) -> None:
        eta = self.eta_seconds
        eta_str = f"{eta:.1f}s" if eta is not None else "unknown"
        self._logger.info(
            "[%s] %d/%d (%.1f%%) elapsed=%.1fs eta=%s",
            self._task,
            self._current,
            self._total,
            self.percentage,
            self.elapsed,
            eta_str,
        )

    def complete(self) -> None:
        """Mark the task as complete and log final stats."""
        self._current = self._total
        self._completed = True
        self._logger.info(
            "[%s] completed %d items in %.2fs",
            self._task,
            self._total,
            self.elapsed,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a snapshot of the current progress state."""
        return {
            "task": self._task,
            "total": self._total,
            "current": self._current,
            "percentage": round(self.percentage, 2),
            "elapsed_sec": round(self.elapsed, 3),
            "eta_sec": round(self.eta_seconds, 3) if self.eta_seconds is not None else None,
            "completed": self._completed,
        }


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------

class MemoryTracker:
    """Track memory usage over time using :mod:`tracemalloc`.

    Example::

        tracker = MemoryTracker.start()
        # ... do work ...
        snapshot = tracker.stop()
        print(snapshot)
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or get_logger()
        self._start_mem: Optional[int] = None
        self._start_time: Optional[float] = None
        self._active = False

    @classmethod
    def start(cls, logger: Optional[logging.Logger] = None) -> MemoryTracker:
        """Begin tracking memory."""
        tracker = cls(logger=logger)
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        current, peak = tracemalloc.get_traced_memory()
        tracker._start_mem = current
        tracker._start_time = time.monotonic()
        tracker._active = True
        tracker._logger.debug("Memory tracking started: current=%d bytes", current)
        return tracker

    def stop(self) -> dict[str, Any]:
        """Stop tracking and return a memory report."""
        if not self._active:
            raise RuntimeError("MemoryTracker is not active")
        current, peak = tracemalloc.get_traced_memory()
        elapsed = time.monotonic() - (self._start_time or time.monotonic())
        delta = current - (self._start_mem or 0)
        self._active = False
        report: dict[str, Any] = {
            "current_bytes": current,
            "peak_bytes": peak,
            "delta_bytes": delta,
            "current_mb": round(current / (1024 * 1024), 2),
            "peak_mb": round(peak / (1024 * 1024), 2),
            "delta_mb": round(delta / (1024 * 1024), 2),
            "elapsed_sec": round(elapsed, 3),
        }
        self._logger.info(
            "Memory report: current=%.2fMB peak=%.2fMB delta=%.2fMB elapsed=%.3fs",
            report["current_mb"],
            report["peak_mb"],
            report["delta_mb"],
            elapsed,
        )
        return report

    @staticmethod
    def get_current_usage_mb() -> float:
        """Return current tracked memory usage in MB."""
        if tracemalloc.is_tracing():
            current, _ = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)
        return 0.0


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timing(
    func: Optional[F] = None,
    *,
    logger: Optional[logging.Logger] = None,
    log_level: int = logging.INFO,
) -> Any:
    """Decorator that logs the execution time of a function.

    Can be used bare (``@timing``) or with arguments (``@timing(logger=lg)``).
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or get_logger()
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                _logger.log(
                    log_level,
                    "%s completed in %.4fs",
                    fn.__qualname__,
                    elapsed,
                )
                return result
            except Exception:
                elapsed = time.perf_counter() - start
                _logger.log(
                    logging.ERROR,
                    "%s failed after %.4fs",
                    fn.__qualname__,
                    elapsed,
                )
                raise

        return wrapper  # type: ignore[return-value]

    if func is not None and callable(func):
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Convenience: extra-data context helper
# ---------------------------------------------------------------------------

def log_with_data(
    logger: logging.Logger,
    level: int,
    message: str,
    **data: Any,
) -> None:
    """Emit a log record that carries arbitrary structured *data*.

    The extra data is included in the JSON output by ``_JsonFormatter``.
    """
    record = logger.makeRecord(
        name=logger.name,
        level=level,
        fn="",
        lno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.extra_data = data  # type: ignore[attr-defined]
    logger.handle(record)
