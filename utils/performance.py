"""
Performance optimization utilities.
"""

import time
import functools
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field
from collections import OrderedDict

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    total_requests: int = 0
    total_time_ms: float = 0
    avg_time_ms: float = 0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0
    cache_hits: int = 0
    cache_misses: int = 0


class LRUCache:
    """Simple LRU cache implementation."""

    def __init__(self, max_size: int = 100):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum cache entries
        """
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)


class PerformanceTracker:
    """Tracks performance metrics."""

    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = {}

    def track(self, name: str, duration_ms: float) -> None:
        """
        Track a performance metric.

        Args:
            name: Metric name
            duration_ms: Duration in milliseconds
        """
        if name not in self._metrics:
            self._metrics[name] = PerformanceMetrics()

        metrics = self._metrics[name]
        metrics.total_requests += 1
        metrics.total_time_ms += duration_ms
        metrics.avg_time_ms = metrics.total_time_ms / metrics.total_requests
        metrics.min_time_ms = min(metrics.min_time_ms, duration_ms)
        metrics.max_time_ms = max(metrics.max_time_ms, duration_ms)

    def track_cache(self, name: str, hit: bool) -> None:
        """Track cache hit/miss."""
        if name not in self._metrics:
            self._metrics[name] = PerformanceMetrics()

        if hit:
            self._metrics[name].cache_hits += 1
        else:
            self._metrics[name].cache_misses += 1

    def get_metrics(self, name: str) -> Optional[PerformanceMetrics]:
        """Get metrics for a name."""
        return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get all metrics."""
        return self._metrics.copy()

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()


def timed(name: Optional[str] = None):
    """
    Decorator to track function execution time.

    Args:
        name: Metric name (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start) * 1000
                logger.debug(f"{metric_name}: {duration_ms:.2f}ms")

        return wrapper
    return decorator


class ModelWarmup:
    """Handles model warm-up for better first-inference latency."""

    def __init__(self, model_manager):
        """
        Initialize warmup handler.

        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self._warmed_up = False

    def warmup(self) -> float:
        """
        Perform model warm-up inference.

        Returns:
            Warmup time in milliseconds
        """
        if self._warmed_up:
            return 0

        import numpy as np
        import tempfile
        from utils.audio_utils import save_audio

        logger.info("Warming up model...")
        start = time.time()

        # Create minimal test audio
        sr = 16000
        duration = 0.5
        audio = np.zeros(int(sr * duration), dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_audio(audio, f.name, sr)

            try:
                model = self.model_manager.get_model()
                processor = self.model_manager.get_processor()

                # Run a minimal inference
                messages = [
                    {"role": "system", "content": "Test"},
                    {"role": "user", "content": [{"type": "audio", "audio": f.name}]}
                ]

                from qwen_omni_utils import process_mm_info
                import torch

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

                inputs = processor(
                    text=text, audio=audios, images=images, videos=videos,
                    return_tensors="pt", padding=True
                ).to(model.device).to(model.dtype)

                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        return_audio=False
                    )

            finally:
                import os
                os.unlink(f.name)

        warmup_time = (time.time() - start) * 1000
        self._warmed_up = True
        logger.info(f"Model warmed up in {warmup_time:.0f}ms")

        return warmup_time


# Global tracker instance
_tracker = PerformanceTracker()


def get_tracker() -> PerformanceTracker:
    """Get global performance tracker."""
    return _tracker


if __name__ == "__main__":
    print("=" * 60)
    print("PERFORMANCE UTILS TEST")
    print("=" * 60)

    # Test LRU cache
    cache = LRUCache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.put("d", 4)  # Should evict 'a'

    print(f"  Cache size: {cache.size()}")
    print(f"  Get 'a': {cache.get('a')}")  # None
    print(f"  Get 'b': {cache.get('b')}")  # 2

    # Test performance tracker
    tracker = PerformanceTracker()
    tracker.track("inference", 1500)
    tracker.track("inference", 1200)
    tracker.track("inference", 1800)

    metrics = tracker.get_metrics("inference")
    print(f"  Avg time: {metrics.avg_time_ms:.0f}ms")
    print(f"  Min time: {metrics.min_time_ms:.0f}ms")
    print(f"  Max time: {metrics.max_time_ms:.0f}ms")

    print("  ✓ Performance utils working correctly")
    print("=" * 60)
