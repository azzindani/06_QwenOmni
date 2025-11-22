"""
Load testing and performance benchmarking scripts.
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_s: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    errors: List[str] = field(default_factory=list)


class LoadTester:
    """Load testing utility."""

    def __init__(self):
        self._results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        name: str,
        func,
        num_requests: int = 100,
        concurrency: int = 10,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a benchmark test.

        Args:
            name: Benchmark name
            func: Function to benchmark
            num_requests: Total number of requests
            concurrency: Concurrent requests
            **kwargs: Arguments to pass to func

        Returns:
            BenchmarkResult
        """
        latencies = []
        errors = []
        successful = 0
        failed = 0

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []

            for _ in range(num_requests):
                future = executor.submit(self._timed_call, func, kwargs)
                futures.append(future)

            for future in futures:
                latency, error = future.result()
                if error:
                    failed += 1
                    errors.append(str(error))
                else:
                    successful += 1
                    latencies.append(latency)

        total_time = time.time() - start_time

        # Calculate statistics
        if latencies:
            sorted_latencies = sorted(latencies)
            result = BenchmarkResult(
                name=name,
                total_requests=num_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time_s=total_time,
                avg_latency_ms=statistics.mean(latencies),
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                p50_latency_ms=sorted_latencies[len(sorted_latencies) // 2],
                p95_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.95)],
                p99_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.99)],
                requests_per_second=successful / total_time if total_time > 0 else 0,
                errors=errors[:10]  # Keep first 10 errors
            )
        else:
            result = BenchmarkResult(
                name=name,
                total_requests=num_requests,
                successful_requests=0,
                failed_requests=failed,
                total_time_s=total_time,
                avg_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                requests_per_second=0,
                errors=errors[:10]
            )

        self._results.append(result)
        return result

    def _timed_call(self, func, kwargs):
        """Call function and return (latency_ms, error)."""
        start = time.time()
        try:
            func(**kwargs)
            latency = (time.time() - start) * 1000
            return latency, None
        except Exception as e:
            return 0, e

    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        return self._results

    def print_report(self) -> str:
        """Generate and print benchmark report."""
        lines = [
            "=" * 70,
            "BENCHMARK REPORT",
            "=" * 70,
        ]

        for result in self._results:
            lines.extend([
                f"\n{result.name}",
                "-" * 40,
                f"Total Requests:     {result.total_requests}",
                f"Successful:         {result.successful_requests}",
                f"Failed:             {result.failed_requests}",
                f"Total Time:         {result.total_time_s:.2f}s",
                f"Requests/Second:    {result.requests_per_second:.2f}",
                f"",
                f"Latency (ms):",
                f"  Average:          {result.avg_latency_ms:.2f}",
                f"  Min:              {result.min_latency_ms:.2f}",
                f"  Max:              {result.max_latency_ms:.2f}",
                f"  P50:              {result.p50_latency_ms:.2f}",
                f"  P95:              {result.p95_latency_ms:.2f}",
                f"  P99:              {result.p99_latency_ms:.2f}",
            ])

            if result.errors:
                lines.append(f"\nFirst errors:")
                for err in result.errors[:3]:
                    lines.append(f"  - {err[:80]}")

        lines.append("\n" + "=" * 70)

        report = "\n".join(lines)
        print(report)
        return report


def benchmark_audio_preprocessing(num_requests: int = 100) -> BenchmarkResult:
    """Benchmark audio preprocessing."""
    from core.audio.preprocessor import AudioPreprocessor

    preprocessor = AudioPreprocessor()

    # Generate test audio
    test_audio = np.random.randn(16000).astype(np.float32)  # 1 second

    def process():
        preprocessor.process(test_audio, 16000)

    tester = LoadTester()
    result = tester.run_benchmark(
        "Audio Preprocessing",
        process,
        num_requests=num_requests,
        concurrency=10
    )

    return result


def benchmark_session_management(num_requests: int = 1000) -> BenchmarkResult:
    """Benchmark session management."""
    from core.conversation.session import SessionManager, ConversationSession

    manager = SessionManager(max_sessions=10000)

    def create_and_use():
        session = manager.create_session()
        session.add_user_text("Test message")
        session.add_assistant_message("Test response")
        manager.delete_session(session.session_id)

    tester = LoadTester()
    result = tester.run_benchmark(
        "Session Management",
        create_and_use,
        num_requests=num_requests,
        concurrency=50
    )

    return result


def benchmark_vad_detection(num_requests: int = 100) -> BenchmarkResult:
    """Benchmark VAD detection."""
    from core.audio.vad import VADDetector

    detector = VADDetector()
    test_audio = np.random.randn(1600).astype(np.float32)  # 100ms chunk

    def detect():
        detector.process_chunk(test_audio)

    tester = LoadTester()
    result = tester.run_benchmark(
        "VAD Detection",
        detect,
        num_requests=num_requests,
        concurrency=10
    )

    return result


def run_all_benchmarks() -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    results = []

    print("Running benchmarks...\n")

    # Preprocessing benchmark
    result = benchmark_audio_preprocessing(100)
    results.append(result)
    print(f"✓ {result.name}: {result.avg_latency_ms:.2f}ms avg")

    # Session management benchmark
    result = benchmark_session_management(1000)
    results.append(result)
    print(f"✓ {result.name}: {result.avg_latency_ms:.2f}ms avg")

    # VAD benchmark (if available)
    try:
        result = benchmark_vad_detection(100)
        results.append(result)
        print(f"✓ {result.name}: {result.avg_latency_ms:.2f}ms avg")
    except Exception as e:
        print(f"✗ VAD Detection: {e}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("LOAD TESTING")
    print("=" * 60)

    results = run_all_benchmarks()

    tester = LoadTester()
    tester._results = results
    tester.print_report()
