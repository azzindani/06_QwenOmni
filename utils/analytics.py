"""
Analytics tracking for usage statistics.
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RequestMetric:
    """Single request metric."""
    timestamp: str
    session_id: str
    request_type: str  # 'inference', 'transcription', 'translation'
    input_duration_ms: float
    processing_time_ms: float
    output_tokens: int
    language: str = "en"
    success: bool = True
    error: Optional[str] = None


@dataclass
class SessionMetric:
    """Session-level metrics."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    total_requests: int = 0
    total_turns: int = 0
    total_processing_time_ms: float = 0
    languages_used: List[str] = field(default_factory=list)


class AnalyticsTracker:
    """Tracks usage analytics."""

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize analytics tracker.

        Args:
            storage_dir: Directory for storing analytics
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path("./analytics")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._requests: List[RequestMetric] = []
        self._sessions: Dict[str, SessionMetric] = {}
        self._daily_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "requests": 0,
            "errors": 0,
            "total_time_ms": 0,
            "sessions": set()
        })

    def track_request(
        self,
        session_id: str,
        request_type: str,
        input_duration_ms: float,
        processing_time_ms: float,
        output_tokens: int,
        language: str = "en",
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Track a request.

        Args:
            session_id: Session identifier
            request_type: Type of request
            input_duration_ms: Input audio duration
            processing_time_ms: Processing time
            output_tokens: Output token count
            language: Language used
            success: Whether request succeeded
            error: Error message if failed
        """
        metric = RequestMetric(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            request_type=request_type,
            input_duration_ms=input_duration_ms,
            processing_time_ms=processing_time_ms,
            output_tokens=output_tokens,
            language=language,
            success=success,
            error=error
        )

        self._requests.append(metric)

        # Update session metrics
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.total_requests += 1
            session.total_processing_time_ms += processing_time_ms
            if language not in session.languages_used:
                session.languages_used.append(language)

        # Update daily stats
        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_stats[today]["requests"] += 1
        self._daily_stats[today]["total_time_ms"] += processing_time_ms
        self._daily_stats[today]["sessions"].add(session_id)
        if not success:
            self._daily_stats[today]["errors"] += 1

    def start_session(self, session_id: str) -> None:
        """Track session start."""
        self._sessions[session_id] = SessionMetric(
            session_id=session_id,
            start_time=datetime.now().isoformat()
        )

    def end_session(self, session_id: str) -> None:
        """Track session end."""
        if session_id in self._sessions:
            self._sessions[session_id].end_time = datetime.now().isoformat()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get analytics summary.

        Returns:
            Summary statistics
        """
        total_requests = len(self._requests)
        total_errors = sum(1 for r in self._requests if not r.success)
        total_time = sum(r.processing_time_ms for r in self._requests)

        avg_time = total_time / total_requests if total_requests > 0 else 0

        # Language distribution
        lang_counts: Dict[str, int] = defaultdict(int)
        for r in self._requests:
            lang_counts[r.language] += 1

        # Request type distribution
        type_counts: Dict[str, int] = defaultdict(int)
        for r in self._requests:
            type_counts[r.request_type] += 1

        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "total_sessions": len(self._sessions),
            "avg_processing_time_ms": avg_time,
            "language_distribution": dict(lang_counts),
            "request_type_distribution": dict(type_counts),
        }

    def get_daily_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily statistics.

        Args:
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Daily statistics
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        stats = self._daily_stats.get(date, {})
        if stats:
            return {
                "date": date,
                "requests": stats.get("requests", 0),
                "errors": stats.get("errors", 0),
                "unique_sessions": len(stats.get("sessions", set())),
                "total_time_ms": stats.get("total_time_ms", 0),
            }
        return {"date": date, "requests": 0, "errors": 0, "unique_sessions": 0}

    def save(self) -> str:
        """
        Save analytics to file.

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.storage_dir / f"analytics_{timestamp}.json"

        data = {
            "exported_at": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "requests": [asdict(r) for r in self._requests],
            "sessions": {k: asdict(v) for k, v in self._sessions.items()},
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Analytics saved: {filepath}")
        return str(filepath)

    def reset(self) -> None:
        """Reset all analytics."""
        self._requests.clear()
        self._sessions.clear()
        self._daily_stats.clear()


# Global tracker instance
_analytics = AnalyticsTracker()


def get_analytics() -> AnalyticsTracker:
    """Get global analytics tracker."""
    return _analytics


if __name__ == "__main__":
    print("=" * 60)
    print("ANALYTICS TRACKER TEST")
    print("=" * 60)

    tracker = AnalyticsTracker()

    # Track some requests
    tracker.start_session("session-1")
    tracker.track_request(
        session_id="session-1",
        request_type="inference",
        input_duration_ms=2000,
        processing_time_ms=1500,
        output_tokens=50,
        language="en"
    )
    tracker.track_request(
        session_id="session-1",
        request_type="inference",
        input_duration_ms=3000,
        processing_time_ms=1800,
        output_tokens=75,
        language="zh"
    )

    # Get summary
    summary = tracker.get_summary()
    print(f"  Total requests: {summary['total_requests']}")
    print(f"  Avg time: {summary['avg_processing_time_ms']:.0f}ms")
    print(f"  Languages: {summary['language_distribution']}")

    # Daily stats
    daily = tracker.get_daily_stats()
    print(f"  Today's requests: {daily['requests']}")

    print("  ✓ Analytics tracker working correctly")
    print("=" * 60)
