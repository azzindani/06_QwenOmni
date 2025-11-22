"""
Test utilities.

Run with: python -m utils.test_utils
"""

import sys


def test_logger():
    """Test logger."""
    print("\n[1/5] Testing Logger...")

    try:
        from utils.logger import get_logger

        logger = get_logger("test")
        assert logger is not None

        # Test logging (will print to stdout)
        logger.info("Test message")

        print(f"  ✓ Logger created")
        print(f"  ✓ Logging working")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_languages():
    """Test language support."""
    print("\n[2/5] Testing Languages...")

    try:
        from utils.languages import (
            list_languages, get_language_config,
            get_system_prompt, get_translation_prompt
        )

        # List languages
        langs = list_languages()
        assert len(langs) >= 8

        # Get config
        en = get_language_config("en")
        assert en is not None
        assert en.code == "en"

        zh = get_language_config("zh")
        assert zh is not None

        # Get prompts
        prompt = get_system_prompt("en")
        assert "Qwen" in prompt

        trans = get_translation_prompt("en", "zh")
        assert "English" in trans and "Chinese" in trans

        print(f"  ✓ {len(langs)} languages supported")
        print(f"  ✓ Language configs")
        print(f"  ✓ System prompts")
        print(f"  ✓ Translation prompts")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_performance():
    """Test performance utilities."""
    print("\n[3/5] Testing Performance...")

    try:
        from utils.performance import LRUCache, PerformanceTracker

        # Test cache
        cache = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        assert cache.get("a") == 1
        assert cache.size() == 3

        cache.put("d", 4)  # Should evict oldest
        assert cache.size() == 3

        # Test tracker
        tracker = PerformanceTracker()
        tracker.track("test", 100)
        tracker.track("test", 200)
        tracker.track("test", 150)

        metrics = tracker.get_metrics("test")
        assert metrics.total_requests == 3
        assert metrics.avg_time_ms == 150
        assert metrics.min_time_ms == 100
        assert metrics.max_time_ms == 200

        print(f"  ✓ LRU Cache")
        print(f"  ✓ Performance Tracker")
        print(f"  ✓ Metrics calculation")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_analytics():
    """Test analytics tracking."""
    print("\n[4/5] Testing Analytics...")

    try:
        from utils.analytics import AnalyticsTracker

        tracker = AnalyticsTracker()

        # Start session
        tracker.start_session("session-1")

        # Track requests
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
        assert summary["total_requests"] == 2
        assert summary["total_sessions"] == 1
        assert "en" in summary["language_distribution"]

        print(f"  ✓ Session tracking")
        print(f"  ✓ Request tracking")
        print(f"  ✓ Summary generation")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_auth():
    """Test authentication."""
    print("\n[5/5] Testing Auth...")

    try:
        from utils.auth import AuthManager

        auth = AuthManager()

        # API key
        key = auth.generate_api_key("test", rate_limit=100)
        assert auth.validate_api_key(key.key) is not None

        # JWT
        token = auth.generate_jwt("user1")
        payload = auth.validate_jwt(token)
        assert payload["sub"] == "user1"

        print(f"  ✓ API key management")
        print(f"  ✓ JWT tokens")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all utility tests."""
    print("=" * 60)
    print("UTILITIES TEST")
    print("=" * 60)

    results = []

    results.append(("Logger", test_logger()))
    results.append(("Languages", test_languages()))
    results.append(("Performance", test_performance()))
    results.append(("Analytics", test_analytics()))
    results.append(("Auth", test_auth()))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
