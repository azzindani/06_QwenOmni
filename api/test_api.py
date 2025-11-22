"""
Test API components.

Run with: python -m api.test_api
"""

import sys


def test_schemas():
    """Test API schemas."""
    print("\n[1/3] Testing Schemas...")

    try:
        from api.schemas import (
            HealthResponse, SessionResponse, ErrorResponse,
            WebSocketMessage, to_dict
        )

        # Test HealthResponse
        health = HealthResponse(
            status="healthy",
            model_loaded=True,
            gpu_available=True,
            active_sessions=5
        )
        assert health.status == "healthy"

        # Test to_dict
        d = to_dict(health)
        assert isinstance(d, dict)
        assert d["status"] == "healthy"

        # Test SessionResponse
        session = SessionResponse(
            session_id="test-123",
            created_at="2024-01-01",
            message_count=5,
            turn_count=2
        )
        assert session.turn_count == 2

        # Test ErrorResponse
        error = ErrorResponse(
            error="Test error",
            detail="Details"
        )
        assert error.timestamp is not None

        print(f"  ✓ HealthResponse")
        print(f"  ✓ SessionResponse")
        print(f"  ✓ ErrorResponse")
        print(f"  ✓ to_dict conversion")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_routes():
    """Test API routes."""
    print("\n[2/3] Testing Routes...")

    try:
        from api.routes import APIRoutes
        from core.conversation.session import SessionManager

        manager = SessionManager()
        routes = APIRoutes(manager)

        # Test health
        health = routes.health()
        assert health["status"] == "healthy"

        # Test create session
        session = routes.create_session()
        assert "session_id" in session

        # Test list sessions
        sessions = routes.list_sessions()
        assert sessions["count"] == 1

        # Test get session
        info = routes.get_session(session["session_id"])
        assert info["session_id"] == session["session_id"]

        # Test delete session
        result = routes.delete_session(session["session_id"])
        assert result["status"] == "deleted"

        # Test get nonexistent
        error = routes.get_session("nonexistent")
        assert "error" in error

        print(f"  ✓ Health endpoint")
        print(f"  ✓ Create session")
        print(f"  ✓ List sessions")
        print(f"  ✓ Get/delete session")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_auth():
    """Test authentication."""
    print("\n[3/3] Testing Auth...")

    try:
        from utils.auth import AuthManager

        auth = AuthManager()

        # Generate API key
        key = auth.generate_api_key("test-key", rate_limit=10)
        assert key.key.startswith("qo_")

        # Validate key
        valid = auth.validate_api_key(key.key)
        assert valid is not None

        # Rate limiting
        for i in range(12):
            allowed, info = auth.check_rate_limit(key.key)
            if not allowed:
                assert i >= 10  # Should be limited after 10
                break

        # JWT
        token = auth.generate_jwt("user123")
        assert token is not None

        payload = auth.validate_jwt(token)
        assert payload["sub"] == "user123"

        # Revoke key
        auth.revoke_api_key(key.key)
        assert auth.validate_api_key(key.key) is None

        print(f"  ✓ API key generation")
        print(f"  ✓ Key validation")
        print(f"  ✓ Rate limiting")
        print(f"  ✓ JWT tokens")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all API tests."""
    print("=" * 60)
    print("API COMPONENTS TEST")
    print("=" * 60)

    results = []

    results.append(("Schemas", test_schemas()))
    results.append(("Routes", test_routes()))
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
