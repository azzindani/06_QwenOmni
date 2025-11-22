"""
Unit tests for API schemas.
"""

import pytest


@pytest.mark.unit
class TestSchemas:
    """Tests for API schemas."""

    def test_health_response(self):
        """Test HealthResponse schema."""
        from api.schemas import HealthResponse

        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            gpu_available=True,
            active_sessions=5
        )

        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.timestamp is not None

    def test_session_response(self):
        """Test SessionResponse schema."""
        from api.schemas import SessionResponse

        response = SessionResponse(
            session_id="test-123",
            created_at="2024-01-01T00:00:00",
            message_count=5,
            turn_count=2
        )

        assert response.session_id == "test-123"
        assert response.turn_count == 2

    def test_error_response(self):
        """Test ErrorResponse schema."""
        from api.schemas import ErrorResponse

        response = ErrorResponse(
            error="Not found",
            detail="Session not found"
        )

        assert response.error == "Not found"
        assert response.detail == "Session not found"
        assert response.timestamp is not None

    def test_websocket_message(self):
        """Test WebSocketMessage schema."""
        from api.schemas import WebSocketMessage

        message = WebSocketMessage(
            type="response",
            data={"text": "Hello"},
            session_id="test-123"
        )

        assert message.type == "response"
        assert message.data["text"] == "Hello"

    def test_to_dict(self):
        """Test to_dict function."""
        from api.schemas import HealthResponse, to_dict

        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            gpu_available=False,
            active_sessions=3
        )

        d = to_dict(response)

        assert isinstance(d, dict)
        assert d["status"] == "healthy"
        assert d["model_loaded"] is True


@pytest.mark.unit
class TestAPIRoutes:
    """Tests for API routes."""

    def test_health_endpoint(self):
        """Test health endpoint."""
        from api.routes import APIRoutes
        from core.conversation.session import SessionManager

        manager = SessionManager()
        routes = APIRoutes(manager)

        health = routes.health()

        assert health["status"] == "healthy"
        assert "model_loaded" in health
        assert "gpu_available" in health

    def test_create_session_endpoint(self):
        """Test create session endpoint."""
        from api.routes import APIRoutes
        from core.conversation.session import SessionManager

        manager = SessionManager()
        routes = APIRoutes(manager)

        result = routes.create_session()

        assert "session_id" in result
        assert "created_at" in result

    def test_list_sessions_endpoint(self):
        """Test list sessions endpoint."""
        from api.routes import APIRoutes
        from core.conversation.session import SessionManager

        manager = SessionManager()
        routes = APIRoutes(manager)

        # Create some sessions
        routes.create_session()
        routes.create_session()

        result = routes.list_sessions()

        assert result["count"] == 2
        assert len(result["sessions"]) == 2

    def test_delete_session_endpoint(self):
        """Test delete session endpoint."""
        from api.routes import APIRoutes
        from core.conversation.session import SessionManager

        manager = SessionManager()
        routes = APIRoutes(manager)

        session = routes.create_session()
        result = routes.delete_session(session["session_id"])

        assert result["status"] == "deleted"

    def test_get_nonexistent_session(self):
        """Test getting non-existent session."""
        from api.routes import APIRoutes
        from core.conversation.session import SessionManager

        manager = SessionManager()
        routes = APIRoutes(manager)

        result = routes.get_session("nonexistent")

        assert "error" in result
