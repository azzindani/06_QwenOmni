"""
Authentication and security middleware.
"""

import hashlib
import secrets
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class APIKey:
    """API key data."""
    key: str
    name: str
    created_at: str
    expires_at: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    scopes: List[str] = field(default_factory=lambda: ["read", "write"])
    is_active: bool = True


@dataclass
class RateLimitInfo:
    """Rate limiting information."""
    requests: int = 0
    window_start: float = 0
    blocked_until: float = 0


class AuthManager:
    """Manages authentication and API keys."""

    def __init__(self):
        self._api_keys: Dict[str, APIKey] = {}
        self._rate_limits: Dict[str, RateLimitInfo] = {}
        self._jwt_secret = secrets.token_hex(32)

    def generate_api_key(self, name: str, rate_limit: int = 100,
                         expires_days: Optional[int] = None) -> APIKey:
        """
        Generate a new API key.

        Args:
            name: Key name/description
            rate_limit: Requests per minute
            expires_days: Days until expiration

        Returns:
            New APIKey
        """
        key = f"qo_{secrets.token_hex(24)}"

        expires_at = None
        if expires_days:
            expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()

        api_key = APIKey(
            key=key,
            name=name,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at,
            rate_limit=rate_limit
        )

        self._api_keys[key] = api_key
        logger.info(f"Generated API key: {name}")
        return api_key

    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """
        Validate an API key.

        Args:
            key: API key to validate

        Returns:
            APIKey if valid, None otherwise
        """
        api_key = self._api_keys.get(key)

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        if api_key.expires_at:
            expires = datetime.fromisoformat(api_key.expires_at)
            if datetime.now() > expires:
                return None

        return api_key

    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key."""
        if key in self._api_keys:
            self._api_keys[key].is_active = False
            logger.info(f"Revoked API key: {self._api_keys[key].name}")
            return True
        return False

    def check_rate_limit(self, key: str) -> tuple[bool, dict]:
        """
        Check rate limit for API key.

        Args:
            key: API key

        Returns:
            Tuple of (allowed, info_dict)
        """
        api_key = self._api_keys.get(key)
        if not api_key:
            return False, {"error": "Invalid API key"}

        now = time.time()

        if key not in self._rate_limits:
            self._rate_limits[key] = RateLimitInfo(window_start=now)

        limit_info = self._rate_limits[key]

        # Check if blocked
        if limit_info.blocked_until > now:
            return False, {
                "error": "Rate limited",
                "retry_after": int(limit_info.blocked_until - now)
            }

        # Reset window if needed (1 minute window)
        if now - limit_info.window_start > 60:
            limit_info.requests = 0
            limit_info.window_start = now

        # Check limit
        if limit_info.requests >= api_key.rate_limit:
            limit_info.blocked_until = now + 60
            return False, {
                "error": "Rate limited",
                "retry_after": 60
            }

        limit_info.requests += 1

        return True, {
            "remaining": api_key.rate_limit - limit_info.requests,
            "reset": int(limit_info.window_start + 60 - now)
        }

    def generate_jwt(self, user_id: str, expires_hours: int = 24) -> str:
        """
        Generate a JWT token.

        Args:
            user_id: User identifier
            expires_hours: Hours until expiration

        Returns:
            JWT token string
        """
        import base64
        import json
        import hmac

        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": user_id,
            "iat": int(time.time()),
            "exp": int(time.time() + expires_hours * 3600)
        }

        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).rstrip(b'=').decode()

        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).rstrip(b'=').decode()

        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self._jwt_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()

        signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b'=').decode()

        return f"{message}.{signature_b64}"

    def validate_jwt(self, token: str) -> Optional[dict]:
        """
        Validate a JWT token.

        Args:
            token: JWT token

        Returns:
            Payload dict if valid, None otherwise
        """
        import base64
        import json
        import hmac

        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_sig = hmac.new(
                self._jwt_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()

            # Pad base64
            signature_b64 += '=' * (4 - len(signature_b64) % 4)
            actual_sig = base64.urlsafe_b64decode(signature_b64)

            if not hmac.compare_digest(expected_sig, actual_sig):
                return None

            # Decode payload
            payload_b64 += '=' * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))

            # Check expiration
            if payload.get('exp', 0) < time.time():
                return None

            return payload

        except Exception:
            return None

    def list_api_keys(self) -> List[dict]:
        """List all API keys (without actual key values)."""
        return [
            {
                "name": k.name,
                "created_at": k.created_at,
                "expires_at": k.expires_at,
                "rate_limit": k.rate_limit,
                "is_active": k.is_active,
                "key_prefix": k.key[:10] + "..."
            }
            for k in self._api_keys.values()
        ]


# Global auth manager
_auth_manager = AuthManager()


def get_auth_manager() -> AuthManager:
    """Get global auth manager."""
    return _auth_manager


def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Check for API key in kwargs or headers
        api_key = kwargs.get('api_key')
        if not api_key:
            return {"error": "API key required"}, 401

        auth = get_auth_manager()
        if not auth.validate_api_key(api_key):
            return {"error": "Invalid API key"}, 403

        allowed, info = auth.check_rate_limit(api_key)
        if not allowed:
            return info, 429

        return f(*args, **kwargs)
    return wrapper


if __name__ == "__main__":
    print("=" * 60)
    print("AUTH MANAGER TEST")
    print("=" * 60)

    auth = AuthManager()

    # Generate API key
    key = auth.generate_api_key("test-key", rate_limit=10)
    print(f"  Generated key: {key.key[:20]}...")

    # Validate
    valid = auth.validate_api_key(key.key)
    print(f"  Valid: {valid is not None}")

    # Rate limit
    for i in range(12):
        allowed, info = auth.check_rate_limit(key.key)
        if not allowed:
            print(f"  Rate limited at request {i+1}")
            break

    # JWT
    token = auth.generate_jwt("user123")
    print(f"  JWT: {token[:50]}...")

    payload = auth.validate_jwt(token)
    print(f"  JWT valid: {payload is not None}")

    print("  ✓ Auth manager working correctly")
    print("=" * 60)
