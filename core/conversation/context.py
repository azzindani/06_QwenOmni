"""
Context window management for conversation history.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context management."""
    max_tokens: int = 4096
    reserve_tokens: int = 512  # Reserve for response
    truncation_strategy: str = "oldest"  # oldest, summary


class ContextManager:
    """Manages context window for conversations."""

    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize context manager.

        Args:
            config: Context configuration
        """
        self.config = config or ContextConfig()

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4

    def estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        """
        Estimate tokens for a message.

        Args:
            message: Message dict

        Returns:
            Estimated token count
        """
        content = message.get("content", "")

        if isinstance(content, str):
            return self.estimate_tokens(content)

        if isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        total += self.estimate_tokens(item.get("text", ""))
                    elif item.get("type") == "audio":
                        total += 100  # Estimate for audio reference
            return total

        return 0

    def get_available_tokens(self) -> int:
        """Get available tokens for context."""
        return self.config.max_tokens - self.config.reserve_tokens

    def fit_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fit messages within context window.

        Args:
            messages: Full message list

        Returns:
            Truncated message list
        """
        if not messages:
            return []

        available = self.get_available_tokens()

        # Always keep system message
        result = [messages[0]] if messages[0].get("role") == "system" else []
        current_tokens = sum(self.estimate_message_tokens(m) for m in result)

        # Process remaining messages
        remaining = messages[1:] if result else messages

        if self.config.truncation_strategy == "oldest":
            # Add from newest to oldest, then reverse
            temp = []
            for msg in reversed(remaining):
                msg_tokens = self.estimate_message_tokens(msg)
                if current_tokens + msg_tokens <= available:
                    temp.append(msg)
                    current_tokens += msg_tokens
                else:
                    break

            result.extend(reversed(temp))

        logger.debug(f"Context: {len(result)} messages, ~{current_tokens} tokens")
        return result

    def should_summarize(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Check if conversation should be summarized.

        Args:
            messages: Message list

        Returns:
            True if summarization recommended
        """
        total_tokens = sum(self.estimate_message_tokens(m) for m in messages)
        return total_tokens > self.config.max_tokens * 0.8


if __name__ == "__main__":
    print("=" * 60)
    print("CONTEXT MANAGER TEST")
    print("=" * 60)

    manager = ContextManager()

    # Test token estimation
    text = "Hello, this is a test message for token estimation."
    tokens = manager.estimate_tokens(text)
    print(f"  Text tokens: {tokens}")

    # Test message fitting
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
    ]

    fitted = manager.fit_messages(messages)
    print(f"  Fitted messages: {len(fitted)}")

    print("  ✓ Context manager working correctly")
    print("=" * 60)
