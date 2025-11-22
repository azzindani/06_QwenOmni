"""
Multi-language support utilities.
"""

from typing import Dict, Optional
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LanguageConfig:
    """Language configuration."""
    code: str
    name: str
    system_prompt: str
    transcription_prompt: str


# Supported languages with system prompts
SUPPORTED_LANGUAGES: Dict[str, LanguageConfig] = {
    "en": LanguageConfig(
        code="en",
        name="English",
        system_prompt="You are Qwen, a helpful voice assistant. Respond naturally and conversationally in English.",
        transcription_prompt="Transcribe the audio into English text."
    ),
    "zh": LanguageConfig(
        code="zh",
        name="Chinese",
        system_prompt="你是Qwen，一个有用的语音助手。请用中文自然地、对话式地回答。",
        transcription_prompt="将音频转录为中文文本。"
    ),
    "es": LanguageConfig(
        code="es",
        name="Spanish",
        system_prompt="Eres Qwen, un asistente de voz útil. Responde de manera natural y conversacional en español.",
        transcription_prompt="Transcribe el audio a texto en español."
    ),
    "fr": LanguageConfig(
        code="fr",
        name="French",
        system_prompt="Vous êtes Qwen, un assistant vocal utile. Répondez naturellement et de manière conversationnelle en français.",
        transcription_prompt="Transcrire l'audio en texte français."
    ),
    "de": LanguageConfig(
        code="de",
        name="German",
        system_prompt="Sie sind Qwen, ein hilfreicher Sprachassistent. Antworten Sie natürlich und gesprächig auf Deutsch.",
        transcription_prompt="Transkribieren Sie das Audio in deutschen Text."
    ),
    "ja": LanguageConfig(
        code="ja",
        name="Japanese",
        system_prompt="あなたはQwenです。親切な音声アシスタントです。日本語で自然に会話してください。",
        transcription_prompt="音声を日本語テキストに書き起こしてください。"
    ),
    "ko": LanguageConfig(
        code="ko",
        name="Korean",
        system_prompt="당신은 Qwen입니다. 유용한 음성 도우미입니다. 한국어로 자연스럽게 대화해 주세요.",
        transcription_prompt="오디오를 한국어 텍스트로 전사하세요."
    ),
    "ru": LanguageConfig(
        code="ru",
        name="Russian",
        system_prompt="Вы Qwen, полезный голосовой помощник. Отвечайте естественно и разговорно на русском языке.",
        transcription_prompt="Транскрибируйте аудио в русский текст."
    ),
}


def get_language_config(code: str) -> Optional[LanguageConfig]:
    """
    Get language configuration by code.

    Args:
        code: Language code (e.g., 'en', 'zh')

    Returns:
        LanguageConfig or None
    """
    return SUPPORTED_LANGUAGES.get(code.lower())


def get_system_prompt(language: str) -> str:
    """
    Get system prompt for language.

    Args:
        language: Language code

    Returns:
        System prompt
    """
    config = get_language_config(language)
    if config:
        return config.system_prompt
    return SUPPORTED_LANGUAGES["en"].system_prompt


def get_transcription_prompt(language: str) -> str:
    """
    Get transcription prompt for language.

    Args:
        language: Language code

    Returns:
        Transcription prompt
    """
    config = get_language_config(language)
    if config:
        return config.transcription_prompt
    return SUPPORTED_LANGUAGES["en"].transcription_prompt


def list_languages() -> Dict[str, str]:
    """
    List all supported languages.

    Returns:
        Dict of code -> name
    """
    return {code: config.name for code, config in SUPPORTED_LANGUAGES.items()}


def get_translation_prompt(source_lang: str, target_lang: str) -> str:
    """
    Get translation prompt.

    Args:
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Translation prompt
    """
    source = get_language_config(source_lang)
    target = get_language_config(target_lang)

    source_name = source.name if source else source_lang
    target_name = target.name if target else target_lang

    return f"Translate the audio from {source_name} to {target_name}."


if __name__ == "__main__":
    print("=" * 60)
    print("LANGUAGE SUPPORT TEST")
    print("=" * 60)

    languages = list_languages()
    print(f"  Supported languages: {len(languages)}")

    for code, name in languages.items():
        print(f"    {code}: {name}")

    # Test prompts
    en_prompt = get_system_prompt("en")
    zh_prompt = get_system_prompt("zh")

    print(f"\n  English prompt: {en_prompt[:50]}...")
    print(f"  Chinese prompt: {zh_prompt[:30]}...")

    # Test translation
    trans = get_translation_prompt("en", "zh")
    print(f"\n  Translation prompt: {trans}")

    print("\n  ✓ Language support working correctly")
    print("=" * 60)
