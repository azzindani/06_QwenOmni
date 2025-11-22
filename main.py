"""
Main entry point for Qwen Omni Voice Assistant.
"""

import argparse

from config import Config, set_config
from ui.gradio_app import GradioApp
from utils.logger import get_logger
from utils.hardware import get_device_info

logger = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Qwen Omni Voice Assistant")

    # Server options
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create public link")

    # Model options
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Omni-3B", help="Model path")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"])

    # Generation options
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")

    args = parser.parse_args()

    # Create config from arguments
    config = Config(
        model_path=args.model,
        quantization=args.quantization,
        host=args.host,
        port=args.port,
        share=args.share,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )
    set_config(config)

    # Log startup info
    logger.info("=" * 60)
    logger.info("Qwen Omni Voice Assistant")
    logger.info("=" * 60)

    # Log hardware info
    device_info = get_device_info()
    if device_info["cuda_available"]:
        for device in device_info["devices"]:
            logger.info(f"GPU {device['index']}: {device['name']} ({device['total_memory_gb']} GB)")
    else:
        logger.warning("No GPU detected - model will run on CPU (very slow)")

    # Log config
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Quantization: {config.quantization}")
    logger.info(f"Server: {config.host}:{config.port}")

    # Launch app
    app = GradioApp(config)
    app.launch()


if __name__ == "__main__":
    main()
