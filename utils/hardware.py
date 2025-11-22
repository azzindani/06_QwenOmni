"""
Hardware detection utilities.
"""

from typing import Dict, Any, Optional


def detect_gpu() -> bool:
    """
    Detect if CUDA GPU is available.

    Returns:
        True if GPU is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed device information.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "current_device": None,
    }

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()

        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = torch.cuda.current_device()

            for i in range(info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "major": props.major,
                    "minor": props.minor,
                }
                info["devices"].append(device_info)

    except ImportError:
        pass

    return info


def get_optimal_device() -> str:
    """
    Get the optimal device for model loading.

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if detect_gpu():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    print("=" * 60)
    print("HARDWARE DETECTION TEST")
    print("=" * 60)

    gpu_available = detect_gpu()
    print(f"  GPU Available: {gpu_available}")

    info = get_device_info()
    print(f"  Device Count: {info['device_count']}")

    for device in info["devices"]:
        print(f"  GPU {device['index']}: {device['name']} ({device['total_memory_gb']} GB)")

    optimal = get_optimal_device()
    print(f"  Optimal Device: {optimal}")

    print("  ✓ Hardware detection complete")
    print("=" * 60)
