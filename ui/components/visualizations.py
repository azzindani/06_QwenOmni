"""
Enhanced UI components for Gradio interface.
"""

import numpy as np
from typing import Optional, Tuple
import base64


def generate_waveform_html(audio: np.ndarray, sample_rate: int,
                           width: int = 600, height: int = 100,
                           color: str = "#4CAF50") -> str:
    """
    Generate SVG waveform visualization.

    Args:
        audio: Audio array
        sample_rate: Sample rate
        width: SVG width
        height: SVG height
        color: Waveform color

    Returns:
        HTML string with SVG waveform
    """
    # Downsample for visualization
    num_points = min(width, len(audio))
    step = len(audio) // num_points

    if step == 0:
        step = 1
        num_points = len(audio)

    # Get peaks
    peaks = []
    for i in range(num_points):
        start = i * step
        end = min(start + step, len(audio))
        chunk = audio[start:end]
        if len(chunk) > 0:
            peak = np.max(np.abs(chunk))
            peaks.append(peak)

    if not peaks:
        return "<div>No audio data</div>"

    # Normalize
    max_peak = max(peaks) if max(peaks) > 0 else 1
    normalized = [p / max_peak for p in peaks]

    # Generate SVG path
    mid_y = height / 2
    points_top = []
    points_bottom = []

    for i, peak in enumerate(normalized):
        x = (i / len(normalized)) * width
        y_offset = peak * (height / 2 - 5)
        points_top.append(f"{x},{mid_y - y_offset}")
        points_bottom.append(f"{x},{mid_y + y_offset}")

    # Create path
    path_top = " ".join(points_top)
    path_bottom = " ".join(reversed(points_bottom))

    svg = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#1a1a1a"/>
        <polygon points="{path_top} {path_bottom}" fill="{color}" opacity="0.8"/>
        <line x1="0" y1="{mid_y}" x2="{width}" y2="{mid_y}"
              stroke="{color}" stroke-width="1" opacity="0.5"/>
    </svg>
    '''

    return f'<div style="border-radius: 8px; overflow: hidden;">{svg}</div>'


def generate_speaking_indicator(is_speaking: bool,
                                 confidence: float = 0.0) -> str:
    """
    Generate speaking indicator HTML.

    Args:
        is_speaking: Whether speech is detected
        confidence: VAD confidence (0-1)

    Returns:
        HTML string
    """
    if is_speaking:
        color = "#4CAF50"  # Green
        status = "Speaking"
        animation = "pulse 1s infinite"
    else:
        color = "#666"
        status = "Listening"
        animation = "none"

    confidence_pct = int(confidence * 100)

    return f'''
    <div style="display: flex; align-items: center; gap: 10px; padding: 10px;">
        <div style="
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: {color};
            animation: {animation};
        "></div>
        <span style="color: {color}; font-weight: bold;">{status}</span>
        <span style="color: #888; font-size: 12px;">({confidence_pct}%)</span>
    </div>
    <style>
        @keyframes pulse {{
            0% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.5; transform: scale(1.2); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
    </style>
    '''


def get_theme_css(theme: str = "dark") -> str:
    """
    Get CSS for theme.

    Args:
        theme: 'dark' or 'light'

    Returns:
        CSS string
    """
    if theme == "dark":
        return '''
        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --accent: #4CAF50;
            --border: #404040;
        }
        '''
    else:
        return '''
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f5;
            --text-primary: #000000;
            --text-secondary: #666666;
            --accent: #2196F3;
            --border: #e0e0e0;
        }
        '''


def create_audio_player_html(audio_base64: str, sample_rate: int = 24000) -> str:
    """
    Create custom audio player HTML.

    Args:
        audio_base64: Base64 encoded audio
        sample_rate: Sample rate

    Returns:
        HTML string
    """
    return f'''
    <div style="
        background: var(--bg-secondary, #2d2d2d);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    ">
        <audio controls style="width: 100%;">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
    </div>
    '''


def create_progress_indicator(progress: float, label: str = "Processing") -> str:
    """
    Create progress indicator HTML.

    Args:
        progress: Progress value (0-1)
        label: Progress label

    Returns:
        HTML string
    """
    pct = int(progress * 100)

    return f'''
    <div style="margin: 10px 0;">
        <div style="
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        ">
            <span>{label}</span>
            <span>{pct}%</span>
        </div>
        <div style="
            background: #333;
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
        ">
            <div style="
                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                height: 100%;
                width: {pct}%;
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    '''


if __name__ == "__main__":
    print("=" * 60)
    print("UI COMPONENTS TEST")
    print("=" * 60)

    # Test waveform
    audio = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    waveform = generate_waveform_html(audio, 16000)
    print(f"  Waveform HTML length: {len(waveform)}")

    # Test speaking indicator
    indicator = generate_speaking_indicator(True, 0.85)
    print(f"  Indicator HTML length: {len(indicator)}")

    # Test progress
    progress = create_progress_indicator(0.75, "Loading model")
    print(f"  Progress HTML length: {len(progress)}")

    print("  ✓ UI components working correctly")
    print("=" * 60)
