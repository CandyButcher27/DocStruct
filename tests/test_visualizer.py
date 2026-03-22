"""Tests for side-by-side visualizer helpers."""

from PIL import Image, ImageFont

from visualize_overlay import _compose_side_by_side


def test_compose_side_by_side_dimensions():
    left = Image.new("RGB", (100, 80), (255, 255, 255))
    right = Image.new("RGB", (120, 60), (255, 255, 255))
    font = ImageFont.load_default()
    combined = _compose_side_by_side(left, right, "LEFT", "RIGHT", font)
    assert combined.width == 100 + 120 + 20
    assert combined.height == max(80, 60) + 24
