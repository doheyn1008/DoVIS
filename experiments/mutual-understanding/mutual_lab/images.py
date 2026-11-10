"""Deterministic PNG rendering with the standard library; no network or font dependency."""

from __future__ import annotations

import base64
import functools
import struct
import zlib

from .world import Scene

WIDTH, HEIGHT = 256, 176
RGB = {"red": (219, 66, 73), "blue": (54, 110, 209), "green": (31, 151, 110)}
BACKGROUND = (247, 249, 252)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode_png(pixels: bytearray, width: int, height: int) -> bytes:
    stride = width * 3
    raw = b"".join(b"\x00" + pixels[y * stride:(y + 1) * stride] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b""))


def _pixels(scene: Scene) -> bytearray:
    pixels = bytearray(BACKGROUND * (WIDTH * HEIGHT))
    centers = ((65, 88), (191, 88)) if scene.relation == "left_of" else ((128, 43), (128, 133))
    for obj, (cx, cy) in zip((scene.subject, scene.object), centers):
        radius = 28
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                dx, dy = x - cx, y - cy
                inside = (obj.shape == "square" or
                          (obj.shape == "circle" and dx * dx + dy * dy <= radius * radius) or
                          (obj.shape == "triangle" and abs(dx) <= (dy + radius) / 2))
                if inside:
                    index = (y * WIDTH + x) * 3
                    pixels[index:index + 3] = bytes(RGB[obj.color])
    return pixels


@functools.lru_cache(maxsize=256)
def scene_png(scene: Scene) -> bytes:
    return encode_png(_pixels(scene), WIDTH, HEIGHT)


_DIGITS = {
    1: ("010", "110", "010", "010", "111"),
    2: ("111", "001", "111", "100", "111"),
    3: ("111", "001", "111", "001", "111"),
    4: ("101", "101", "111", "001", "001"),
}


@functools.lru_cache(maxsize=256)
def choice_sheet(choices: tuple[Scene, ...]) -> bytes:
    """One image, four opaque numbered panels, no filenames or semantic labels."""
    width, height, label_height = WIDTH * 2, (HEIGHT + 24) * 2, 24
    pixels = bytearray((225, 231, 239) * (width * height))
    for index, scene in enumerate(choices):
        ox, oy = (index % 2) * WIDTH, (index // 2) * (HEIGHT + label_height)
        panel = _pixels(scene)
        for y in range(HEIGHT):
            target = ((oy + label_height + y) * width + ox) * 3
            pixels[target:target + WIDTH * 3] = panel[y * WIDTH * 3:(y + 1) * WIDTH * 3]
        for y, line in enumerate(_DIGITS[index + 1]):
            for x, bit in enumerate(line):
                if bit == "1":
                    for dy in range(3):
                        for dx in range(3):
                            target = ((oy + 4 + y * 3 + dy) * width + ox + 12 + x * 3 + dx) * 3
                            pixels[target:target + 3] = bytes((25, 40, 60))
    return encode_png(pixels, width, height)


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
