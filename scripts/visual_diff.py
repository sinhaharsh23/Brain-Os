#!/usr/bin/env python3
"""Small dependency-free RGBA PNG diff for BrainOS visual regression captures."""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path


def read_png(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    width = height = color_type = bit_depth = None
    compressed = bytearray()
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[:10])
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if (width, height, bit_depth, color_type) == (None, None, None, None):
        raise ValueError(f"missing PNG header: {path}")
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError("visual diff supports only 8-bit RGB/RGBA PNGs")
    channels = 3 if color_type == 2 else 4
    raw = zlib.decompress(bytes(compressed))
    stride = width * channels
    rows: list[bytearray] = []
    cursor = 0
    previous = bytearray(stride)
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        row = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        for index in range(stride):
            left = row[index - channels] if index >= channels else 0
            up = previous[index]
            up_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                row[index] = (row[index] + left) & 255
            elif filter_type == 2:
                row[index] = (row[index] + up) & 255
            elif filter_type == 3:
                row[index] = (row[index] + ((left + up) // 2)) & 255
            elif filter_type == 4:
                estimate = left + up - up_left
                distances = (abs(estimate - left), abs(estimate - up), abs(estimate - up_left))
                predictor = (left, up, up_left)[distances.index(min(distances))]
                row[index] = (row[index] + predictor) & 255
            elif filter_type != 0:
                raise ValueError(f"unsupported PNG filter {filter_type}")
        rows.append(row)
        previous = row
    rgba = bytearray()
    for row in rows:
        if channels == 4:
            rgba.extend(row)
        else:
            for index in range(0, len(row), 3):
                rgba.extend(row[index : index + 3])
                rgba.append(255)
    return int(width), int(height), bytes(rgba)


def write_png(path: Path, width: int, height: int, rgba: bytes) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    scanlines = bytearray()
    stride = width * 4
    for row in range(height):
        scanlines.append(0)
        scanlines.extend(rgba[row * stride : (row + 1) * stride])
    output = b"\x89PNG\r\n\x1a\n"
    output += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    output += chunk(b"IDAT", zlib.compress(bytes(scanlines), 6))
    output += chunk(b"IEND", b"")
    path.write_bytes(output)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: visual_diff.py TARGET ACTUAL DIFF", file=sys.stderr)
        return 2
    target_path, actual_path, diff_path = map(Path, sys.argv[1:])
    width_a, height_a, target = read_png(target_path)
    width_b, height_b, actual = read_png(actual_path)
    if (width_a, height_a) != (width_b, height_b):
        raise SystemExit(f"dimension mismatch: target={width_a}x{height_a}, actual={width_b}x{height_b}")
    diff = bytearray(width_a * height_a * 4)
    different = 0
    for index in range(0, len(target), 4):
        distance = max(abs(target[index + channel] - actual[index + channel]) for channel in range(3))
        if distance > 24:
            different += 1
            diff[index : index + 4] = bytes((255, min(255, distance * 3), 42, 210))
        else:
            average = sum(actual[index : index + 3]) // 3
            diff[index : index + 4] = bytes((average // 3, average // 3, average // 2, 120))
    write_png(diff_path, width_a, height_a, bytes(diff))
    total = width_a * height_a
    print(f"different_pixels={different}/{total} ({different / total:.2%}) threshold=24")
    print(f"target={target_path}\nactual={actual_path}\ndiff={diff_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
