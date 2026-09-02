#!/usr/bin/env python3
"""Plot short, evenly spaced pose chunks from two place-field rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


XBOUND = (100.0, 2000.0)
YBOUND = (100.0, 2000.0)
PANEL_SIZE = 660
PANEL_GAP = 90
LEFT_MARGIN = 90
TOP_MARGIN = 170
ROW_GAP = 110


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-length", type=int, default=300)
    parser.add_argument("--num-chunks", type=int, default=4)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def load_pose(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"frame", "x", "y", "num_traj"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.sort_values("frame").reset_index(drop=True)


def chunk_starts(length: int, chunk_length: int, num_chunks: int) -> np.ndarray:
    if chunk_length < 2 or chunk_length > length:
        raise ValueError(f"chunk length {chunk_length} is invalid for {length} rows")
    return np.linspace(0, length - chunk_length, num_chunks, dtype=int)


def viridis(value: float) -> tuple[int, int, int]:
    stops = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37))
    scaled = min(max(value, 0.0), 1.0) * (len(stops) - 1)
    index = min(int(scaled), len(stops) - 2)
    fraction = scaled - index
    return tuple(round(stops[index][channel] * (1.0 - fraction) + stops[index + 1][channel] * fraction) for channel in range(3))


def position_to_pixel(x: float, y: float, left: int, top: int) -> tuple[int, int]:
    px = left + round((x - XBOUND[0]) / (XBOUND[1] - XBOUND[0]) * PANEL_SIZE)
    py = top + PANEL_SIZE - round((y - YBOUND[0]) / (YBOUND[1] - YBOUND[0]) * PANEL_SIZE)
    return px, py


def draw_chunk(
    draw: ImageDraw.ImageDraw,
    frame: pd.DataFrame,
    start: int,
    length: int,
    left: int,
    top: int,
) -> None:
    chunk = frame.iloc[start : start + length]
    points = chunk[["x", "y"]].to_numpy(dtype=np.float64)
    trajectories = chunk["num_traj"].to_numpy()
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    discontinuities = (trajectories[:-1] != trajectories[1:]) | (distances > 150.0)

    draw.rectangle((left, top, left + PANEL_SIZE, top + PANEL_SIZE), fill="#f8fafc", outline="#475569", width=2)
    for coordinate in range(100, 2001, 200):
        grid_x, _ = position_to_pixel(coordinate, 100, left, top)
        _, grid_y = position_to_pixel(100, coordinate, left, top)
        draw.line((grid_x, top, grid_x, top + PANEL_SIZE), fill="#e2e8f0", width=1)
        draw.line((left, grid_y, left + PANEL_SIZE, grid_y), fill="#e2e8f0", width=1)

    for index in range(len(points) - 1):
        if discontinuities[index]:
            continue
        first = position_to_pixel(points[index, 0], points[index, 1], left, top)
        second = position_to_pixel(points[index + 1, 0], points[index + 1, 1], left, top)
        draw.line((*first, *second), fill=viridis(index / max(1, len(points) - 2)), width=3)

    boundaries = np.concatenate(([0], np.flatnonzero(discontinuities) + 1, [len(points)]))
    for boundary_start, boundary_end in zip(boundaries[:-1], boundaries[1:]):
        start_point = position_to_pixel(*points[boundary_start], left, top)
        end_point = position_to_pixel(*points[boundary_end - 1], left, top)
        radius = 7
        draw.ellipse(
            (start_point[0] - radius, start_point[1] - radius, start_point[0] + radius, start_point[1] + radius),
            fill="#16a34a",
            outline="white",
            width=2,
        )
        cross = 8
        draw.line((end_point[0] - cross, end_point[1] - cross, end_point[0] + cross, end_point[1] + cross), fill="#dc2626", width=3)
        draw.line((end_point[0] - cross, end_point[1] + cross, end_point[0] + cross, end_point[1] - cross), fill="#dc2626", width=3)

    tick_font = font(15)
    for coordinate in range(100, 2001, 400):
        tick_x, _ = position_to_pixel(coordinate, 100, left, top)
        _, tick_y = position_to_pixel(100, coordinate, left, top)
        label = str(coordinate)
        draw.text((tick_x - 18, top + PANEL_SIZE + 6), label, fill="#475569", font=tick_font)
        draw.text((left - 53, tick_y - 8), label, fill="#475569", font=tick_font)

    first_frame = int(chunk.iloc[0]["frame"])
    last_frame = int(chunk.iloc[-1]["frame"])
    panel_title = f"decisions {first_frame:,}–{last_frame:,}"
    title_font = font(21, bold=True)
    title_box = draw.textbbox((0, 0), panel_title, font=title_font)
    draw.text((left + (PANEL_SIZE - (title_box[2] - title_box[0])) / 2, top - 34), panel_title, fill="#0f172a", font=title_font)


def main() -> None:
    args = parse_args()
    left_frame = load_pose(args.left)
    right_frame = load_pose(args.right)
    common_length = min(len(left_frame), len(right_frame))
    starts = chunk_starts(common_length, args.chunk_length, args.num_chunks)

    width = LEFT_MARGIN * 2 + PANEL_SIZE * 2 + PANEL_GAP
    height = TOP_MARGIN + args.num_chunks * PANEL_SIZE + (args.num_chunks - 1) * ROW_GAP + 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = font(33, bold=True)
    label_font = font(26, bold=True)
    note_font = font(17)
    title_box = draw.textbbox((0, 0), args.title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 24), args.title, fill="#0f172a", font=title_font)

    left_center = LEFT_MARGIN + PANEL_SIZE / 2
    right_left = LEFT_MARGIN + PANEL_SIZE + PANEL_GAP
    right_center = right_left + PANEL_SIZE / 2
    for label, center in ((args.left_label, left_center), (args.right_label, right_center)):
        label_box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((center - (label_box[2] - label_box[0]) / 2, 92), label, fill="#1e293b", font=label_font)

    for row, start in enumerate(starts):
        top = TOP_MARGIN + row * (PANEL_SIZE + ROW_GAP)
        draw_chunk(draw, left_frame, int(start), args.chunk_length, LEFT_MARGIN, top)
        draw_chunk(draw, right_frame, int(start), args.chunk_length, right_left, top)

    note = (
        f"Four evenly spaced {args.chunk_length}-decision windows. Color runs early → late within each window. "
        "Green circles mark segment starts; red crosses mark segment ends. Episode resets are not connected."
    )
    note_box = draw.textbbox((0, 0), note, font=note_font)
    draw.text(((width - (note_box[2] - note_box[0])) / 2, height - 62), note, fill="#334155", font=note_font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
