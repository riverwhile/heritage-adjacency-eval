#!/usr/bin/env python3
"""Extract publication-safe stream metadata from mapped MP4 source files."""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path


def boxes(data: bytes, start: int = 0, end: int | None = None):
    end = len(data) if end is None else end
    while start + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, start)
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, start + 8)[0]
            header = 16
        elif size == 0:
            size = end - start
        if size < header or start + size > end:
            raise ValueError("invalid MP4 box")
        yield kind, start + header, start + size
        start += size


def time_fields(data: bytes, offset: int) -> tuple[int, int]:
    position = offset + (20 if data[offset] == 1 else 12)
    scale = struct.unpack_from(">I", data, position)[0]
    duration = struct.unpack_from(">Q" if data[offset] == 1 else ">I", data, position + 4)[0]
    return scale, duration


def inspect_video(path: Path) -> dict[str, float | int]:
    data = path.read_bytes()
    movie = next((start, end) for kind, start, end in boxes(data) if kind == b"moov")
    for kind, start, end in boxes(data, *movie):
        if kind != b"trak":
            continue
        track = {child: (left, right) for child, left, right in boxes(data, start, end)}
        media = {child: (left, right) for child, left, right in boxes(data, *track[b"mdia"])}
        handler = media[b"hdlr"][0]
        if data[handler + 8 : handler + 12] != b"vide":
            continue
        scale, duration = time_fields(data, media[b"mdhd"][0])
        track_header_end = track[b"tkhd"][1]
        minimum = {child: (left, right) for child, left, right in boxes(data, *media[b"minf"])}
        sample_table = {child: (left, right) for child, left, right in boxes(data, *minimum[b"stbl"])}
        sample_size = sample_table[b"stsz"][0]
        sample_count = struct.unpack_from(">I", data, sample_size + 8)[0]
        timing = sample_table[b"stts"][0]
        entry_count = struct.unpack_from(">I", data, timing + 4)[0]
        entries = [struct.unpack_from(">II", data, timing + 8 + 8 * index) for index in range(entry_count)]
        if sum(count for count, _ in entries) != sample_count or sum(count * delta for count, delta in entries) != duration:
            raise ValueError(f"inconsistent sample timing in {path.name}")
        return {
            "encoded_width": int(struct.unpack_from(">I", data, track_header_end - 8)[0] / 65536),
            "encoded_height": int(struct.unpack_from(">I", data, track_header_end - 4)[0] / 65536),
            "video_fps": sample_count * scale / duration,
            "duration_s": duration / scale,
            "video_samples": sample_count,
        }
    raise ValueError(f"no video track found in {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("mapping_csv", type=Path, help="CSV columns: object, in_filename, out_filename")
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    with args.mapping_csv.open(newline="", encoding="utf-8-sig") as handle:
        mapping = list(csv.DictReader(handle))
    rows = []
    for item in mapping:
        inside = inspect_video(args.source_dir / item["in_filename"])
        outside = inspect_video(args.source_dir / item["out_filename"])
        for key in ("encoded_width", "encoded_height", "video_fps"):
            if abs(float(inside[key]) - float(outside[key])) > 1e-9:
                raise ValueError(f"IN/OUT {key} mismatch for {item['object']}")
        rows.append(
            {
                "object": item["object"],
                "encoded_width": inside["encoded_width"],
                "encoded_height": inside["encoded_height"],
                "video_fps": f"{float(inside['video_fps']):.6f}",
                "in_duration_s": f"{float(inside['duration_s']):.6f}",
                "out_duration_s": f"{float(outside['duration_s']):.6f}",
                "in_video_samples": inside["video_samples"],
                "out_video_samples": outside["video_samples"],
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote metadata for {len(rows)} object pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
