#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

DEFAULT_QUESTION = "Question"
DEFAULT_OPTIONS = {
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D",
}
CHOICES = ("A", "B", "C", "D")
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class MotionSpec:
    zoom_start: float
    zoom_end: float
    x_shift_start: float
    x_shift_end: float
    y_shift_start: float
    y_shift_end: float


def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def iter_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = sorted(input_dir.glob(pattern))
    return [p for p in files if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    if words:
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            test = f"{current} {word}"
            bbox = draw.textbbox((0, 0), test, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    if not text:
        return [""]

    chars = list(text)
    lines: list[str] = []
    current = chars[0]
    for ch in chars[1:]:
        test = f"{current}{ch}"
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = test
        else:
            lines.append(current)
            current = ch
    lines.append(current)
    return lines


def ease_in_out(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


def render_base_frame(
    source: Image.Image,
    width: int,
    height: int,
    motion: MotionSpec,
    frame_idx: int,
    num_frames: int,
) -> Image.Image:
    t = 0.0 if num_frames <= 1 else frame_idx / float(num_frames - 1)
    p = ease_in_out(t)

    zoom = motion.zoom_start + (motion.zoom_end - motion.zoom_start) * p
    x_shift = motion.x_shift_start + (motion.x_shift_end - motion.x_shift_start) * p
    y_shift = motion.y_shift_start + (motion.y_shift_end - motion.y_shift_start) * p

    cover_scale = max(width / source.width, height / source.height)
    scale = cover_scale * zoom
    new_w = max(width, int(source.width * scale))
    new_h = max(height, int(source.height * scale))
    resized = source.resize((new_w, new_h), Image.Resampling.LANCZOS)

    extra_w = max(0, new_w - width)
    extra_h = max(0, new_h - height)

    center_x = extra_w // 2
    center_y = extra_h // 2
    offset_x = center_x + int(x_shift * extra_w * 0.5)
    offset_y = center_y + int(y_shift * extra_h * 0.5)
    offset_x = min(max(0, offset_x), extra_w)
    offset_y = min(max(0, offset_y), extra_h)

    crop = resized.crop((offset_x, offset_y, offset_x + width, offset_y + height))
    return crop.convert("RGB")


def draw_overlay(
    image: Image.Image,
    question: str,
    options: dict[str, str],
    show_panel: bool,
    lit_choice: str | None,
    font_path: str | None,
) -> Image.Image:
    w, h = image.size
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    border = max(3, w // 500)
    margin = max(16, w // 50)

    corner_w = int(w * 0.15)
    corner_h = int(h * 0.11)
    corners = {
        "A": (margin, margin, margin + corner_w, margin + corner_h),
        "B": (w - margin - corner_w, margin, w - margin, margin + corner_h),
        "C": (margin, h - margin - corner_h, margin + corner_w, h - margin),
        "D": (w - margin - corner_w, h - margin - corner_h, w - margin, h - margin),
    }

    label_font = load_font(font_path, size=max(18, int(h * 0.06)))
    for label, rect in corners.items():
        is_lit = lit_choice == label
        fill = (55, 55, 55, 220) if is_lit else (245, 245, 245, 220)
        text_fill = (255, 255, 255, 255) if is_lit else (20, 20, 20, 255)
        draw.rounded_rectangle(rect, radius=max(8, corner_h // 8), fill=fill, outline=(0, 0, 0, 255), width=border)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = rect[0] + (corner_w - tw) // 2
        ty = rect[1] + (corner_h - th) // 2 - 1
        draw.text((tx, ty), label, fill=text_fill, font=label_font)

    if show_panel:
        panel_w = int(w * 0.72)
        panel_h = int(h * 0.36)
        px1 = (w - panel_w) // 2
        py1 = (h - panel_h) // 2
        px2 = px1 + panel_w
        py2 = py1 + panel_h

        draw.rounded_rectangle(
            (px1, py1, px2, py2),
            radius=max(14, min(w, h) // 40),
            fill=(245, 245, 245, 222),
            outline=(0, 0, 0, 230),
            width=border,
        )

        padding = max(16, w // 60)
        title_font = load_font(font_path, size=max(18, int(h * 0.04)))
        body_font = load_font(font_path, size=max(16, int(h * 0.032)))

        text_x = px1 + padding
        text_y = py1 + padding
        text_max_w = panel_w - 2 * padding

        question_lines = wrap_text(draw, question, title_font, text_max_w)
        for line in question_lines[:3]:
            draw.text((text_x, text_y), line, fill=(10, 10, 10, 255), font=title_font)
            bbox = draw.textbbox((0, 0), line, font=title_font)
            text_y += (bbox[3] - bbox[1]) + 8

        text_y += 4
        for choice in CHOICES:
            line = f"{choice}. {options[choice]}"
            wrapped = wrap_text(draw, line, body_font, text_max_w)
            for sub in wrapped[:2]:
                draw.text((text_x, text_y), sub, fill=(20, 20, 20, 255), font=body_font)
                bbox = draw.textbbox((0, 0), sub, font=body_font)
                text_y += (bbox[3] - bbox[1]) + 6
            if text_y > py2 - padding:
                break

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def render_sample_frames(
    source_path: Path,
    frames_dir: Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    motion: MotionSpec,
    question: str,
    options: dict[str, str],
    answer: str,
    overlay_mode: str,
    highlight_start: int,
    highlight_end: int,
    font_path: str | None,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as src:
        src_rgb = src.convert("RGB")
        for frame_idx in range(num_frames):
            base = render_base_frame(
                source=src_rgb,
                width=width,
                height=height,
                motion=motion,
                frame_idx=frame_idx,
                num_frames=num_frames,
            )
            show_panel = (overlay_mode == "all_frames") or (frame_idx == 0)
            lit_choice = answer if highlight_start <= frame_idx <= highlight_end else None
            composed = draw_overlay(
                image=base,
                question=question,
                options=options,
                show_panel=show_panel,
                lit_choice=lit_choice,
                font_path=font_path,
            )
            frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
            composed.save(frame_path, format="PNG")


def build_motion(rng: random.Random) -> MotionSpec:
    zoom_start = rng.uniform(1.00, 1.06)
    zoom_end = zoom_start + rng.uniform(0.03, 0.08)
    return MotionSpec(
        zoom_start=zoom_start,
        zoom_end=zoom_end,
        x_shift_start=rng.uniform(-0.8, 0.8),
        x_shift_end=rng.uniform(-0.8, 0.8),
        y_shift_start=rng.uniform(-0.6, 0.6),
        y_shift_end=rng.uniform(-0.6, 0.6),
    )


def process_one(
    source_path: Path,
    sample_id: str,
    source_rel: str,
    out_split: Path,
    split_name: str,
    width: int,
    height: int,
    fps: int,
    seconds: float,
    num_frames: int,
    question: str,
    options: dict[str, str],
    answer: str,
    overlay_mode: str,
    highlight_start: int,
    highlight_end: int,
    font_path: str | None,
    motion: MotionSpec,
) -> tuple[dict[str, object] | None, str | None]:
    try:
        sample_dir = out_split / sample_id
        frames_dir = sample_dir / "frames"
        render_sample_frames(
            source_path=source_path,
            frames_dir=frames_dir,
            width=width,
            height=height,
            num_frames=num_frames,
            motion=motion,
            question=question,
            options=options,
            answer=answer,
            overlay_mode=overlay_mode,
            highlight_start=highlight_start,
            highlight_end=highlight_end,
            font_path=font_path,
        )

        rec = {
            "sample_id": sample_id,
            "split": split_name,
            "source_image": source_rel,
            "width": width,
            "height": height,
            "fps": fps,
            "seconds": seconds,
            "num_frames": num_frames,
            "question": question,
            "choices": [f"A: {options['A']}", f"B: {options['B']}", f"C: {options['C']}", f"D: {options['D']}"],
            "answer": answer,
            "overlay_mode": overlay_mode,
            "highlight_start": highlight_start,
            "highlight_end": highlight_end,
            "frames_dir": str(frames_dir.relative_to(out_split)),
        }
        return rec, None
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"{source_path}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-frame video-style MCQ samples from existing image datasets."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Image dataset root directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output dataset root directory")
    parser.add_argument("--split", default="train", help="Split name under output-dir")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question text")
    parser.add_argument("--option-a", default=DEFAULT_OPTIONS["A"], help="Choice A")
    parser.add_argument("--option-b", default=DEFAULT_OPTIONS["B"], help="Choice B")
    parser.add_argument("--option-c", default=DEFAULT_OPTIONS["C"], help="Choice C")
    parser.add_argument("--option-d", default=DEFAULT_OPTIONS["D"], help="Choice D")
    parser.add_argument("--answer", choices=list(CHOICES), default="A", help="Correct answer choice")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second")
    parser.add_argument("--seconds", type=float, default=3.0, help="Video duration per sample")
    parser.add_argument("--width", type=int, default=1024, help="Output frame width")
    parser.add_argument("--height", type=int, default=768, help="Output frame height")
    parser.add_argument(
        "--overlay-mode",
        choices=["first_frame_only", "all_frames"],
        default="all_frames",
        help="Whether to show question panel on first frame only or on all frames",
    )
    parser.add_argument("--highlight-start", type=int, default=1, help="Start frame index to highlight answer box")
    parser.add_argument("--highlight-end", type=int, default=None, help="End frame index (inclusive), default last frame")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic motion")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of input images")
    parser.add_argument("--font-path", default=None, help="Optional TTF/TTC font path")
    parser.add_argument("--recursive", action="store_true", help="Recursively load images under input-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    out_split = output_dir / args.split
    out_split.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] input-dir not found: {input_dir}")
        return 1

    if args.fps <= 0:
        print("[ERROR] fps must be > 0")
        return 1
    if args.seconds <= 0:
        print("[ERROR] seconds must be > 0")
        return 1
    if args.width <= 0 or args.height <= 0:
        print("[ERROR] width/height must be > 0")
        return 1

    num_frames = int(round(args.fps * args.seconds))
    if num_frames <= 0:
        print("[ERROR] computed num_frames <= 0")
        return 1

    highlight_end = (num_frames - 1) if args.highlight_end is None else args.highlight_end
    highlight_end = min(highlight_end, num_frames - 1)
    if args.highlight_start < 0 or highlight_end < args.highlight_start:
        print("[ERROR] invalid highlight frame range")
        return 1

    images = iter_images(input_dir, recursive=args.recursive)
    if args.max_samples is not None:
        images = images[: max(0, args.max_samples)]
    if not images:
        print(f"[ERROR] no image files (PNG/JPG/JPEG) found in: {input_dir}")
        return 1

    options = {
        "A": args.option_a,
        "B": args.option_b,
        "C": args.option_c,
        "D": args.option_d,
    }

    rng = random.Random(args.seed)
    tasks = []
    for idx, source_path in enumerate(images):
        sample_id = f"{idx:06d}"
        try:
            source_rel = str(source_path.relative_to(input_dir))
        except ValueError:
            source_rel = source_path.name
        motion = build_motion(rng)
        tasks.append((source_path, sample_id, source_rel, motion))

    metadata_path = out_split / "metadata.jsonl"
    total = len(tasks)
    failed: list[str] = []
    records: list[dict[str, object]] = []

    print(f"[INFO] Found {total} images. Generating {num_frames} frames/sample...")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                process_one,
                source_path=source_path,
                sample_id=sample_id,
                source_rel=source_rel,
                out_split=out_split,
                split_name=args.split,
                width=args.width,
                height=args.height,
                fps=args.fps,
                seconds=args.seconds,
                num_frames=num_frames,
                question=args.question,
                options=options,
                answer=args.answer,
                overlay_mode=args.overlay_mode,
                highlight_start=args.highlight_start,
                highlight_end=highlight_end,
                font_path=args.font_path,
                motion=motion,
            )
            for source_path, sample_id, source_rel, motion in tasks
        ]

        done = 0
        for fut in as_completed(futures):
            rec, err = fut.result()
            done += 1
            if err is not None:
                failed.append(err)
            elif rec is not None:
                records.append(rec)
            if done % 20 == 0 or done == total:
                print(f"[INFO] Progress: {done}/{total}")

    records.sort(key=lambda r: str(r["sample_id"]))
    with metadata_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    success = len(records)
    print(f"[INFO] Completed. Success: {success}, Failed: {len(failed)}")
    print(f"[INFO] metadata: {metadata_path}")
    if failed:
        print("[ERROR] Failed samples:")
        for line in failed[:20]:
            print(f"  - {line}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
