#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
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


def fit_rect(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    scale = min(max_w / src_w, max_h / src_h)
    return max(1, int(src_w * scale)), max(1, int(src_h * scale))


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    x1, y1, x2, y2 = rect
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 - th) // 2
    draw.text((tx, ty), text, fill=fill, font=font)


def wrap_text_multiline(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    words = text.split()
    if not words:
        chars = list(text)
        if not chars:
            return text
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
        return "\n".join(lines)

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
    return "\n".join(lines)


def ease_in_out(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


def render_motion_image_for_slot(
    source: Image.Image,
    slot_w: int,
    slot_h: int,
    motion: MotionSpec,
    frame_idx: int,
    num_frames: int,
) -> Image.Image:
    t = 0.0 if num_frames <= 1 else frame_idx / float(num_frames - 1)
    p = ease_in_out(t)

    zoom = motion.zoom_start + (motion.zoom_end - motion.zoom_start) * p
    x_shift = motion.x_shift_start + (motion.x_shift_end - motion.x_shift_start) * p
    y_shift = motion.y_shift_start + (motion.y_shift_end - motion.y_shift_start) * p

    base_w, base_h = fit_rect(source.width, source.height, slot_w, slot_h)
    new_w = max(1, int(base_w * zoom))
    new_h = max(1, int(base_h * zoom))
    resized = source.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if new_w <= slot_w and new_h <= slot_h:
        slot = Image.new("RGB", (slot_w, slot_h), "white")
        extra_w = slot_w - new_w
        extra_h = slot_h - new_h
        px = (extra_w // 2) + int(x_shift * extra_w * 0.5)
        py = (extra_h // 2) + int(y_shift * extra_h * 0.5)
        px = min(max(0, px), extra_w)
        py = min(max(0, py), extra_h)
        slot.paste(resized, (px, py))
        return slot

    extra_w = max(0, new_w - slot_w)
    extra_h = max(0, new_h - slot_h)
    ox = (extra_w // 2) + int(x_shift * extra_w * 0.5)
    oy = (extra_h // 2) + int(y_shift * extra_h * 0.5)
    ox = min(max(0, ox), extra_w)
    oy = min(max(0, oy), extra_h)
    return resized.crop((ox, oy, ox + slot_w, oy + slot_h)).convert("RGB")


def render_layout_frame(
    source: Image.Image,
    *,
    width: int,
    height: int,
    frame_idx: int,
    num_frames: int,
    motion: MotionSpec,
    question: str,
    options: dict[str, str],
    show_panel: bool,
    lit_choice: str | None,
    font_path: str | None,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    border = max(6, width // 320)
    margin = max(20, width // 48)

    outer = (margin, margin, width - margin, height - margin)
    draw.rectangle(outer, outline="black", width=border)

    corner_w = int(width * 0.16)
    corner_h = int(height * 0.12)
    corner_inset_x = int(width * 0.02)
    corner_inset_y = int(height * 0.015)

    ax1 = outer[0] + corner_inset_x
    ay1 = outer[1] + corner_inset_y
    bx1 = outer[2] - corner_inset_x - corner_w
    by1 = ay1
    cx1 = ax1
    cy1 = outer[3] - corner_inset_y - corner_h
    dx1 = bx1
    dy1 = cy1

    corners = {
        "A": (ax1, ay1, ax1 + corner_w, ay1 + corner_h),
        "B": (bx1, by1, bx1 + corner_w, by1 + corner_h),
        "C": (cx1, cy1, cx1 + corner_w, cy1 + corner_h),
        "D": (dx1, dy1, dx1 + corner_w, dy1 + corner_h),
    }

    label_font = load_font(font_path, size=max(16, int(height * 0.065)))
    for label, rect in corners.items():
        fill = (35, 35, 35) if lit_choice == label else (255, 255, 255)
        text_fill = (255, 255, 255) if lit_choice == label else (0, 0, 0)
        draw.rectangle(rect, outline="black", width=border, fill=fill)
        draw_centered_text(draw, rect, label, label_font, fill=text_fill)

    center_w = int(width * 0.5)
    center_h = int(height * 0.54)
    center_x1 = (width - center_w) // 2
    center_y1 = (height - center_h) // 2
    center_rect = (center_x1, center_y1, center_x1 + center_w, center_y1 + center_h)
    draw.rectangle(center_rect, outline="black", width=border)

    padding = int(width * 0.03)
    question_font = load_font(font_path, size=max(16, int(height * 0.05)))
    option_font = load_font(font_path, size=max(14, int(height * 0.04)))

    text_x = center_rect[0] + padding
    text_max_w = center_w - 2 * padding
    image_top = center_rect[1] + padding
    if show_panel:
        text_y = center_rect[1] + padding
        wrapped_question = wrap_text_multiline(draw, question, question_font, text_max_w)
        draw.multiline_text((text_x, text_y), wrapped_question, fill="black", font=question_font, spacing=10)

        qbbox = draw.multiline_textbbox((text_x, text_y), wrapped_question, font=question_font, spacing=10)
        options_y = qbbox[3] + int(height * 0.02)
        options_text = (
            f"A. {options['A']}   |   B. {options['B']}   |   "
            f"C. {options['C']}   |   D. {options['D']}"
        )
        wrapped_options = wrap_text_multiline(draw, options_text, option_font, text_max_w)
        draw.multiline_text((text_x, options_y), wrapped_options, fill="black", font=option_font, spacing=8)

        obbox = draw.multiline_textbbox((text_x, options_y), wrapped_options, font=option_font, spacing=8)
        image_top = obbox[3] + int(height * 0.03)

    image_bottom = center_rect[3] - padding
    min_image_h = max(48, int(height * 0.18))
    if (image_bottom - image_top) < min_image_h:
        image_top = max(center_rect[1] + padding, image_bottom - min_image_h)

    image_area = (
        center_rect[0] + padding,
        image_top,
        center_rect[2] - padding,
        image_bottom,
    )
    draw.rectangle(image_area, outline="black", width=max(2, border // 2))

    slot_w = max(1, image_area[2] - image_area[0] - int(width * 0.02))
    slot_h = max(1, image_area[3] - image_area[1] - int(height * 0.02))
    moving_img = render_motion_image_for_slot(
        source=source,
        slot_w=slot_w,
        slot_h=slot_h,
        motion=motion,
        frame_idx=frame_idx,
        num_frames=num_frames,
    )
    px = image_area[0] + (image_area[2] - image_area[0] - moving_img.width) // 2
    py = image_area[1] + (image_area[3] - image_area[1] - moving_img.height) // 2
    canvas.paste(moving_img, (px, py))
    return canvas


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
            show_panel = (overlay_mode == "all_frames") or (frame_idx == 0)
            lit_choice = answer if highlight_start <= frame_idx <= highlight_end else None
            composed = render_layout_frame(
                source=src_rgb,
                width=width,
                height=height,
                frame_idx=frame_idx,
                num_frames=num_frames,
                motion=motion,
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
