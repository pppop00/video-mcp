#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw, ImageFont

DEFAULT_QUESTION = "Question"
DEFAULT_OPTIONS = {
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D",
}


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


def fit_rect(src_w: int, src_h: int, max_w: int, max_h: int) -> Tuple[int, int]:
    scale = min(max_w / src_w, max_h / src_h)
    return max(1, int(src_w * scale)), max(1, int(src_h * scale))


def draw_centered_text(draw: ImageDraw.ImageDraw, rect: Tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, fill=(0, 0, 0), stroke_width=0) -> None:
    x1, y1, x2, y2 = rect
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 - th) // 2
    draw.text((tx, ty), text, fill=fill, font=font, stroke_width=stroke_width)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    words = text.split()
    if not words:
        # Fallback for CJK text without spaces.
        chars = list(text)
        if not chars:
            return text
        lines = []
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

    lines = []
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


def render_first_frame(
    source_path: Path,
    output_path: Path,
    question: str,
    options: dict[str, str],
    width: int,
    height: int,
    font_path: str | None,
) -> None:
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

    label_font = load_font(font_path, size=int(height * 0.065))
    for label, rect in corners.items():
        draw.rectangle(rect, outline="black", width=border)
        draw_centered_text(draw, rect, label, label_font)

    center_w = int(width * 0.5)
    center_h = int(height * 0.54)
    center_x1 = (width - center_w) // 2
    center_y1 = (height - center_h) // 2
    center_rect = (center_x1, center_y1, center_x1 + center_w, center_y1 + center_h)
    draw.rectangle(center_rect, outline="black", width=border)

    padding = int(width * 0.03)
    question_font = load_font(font_path, size=int(height * 0.05))
    option_font = load_font(font_path, size=int(height * 0.04))

    text_x = center_rect[0] + padding
    text_y = center_rect[1] + padding
    text_max_w = center_w - 2 * padding

    wrapped_question = wrap_text(draw, question, question_font, text_max_w)
    draw.multiline_text((text_x, text_y), wrapped_question, fill="black", font=question_font, spacing=10)

    qbbox = draw.multiline_textbbox((text_x, text_y), wrapped_question, font=question_font, spacing=10)
    options_y = qbbox[3] + int(height * 0.02)
    options_text = f"A. {options['A']}   |   B. {options['B']}   |   C. {options['C']}   |   D. {options['D']}"
    wrapped_options = wrap_text(draw, options_text, option_font, text_max_w)
    draw.multiline_text((text_x, options_y), wrapped_options, fill="black", font=option_font, spacing=8)

    obbox = draw.multiline_textbbox((text_x, options_y), wrapped_options, font=option_font, spacing=8)
    image_top = obbox[3] + int(height * 0.03)
    image_area = (
        center_rect[0] + padding,
        image_top,
        center_rect[2] - padding,
        center_rect[3] - padding,
    )
    draw.rectangle(image_area, outline="black", width=max(2, border // 2))

    with Image.open(source_path) as src:
        src = src.convert("RGB")
        slot_w = image_area[2] - image_area[0] - int(width * 0.02)
        slot_h = image_area[3] - image_area[1] - int(height * 0.02)
        new_w, new_h = fit_rect(src.width, src.height, slot_w, slot_h)
        resized = src.resize((new_w, new_h), Image.Resampling.LANCZOS)

    px = image_area[0] + (image_area[2] - image_area[0] - new_w) // 2
    py = image_area[1] + (image_area[3] - image_area[1] - new_h) // 2
    canvas.paste(resized, (px, py))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    files = sorted(input_dir.glob(pattern))
    return [p for p in files if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def process_one(
    source: Path,
    output_path: Path,
    question: str,
    options: dict[str, str],
    width: int,
    height: int,
    font_path: str | None,
) -> Tuple[Path, str | None]:
    try:
        render_first_frame(
            source_path=source,
            output_path=output_path,
            question=question,
            options=options,
            width=width,
            height=height,
            font_path=font_path,
        )
        return source, None
    except Exception as exc:  # pylint: disable=broad-except
        return source, str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch generate first_frame layout images from PNG files."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing source PNG files")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination folder")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question text shown in center")
    parser.add_argument("--option-a", default=DEFAULT_OPTIONS["A"], help="Option A text")
    parser.add_argument("--option-b", default=DEFAULT_OPTIONS["B"], help="Option B text")
    parser.add_argument("--option-c", default=DEFAULT_OPTIONS["C"], help="Option C text")
    parser.add_argument("--option-d", default=DEFAULT_OPTIONS["D"], help="Option D text")
    parser.add_argument("--width", type=int, default=1920, help="Output width")
    parser.add_argument("--height", type=int, default=1080, help="Output height")
    parser.add_argument("--font-path", default=None, help="Optional TTF/TTC font path")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Read PNG files recursively under input-dir",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] input-dir not found: {input_dir}")
        return 1

    pngs = list(iter_images(input_dir, recursive=args.recursive))
    if not pngs:
        print(f"[ERROR] no image files (PNG/JPG) found in: {input_dir}")
        return 1

    options = {
        "A": args.option_a,
        "B": args.option_b,
        "C": args.option_c,
        "D": args.option_d,
    }

    failed = []
    total = len(pngs)
    print(f"[INFO] Found {total} PNG files. Processing...")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        tasks = []
        for idx, source in enumerate(pngs, start=1):
            ts = datetime.now().strftime("%m-%d-%y-%H%M%S")
            stem = source.stem  # e.g. "image1"
            output_path = output_dir / f"{stem}_first_frame_{ts}.png"
            tasks.append((source, output_path))

        futures = [
            pool.submit(
                process_one,
                source=source,
                output_path=output_path,
                question=args.question,
                options=options,
                width=args.width,
                height=args.height,
                font_path=args.font_path,
            )
            for source, output_path in tasks
        ]
        done = 0
        for fut in as_completed(futures):
            source, err = fut.result()
            done += 1
            if err:
                failed.append((source, err))
            if done % 50 == 0 or done == total:
                print(f"[INFO] Progress: {done}/{total}")

    success = total - len(failed)
    print(f"[INFO] Completed. Success: {success}, Failed: {len(failed)}")
    if failed:
        print("[ERROR] Failed files:")
        for source, err in failed[:20]:
            print(f"  - {source}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        return 2

    print(f"[INFO] Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
