"""
Visualize each stage of the manga translation pipeline for a single page.

Outputs:
1. Detection: Shows detected text blocks with bounding boxes
2. Inpainting: Shows the cleaned image with text removed
3. Final: Shows the translated text rendered back
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Core modules
from modules.detection import TextBlockDetector
from modules.ocr.processor import OCRProcessor
from modules.utils.pipeline_utils import (
    inpaint_map, generate_mask, get_config
)
from modules.utils.textblock import sort_blk_list
from modules.utils.translator_utils import set_upper_case

# Try to import translation - make it optional
try:
    from processing import run_contextual_translation
    TRANSLATION_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Translation not available: {e}")
    TRANSLATION_AVAILABLE = False

from translate_pdf import HeadlessSettings, HeadlessMainPage, render_translations


def visualize_detection(image_bgr, blk_list, output_path):
    """Draw bounding boxes on detected text blocks."""
    print(f"[STAGE 1: DETECTION] Detected {len(blk_list)} text blocks")

    # Convert to RGB for PIL
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Draw each text block
    for i, blk in enumerate(blk_list):
        # Get bounding box coordinates
        xyxy = blk.xyxy  # [x1, y1, x2, y2]

        # Draw rectangle
        draw.rectangle(
            [(xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])],
            outline="red",
            width=3
        )

        # Draw block number
        draw.text(
            (xyxy[0], xyxy[1] - 20),
            f"Block {i}",
            fill="red"
        )

    # Save
    pil_image.save(output_path)
    print(f"  ✓ Saved detection visualization to: {output_path}")


def visualize_inpainting(clean_bgr, output_path):
    """Save the inpainted (cleaned) image."""
    print(f"[STAGE 2: INPAINTING] Saving cleaned image...")

    cv2.imwrite(output_path, clean_bgr)
    print(f"  ✓ Saved inpainted image to: {output_path}")


def visualize_final(rendered_bgr, output_path):
    """Save the final rendered image with translations."""
    print(f"[STAGE 3: FINAL RENDERING] Saving final translated image...")

    cv2.imwrite(output_path, rendered_bgr)
    print(f"  ✓ Saved final image to: {output_path}")


def process_single_page(
    image_path,
    output_dir,
    source_lang,
    target_lang,
    inpainter_name="lama_manga",
    gpu=False,
):
    """Process a single page and save intermediate visualizations."""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image {image_path}")

    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"Image size: {image_bgr.shape[1]}x{image_bgr.shape[0]}")
    print(f"{'='*60}\n")

    # Initialize components
    settings_stub = HeadlessSettings(
        inpainter=inpainter_name,
        gpu=gpu,
        uppercase=False,
        extra_context="",
    )
    main_page_stub = HeadlessMainPage(settings_stub)
    detector = TextBlockDetector(settings_stub)
    ocr_proc = OCRProcessor()

    # ========== STAGE 1: DETECTION ==========
    print("\n" + "="*60)
    blk_list = detector.detect(image_bgr)

    if not blk_list:
        print("No text blocks detected! Saving original image.")
        cv2.imwrite(str(output_dir / "0_original.png"), image_bgr)
        return

    # Save detection visualization
    visualize_detection(
        image_bgr,
        blk_list,
        output_dir / "1_detection.png"
    )

    # ========== STAGE 1.5: OCR ==========
    print("\n" + "="*60)
    print("[STAGE 1.5: OCR] Extracting text from blocks...")
    ocr_proc.initialize(main_page_stub, source_lang)
    ocr_proc.process(image_bgr, blk_list)

    # Sort blocks
    rtl = source_lang == "Japanese"
    blk_list = sort_blk_list(blk_list, rtl)

    # Print detected text
    print(f"  ✓ OCR completed. Text blocks:")
    for i, blk in enumerate(blk_list):
        text = getattr(blk, "text", "") or ""
        text = text.strip()
        if text:
            print(f"    Block {i}: {text[:50]}...")

    # ========== STAGE 1.6: TRANSLATION ==========
    print("\n" + "="*60)
    print("[STAGE 1.6: TRANSLATION] Translating text...")

    # Build transcript
    transcript = []
    for idx, blk in enumerate(blk_list):
        text = getattr(blk, "text", "") or ""
        text = text.strip()
        if not text:
            continue
        transcript.append({
            "index": idx,
            "character": "Narrator",
            "text": text,
        })

    # Translate
    if transcript:
        if TRANSLATION_AVAILABLE:
            try:
                translations = run_contextual_translation(transcript, source_lang, target_lang)
                idx_to_vi = {
                    item["index"]: item.get("text_vi", "") for item in translations
                }
                for idx, blk in enumerate(blk_list):
                    if idx in idx_to_vi:
                        blk.translation = idx_to_vi[idx]

                print(f"  ✓ Translation completed:")
                for i, blk in enumerate(blk_list):
                    trans = getattr(blk, "translation", "")
                    if trans:
                        print(f"    Block {i}: {trans[:50]}...")
            except Exception as e:
                print(f"  ✗ Translation failed: {e}")
        else:
            print(f"  ⚠ Translation skipped (OPENAI_API_KEY not set)")
            print(f"  → Set OPENAI_API_KEY in .env to enable translation")
            # Set dummy translations for visualization
            for idx, blk in enumerate(blk_list):
                blk.translation = "[Translation unavailable]"

    # ========== STAGE 2: INPAINTING ==========
    print("\n" + "="*60)
    print("[STAGE 2: INPAINTING] Removing original text...")

    device = "cuda" if settings_stub.is_gpu_enabled() else "cpu"
    inpainter_key = settings_stub.get_tool_selection("inpainter")
    inpainter_map_dict = {
        "lama_manga": "LaMa",
        "lama": "LaMa",
        "aot": "AOT",
        "mi-gan": "MI-GAN",
        "migan": "MI-GAN",
    }
    inpainter_key = inpainter_map_dict.get(inpainter_key.lower(), inpainter_key)
    InpainterClass = inpaint_map[inpainter_key]
    inpainter = InpainterClass(device, backend="onnx")

    config = get_config(settings_stub)
    mask = generate_mask(image_bgr, blk_list)
    clean_bgr = inpainter(image_bgr, mask, config)
    clean_bgr = cv2.convertScaleAbs(clean_bgr)

    # Save inpainted image
    visualize_inpainting(clean_bgr, output_dir / "2_inpainted.png")

    # Also save the mask for reference
    cv2.imwrite(str(output_dir / "2a_mask.png"), mask)
    print(f"  ✓ Saved mask to: {output_dir / '2a_mask.png'}")

    # ========== STAGE 3: RENDERING ==========
    print("\n" + "="*60)

    rendered_bgr, ok = render_translations(clean_bgr, blk_list, settings_stub, target_lang)

    if not ok:
        print("[WARN] Text rendering failed. Using inpainted image as final output.")
        rendered_bgr = clean_bgr

    # Save final image
    visualize_final(rendered_bgr, output_dir / "3_final.png")

    # Save original for comparison
    cv2.imwrite(str(output_dir / "0_original.png"), image_bgr)
    print(f"  ✓ Saved original image to: {output_dir / '0_original.png'}")

    print("\n" + "="*60)
    print("✓ ALL STAGES COMPLETED!")
    print(f"Check outputs in: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize manga translation pipeline stages for a single page"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image (PNG/JPG)"
    )
    parser.add_argument(
        "--output-dir",
        default="./visualization_output",
        help="Directory to save stage outputs (default: ./visualization_output)"
    )
    parser.add_argument(
        "--source",
        default="English",
        help="Source language (default: English)"
    )
    parser.add_argument(
        "--target",
        default="Vietnamese",
        help="Target language (default: Vietnamese)"
    )
    parser.add_argument(
        "--inpainter",
        default="lama_manga",
        help="Inpainter model (default: lama_manga)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available"
    )

    args = parser.parse_args()

    process_single_page(
        image_path=args.image,
        output_dir=args.output_dir,
        source_lang=args.source,
        target_lang=args.target,
        inpainter_name=args.inpainter,
        gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
