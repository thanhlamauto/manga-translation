"""
Minimal standalone helper to run detection + OCR and inspect blk.text values.

Usage:
    python scripts/ocr_preview.py --image /path/to/page.png \
        --source-lang Japanese --detector "RT-DETR-V2" --ocr "Default"

This reuses the project's detection/OCR processors but stubs out the GUI
`main_page` and `settings_page` so you can see OCR output in the console.
"""

import argparse
import sys
from pathlib import Path

import imkit as imk

from modules.detection.processor import TextBlockDetector
from modules.ocr.processor import OCRProcessor


class DummyUI:
    def tr(self, text: str) -> str:
        # Pass-through translation stub
        return text


class DummySettings:
    """
    Minimal settings stub to satisfy detectors/OCR.
    """

    def __init__(self, detector_key: str, ocr_key: str):
        self.ui = DummyUI()
        self._detector_key = detector_key
        self._ocr_key = ocr_key

    def get_tool_selection(self, key: str):
        if key == "detector":
            return self._detector_key
        if key == "ocr":
            return self._ocr_key
        return None

    def is_gpu_enabled(self):
        # Detection/OCR factories query this; adjust if you want GPU usage.
        return False


class DummyMainPage:
    """
    Minimal main_page stub providing only what OCRProcessor needs:
    - lang_mapping: localized -> English map
    - settings_page: access to tool selections and ui.tr
    """

    def __init__(self, settings_page, lang_mapping):
        self.settings_page = settings_page
        self.lang_mapping = lang_mapping


def run_ocr(image_path: Path, source_lang: str, detector_key: str, ocr_key: str):
    # Set up stubs
    settings = DummySettings(detector_key, ocr_key)
    lang_mapping = {
        # Extend as needed; keys are localized names, values are English labels
        "Japanese": "Japanese",
        "Korean": "Korean",
        "Chinese": "Chinese",
        "English": "English",
    }
    main_page = DummyMainPage(settings, lang_mapping)

    # Load image
    image = imk.read_image(str(image_path))
    if image is None:
        raise SystemExit(f"Failed to load image: {image_path}")

    # Detect text blocks
    detector = TextBlockDetector(settings)
    blk_list = detector.detect(image)
    print(f"Detected {len(blk_list)} blocks")

    if not blk_list:
        return

    # Run OCR
    ocr = OCRProcessor()
    ocr.initialize(main_page, source_lang)
    ocr.process(image, blk_list)

    # Show results
    for i, blk in enumerate(blk_list, 1):
        x1, y1, x2, y2 = blk.xyxy
        text = getattr(blk, "text", "")
        print(f"[{i:03d}] ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}) :: {text!r}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run OCR and print blk.text")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument(
        "--source-lang",
        default="Japanese",
        help="Localized source language name (default: Japanese)",
    )
    parser.add_argument(
        "--detector",
        default="RT-DETR-V2",
        help="Detector key as configured in the app (default: RT-DETR-V2)",
    )
    parser.add_argument(
        "--ocr",
        default="Default",
        help="OCR engine key as configured in the app (default: Default)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run_ocr(Path(args.image), args.source_lang, args.detector, args.ocr)

