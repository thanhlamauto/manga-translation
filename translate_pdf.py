import os, argparse, shutil, json
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import img2pdf
from tqdm import tqdm

# Core modules used by GUI pipeline
from modules.detection import TextBlockDetector
from modules.ocr.processor import OCRProcessor
from modules.utils.pipeline_utils import (
    inpaint_map, generate_mask, get_config
)
from modules.utils.textblock import sort_blk_list
from modules.utils.translator_utils import (
    set_upper_case,
    get_raw_text,
    get_raw_translation,
)

# Translation factory used by GUI
from modules.translation.factory import TranslationFactory  # <-- GUI uses this :contentReference[oaicite:2]{index=2}
from modules.translation.base import LLMTranslation

# Rendering module
from modules.rendering import render as render_mod  # <-- text rendering stage :contentReference[oaicite:3]{index=3}


def pdf_to_images(pdf_path, out_dir, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for i, page in enumerate(pages):
        p = out_dir / f"page_{i:04d}.png"
        page.save(p, "PNG")
        image_paths.append(str(p))
    return image_paths


def images_to_pdf(image_paths, out_pdf):
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert(image_paths))


class HeadlessSettings:
    """
    Minimal settings stub to satisfy get_config() and tool selection.
    We also expose translator key so TranslationFactory can read it.
    """
    def __init__(self, inpainter="lama_manga", translator="google",
                 gpu=False, uppercase=False, extra_context="", ocr="Default"):
        self._inpainter = inpainter
        self._translator = translator
        self._gpu = gpu
        self._uppercase = uppercase
        self._extra_context = extra_context
        self._ocr = ocr

    def is_gpu_enabled(self): 
        return self._gpu

    def get_tool_selection(self, name):
        if name == "inpainter":
            return self._inpainter
        if name == "translator":
            return self._translator
        if name == "ocr":
            return self._ocr
        return None

    def get_llm_settings(self):
        # Used only by LLM engines; harmless for free engines.
        return {"extra_context": self._extra_context}

    def get_credentials(self, service: str = ""):
        """
        Return credentials for the currently selected translator.

        Logic:
        - Dựa trên giá trị truyền vào --translator khi chạy CLI
        - Mỗi translator sẽ đọc API key từ một biến môi trường tương ứng
        - Nếu không có key trong env thì trả về {} để tránh crash
        """
        tr = (self._translator or "").lower()

        # Google Translate dùng thư viện free, không cần API key
        if tr in ("google", "google translate"):
            return {}

        # OpenAI GPT / gpt-4o / GPT-4.1 / Custom (OpenAI-compatible)
        if tr in ("gpt-4o", "gpt-4.1", "gpt-4.1-mini", "custom", "openai", "gpt"):
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return {}
            return {"api_key": api_key, "model": "gpt-4o"}

        # DeepL
        if tr in ("deepl",):
            api_key = os.environ.get("DEEPL_API_KEY", "")
            if not api_key:
                return {}
            return {"api_key": api_key}

        # Yandex
        if tr in ("yandex",):
            api_key = os.environ.get("YANDEX_API_KEY", "")
            if not api_key:
                return {}
            return {"api_key": api_key}

        # Microsoft Translator
        if tr in ("microsoft", "microsoft translator", "azure"):
            api_key = os.environ.get("AZURE_TRANSLATOR_KEY", "")
            if not api_key:
                return {}
            return {"api_key": api_key}

        # Gemini
        if tr in ("gemini", "gemini-2.5", "gemini-2.0", "gemini-2.5-flash", "gemini-2.0-flash"):
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return {}
            return {"api_key": api_key}

        # DeepSeek
        if tr in ("deepseek", "deepseek-v3"):
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                return {}
            return {"api_key": api_key}

        # Mặc định: không có credential
        return {}

    def get_hd_strategy_settings(self):
        """
        Return HD strategy settings for inpainting.
        Defaults to "Resize" strategy with default values.
        """
        return {
            'strategy': self.ui.tr("Resize"),
            'resize_limit': 512,
            'crop_margin': 512,
            'crop_trigger_size': 512
        }

    # dummy container used by get_config()
    @property
    def ui(self):
        class UI:
            def __init__(self, uppercase):
                self.uppercase_checkbox = type(
                    "", (), {"isChecked": lambda s: uppercase}
                )()
            def tr(self, text):
                # Simple translation stub - just return the text as-is
                return text
        return UI(self._uppercase)


class HeadlessMainPage:
    """
    Minimal main_page stub for headless operation.
    Provides settings_page and lang_mapping that OCRProcessor expects.
    """
    def __init__(self, settings_stub):
        self.settings_page = settings_stub
        # Language mapping: maps English names to themselves (since we use English in headless mode)
        self.lang_mapping = {
            "English": "English",
            "Korean": "Korean",
            "Japanese": "Japanese",
            "French": "French",
            "Simplified Chinese": "Simplified Chinese",
            "Traditional Chinese": "Traditional Chinese",
            "Chinese": "Chinese",
            "Russian": "Russian",
            "German": "German",
            "Dutch": "Dutch",
            "Spanish": "Spanish",
            "Italian": "Italian",
            "Turkish": "Turkish",
            "Polish": "Polish",
            "Portuguese": "Portuguese",
            "Brazilian Portuguese": "Brazilian Portuguese",
            "Thai": "Thai",
            "Vietnamese": "Vietnamese",
            "Indonesian": "Indonesian",
            "Hungarian": "Hungarian",
            "Finnish": "Finnish",
            "Arabic": "Arabic",
            "Czech": "Czech",
            "Persian": "Persian",
            "Romanian": "Romanian",
            "Mongolian": "Mongolian",
        }


def render_translations(clean_bgr, blk_list, settings_stub, target_lang):
    """
    Render translated text back into bubbles.
    We try common function names to be version-robust.
    """
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

    # Prefer user-specified font: anime_ace_bb (absolute path provided)
    font_path = None
    preferred_font_root = Path("/Users/nguyenthanhlam/manga-translation/font-hung-lan/hunglan/hlcomic/HLcomic1_normal.ttf")
    preferred_fonts = []
    if preferred_font_root.exists():
        if preferred_font_root.is_file():
            preferred_fonts = [preferred_font_root]
        else:
            preferred_fonts = list(preferred_font_root.glob("*.ttf")) + list(preferred_font_root.glob("*.otf"))
    if preferred_fonts:
        font_path = str(preferred_fonts[0])

    # Fallback to bundled fonts if preferred font not found
    if font_path is None:
        font_folder = Path(__file__).parent / "resources" / "fonts"
        if font_folder.exists():
            font_files = list(font_folder.glob("*.ttf")) + list(font_folder.glob("*.otf"))
            if font_files:
                font_path = str(font_files[0])
    
    # Fallback to system font if no custom font found
    if font_path is None:
        import glob
        # Try common system font paths (prioritize readable fonts)
        system_fonts = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",  # macOS
            "/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows
        ]
        for sys_font in system_fonts:
            if os.path.exists(sys_font):
                font_path = sys_font
                break
        
        # If still no font, try to find any .ttf or .ttc in system font directories
        if font_path is None:
            for font_dir in ["/System/Library/Fonts/Supplemental", "/System/Library/Fonts", "/Library/Fonts"]:
                if os.path.exists(font_dir):
                    fonts = glob.glob(os.path.join(font_dir, "*.ttf")) + glob.glob(os.path.join(font_dir, "*.ttc"))
                    # Prefer fonts with "Arial" or "Helvetica" in the name
                    preferred = [f for f in fonts if "Arial" in f or "Helvetica" in f]
                    if preferred:
                        font_path = preferred[0]
                        break
                    elif fonts:
                        font_path = fonts[0]
                        break
    
    # If still no font found, skip rendering
    if font_path is None:
        print("[WARN] No font found, skipping text rendering")
        return clean_bgr, False

    # Check if we have translations to render
    if not blk_list or not any(getattr(blk, 'translation', None) for blk in blk_list):
        print("[WARN] No translations found in text blocks, skipping text rendering")
        return clean_bgr, False

    # Normalize alignment and font sizes per block
    for blk in blk_list:
        if not hasattr(blk, 'alignment') or not blk.alignment:
            blk.alignment = "center"  # Default to center
        else:
            # Normalize alignment to valid PIL values
            align_lower = str(blk.alignment).lower()
            if align_lower in ["left", "l"]:
                blk.alignment = "left"
            elif align_lower in ["right", "r"]:
                blk.alignment = "right"
            elif align_lower in ["center", "centre", "c", ""]:
                blk.alignment = "center"
            else:
                blk.alignment = "center"  # Default fallback
        
        # Ensure line_spacing is a number
        if not hasattr(blk, 'line_spacing') or not blk.line_spacing:
            blk.line_spacing = 1.0
        try:
            blk.line_spacing = float(blk.line_spacing)
        except (ValueError, TypeError):
            blk.line_spacing = 1.0

        # Force fixed font sizes per block so draw_text doesn't override them
        blk.max_font_size = 60
        blk.min_font_size = 60

    # Fixed font sizes
    init_font_size = 60
    min_font_size = 60

    rendered_rgb = None
    # Try draw_text with font path
    fn = getattr(render_mod, "draw_text", None)
    if fn is not None and font_path:
        try:
            print(f"[INFO] Rendering text with font: {font_path}")
            # draw_text signature: (image, blk_list, font_pth, colour, init_font_size, min_font_size, outline)
            rendered_rgb = fn(clean_rgb, blk_list, font_path, "#000", init_font_size, min_font_size, True)
            print(f"[INFO] Text rendering successful")
        except Exception as e:
            # Debug: print error but don't fail completely
            import traceback
            print(f"[ERROR] Text rendering failed with font {font_path}: {e}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            pass

    if rendered_rgb is None:
        return clean_bgr, False

    if isinstance(rendered_rgb, Image.Image):
        rendered_rgb = np.array(rendered_rgb)

    rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
    return rendered_bgr, True


def translate_page_image(
    image_path, out_path, source_lang, target_lang,
    detector, ocr_proc, main_page_stub
):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image {image_path}")

    # 1) detect text blocks
    blk_list = detector.detect(image_bgr)
    if not blk_list:
        cv2.imwrite(out_path, image_bgr)
        return

    # 2) OCR
    ocr_proc.initialize(main_page_stub, source_lang)
    ocr_proc.process(image_bgr, blk_list)

    # sort RTL if Japanese
    rtl = (source_lang == "Japanese")
    blk_list = sort_blk_list(blk_list, rtl)

    # 3) translate (factory like GUI)
    settings_stub = main_page_stub.settings_page
    translator_key = settings_stub.get_tool_selection("translator")
    # Normalize translator key to match factory expectations
    translator_map = {
        "google": "Google Translate",
        "microsoft": "Microsoft Translator",
        "deepl": "DeepL",
        "yandex": "Yandex"
    }
    translator_key = translator_map.get(translator_key.lower(), translator_key)
    engine = TranslationFactory.create_engine(
        settings_stub,
        source_lang,
        target_lang,
        translator_key=translator_key
    )
    extra_context = settings_stub.get_llm_settings()["extra_context"]
    # Traditional engines only take blk_list, LLM engines also take image and extra_context
    if isinstance(engine, LLMTranslation):
        engine.translate(blk_list, image_bgr, extra_context)
    else:
        engine.translate(blk_list)

    set_upper_case(blk_list, settings_stub.ui.uppercase_checkbox.isChecked())

    # 3b) export OCR & translation text for this page
    try:
        text_dir = Path("._ct_work") / "text"
        text_dir.mkdir(parents=True, exist_ok=True)
        page_stem = Path(image_path).stem  # e.g. page_0000

        # Raw OCR text (trước khi render)
        ocr_json = get_raw_text(blk_list)
        with open(text_dir / f"{page_stem}_ocr.json", "w", encoding="utf-8") as f:
            f.write(ocr_json)

        # Translated text
        translated_json = get_raw_translation(blk_list)
        with open(text_dir / f"{page_stem}_translated.json", "w", encoding="utf-8") as f:
            f.write(translated_json)
    except Exception as e:
        # Không để việc export text làm hỏng pipeline chính
        print(f"[WARN] Failed to export text for {image_path}: {e}")

    # 4) inpaint/remove original text
    device = "cuda" if settings_stub.is_gpu_enabled() else "cpu"
    inpainter_key = settings_stub.get_tool_selection("inpainter")
    # Normalize inpainter key to match inpaint_map expectations
    inpainter_map = {
        "lama_manga": "LaMa",
        "lama": "LaMa",
        "aot": "AOT",
        "mi-gan": "MI-GAN",
        "migan": "MI-GAN"
    }
    inpainter_key = inpainter_map.get(inpainter_key.lower(), inpainter_key)
    InpainterClass = inpaint_map[inpainter_key]
    inpainter = InpainterClass(device, backend='onnx')

    config = get_config(settings_stub)
    mask = generate_mask(image_bgr, blk_list)
    clean_bgr = inpainter(image_bgr, mask, config)
    clean_bgr = cv2.convertScaleAbs(clean_bgr)

    # 5) render translated text back in
    rendered_bgr, ok = render_translations(clean_bgr, blk_list, settings_stub, target_lang)
    if not ok:
        # Debug: check why rendering failed
        print(f"[WARN] Text rendering failed. Checking blocks: {len(blk_list) if blk_list else 0} blocks")
        if blk_list:
            trans_count = sum(1 for blk in blk_list if getattr(blk, 'translation', None))
            print(f"[WARN] Blocks with translations: {trans_count}/{len(blk_list)}")
        print("[WARN] Output will be cleaned-only (no translated text).")

    cv2.imwrite(out_path, rendered_bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--inpainter", default="lama_manga")
    ap.add_argument("--translator", default="google",
                    help="Force a translator. 'google' is free/no-key.")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--uppercase", action="store_true")
    ap.add_argument("--extra_context", default="")
    args = ap.parse_args()

    work_dir = Path("._ct_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    (work_dir / "pages").mkdir(parents=True)
    (work_dir / "translated_pages").mkdir(parents=True)

    # init core handlers once
    settings_stub = HeadlessSettings(
        inpainter=args.inpainter,
        translator=args.translator,
        gpu=args.gpu,
        uppercase=args.uppercase,
        extra_context=args.extra_context
    )
    
    main_page_stub = HeadlessMainPage(settings_stub)
    detector = TextBlockDetector(settings_stub)
    ocr_proc = OCRProcessor()

    # 1) pdf → images
    page_images = pdf_to_images(args.pdf, work_dir / "pages", dpi=args.dpi)

    # 2) translate each page
    translated_paths = []
    for p in tqdm(page_images, desc="Translating pages"):
        out_p = str((work_dir / "translated_pages" / Path(p).name))
        translate_page_image(
            p, out_p,
            args.source, args.target,
            detector, ocr_proc, main_page_stub
        )
        translated_paths.append(out_p)

    # 3) images → pdf
    images_to_pdf(translated_paths, args.out)
    print("Done:", args.out)


if __name__ == "__main__":
    main()
