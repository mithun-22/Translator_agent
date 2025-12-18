import pytesseract
import cv2
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------
# CONFIGURE TESSERACT (USER PROVIDED)
# -------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\p90023739\Tesseract-OCR\tesseract.exe"

# -------------------------------------------
# FONT MAP (safe and corrected)
# -------------------------------------------
FONTS = {
    "default": ("NotoSans", "NotoSans-Regular.ttf"),

    # Indian languages
    "hi": ("NotoSansDevanagari", "NotoSansDevanagari-Regular.ttf"),
    "bn": ("NotoSansBengali", "NotoSansBengali-Regular.ttf"),
    "ta": ("NotoSansTamil", "NotoSansTamil-Regular.ttf"),
    "te": ("NotoSansTelugu", "NotoSansTelugu-Regular.ttf"),
    "gu": ("NotoSansGujarati", "NotoSansGujarati-Regular.ttf"),
    "pa": ("NotoSansGurmukhi", "NotoSansGurmukhi-Regular.ttf"),
    "mr": ("NotoSansDevanagari", "NotoSansDevanagari-Regular.ttf"),

    # Asian
    "zh": ("NotoSansSC", "NotoSansSC-Regular.otf"),
    "ja": ("NotoSansJP", "NotoSansJP-Regular.otf"),
    "ko": ("NotoSansKR", "NotoSansKR-Regular.otf"),

    # Middle East
    "ar": ("NotoNaskhArabic", "NotoNaskhArabic-Regular.ttf"),

    # European
    "ru": ("NotoSans", "NotoSans-Regular.ttf"),
}


# -------------------------------------------
# LANGUAGE CODE MAP (Tesseract)
# -------------------------------------------
def get_tesseract_lang_code(lang_code):
    tesseract_codes = {
        "en": "eng",
        "hi": "hin",
        "ru": "rus",
        "fr": "fra",
        "de": "deu",
        "es": "spa",
        "pt": "por",
        "it": "ita",
        "bn": "ben",
        "ta": "tam",
        "te": "tel",
        "gu": "guj",
        "pa": "pan",
        "zh": "chi_sim",
        "ja": "jpn",
        "ko": "kor",
        "ar": "ara",
    }

    return tesseract_codes.get(lang_code, "eng")


# -------------------------------------------
# TRANSLATION (safe wrapper)
# -------------------------------------------
def translate_text(text, source_lang, target_lang):
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        logger.error(f"Translation failed ({text}): {e}")
        return text  # fallback


# -------------------------------------------
# LOAD FONT SAFELY
# -------------------------------------------
def load_best_font(target_lang, font_size, font_path):
    try:
        if font_path:
            name, file = FONTS.get(target_lang, FONTS["default"])
            full_path = os.path.join(font_path, file)

            if os.path.exists(full_path):
                return ImageFont.truetype(full_path, font_size)
    except Exception as e:
        logger.warning(f"Font load failed: {e}")

    return ImageFont.load_default()


# -------------------------------------------
# TRANSLATE IMAGE WITH TEXT OVERLAY
# -------------------------------------------
def translate_image_with_overlay(image_path, source_lang="ru", target_lang="en",
                                 output_path=None, font_path=None):

    logger.info(f"Loading image: {image_path}")

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError("Invalid image path!")

    tess_lang = get_tesseract_lang_code(source_lang)

    logger.info("Running OCR...")
    ocr_data = pytesseract.image_to_data(
        img_cv,
        lang=f"{tess_lang}+eng",
        output_type=pytesseract.Output.DICT
    )

    # convert to PIL
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    found_lines = sum(1 for t in ocr_data["text"] if t.strip())
    logger.info(f"OCR detected {found_lines} text segments")

    for i, text in enumerate(ocr_data["text"]):
        if not text.strip():
            continue

        x, y, w, h = (
            ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i],
        )

        translated = translate_text(text, source_lang, target_lang)
        logger.info(f"{text} â†’ {translated}")

        # draw white background
        draw.rectangle([x, y, x + w, y + h], fill="white")

        # choose best font size (shrink to fit)
        font_size = max(int(h * 0.8), 14)
        font = load_best_font(target_lang, font_size, font_path)

        # wrapping logic
        words = translated.split()
        lines = []
        line = ""

        for word in words:
            test_line = (line + " " + word).strip()
            try:
                width_test = draw.textlength(test_line, font=font)
            except:
                width_test = len(test_line) * font_size * 0.6  # fallback

            if width_test <= w:
                line = test_line
            else:
                lines.append(line)
                line = word

        if line:
            lines.append(line)

        # shrink font if needed
        total_height = len(lines) * (font_size + 2)
        if total_height > h:
            scale = h / total_height
            new_size = max(int(font_size * scale), 10)
            font = load_best_font(target_lang, new_size, font_path)

        # draw text
        cy = y
        for ln in lines:
            draw.text((x, cy), ln, fill="black", font=font)
            cy += font.size + 2

    if output_path:
        pil_img.save(output_path)
        logger.info(f"Saved translated output â†’ {output_path}")

    return pil_img


# -------------------------------------------
# BATCH PROCESSOR
# -------------------------------------------
def translate_image_batch(image_paths, source_lang="ru", target_lang="en",
                          output_dir="translated_images", font_path=None):

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, path in enumerate(image_paths, 1):
        logger.info(f"[{i}/{len(image_paths)}] Translating: {path}")

        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        out = os.path.join(output_dir, f"{name}_translated{ext}")

        try:
            translate_image_with_overlay(path, source_lang, target_lang, out, font_path)
            results.append(out)
        except Exception as e:
            logger.error(f"âŒ Failed: {e}")

    return results


# -------------------------------------------
# DIRECT RUN (example)
# -------------------------------------------
if __name__ == "__main__":
    IMG = r"C:\Users\p90023739\Downloads\russian-text.png"
    OUT = "translated.png"
    FONTS_DIR = "translator/fonts"

    if os.path.exists(IMG):
        translate_image_with_overlay(
            image_path=IMG,
            source_lang="ru",
            target_lang="en",
            output_path=OUT,
            font_path=FONTS_DIR,
        )
        print("DONE")
    else:
        print("Invalid image path")
