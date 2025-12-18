
# utils.py
# ------------------------------------------------------------
# Translation utilities + PDF rebuild pipelines
#  - PyMuPDF overlay (layout-preserving)
#  - ReportLab paragraph wrapping
#  - Hybrid (overlay + wrapped fragments)
# ------------------------------------------------------------

from __future__ import annotations
import io
import os
import base64
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# External libs (guarded imports)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

try:
    from PIL import Image as PILImage  # noqa: F401
except Exception:
    PILImage = None  # type: ignore

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None  # type: ignore

try:
    import google.generativeai as genai
    # Optional: Auto-configure Gemini from env var if present
    _GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if _GEMINI_KEY:
        try:
            genai.configure(api_key=_GEMINI_KEY)
        except Exception:
            pass
except Exception:
    genai = None  # type: ignore

# ReportLab imports
try:
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
except Exception:
    ImageReader = canvas = A4 = pdfmetrics = TTFont = stringWidth = Paragraph = ParagraphStyle = TA_LEFT = None  # type: ignore

# Django font settings (optional)
FONT_PATH: str = ''
FONTS: Dict[str, tuple] = {
    'default': ('Helvetica', ''),  # core fallback if no TTF available
}
try:
    from django.conf import settings  # type: ignore
    if hasattr(settings, 'FONT_PATH'):
        FONT_PATH = settings.FONT_PATH
    if hasattr(settings, 'FONTS'):
        FONTS = settings.FONTS
except Exception:
    pass

LANGUAGE_SLUGS: Dict[str, str] = {
    'en': 'English', 'es': 'Spanish', 'zh': 'Chinese', 'hi': 'Hindi',
    'ar': 'Arabic', 'pt': 'Portuguese', 'bn': 'Bengali', 'ru': 'Russian',
    'ur': 'Urdu', 'fr': 'French', 'id': 'Indonesian', 'de': 'German',
    'ja': 'Japanese', 'pa': 'Punjabi', 'te': 'Telugu', 'mr': 'Marathi',
    'vi': 'Vietnamese', 'ko': 'Korean', 'ta': 'Tamil', 'it': 'Italian',
    'tr': 'Turkish', 'th': 'Thai', 'gu': 'Gujarati', 'pl': 'Polish',
    'uk': 'Ukrainian', 'nl': 'Dutch', 'fa': 'Persian', 'ms': 'Malay',
    'sv': 'Swedish', 'he': 'Hebrew', 'ro': 'Romanian', 'hu': 'Hungarian',
    'cs': 'Czech', 'fi': 'Finnish', 'el': 'Greek', 'da': 'Danish',
    'bg': 'Bulgarian', 'sk': 'Slovak', 'hr': 'Croatian', 'sl': 'Slovenian',
    'no': 'Norwegian', 'sq': 'Albanian', 'lt': 'Lithuanian', 'lv': 'Latvian',
    'et': 'Estonian', 'mk': 'Macedonian', 'is': 'Icelandic', 'ga': 'Irish',
    'kk': 'Kazakh',
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------
# Translation utilities
# ------------------------------------------------------------
@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    engine: str
    success: bool
    error_message: Optional[str] = None


class SimpleTranslationManager:
    """Basic translation manager for GoogleTranslator and Gemini."""

    def translate_text(self, text: str, source_lang: str, target_lang: str,
                       engine: str = 'google') -> TranslationResult:
        if not text or not text.strip():
            return TranslationResult(text, '', source_lang, target_lang, engine, False, 'Empty text')
        try:
            if engine == 'google':
                translated = self._google_translate(text, source_lang, target_lang)
            elif engine == 'gemini':
                translated = self._gemini_translate(text, source_lang, target_lang)
            else:
                return TranslationResult(text, text, source_lang, target_lang, engine, False, 'Unknown engine')
            return TranslationResult(text, translated, source_lang, target_lang, engine, True)
        except Exception as e:
            logger.error(f'Translation failed: {e}')
            return TranslationResult(text, text, source_lang, target_lang, engine, False, str(e))

    def _google_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if GoogleTranslator is None:
            raise RuntimeError('deep_translator.GoogleTranslator is not available.')
        if len(text) <= 4000:
            return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        chunks = chunk_text(text)
        return translate_chunks(chunks, source_lang, target_lang)

    def _gemini_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if genai is None:
            raise RuntimeError('google.generativeai is not available.')
        if len(text) <= 4000:
            return gemini_translate_text(text, source_lang, target_lang)
        chunks = chunk_text(text)
        return gemini_translate_chunks(chunks, source_lang, target_lang)


translation_manager = SimpleTranslationManager()


def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """Split text into chunks under `max_chars`, preserving paragraphs."""
    paragraphs = text.split("\n\n")  # âœ… CORRECT literal (double quotes, \n\n)
    chunks: List[str] = []
    current_chunk = ''
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ''
            if len(para) > max_chars:
                start = 0
                while start < len(para):
                    end = min(start + max_chars, len(para))
                    chunks.append(para[start:end])
                    start = end
            else:
                current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def translate_chunks(chunks: List[str], source_lang: str, target_lang: str) -> str:
    translated_chunks: List[str] = []
    for chunk in chunks:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(chunk)
        translated_chunks.append(translated)
    return "\n\n".join(translated_chunks)


def gemini_translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate a text using Gemini (Flash)."""
    if genai is None:
        raise RuntimeError('google.generativeai is not available.')
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = (
        f'Translate the following text from {source_lang} to {target_lang}.\n'
        f'Output ONLY the translated text; keep line breaks and formatting.\n'
        f'Text:\n{text}'
    )
    response = model.generate_content(prompt, request_options={'timeout': 60})
    if hasattr(response, 'text') and response.text:
        return response.text.strip()
    if hasattr(response, 'candidates') and response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    logger.warning(f'Unexpected Gemini response format: {response}')
    return ''


def gemini_translate_chunks(chunks: List[str], source_lang: str, target_lang: str) -> str:
    translated_chunks: List[str] = []
    for chunk in chunks:
        translated = gemini_translate_text(chunk, source_lang, target_lang)
        translated_chunks.append(translated)
    return "\n\n".join(translated_chunks)


def translate_text_simple(text: str, source_lang: str, target_lang: str, engine: str = 'google') -> str:
    """Return translated text or original on failure."""
    result = translation_manager.translate_text(text, source_lang, target_lang, engine)
    return result.translated_text if result.success else text


# ------------------------------------------------------------
# PDF rebuild pipelines â€” overlay
# ------------------------------------------------------------
def rebuild_pdf_overlay(pages: List[Dict[str, Any]], original_pdf_path: str, target_lang: str) -> bytes:
    """
    Layout-preserving overlay using PyMuPDF.
    `pages` structure: [{"number": 1-based page index, "blocks": [ {type, bbox, text, ...} ]}]
    """
    if fitz is None:
        raise RuntimeError('PyMuPDF is not available.')

    logger.info('ðŸ”§ Starting PDF rebuild (overlay mode)...')
    try:
        original_doc = fitz.open(original_pdf_path)
    except Exception as e:
        logger.error(f'âŒ Cannot open original PDF: {e}')
        raise

    output_doc = fitz.open()
    for p in pages:
        try:
            page_num = p.get('number', 1) - 1
            if page_num >= len(original_doc):
                logger.warning(f'âš  Page {page_num+1} out of range in original PDF')
                continue
            original_page = original_doc.load_page(page_num)
            rect = original_page.rect

            # 1) draw original content as background
            output_page = output_doc.new_page(width=rect.width, height=rect.height)
            output_page.show_pdf_page(rect, original_doc, page_num)

            # 2) overlay translated blocks
            for block in p.get('blocks', []):
                try:
                    t = block.get('type')
                    if t == 'text':
                        _add_text_block(output_page, block, target_lang)
                    elif t == 'image' and block.get('image_data'):
                        _replace_image_block(output_page, block)
                    elif t == 'table':
                        _add_table_block(output_page, block)
                    elif t == 'flowchart_node':
                        _add_flowchart_node(output_page, block, target_lang)
                except Exception as e:
                    logger.warning(f'Block failed on pg {page_num+1}: {e}')
        except Exception as e:
            logger.error(f'Page rebuild failed: {e}', exc_info=True)
            continue

    pdf_bytes = output_doc.write()
    output_doc.close()
    original_doc.close()
    logger.info('âœ… PDF rebuild (overlay) complete.')
    return pdf_bytes


def _add_text_block(page: 'fitz.Page', block: Dict[str, Any], target_lang: str):
    """Overlay translated text within its bounding box."""
    try:
        bbox = block.get('bbox')
        if not bbox:
            return
        text = block.get('text', '').strip()
        if not text:
            return
        rect = fitz.Rect(bbox)
        max_height = rect.height
        font_size = min(max(int(max_height * 0.65), 8), 36)
        page.insert_textbox(
            rect,
            text,
            fontsize=font_size,
            fontname='helv',
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_LEFT,
        )
    except Exception as e:
        logger.warning(f'âš  Text block failed: {e}')


def _replace_image_block(page: 'fitz.Page', block: Dict[str, Any]):
    """Place base64-encoded translated image at the same coordinates."""
    try:
        bbox = block.get('bbox')
        if not bbox:
            return
        img_data_b64 = block.get('image_data')
        if not img_data_b64:
            return
        img_bytes = base64.b64decode(img_data_b64)
        rect = fitz.Rect(bbox)
        page.insert_image(rect, stream=img_bytes)
    except Exception as e:
        logger.warning(f'âš  Image insertion failed: {e}')


def _add_table_block(page: 'fitz.Page', table: Dict[str, Any]):
    """Render a simple grid and place cell text."""
    try:
        bbox = table.get('bbox')
        if not bbox:
            return
        rect = fitz.Rect(bbox)
        rows = table.get('rows', [])
        if not rows:
            return
        n_rows = len(rows)
        n_cols = table.get('num_cols', 1)
        W = rect.width
        H = rect.height
        row_height = H / n_rows if n_rows else 20
        col_width = W / n_cols if n_cols else W
        page.draw_rect(rect, width=0.8, color=(0, 0, 0))
        for r_idx, row in enumerate(rows):
            for c_idx, cell in enumerate(row['cells']):
                x0 = rect.x0 + (c_idx * col_width)
                y0 = rect.y0 + (r_idx * row_height)
                cell_rect = fitz.Rect(x0, y0, x0 + col_width, y0 + row_height)
                page.draw_rect(cell_rect, width=0.3, color=(0, 0, 0))
                txt = cell.get('translated_text', cell.get('text', ''))
                if not txt:
                    continue
                fontsize = max(8, int(row_height * 0.35))
                try:
                    page.insert_textbox(
                        cell_rect,
                        txt,
                        fontsize=fontsize,
                        fontname='helv',
                        color=(0, 0, 0),
                        align=fitz.TEXT_ALIGN_LEFT,
                    )
                except Exception as e:
                    logger.warning(f'Table cell text error: {e}')
    except Exception as e:
        logger.warning(f'âš  Table render failed: {e}')


def _add_flowchart_node(page: 'fitz.Page', node: Dict[str, Any], target_lang: str):
    """Render a filled rectangle and a centered label."""
    try:
        bbox = node.get('bbox')
        if not bbox:
            return
        text = node.get('translated_text', node.get('text', ''))
        color = node.get('color', (0.85, 0.92, 1.0))  # soft blue
        rect = fitz.Rect(bbox)
        page.draw_rect(rect, fill=color, color=color, width=0)
        page.draw_rect(rect, width=1.2, color=(0, 0, 0))
        if not text.strip():
            return
        fontsize = max(8, int(rect.height * 0.25))
        try:
            page.insert_textbox(
                rect,
                text,
                fontsize=fontsize,
                fontname='helv',
                color=(0, 0, 0),
                align=fitz.TEXT_ALIGN_CENTER,
            )
        except Exception as e:
            logger.warning(f'âš  Flowchart text error: {e}')
    except Exception as e:
        logger.warning(f'âš  Flowchart rendering failed: {e}')


# ------------------------------------------------------------
# Hybrid helpers and pipeline
# ------------------------------------------------------------
def _should_wrap_text(block: Dict[str, Any]) -> bool:
    """
    Heuristic: wrap if the text is long relative to bbox width,
    or if block explicitly requests wrapping via block['wrap'] = True.
    """
    text = (block.get("text") or "").strip()
    if not text:
        return False
    if block.get("wrap") is True:
        return True

    x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
    box_width = max(1.0, x1 - x0)
    est_char_capacity_per_line = (box_width / 5.0)  # crude predictor; tune for your fonts
    return len(text) > est_char_capacity_per_line * 3  # > ~3 lines worth â†’ wrap


def _render_wrapped_fragment_pdf(text: str, box_width: float, box_height: float,
                                 font_name: str, font_size: int,
                                 fonts_map: Dict[str, tuple], font_path_base: str) -> bytes:
    """
    Create a single-page PDF fragment sized to (box_width, box_height),
    draw a ReportLab Paragraph inside it, and return fragment bytes.
    """
    if canvas is None:
        raise RuntimeError("ReportLab is not available in this environment.")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(box_width, box_height))
    try:
        c.setFont(font_name or "Helvetica", font_size)
    except Exception:
        c.setFont("Helvetica", font_size)

    p_style = ParagraphStyle(
        name="wrapped_block",
        fontName=(font_name or "Helvetica"),
        fontSize=font_size,
        leading=font_size * 1.2,
        alignment=TA_LEFT,
    )
    p = Paragraph(text, p_style)

    margin = 6
    avail_w = max(1, box_width - 2 * margin)
    avail_h = max(1, box_height - 2 * margin)

    p.wrapOn(c, avail_w, avail_h)
    p.drawOn(c, margin, box_height - margin - p.height)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def rebuild_pdf_hybrid(pages: List[Dict[str, Any]], original_pdf_path: str,
                       target_lang: str, fonts_map: Dict[str, tuple] = None,
                       font_path_base: str = "") -> bytes:
    """
    Hybrid pipeline:
      - Draw original page as background (PyMuPDF).
      - image/table/flowchart_node â†’ overlay (precise placement).
      - text â†’ overlay for short labels; wrapped fragment (ReportLab) for long text.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF is not available in this environment.")

    logger.info("ðŸ”§ Starting PDF rebuild (hybrid mode)...")
    fonts_map = fonts_map or FONTS

    try:
        original_doc = fitz.open(original_pdf_path)
    except Exception as e:
        logger.error(f"âŒ Cannot open original PDF: {e}")
        raise

    output_doc = fitz.open()
    for p in pages:
        try:
            page_num = p.get("number", 1) - 1
            if page_num >= len(original_doc):
                logger.warning(f"âš  Page {page_num+1} out of range in original PDF")
                continue

            original_page = original_doc.load_page(page_num)
            rect = original_page.rect

            out_page = output_doc.new_page(width=rect.width, height=rect.height)
            out_page.show_pdf_page(rect, original_doc, page_num)

            for block in p.get("blocks", []):
                try:
                    btype = block.get("type")
                    if btype in ("image", "table", "flowchart_node"):
                        if btype == "image" and block.get("image_data"):
                            _replace_image_block(out_page, block)
                        elif btype == "table":
                            _add_table_block(out_page, block)
                        elif btype == "flowchart_node":
                            _add_flowchart_node(out_page, block, target_lang)
                        continue

                    if btype == "text":
                        text = (block.get("text") or "").strip()
                        if not text:
                            continue
                        x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
                        box_w, box_h = (x1 - x0), (y1 - y0)
                        font_name = fonts_map.get(target_lang, FONTS["default"])[0]
                        font_size = max(10, int(box_h * 0.65))

                        if _should_wrap_text(block):
                            frag_bytes = _render_wrapped_fragment_pdf(
                                text, box_w, box_h,
                                font_name=font_name,
                                font_size=font_size,
                                fonts_map=fonts_map,
                                font_path_base=FONT_PATH,
                            )
                            frag_doc = fitz.open(stream=frag_bytes, filetype="pdf")
                            try:
                                frag_rect = fitz.Rect(x0, y0, x1, y1)
                                out_page.show_pdf_page(frag_rect, frag_doc, 0)
                            finally:
                                frag_doc.close()
                        else:
                            _add_text_block(out_page, block, target_lang)

                except Exception as e:
                    logger.warning(f"Block failed on pg {page_num+1}: {e}")

        except Exception as e:
            logger.error(f"Page rebuild failed: {e}", exc_info=True)
            continue

    pdf_bytes = output_doc.write()
    output_doc.close()
    original_doc.close()
    logger.info("âœ… PDF rebuild (hybrid) complete.")
    return pdf_bytes


# ------------------------------------------------------------
# ReportLab paragraph-wrapping pipeline
# ------------------------------------------------------------
def rebuild_pdf_reportlab(translated_pages: List[Dict[str, Any]], original_pdf_path: str, target_lang: str = 'default') -> bytes:
    """
    Rebuild a PDF drawing translated text blocks using ReportLab paragraphs.
    Expects `translated_pages` with blocks having `type`, `bbox`, `text`.
    """
    if canvas is None:
        raise RuntimeError('ReportLab is not available.')
    if fitz is None:
        raise RuntimeError('PyMuPDF is required to iterate original pages and extract images.')

    doc = fitz.open(original_pdf_path)
    buffer = io.BytesIO()
    page_width, page_height = A4
    c = canvas.Canvas(buffer, pagesize=(page_width, page_height))

    # Font selection/registration
    try:
        font_name, font_file = FONTS.get(target_lang, FONTS['default'])
        if font_file:
            font_path = os.path.join(FONT_PATH, font_file)
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            logger.info(f"Registered font '{font_name}' for language '{target_lang}'.")
        else:
            font_name = font_name  # core fallback
    except Exception as e:
        logger.warning(f'Font registration failed; using fallback. Error: {e}')
        font_name = 'Helvetica'

    for page_index, translated_page in enumerate(translated_pages):
        if page_index >= len(doc):
            break
        page = doc[page_index]
        translated_blocks = translated_page.get('blocks', [])
        for block_idx, block in enumerate(translated_blocks):
            if block.get('type') == 'text' and block.get('text'):
                try:
                    x0, y0, x1, y1 = block['bbox']
                    box_width = x1 - x0
                    box_height = y1 - y0
                    font_size = block.get('size', 10) - 2
                    p_style = ParagraphStyle(
                        name=f'p_style_{page_index}_{block_idx}',
                        fontName=font_name,
                        fontSize=font_size,
                        leading=font_size * 1.2,
                        alignment=TA_LEFT,
                    )
                    p = Paragraph(block['text'], p_style)
                    p.wrapOn(c, box_width, box_height)
                    c.saveState()
                    # Flip Y (ReportLab origin at bottom-left)
                    p.drawOn(c, x0, page_height - y1)
                    c.restoreState()
                except Exception as e:
                    logger.error(f'Error rendering text block: {e}')
            elif block.get('type') == 'image':
                _render_image_block_reportlab(c, page, block, block, page_height, font_name, 10)
        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def _render_image_block_reportlab(c: 'canvas.Canvas', page: 'fitz.Page', original_block: Dict[str, Any],
                                  translated_block: Dict[str, Any], height: float, font_name: str, font_size: int):
    """Render image blocks and optional OCR text below the image."""
    try:
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            img_bytes = base_image['image']
            x0, y0, x1, y1 = original_block['bbox']
            w, h = (x1 - x0), (y1 - y0)
            c.drawImage(ImageReader(io.BytesIO(img_bytes)), x0, height - y1,
                        width=w, height=h, preserveAspectRatio=True, mask='auto')
            ocr_text = translated_block.get('image_text') or translated_block.get('translated_ocr')
            if ocr_text and ocr_text.strip():
                text_y = height - y1 - font_size - 5
                c.setFont(font_name, font_size - 1)
                words = ocr_text.split()
                current_line = ''
                max_width = w
                for word in words:
                    trial_line = (current_line + ' ' + word).strip()
                    if stringWidth(trial_line, font_name, font_size - 1) <= max_width:
                        current_line = trial_line
                    else:
                        if current_line:
                            c.drawString(x0, text_y, current_line)
                            text_y -= (font_size - 1) + 2
                        current_line = word
                if current_line:
                    c.drawString(x0, text_y, current_line)
                break
    except Exception as e:
        logger.error(f'Image block rendering failed: {e}')


# ------------------------------------------------------------
# Master dispatcher
# ------------------------------------------------------------
def rebuild_pdf(pages_or_translated_pages: List[Dict[str, Any]], original_pdf_path: str,
                target_lang: str = 'default', mode: str = 'overlay') -> bytes:
    """
    Rebuild the PDF in the requested mode.
      - mode='overlay': PyMuPDF overlay (expects `pages`)
      - mode='reportlab': ReportLab paragraphs (expects `translated_pages`)
      - mode='hybrid': overlay + wrapped fragments (expects `pages`)
    """
    if mode == 'overlay':
        return rebuild_pdf_overlay(pages_or_translated_pages, original_pdf_path, target_lang)
    elif mode == 'reportlab':
        return rebuild_pdf_reportlab(pages_or_translated_pages, original_pdf_path, target_lang)
    elif mode == 'hybrid':
        return rebuild_pdf_hybrid(pages_or_translated_pages, original_pdf_path, target_lang, FONTS, FONT_PATH)
    else:
        raise ValueError("Unknown mode. Use 'overlay', 'reportlab', or 'hybrid'.")


__all__ = [
    'TranslationResult',
    'SimpleTranslationManager',
    'translation_manager',
    'chunk_text',
    'translate_chunks',
    'gemini_translate_text',
    'gemini_translate_chunks',
    'translate_text_simple',
    'rebuild_pdf_overlay',
    'rebuild_pdf_reportlab',
    'rebuild_pdf_hybrid',
    'rebuild_pdf',
    'LANGUAGE_SLUGS',
]
