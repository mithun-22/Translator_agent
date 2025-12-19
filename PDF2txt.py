# pdf2text.py
# Completed module (SmartPDFProcessor) — OCR, translation, overlay, table & flowchart detection,
# plus helpers: extract_blocks_from_pdf, extract_text_for_validation, rebuild_pdf.

import fitz  # PyMuPDF
from typing import Union, IO, Dict, List, Any, Optional, Tuple
import pytesseract
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import logging
import json
import re
import statistics
import base64
import os
from dataclasses import dataclass, asdict
import numpy as np

# Configure Tesseract binary path (user provided)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Optional translation manager (best-effort import)
try:
    from utils import translation_manager, TranslationResult, gemini_regenerate_image, estimate_tokens_and_cost
except ImportError as e:
    logger.warning(f"Direct import from utils failed: {e}")
    try:
        from .utils import translation_manager, TranslationResult, gemini_regenerate_image, estimate_tokens_and_cost
    except Exception as e2:
        logger.error(f"Failed to import from utils: {e2}")
        translation_manager = None
        TranslationResult = None
        gemini_regenerate_image = None
        estimate_tokens_and_cost = None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Small helpers / dataclasses
# ----------------------------
@dataclass
class SpanStyle:
    size: float
    flags: int
    font: str
    color: int
    alpha: int
    ascender: float
    descender: float
    bidi: int = 0
    char_flags: int = 0

    def __hash__(self):
        return hash((round(self.size, 2), self.flags, self.font, self.color, self.alpha,
                     round(self.ascender, 3), round(self.descender, 3), self.bidi, self.char_flags))

    def __eq__(self, other):
        if not isinstance(other, SpanStyle):
            return False
        return (abs(self.size - other.size) < 0.1 and self.flags == other.flags and self.font == other.font and
                self.color == other.color and self.alpha == other.alpha and abs(self.ascender - other.ascender) < 0.01 and
                abs(self.descender - other.descender) < 0.01 and self.bidi == other.bidi and self.char_flags == other.char_flags)

@dataclass
class TextRun:
    style: SpanStyle
    text: str
    spans: List[Dict[str, Any]]
    line_breaks: List[int]

    def add_span(self, span: Dict[str, Any], text: str, is_line_break: bool = False):
        if is_line_break and self.text and not self.text.endswith(' '):
            self.text += ' '
        start_pos = len(self.text)
        self.text += text
        if is_line_break:
            self.line_breaks.append(start_pos)
        self.spans.append(span)

    def clean_text(self) -> str:
        text = self.text
        text = re.sub(r'-\s+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ----------------------------
# Font helper
# ----------------------------
def get_font_for_language(target_lang: str) -> Tuple[str, Optional[str]]:
    """Get appropriate font name and path for target language (tries Django settings, local fonts)."""
    try:
        from django.conf import settings
        FONTS = getattr(settings, "FONTS", {})
        FONT_PATH = getattr(settings, "FONT_PATH", "")
        font_name, font_file = FONTS.get(target_lang, FONTS.get("default", ("NotoSans", "NotoSans.ttf")))
        font_path = os.path.join(FONT_PATH, font_file) if FONT_PATH else font_file
        return font_name, font_path if os.path.exists(font_path) else None
    except Exception:
        # fallback map
        FONTS = {
            "hi": ("NotoSansDevanagari", "NotoSansDevanagari.ttf"),
            "bn": ("NotoSansBengali", "NotoSansBengali.ttf"),
            "ta": ("NotoSansTamil", "NotoSansTamil.ttf"),
            "te": ("NotoSansTelugu", "NotoSansTelugu.ttf"),
            "default": ("NotoSans", "NotoSans.ttf")
        }
        font_name, font_file = FONTS.get(target_lang, FONTS["default"])
        # search common folders
        for font_dir in ["translator/fonts", "fonts", os.path.join(os.path.dirname(__file__), "fonts")]:
            font_path = os.path.join(font_dir, font_file)
            if os.path.exists(font_path):
                return font_name, font_path
        return font_name, None

# ----------------------------
# Tesseract language mapping
# ----------------------------
def _get_tesseract_lang_code(lang_code: str) -> str:
    mapping = {
        "en": "eng", "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel",
        "gu": "guj", "mr": "mar", "pa": "pan", "zh": "chi_sim", "ja": "jpn",
        "ko": "kor", "ar": "ara", "ru": "rus", "fa": "fas", "ur": "urd",
        "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "pt": "por"
    }
    if not lang_code:
        return "eng"
    return mapping.get(lang_code.lower(), lang_code.lower())

# ----------------------------
# Table & Flowchart detectors (from your code)
# ----------------------------
class TableDetector:
    @staticmethod
    def detect_table_structure(blocks: List[Dict], page_height: float) -> List[List[Dict]]:
        tables = []
        y_positions = {}
        for block in blocks:
            if block.get("type") != "text":
                continue
            bbox = block.get("bbox", [0, 0, 0, 0])
            y = round(bbox[1], 1)
            y_positions.setdefault(y, []).append(block)
        sorted_rows = sorted(y_positions.items())
        current_table = []
        prev_y = None
        for y, row_blocks in sorted_rows:
            if len(row_blocks) >= 2:
                row_blocks_sorted = sorted(row_blocks, key=lambda b: b["bbox"][0])
                y_values = [b["bbox"][1] for b in row_blocks_sorted]
                if max(y_values) - min(y_values) < 10:
                    current_table.append({"y": y, "cells": row_blocks_sorted})
                    prev_y = y
                else:
                    if len(current_table) >= 2:
                        tables.append(current_table)
                    current_table = []
            else:
                if prev_y and (y - prev_y) > 30:
                    if len(current_table) >= 2:
                        tables.append(current_table)
                    current_table = []
        if len(current_table) >= 2:
            tables.append(current_table)
        return tables

    @staticmethod
    def reconstruct_table(table_rows: List[Dict]) -> Dict:
        if not table_rows:
            return {}
        all_x_positions = set()
        for row in table_rows:
            for cell in row["cells"]:
                all_x_positions.add(round(cell["bbox"][0]))
                all_x_positions.add(round(cell["bbox"][2]))
        column_boundaries = sorted(all_x_positions)
        table_data = {
            "type": "table",
            "rows": [],
            "num_rows": len(table_rows),
            "num_cols": len(column_boundaries) - 1 if column_boundaries else 0,
            "bbox": [
                min(cell["bbox"][0] for row in table_rows for cell in row["cells"]),
                min(row["y"] for row in table_rows),
                max(cell["bbox"][2] for row in table_rows for cell in row["cells"]),
                max(cell["bbox"][3] for row in table_rows for cell in row["cells"])
            ]
        }
        for row in table_rows:
            row_data = {"y": row["y"], "cells": []}
            for cell in row["cells"]:
                row_data["cells"].append({
                    "text": cell.get("text", ""),
                    "bbox": cell["bbox"],
                    "font": cell.get("font", "Helvetica"),
                    "size": cell.get("size", 10)
                })
            table_data["rows"].append(row_data)
        return table_data

class FlowchartDetector:
    @staticmethod
    def detect_shapes(page: fitz.Page) -> List[Dict]:
        shapes = []
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []
        for drawing in drawings:
            if drawing.get("type") == "f":
                rect = drawing.get("rect")
                if rect:
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    shape_type = "rectangle"
                    if abs(width - height) < 5:
                        shape_type = "square"
                    elif width > height * 2:
                        shape_type = "horizontal_bar"
                    shapes.append({"type": shape_type, "bbox": list(rect), "color": drawing.get("color", (0, 0, 0))})
            elif drawing.get("type") == "l":
                items = drawing.get("items", [])
                if items:
                    shapes.append({"type": "line", "points": items, "color": drawing.get("color", (0, 0, 0))})
        return shapes

    @staticmethod
    def match_text_to_shapes(shapes: List[Dict], text_blocks: List[Dict]) -> List[Dict]:
        flowchart_elements = []
        for shape in shapes:
            if shape["type"] in ["rectangle", "square"]:
                shape_bbox = shape["bbox"]
                contained_text = []
                for block in text_blocks:
                    if block.get("type") != "text":
                        continue
                    text_bbox = block["bbox"]
                    if (text_bbox[0] >= shape_bbox[0] and text_bbox[1] >= shape_bbox[1] and
                        text_bbox[2] <= shape_bbox[2] and text_bbox[3] <= shape_bbox[3]):
                        contained_text.append(block.get("text", ""))
                if contained_text:
                    flowchart_elements.append({
                        "type": "flowchart_node",
                        "shape": shape["type"],
                        "text": " ".join(contained_text),
                        "bbox": shape["bbox"],
                        "color": shape["color"]
                    })
        return flowchart_elements

# -----------------------------------
# SmartPDFProcessor class
# -----------------------------------
class SmartPDFProcessor:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.debug_files = {}
        if debug_mode:
            self.debug_files = {
                'output': open("output.txt", "w", encoding="utf-8"),
                'blocks': open("blocks.json", "w", encoding="utf-8"),
                'runs': open("text_runs.json", "w", encoding="utf-8"),
                'results': open("results.json", "w", encoding="utf-8")
            }
        self.total_text_chars = 0
        self.total_images = 0


    def __del__(self):
        for file in self.debug_files.values():
            if file and not file.closed:
                file.close()

    def extract_blocks_from_pdf(self, pdf_file: Union[str, IO], source_lang: str = "en",
                               target_lang: str = "hi", engine: str = "gemini") -> Dict[str, Any]:
        try:
            doc = self._open_pdf(pdf_file)
            if not doc:
                raise ValueError("Could not open PDF file")

            results = {
                "pages": [],
                "metadata": {
                    "total_pages": doc.page_count,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "engine": engine,
                    "errors": [],
                    "tables_found": 0,
                    "flowcharts_found": 0,
                    "text_tokens": 0,
                    "image_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0
                }
            }

            for page_num, page in enumerate(doc, start=1):
                try:
                    page_data = self._process_page(page, page_num, source_lang, target_lang, engine)
                    results["pages"].append(page_data)
                    results["metadata"]["tables_found"] += len([b for b in page_data["blocks"] if b.get("type") == "table"])
                    results["metadata"]["flowcharts_found"] += len([b for b in page_data["blocks"] if b.get("type") == "flowchart_node"])
                except Exception as e:
                    error_msg = f"Page {page_num} processing failed: {e}"
                    logger.error(error_msg)
                    results["metadata"]["errors"].append(error_msg)
                    results["pages"].append({"number": page_num, "blocks": [], "error": str(e)})

            # Add Estimation Stats
            if estimate_tokens_and_cost:
                stats = estimate_tokens_and_cost(self.total_text_chars, self.total_images)
                results["metadata"].update(stats)


            if self.debug_mode:
                with open("final_results.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            doc.close()
            return results

        except Exception as e:
            logger.error(f"PDF processing failed: {e}", exc_info=True)
            return {"pages": [], "metadata": {"total_pages": 0, "errors": [str(e)]}}

    def _open_pdf(self, pdf_file: Union[str, IO]) -> Optional[fitz.Document]:
        try:
            if isinstance(pdf_file, str):
                return fitz.open(pdf_file)
            elif isinstance(pdf_file, bytes):
                return fitz.open(stream=pdf_file, filetype="pdf")
            else:
                pdf_file.seek(0)
                return fitz.open(stream=pdf_file.read(), filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}", exc_info=True)
            return None

    def _process_page(self, page: fitz.Page, page_num: int, source_lang: str,
                      target_lang: str, engine: str) -> Dict[str, Any]:
        blocks = page.get_text("dict")["blocks"]
        page_data = {"number": page_num, "blocks": []}
        processed_blocks = []

        for block_idx, block in enumerate(blocks):
            try:
                if "lines" in block:
                    paragraph_blocks = self._process_text_block(block, page_num, block_idx, source_lang, target_lang, engine)
                    processed_blocks.extend(paragraph_blocks)
                elif "image" in block:
                    processed_block = self._process_image_block(page, block, page_num, block_idx, source_lang, target_lang, engine)
                    processed_blocks.append(processed_block)
                else:
                    processed_blocks.append({"type": "other", "bbox": block.get("bbox")})
            except Exception as e:
                logger.warning(f"Block {block_idx} on page {page_num} failed: {e}", exc_info=True)
                continue

        # Detect tables and add them
        tables = TableDetector.detect_table_structure(processed_blocks, page.rect.height)
        for table_rows in tables:
            table_data = TableDetector.reconstruct_table(table_rows)
            if table_data:
                table_data = self._translate_table(table_data, source_lang, target_lang, engine)
                processed_blocks.append(table_data)

        # Detect flowcharts and map text into nodes
        shapes = FlowchartDetector.detect_shapes(page)
        flowchart_elements = FlowchartDetector.match_text_to_shapes(shapes, processed_blocks)
        for element in flowchart_elements:
            element = self._translate_flowchart_element(element, source_lang, target_lang, engine)
            processed_blocks.append(element)

        page_data["blocks"] = processed_blocks

        if self.debug_mode:
            page_data_json = f"page_{page_num}_data.json"
            try:
                with open(page_data_json, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error writing page {page_num} data to file: {e}")

        return page_data

    def _process_text_block(self, block: Dict[str, Any], page_num: int, block_idx: int,
                           source_lang: str, target_lang: str, engine: str) -> List[Dict[str, Any]]:
        paragraphs = self._segment_block_into_paragraphs(block)
        final_paragraph_blocks = []

        for paragraph_spans in paragraphs:
            if not paragraph_spans:
                continue

            original_text = " ".join(span.get("text", "") for span in paragraph_spans).strip()
            original_text = re.sub(r'\s+', ' ', original_text)

            if not original_text:
                continue

            self.total_text_chars += len(original_text)


            translated_text = original_text
            if translation_manager:
                try:
                    res = translation_manager.translate_text(original_text, source_lang, target_lang, engine)
                    # Accept various shapes of response
                    if isinstance(res, str):
                        translated_text = res
                    elif hasattr(res, "translated_text"):
                        translated_text = res.translated_text
                    elif isinstance(res, dict):
                        translated_text = res.get("translated_text") or res.get("text") or translated_text
                except Exception as e:
                    logger.warning(f"Paragraph translation failed (Manager): {e}")
            else:
                # Emergency Fallback if manager failed to load but googletrans is available
                try:
                    from deep_translator import GoogleTranslator
                    translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(original_text)
                except Exception as e:
                    logger.warning(f"Paragraph translation failed (Emergency Fallback): {e}")

            min_x = min(s['bbox'][0] for s in paragraph_spans)
            min_y = min(s['bbox'][1] for s in paragraph_spans)
            max_x = max(s['bbox'][2] for s in paragraph_spans)
            max_y = max(s['bbox'][3] for s in paragraph_spans)
            paragraph_bbox = (min_x, min_y, max_x, max_y)

            first_span = paragraph_spans[0]

            final_paragraph_blocks.append({
                "type": "text",
                "bbox": paragraph_bbox,
                "text": translated_text,
                "original_text": original_text,
                "size": first_span.get("size", 10),
                "font": first_span.get("font", "Helvetica")
            })

        return final_paragraph_blocks

    def _segment_block_into_paragraphs(self, block: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        if not block.get("lines"):
            return []

        line_data = []
        for line in block["lines"]:
            line_bbox = line["bbox"]
            line_text = " ".join(span.get("text", "") for span in line.get("spans", [])).strip()
            line_data.append({
                "bbox": line_bbox,
                "text": line_text,
                "y_pos": line_bbox[1],
                "x_start": line_bbox[0],
                "line_width": line_bbox[2] - line_bbox[0],
                "spans": line.get("spans", [])
            })

        line_gaps = [abs(line_data[i]["y_pos"] - line_data[i-1]["y_pos"]) for i in range(1, len(line_data))]
        base_line_height = statistics.median(g for g in line_gaps if g > 0) if any(g > 0 for g in line_gaps) else 15
        vertical_gap_threshold = base_line_height * 1.5
        line_widths = [ld["line_width"] for ld in line_data if ld["line_width"] > 0]
        avg_line_width = statistics.mean(line_widths) if line_widths else 400
        short_line_threshold = avg_line_width * 0.7

        all_paragraphs = []
        current_paragraph_spans = []

        for i, line_info in enumerate(line_data):
            is_paragraph_break = False
            if i > 0:
                previous_line = line_data[i-1]
                vertical_gap = abs(line_info["y_pos"] - previous_line["y_pos"])
                if vertical_gap > vertical_gap_threshold:
                    is_paragraph_break = True
                else:
                    previous_is_short = previous_line["line_width"] < short_line_threshold
                    is_indented = abs(line_info["x_start"] - previous_line["x_start"]) > 10
                    previous_ends_punct = previous_line["text"].rstrip().endswith(('.', '!', '?', ':'))
                    current_starts_capital = line_info["text"] and line_info["text"][0].isupper()
                    if previous_is_short and is_indented:
                        is_paragraph_break = True
                    elif (previous_ends_punct and current_starts_capital and vertical_gap > base_line_height * 1.2):
                        is_paragraph_break = True

            if is_paragraph_break and current_paragraph_spans:
                all_paragraphs.append(current_paragraph_spans)
                current_paragraph_spans = []

            current_paragraph_spans.extend(line_info["spans"])

        if current_paragraph_spans:
            all_paragraphs.append(current_paragraph_spans)

        return all_paragraphs

    def _process_image_block(self, page, block, page_num, block_idx, source_lang, target_lang, engine):
        """
        Processes an image block: extracts it, and if it looks like a diagram/table, 
        uses Gemini Vision to translate and inpaint it.
        """
        try:
            # Extract image bytes from PDF
            bbox = block.get("bbox")
            if not bbox:
                return {"type": "image", "bbox": None}
            
            pix = page.get_pixmap(clip=fitz.Rect(bbox), matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            
            # Use Gemini Vision for translation if engine is gemini
            # Use Gemini Vision for translation if engine is gemini
            self.total_images += 1
            if engine == "gemini" and gemini_regenerate_image:
                modified_bytes = gemini_regenerate_image(img_bytes, target_lang)
                if modified_bytes:
                    return {
                        "type": "image",
                        "bbox": bbox,
                        "image_data": base64.b64encode(modified_bytes).decode()
                    }
                else:
                    logger.warning(f"Gemini image regeneration failed for block {block_idx}. Using fallback.")

            # Fallback: OCR or Return Original
            # If no text was extracted from this page (scan?), we might want to OCR this image.
            # But here we just return the image. 
            # TODO: Add OCR call if needed.
            
            # OCR Fallback
            image_text = ""
            # Check if we can run OCR
            if pytesseract and hasattr(pytesseract, "image_to_string"):
                try:
                    from PIL import Image as PILImage
                    img_pil = PILImage.open(io.BytesIO(img_bytes))
                    # Use provided path or default
                    tesseract_lang = self._get_tesseract_lang_code(source_lang)
                    image_text = pytesseract.image_to_string(img_pil, lang=tesseract_lang)
                    image_text = image_text.strip()
                    if image_text:
                        logger.info(f"OCR extracted text from image block {block_idx} on page {page_num}: {image_text[:50]}...")
                        # Translate OCR'd text
                        translated_image_text = image_text
                        if translation_manager:
                            try:
                                res = translation_manager.translate_text(image_text, source_lang, target_lang, engine)
                                if isinstance(res, str):
                                    translated_image_text = res
                                elif hasattr(res, "translated_text"):
                                    translated_image_text = res.translated_text
                                elif isinstance(res, dict):
                                    translated_image_text = res.get("translated_text") or res.get("text") or translated_image_text
                            except Exception as e:
                                logger.warning(f"OCR text translation failed: {e}")
                        else:
                            try:
                                from deep_translator import GoogleTranslator
                                translated_image_text = GoogleTranslator(source=source_lang, target=target_lang).translate(image_text)
                            except Exception as e:
                                logger.warning(f"OCR text translation (Emergency Fallback) failed: {e}")
                        
                        return {
                            "type": "image",
                            "bbox": bbox,
                            "image_data": base64.b64encode(img_bytes).decode(),
                            "image_text": image_text,
                            "translated_image_text": translated_image_text
                        }
                except Exception as e:
                    logger.warning(f"Tesseract OCR failed for image block {block_idx} on page {page_num}: {e}")


            return {
                "type": "image",
                "bbox": bbox,
                "image_data": base64.b64encode(img_bytes).decode()
            }
        except Exception as e:
            logger.warning(f"Image processing failed on page {page_num}: {e}")
            return {"type": "image", "bbox": block.get("bbox")}


    def _translate_image_content(self, img_bytes, target_lang):
        """
        Sends image to Gemini -> Gets Text Coordinates -> Inpaints -> Draws Translated Text
        """
        if genai is None:
            return None
            
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Analyze this image (it may be a table, diagram, or chart).
        1. Identify every text element.
        2. Translate the text to {target_lang}.
        3. Return a JSON list: [{{"box_2d": [ymin, xmin, ymax, xmax], "translated_text": "..."}}]
        4. Coordinates must be 0-1000 normalized.
        5. Return ONLY JSON.
        """
        
        try:
            # 1. Gemini Call
            response = model.generate_content([{'mime_type': 'image/png', 'data': img_bytes}, prompt])
            json_str = response.text.strip()
            if "```json" in json_str: json_str = json_str.split("```json")[1].split("```")[0]
            if "```" in json_str: json_str = json_str.replace("```", "")
            
            data = json.loads(json_str)
            
            # 2. Prepare Image for Inpainting
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            h, w = img_cv.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            pixel_data = []

            # 3. Create Mask & Parse Coords
            for item in data:
                ymin, xmin, ymax, xmax = item['box_2d']
                # Convert 0-1000 to pixels
                x1, y1 = int(xmin/1000 * w), int(ymin/1000 * h)
                x2, y2 = int(xmax/1000 * w), int(ymax/1000 * h)
                
                # Add to mask (remove old text)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
                # Detect text color (simple average)
                roi = img_cv[y1:y2, x1:x2]
                avg_color = np.mean(roi) if roi.size > 0 else 0
                text_color = (255, 255, 255) if avg_color < 128 else (0, 0, 0)
                
                pixel_data.append({
                    "bbox": (x1, y1, x2-x1, y2-y1), # x,y,w,h
                    "text": item.get('translated_text', ''),
                    "color": text_color
                })

            # 4. Inpaint
            dilated_mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
            img_inpainted = cv2.inpaint(img_cv, dilated_mask, 3, cv2.INPAINT_TELEA)
            
            # 5. Draw New Text
            img_pil = Image.fromarray(cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font_path = get_font_path(target_lang)

            for item in pixel_data:
                self._draw_fitted_text(draw, item['text'], item['bbox'], font_path, item['color'])
            
            # 6. Return Base64
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
            
        except Exception as e:
            print(f"Gemini Error: {e}")
            return None

    def _draw_fitted_text(self, draw, text, box, font_path, color):
        """Fits text into box using binary search for font size"""
        x, y, w, h = box
        min_s, max_s = 8, 100
        best_size = min_s
        best_lines = [text]
        best_font = ImageFont.load_default()

        # Binary Search
        low, high = min_s, min(max_s, int(h))
        while low <= high:
            mid = (low + high) // 2
            try: font = ImageFont.truetype(font_path, mid)
            except: font = ImageFont.load_default()
            
            avg_w = font.getlength("a") or 1
            chars_per_line = max(1, int(w / avg_w))
            lines = textwrap.wrap(text, width=chars_per_line)
            
            line_h = font.getbbox("Ay")[3]
            total_h = len(lines) * line_h
            max_line_w = max([font.getlength(l) for l in lines]) if lines else 0
            
            if total_h <= h and max_line_w <= w:
                best_size = mid
                best_lines = lines
                best_font = font
                low = mid + 1
            else:
                high = mid - 1
        
        # Draw Centered
        line_h = best_font.getbbox("Ay")[3]
        current_y = y + (h - (len(best_lines) * line_h)) // 2
        for line in best_lines:
            line_w = best_font.getlength(line)
            current_x = x + (w - line_w) // 2
            draw.text((current_x, current_y), line, font=best_font, fill=color)
            current_y += line_h

    def _get_tesseract_lang_code(self, lang_code: str) -> str:
        return _get_tesseract_lang_code(lang_code)

    def _translate_table(self, table_data: Dict, source_lang: str,
                        target_lang: str, engine: str) -> Dict:
        for row in table_data.get("rows", []):
            for cell in row.get("cells", []):
                original_text = cell.get("text", "")
                if original_text.strip():
                    try:
                        res = translation_manager.translate_text(original_text, source_lang, target_lang, engine) if translation_manager else None
                        if isinstance(res, str):
                            cell["translated_text"] = res
                        elif hasattr(res, "translated_text"):
                            cell["translated_text"] = res.translated_text
                        elif isinstance(res, dict):
                            cell["translated_text"] = res.get("translated_text") or original_text
                        else:
                            cell["translated_text"] = original_text
                    except Exception as e:
                        logger.warning(f"Table cell translation failed: {e}")
                        cell["translated_text"] = original_text
                else:
                    cell["translated_text"] = ""
        return table_data

    def _translate_flowchart_element(self, element: Dict, source_lang: str,
                                    target_lang: str, engine: str) -> Dict:
        original_text = element.get("text", "")
        if original_text.strip():
            try:
                res = translation_manager.translate_text(original_text, source_lang, target_lang, engine) if translation_manager else None
                if isinstance(res, str):
                    element["translated_text"] = res
                elif hasattr(res, "translated_text"):
                    element["translated_text"] = res.translated_text
                elif isinstance(res, dict):
                    element["translated_text"] = res.get("translated_text") or original_text
                else:
                    element["translated_text"] = original_text
            except Exception as e:
                logger.warning(f"Flowchart element translation failed: {e}")
                element["translated_text"] = original_text
        else:
            element["translated_text"] = ""
        return element

# ----------------------------
# Backwards-compatible module-level wrappers
# ----------------------------
def extract_blocks_from_pdf(pdf_file: Union[str, IO], source_lang: str = "en",
                            target_lang: str = "hi", engine: str = "gemini", debug: bool = False) -> Dict[str, Any]:
    processor = SmartPDFProcessor(debug_mode=debug)
    return processor.extract_blocks_from_pdf(pdf_file, source_lang, target_lang, engine)

def extract_text_for_validation(pdf_file: Union[str, IO]) -> str:
    """Extract plain text for quick validation (non-translated preview uses source==target)."""
    try:
        processor = SmartPDFProcessor(debug_mode=False)
        data = processor.extract_blocks_from_pdf(pdf_file, "en", "en", "google")
        text_content = []
        for page in data.get("pages", []):
            page_texts = []
            for block in page.get("blocks", []):
                if block.get("type") == "text" and block.get("text"):
                    page_texts.append(block["text"])
                elif block.get("type") == "image" and block.get("image_text"):
                    page_texts.append(f"[Image: {block['image_text']}]")
                elif block.get("type") == "table":
                    # flatten table translated cells for preview
                    for r in block.get("rows", []):
                        row_texts = [c.get("translated_text", c.get("text", "")) for c in r.get("cells", [])]
                        page_texts.append(" | ".join(row_texts))
                elif block.get("type") == "flowchart_node":
                    page_texts.append(f"[Flowchart: {block.get('translated_text', block.get('text',''))}]")
            if page_texts:
                page_text = "\n".join(page_texts)
                text_content.append(f"--- Page {page.get('number','?')} ---\n{page_text}")
        return "\n\n".join(text_content)
    except Exception as e:
        logger.error(f"extract_text_for_validation failed: {e}", exc_info=True)
        return ""

# ----------------------------
# Rebuild PDF preserving layout
# ----------------------------
def rebuild_pdf(pages: List[Dict[str, Any]], original_pdf_path: Union[str, bytes], target_lang: str = "default") -> bytes:
    """
    Rebuild the PDF by importing original pages and overlaying translated images/text.
    - pages: output from extract_blocks_from_pdf
    - original_pdf_path: path or bytes of original PDF
    Returns bytes of the rebuilt PDF.
    """
    try:
        if isinstance(original_pdf_path, (bytes, bytearray)):
            orig_doc = fitz.open(stream=original_pdf_path, filetype="pdf")
        else:
            orig_doc = fitz.open(original_pdf_path)
        out_doc = fitz.open()

        for page_info in pages:
            page_num = page_info.get("number", 1) - 1
            try:
                orig_page = orig_doc.load_page(page_num)
                rect = orig_page.rect
                new_page = out_doc.new_page(width=rect.width, height=rect.height)
                new_page.show_pdf_page(rect, orig_doc, page_num)
            except Exception:
                new_page = out_doc.new_page()

            # Insert images first (modified overlays)
            for block in page_info.get("blocks", []):
                if block.get("type") == "image":
                    # prefer image_data if present (base64)
                    img_bytes = None
                    if block.get("image_data"):
                        try:
                            img_bytes = base64.b64decode(block["image_data"])
                        except Exception:
                            img_bytes = None
                    # older field original_base64 renamed 'modified_image' in some variants
                    if not img_bytes and block.get("modified_image"):
                        img_bytes = block.get("modified_image")
                    # If we have bytes, insert image in the bbox
                    if img_bytes:
                        try:
                            bbox = block.get("bbox")
                            if bbox:
                                img_rect = fitz.Rect(*bbox)
                                new_page.insert_image(img_rect, stream=img_bytes)
                        except Exception as e:
                            logger.warning(f"Failed to insert image on rebuilt PDF page {page_num+1}: {e}", exc_info=True)

            # Insert text blocks (so text overlays appear above images if needed)
            for block in page_info.get("blocks", []):
                if block.get("type") == "text":
                    try:
                        bbox = block.get("bbox")
                        if not bbox:
                            continue
                        rect = fitz.Rect(*bbox)
                        text = block.get("text", "")
                        height = rect.height if rect.height > 0 else 12
                        fontsize = max(8, min(int(height * 0.7), 36))
                        fontname = block.get("font", "helv")
                        try:
                            new_page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname, align=0)
                        except Exception:
                            new_page.insert_textbox(rect, text, fontsize=fontsize, fontname="helv", align=0)
                    except Exception as e:
                        logger.warning(f"Failed to insert text on rebuilt PDF page {page_num+1}: {e}", exc_info=True)

            # Insert tables (if any)
            for block in page_info.get("blocks", []):
                if block.get("type") == "table":
                    try:
                        bbox = block.get("bbox")
                        rect = fitz.Rect(*bbox)
                        # Render table rows as text inside rect; a better option is to draw lines and cells,
                        # but for simplicity, insert flat text representation.
                        rows_text = []
                        for r in block.get("rows", []):
                            row_cells = [c.get("translated_text", c.get("text", "")) for c in r.get("cells", [])]
                            rows_text.append(" | ".join(row_cells))
                        table_text = "\n".join(rows_text)
                        fontsize = max(8, min(int(rect.height / max(len(rows_text), 1) * 0.6), 12))
                        new_page.insert_textbox(rect, table_text, fontsize=fontsize, fontname="helv", align=0)
                    except Exception as e:
                        logger.warning(f"Failed to insert table on page {page_num+1}: {e}", exc_info=True)

            # Insert flowchart nodes (simple overlay)
            for block in page_info.get("blocks", []):
                if block.get("type") == "flowchart_node":
                    try:
                        bbox = block.get("bbox")
                        rect = fitz.Rect(*bbox)
                        text = block.get("translated_text", block.get("text", ""))
                        fontsize = max(8, min(int(rect.height * 0.5), 14))
                        new_page.insert_textbox(rect, text, fontsize=fontsize, fontname="helv", align=1)
                    except Exception as e:
                        logger.warning(f"Failed to insert flowchart node on page {page_num+1}: {e}", exc_info=True)

        out_bytes = out_doc.write()
        out_doc.close()
        orig_doc.close()
        return out_bytes

    except Exception as e:
        logger.error(f"rebuild_pdf failed: {e}", exc_info=True)
        raise

# End of pdf2text.py
