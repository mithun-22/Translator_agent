
# app.py — Streamlit front-end integrated with translator.utils and OCR
# Path: C:\Users\p90023739\Documents\Doc_trans\Translator\Translator\app.py
import os
import sys
import io
import tempfile
import time
from pathlib import Path
import streamlit as st

# ------------------------------------------------------------------------
# Import paths
# ------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# -------------------- Optional Django Setup --------------------
try:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
    import django
    django.setup()
    HAS_DJANGO = True
except Exception:
    HAS_DJANGO = False

# -------------------- Optional Gemini / Vertex AI credentials setup --------------------
# This block loads environment variables (optionally from .env),
# sets GOOGLE_APPLICATION_CREDENTIALS to your service account JSON,
# and initializes Vertex AI for Gemini calls used in translator.utils.

# Optional: load .env in development (won't break if missing)
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# Default values (can be overridden by .env or system env)
DEFAULT_PROJECT = os.getenv("GCP_PROJECT_ID", "ai-led-drug-discovery-dev")   # from credentials.json
DEFAULT_LOCATION = os.getenv("GCP_LOCATION", "us-central1")  # common Vertex AI region

# If a local credentials file is present, set the env var so Google SDK uses it
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    cred_path = os.path.join(ROOT, "credentials.json")
    if os.path.exists(cred_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

# Initialize Vertex AI once (safe to call multiple times; lightweight)
try:
    from vertexai import init as vertexai_init
    vertexai_init(project=DEFAULT_PROJECT, location=DEFAULT_LOCATION)
except Exception as _vertex_err:
    # Don't crash UI—Gemini path will raise a clear error on use; log for debug
    import logging as _logging
    _logging.getLogger(__name__).warning(f"Vertex AI init warning: {_vertex_err}")
# ----------------------------------------------------------------------------------------

# ------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------
# Only used when engine == "google"
from deep_translator import GoogleTranslator
# Utilities from your fixed translator.utils
# Local imports
from utils import (
    LANGUAGE_SLUGS,
    chunk_text,
    translate_chunks,
    gemini_translate_text,
    gemini_translate_chunks,
    translate_text_simple,
    rebuild_pdf,
    estimate_tokens_and_cost,
)

from PDF2txt import (
    extract_text_for_validation,
    extract_blocks_from_pdf
)
# Optional: a Django model you already use for stats
try:
    from translator.models import Translation
except Exception:
    Translation = None  # allow UI to start even if DB/model isn't ready

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# Helper: save bytes to a temp PDF path (used by rebuild)
# ------------------------------------------------------------------------
def save_bytes_to_temp_pdf(b: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(b)
    tmp.flush()
    tmp.close()
    return tmp.name

# ------------------------------------------------------------------------
# Helper: generate a simple PDF from text (for Text Input path)
# ------------------------------------------------------------------------
def generate_pdf_from_text(text: str, target_lang: str = "default") -> bytes:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.pdfmetrics import stringWidth

    try:
        from django.conf import settings
        FONTS = getattr(settings, "FONTS", {})
        FONT_PATH = getattr(settings, "FONT_PATH", "")
    except Exception:
        FONTS, FONT_PATH = {}, ""

    # Pick a font for the target language (falls back to Helvetica)
    try:
        font_name, font_file = FONTS.get(target_lang, FONTS.get("default", ("Helvetica", "")))
    except Exception:
        font_name, font_file = ("Helvetica", "")
    font_path = os.path.join(FONT_PATH, font_file) if FONT_PATH and font_file else None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    try:
        if font_path and os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            c.setFont(font_name, 12)
        else:
            c.setFont("Helvetica", 12)
    except Exception as e:
        logger.warning(f"Font setup failed: {e}")
        c.setFont("Helvetica", 12)

    max_width = width - 100
    x, y = 50, height - 50
    line_height = 16

    for paragraph in text.split("\n\n"):
        if not paragraph.strip():
            continue
        for line in paragraph.split("\n"):
            words = line.split(" ")
            current_line = ""
            for word in words:
                trial = (current_line + " " + word).strip()
                try:
                    if stringWidth(trial, font_name, 12) <= max_width:
                        current_line = trial
                    else:
                        if current_line:
                            c.drawString(x, y, current_line)
                            y -= line_height
                        current_line = word
                        if y < 50:
                            c.showPage()
                            c.setFont(font_name, 12)
                            y = height - 50
                except Exception:
                    # Fallback width estimate if stringWidth unavailable
                    if len(trial) * 7 <= max_width:
                        current_line = trial
                    else:
                        if current_line:
                            c.drawString(x, y, current_line)
                            y -= line_height
                        current_line = word
                        if y < 50:
                            c.showPage()
                            c.setFont(font_name, 12)
                            y = height - 50
            if current_line:
                c.drawString(x, y, current_line)
                y -= line_height
                if y < 50:
                    c.showPage()
                    c.setFont(font_name, 12)
                    y = height - 50

    y -= line_height // 2
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------
st.set_page_config(page_title="Translator", layout="wide", page_icon="🌐")
st.title("🌐 Translator")
st.caption("Enhanced PDF translation with OCR and smart batching")

with st.sidebar:
    st.header("📊 Stats")
    try:
        if Translation:
            total_translations = Translation.objects.count()
            st.metric("Total Translations", total_translations)
        else:
            st.metric("Total Translations", "N/A")
    except Exception as e:
        st.metric("Status", "DB Error")
        logger.warning(f"Stats error: {e}")

    st.divider()
    st.header("⚙️ Options")
    
    # Dependency Checks
    with st.expander("System Health", expanded=False):
        # Deep Translator
        try:
            import deep_translator
            st.success("✔ deep_translator")
        except ImportError:
            st.error("❌ deep_translator missing")
            
        # Vertex AI
        try:
            import vertexai
            st.success("✔ vertexai")
        except ImportError:
            st.warning("⚠ vertexai missing (using fallback)")
            
        # Tesseract
        import shutil
        if shutil.which("tesseract") or os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
             st.success("✔ Tesseract found")
        else:
             st.warning("⚠ Tesseract not found in PATH")

    # Let the user choose how the output PDF is rebuilt
    pdf_rebuild_mode = st.selectbox(
        "PDF Rebuild Mode",
        options=["overlay (layout-preserving)", "reportlab (paragraph-wrapping)", "hybrid (overlay + reportlab)"],
        index=0
    )
    rebuild_mode = (
        "overlay" if pdf_rebuild_mode.startswith("overlay")
        else "reportlab" if pdf_rebuild_mode.startswith("reportlab")
        else "hybrid"
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input")
    input_method = st.radio("Choose input method:", ["Text Input", "PDF Upload"])
    source_text = ""
    uploaded_file = None

    if input_method == "Text Input":
        source_text = st.text_area("Enter text to translate:", height=250)
    else:
        uploaded_file = st.file_uploader("Upload PDF or DOCX file:", type=["pdf", "docx"])
        if uploaded_file:
            st.success(f"File loaded: {uploaded_file.name}")

    st.markdown("### 🌍 Languages")
    languages = list(LANGUAGE_SLUGS.items())
    lang_dict = dict(languages)

    col_from, col_to = st.columns(2)
    with col_from:
        source_lang = st.selectbox(
            "From:",
            options=[code for code, _ in languages],
            index=0,
            format_func=lambda x: lang_dict[x]
        )
    with col_to:
        target_lang = st.selectbox(
            "To:",
            options=[code for code, _ in languages],
            index=3 if len(languages) > 3 else 1,
            format_func=lambda x: lang_dict[x]
        )

    engine = st.selectbox("Translation Engine:", ["gemini", "google"], index=0)

    # Action
    if st.button("🚀 Translate", type="primary", use_container_width=True):
        start_time = time.time()
        progress = st.progress(0)
        status = st.empty()

        if not source_text and not uploaded_file:
            st.error("Please provide text or upload a file.")
        elif source_lang == target_lang:
            st.error("Source and target languages must be different.")
        else:
            try:
                # File Upload path (PDF or DOCX)
                if uploaded_file:
                    file_ext = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_ext == "pdf":
                        status.info("🔍 Analyzing PDF...")
                        progress.progress(10)
                        pdf_bytes = uploaded_file.read()
                        buffer = io.BytesIO(pdf_bytes)

                        status.info("🧠 Processing with OCR and Layout Analysis...")
                        progress.progress(30)

                        # This function should internally OCR pages and return structured blocks + metadata
                        data = extract_blocks_from_pdf(
                            buffer,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            engine=engine,
                            debug=True
                        )
                        progress.progress(60)

                        pages = data.get("pages", [])
                        metadata = data.get("metadata", {})
                        if not pages:
                            st.error("No content could be extracted from the PDF.")
                        else:
                            status.info("📄 Rebuilding PDF with 1:1 Layout...")
                            progress.progress(85)

                            tmp_path = save_bytes_to_temp_pdf(pdf_bytes)
                            try:
                                # Use master dispatcher with selected mode
                                pdf_out = rebuild_pdf(pages, tmp_path, target_lang, mode=rebuild_mode)
                                st.session_state["translated_pdf_bytes"] = pdf_out
                                st.session_state["output_filename"] = f"translated_{uploaded_file.name}"
                                
                                # Show Stats
                                text_tokens = metadata.get("text_tokens", 0)
                                image_tokens = metadata.get("image_tokens", 0)
                                total_tokens = metadata.get("total_tokens", 0)
                                cost = metadata.get("estimated_cost_usd", 0.0)
                                
                                st.divider()
                                k1, k2, k3, k4 = st.columns(4)
                                k1.metric("Text Tokens", text_tokens)
                                k2.metric("Image Tokens", image_tokens)
                                k3.metric("Total Tokens", total_tokens)
                                k4.metric("Est. Cost", f"${cost:.4f}")
                                st.divider()
                            finally:
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass
                    
                    elif file_ext == "docx":
                        status.info("🔍 Analyzing DOCX...")
                        progress.progress(20)
                        from DOCX2txt import translate_docx
                        
                        status.info("🌐 Translating Word Document...")
                        progress.progress(50)
                        docx_out = translate_docx(
                            uploaded_file,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            engine=engine
                        )
                        
                        st.session_state["translated_docx_bytes"] = docx_out
                        st.session_state["output_filename"] = f"translated_{uploaded_file.name}"
                        progress.progress(100)

                    elapsed = time.time() - start_time
                    status.empty()
                    st.success(
                        f"✅ Document translated successfully! "
                        f"Time: {elapsed:.1f}s"
                    )

                # Text Input path
                else:
                    status.info("🌐 Translating text...")
                    progress.progress(30)

                    if engine == "google":
                        if len(source_text) <= 4000:
                            translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(source_text)
                        else:
                            translated_text = translate_chunks(chunk_text(source_text), source_lang, target_lang)
                    else:  # gemini
                        if len(source_text) <= 4000:
                            translated_text = gemini_translate_text(source_text, source_lang, target_lang)
                        else:
                            translated_text = gemini_translate_chunks(chunk_text(source_text), source_lang, target_lang)

                    progress.progress(80)

                    st.session_state["translated_text"] = translated_text
                    
                    # Calculate stats
                    stats = estimate_tokens_and_cost(len(source_text), 0)
                    
                    st.session_state["translation_metadata"] = {
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "engine": engine,
                        "character_count": len(source_text),
                        "processing_time": time.time() - start_time,
                        **stats
                    }

                    progress.progress(100)
                    status.empty()
                    st.success("✅ Text translated successfully.")

            except Exception as e:
                status.empty()
                progress.empty()
                st.error(f"Translation failed: {str(e)}")
                logger.error(f"Translation error: {e}", exc_info=True)


with col2:
    st.subheader("📄 Results")
    # Text result + download as PDF
    if st.session_state.get("translated_text"):
        translated_text = st.session_state["translated_text"]
        metadata = st.session_state.get("translation_metadata", {})
        
        # Stats display for text
        if "text_tokens" in metadata:
             st.caption(f"Tokens: {metadata['total_tokens']} | Cost: ${metadata['estimated_cost_usd']:.5f}")

        st.text_area("Translated Text:", translated_text, height=250)


        pdf_bytes = generate_pdf_from_text(translated_text, metadata.get("target_lang", "default"))
        st.download_button(
            "📄 Download text as PDF",
            data=pdf_bytes,
            file_name="translated_text.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # PDF result (rebuild output)
    if st.session_state.get("translated_pdf_bytes"):
        st.markdown("### 📄 Translated PDF Ready")
        st.download_button(
            "📥 Download Translated PDF",
            data=st.session_state["translated_pdf_bytes"],
            file_name=st.session_state.get("output_filename", "translated_document.pdf"),
            mime="application/pdf",
            use_container_width=True
        )

    # DOCX result
    if st.session_state.get("translated_docx_bytes"):
        st.markdown("### 📝 Translated DOCX Ready")
        st.download_button(
            "📥 Download Translated DOCX",
            data=st.session_state["translated_docx_bytes"],
            file_name=st.session_state.get("output_filename", "translated_document.docx"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
