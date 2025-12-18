# views.py - Fully Updated for utils.py (Production Ready)
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
import io
import os
import logging

from .models import Translation
from .forms import TranslationForm
from .PDF2text import extract_blocks_from_pdf

# Utils (aligned with your new implementation)
from .utils import (
    chunk_text,
    translate_chunks,
    gemini_translate_text,
    gemini_translate_chunks,
    translate_text_simple,
    rebuild_pdf
)

from deep_translator import GoogleTranslator
from django.conf import settings

logger = logging.getLogger(__name__)

FONTS = settings.FONTS
FONT_PATH = settings.FONT_PATH


def translate_view(request):
    """
    Handles text translation and PDF translation using the new block pipeline.
    """
    translated_text = ''
    form = TranslationForm()

    if request.method == 'POST':
        form = TranslationForm(request.POST, request.FILES)

        if form.is_valid():
            source_text = form.cleaned_data.get('source_text')
            source_file = form.cleaned_data.get('source_file')
            source_lang = form.cleaned_data['source_lang']
            target_lang = form.cleaned_data['target_lang']
            engine = form.cleaned_data['engine']

            try:
                # =========================
                # PDF Upload Translation
                # =========================
                if source_file:
                    logger.info(f"[PDF] Processing uploaded PDF: {source_file.name}")

                    result = extract_blocks_from_pdf(
                        source_file,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        engine=engine,
                        debug=False
                    )

                    pages = result.get("pages", [])
                    metadata = result.get("metadata", {})

                    if not pages:
                        messages.error(request, "No text was extracted from the PDF.")
                        return render(request, 'translator/translate.html', {'form': form})

                    # Save translated pages to session
                    request.session["translated_pages"] = pages
                    request.session["translation_metadata"] = metadata

                    # Save original PDF for rebuild
                    temp_pdf_path = os.path.join(settings.MEDIA_ROOT, f"temp_{request.session.session_key}.pdf")
                    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
                    with open(temp_pdf_path, "wb") as f:
                        for chunk in source_file.chunks():
                            f.write(chunk)
                    request.session["original_pdf_path"] = temp_pdf_path

                    messages.success(request, f"PDF translated successfully!")
                    return redirect("download_pdf")

                # =========================
                # Pure Text Translation
                # =========================
                else:
                    if engine == "google":
                        if len(source_text) <= 4000:
                            translated_text = GoogleTranslator(
                                source=source_lang,
                                target=target_lang
                            ).translate(source_text)
                        else:
                            chunks = chunk_text(source_text)
                            translated_text = translate_chunks(chunks, source_lang, target_lang)

                    elif engine == "gemini":
                        if len(source_text) <= 4000:
                            translated_text = gemini_translate_text(source_text, source_lang, target_lang)
                        else:
                            chunks = chunk_text(source_text)
                            translated_text = gemini_translate_chunks(chunks, source_lang, target_lang)

                    Translation.objects.create(
                        source_text=source_text[:5000],
                        source_lang_slug=source_lang,
                        target_lang_slug=target_lang,
                        translated_text=translated_text[:5000]
                    )

                    request.session["translated_text_only"] = translated_text
                    messages.success(request, "Text translated successfully!")

            except Exception as e:
                logger.error(f"Translation Error: {e}", exc_info=True)
                messages.error(request, f"Translation failed: {str(e)}")

    return render(request, 'translator/translate.html', {
        'form': form,
        'translated_text': translated_text,
    })


def download_pdf(request):
    """
    Generates PDF download using new span-based rebuild logic from utils.py
    """

    try:
        pages = request.session.get("translated_pages", [])
        original_pdf_path = request.session.get("original_pdf_path")
        translated_text = request.session.get("translated_text_only")

        # ===========================
        # Case 1 : PDF Rebuild
        # ===========================
        if pages and original_pdf_path:
            target_lang = request.session.get("translation_metadata", {}).get("target_lang", "default")

            logger.info("[PDF] Rebuilding translated PDF...")

            pdf_output = rebuild_pdf(
                pages=pages,
                original_pdf_path=original_pdf_path,
                target_lang=target_lang,
                mode="unified"
            )

            # cleanup
            try:
                os.remove(original_pdf_path)
            except Exception as e:
                logger.warning(f"Could not remove temp file: {e}")

            response = HttpResponse(pdf_output, content_type="application/pdf")
            response['Content-Disposition'] = "inline; filename=translated.pdf"
            response['Content-Length'] = len(pdf_output)
            return response

        # ===========================
        # Case 2 : Plain Text â†’ PDF
        # ===========================
        if translated_text:
            buffer = io.BytesIO()

            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.pdfbase import pdfmetrics

            c = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4

            target_lang = request.session.get("translation_metadata", {}).get("target_lang", "default")

            try:
                font_name, font_file = FONTS.get(target_lang, FONTS["default"])
            except:
                font_name, font_file = FONTS["default"]

            pdfmetrics.registerFont(TTFont(font_name, os.path.join(FONT_PATH, font_file)))
            c.setFont(font_name, 12)

            x, y = 50, height - 50
            line_height = 15
            max_width = width - 100

            from reportlab.pdfbase.pdfmetrics import stringWidth

            for para in translated_text.split("\n"):
                words = para.split(" ")
                line = ""

                for word in words:
                    test_line = f"{line} {word}".strip()
                    if stringWidth(test_line, font_name, 12) <= max_width:
                        line = test_line
                    else:
                        c.drawString(x, y, line)
                        y -= line_height
                        line = word
                        if y < 50:
                            c.showPage()
                            c.setFont(font_name, 12)
                            y = height - 50
                if line:
                    c.drawString(x, y, line)
                    y -= line_height

            c.save()
            buffer.seek(0)

            response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
            response['Content-Disposition'] = "inline; filename=translated_text.pdf"
            return response

        return HttpResponse("Nothing to download. Please translate first.", status=400)

    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return HttpResponse(f"Unable to download: {e}", status=500)
