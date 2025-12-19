
# vertex_sample.py
import os
from vertexai import init as vertexai_init

# Import your utils (which now use Vertex AI GenerativeModel internally)
from .utils import chunk_text, gemini_translate_text, gemini_translate_chunks

# --- Vertex AI initialization (service account via GOOGLE_APPLICATION_CREDENTIALS) ---
# Make sure .env has:
#   GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
#   GCP_PROJECT_ID=ai-led-drug-discovery-dev
#   GCP_LOCATION=us-central1
vertexai_init(
    project=os.getenv("GCP_PROJECT_ID", "ai-led-drug-discovery-dev"),
    location=os.getenv("GCP_LOCATION", "us-central1")
)

sample_text = """Hello world.
This is a test translation.
Let's see if Gemini can translate it to Hindi."""

print("🔹 Using Gemini (short text):")
print(gemini_translate_text(sample_text, "en", "hi"))

print("\n🔹 Using Gemini (chunked text):")
chunks = chunk_text(sample_text, max_chars=50)
