# 🩺📚 Medical PDF → Structured EPUB → Myanmar Translator (v1.01)

A fast, medical-friendly web application that converts complex **Medical PDF textbooks** into **structured EPUB** files and translates them into **natural, accurate Myanmar** while preserving medical terminology, tables, figures, and formatting.

---

## ✨ What This Software Does

Medical textbooks often contain dense multi-column layouts, clinical tables, diagrams, and specialized terminology that standard translators break. This tool solves that with a **2-step pipeline**:

1. **📄 PDF → Structured EPUB**
   - Supports **Calibre (`ebook-convert`)** for high-fidelity layout/table/image extraction, with automatic fallback to **PyMuPDF (`fitz`)**.
   - Preserves chapter structure, images, and XHTML formatting.

2. **🇲🇲 Structured EPUB → Myanmar Translation**
   - **Single-Pass Medical Textbook Protocol**: Reconstructs fragmented PDF paragraphs/tables and translates into clear Myanmar in one pass.
   - **Bilingual Medical Terms**: Keeps English medical terms in parentheses (e.g., `နှလုံးကြွက်သားပုပ်ခြင်း (Myocardial Infarction)`) for clinical accuracy.
   - **Smart Section Skipping**: Automatically skips References, Bibliography, and Index sections to save time and API cost.
   - **High-Speed Parallelism & Checkpoints**: Translates chunks concurrently (`ThreadPoolExecutor`) with automatic retries and book-level checkpoints so interrupted jobs resume without re-paying for completed chapters.
   - **Flexible AI Providers**: Works with **Google Gemini** and any **OpenAI-compatible API** (such as **OpenRouter**).

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **PDF / EPUB Processing**: Calibre (`ebook-convert`), PyMuPDF (`fitz`), `ebooklib`, `beautifulsoup4`, `lxml`
- **AI / LLM**: Google GenAI SDK (`google-genai`), OpenAI-compatible REST APIs
- **Frontend**: Vanilla HTML/CSS/JS with `localStorage` settings persistence

---

## 🚀 Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web server
uvicorn backend.main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

---

## 🙏 Acknowledgements

Special thanks to everyone whose work made this project possible:

- **🇲🇲 Saturngod (`Sagar`)** — For the inspiring architecture and vision behind the `Sagar` EPUB translator app.
- **🐍 Python & PyMuPDF (`fitz`)** — Special tribute for powering **v1.01**: world-standard PDF table extraction, clustered 180 DPI Figure-Region Snapshots, running header/footer filtering, and clean line-by-line reference preservation.
- **🤖 Google Gemini & OpenRouter** — For fast, high-quality medical translation capabilities.
- **🧰 FastAPI & ebooklib** — For the clean Python web and EPUB tooling ecosystem.
