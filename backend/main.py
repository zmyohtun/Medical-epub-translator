"""
FastAPI backend for the Medical PDF -> Structured EPUB -> Myanmar Translator.
"""

import os
import shutil
import threading
import uuid
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.epub_translator import fix_epub_xhtml, inspect_epub_chapters, translate_epub
from backend.pdf_to_epub import convert_pdf_to_epub

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(BASE_DIR, "workspace_files")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Medical PDF/EPUB Myanmar Translator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory job tracker
jobs: dict[str, dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/pdf-to-epub")
async def api_pdf_to_epub(
    file: UploadFile = File(...),
    title: str = Form(""),
    author: str = Form(""),
    pages_per_chapter: int = Form(15),
    engine: str = Form("calibre"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid .pdf file.")

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(WORK_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    pdf_path = os.path.join(job_dir, file.filename)
    base_name = os.path.splitext(file.filename)[0]
    epub_filename = f"{base_name}_structured.epub"
    epub_path = os.path.join(job_dir, epub_filename)

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "status": "running",
        "step": "pdf_to_epub",
        "current": 0,
        "total": 1,
        "message": "Starting PDF to EPUB conversion...",
        "output_file": epub_filename,
        "output_path": epub_path,
        "error": None,
    }

    def _run():
        try:
            def _cb(cur: int, tot: int, msg: str):
                jobs[job_id]["current"] = cur
                jobs[job_id]["total"] = tot
                jobs[job_id]["message"] = msg

            convert_pdf_to_epub(
                pdf_path=pdf_path,
                output_epub_path=epub_path,
                title=title.strip() or None,
                author=author.strip() or None,
                pages_per_chapter=max(1, pages_per_chapter),
                engine=engine,
                progress_callback=_cb,
            )
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "Structured EPUB created successfully!"
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["message"] = f"Error: {e}"

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/inspect-epub")
async def api_inspect_epub(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="Please upload a valid .epub file.")
    tmp_id = str(uuid.uuid4())[:8]
    tmp_dir = os.path.join(WORK_DIR, tmp_id)
    os.makedirs(tmp_dir, exist_ok=True)
    epub_path = os.path.join(tmp_dir, file.filename)
    with open(epub_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chapters = inspect_epub_chapters(epub_path)
    return {"chapters": chapters}


@app.post("/api/fix-epub-xhtml")
async def api_fix_epub_xhtml(
    api_key: str = Form(...),
    model_name: str = Form("gemini-2.5-flash"),
    provider: str = Form("gemini"),
    base_url: str = Form(""),
    source_job_id: str = Form(""),
    file: Optional[UploadFile] = File(None),
):
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="API Key is required.")
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(WORK_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    if file and file.filename:
        input_epub_path = os.path.join(job_dir, file.filename)
        with open(input_epub_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        base_name = os.path.splitext(file.filename)[0]
    elif source_job_id and source_job_id in jobs:
        input_epub_path = jobs[source_job_id]["output_path"]
        base_name = os.path.splitext(jobs[source_job_id]["output_file"])[0]
    else:
        raise HTTPException(status_code=400, detail="Provide an EPUB file or completed Step 1 job.")

    out_filename = f"{base_name}_fixed.epub"
    out_path = os.path.join(job_dir, out_filename)
    jobs[job_id] = {
        "status": "running",
        "step": "fix_xhtml",
        "current": 0,
        "total": 1,
        "message": "Starting Fix with AI (XHTML Repair)...",
        "output_file": out_filename,
        "output_path": out_path,
        "error": None,
    }

    def _run():
        try:
            def _cb(cur: int, tot: int, msg: str):
                jobs[job_id]["current"] = cur
                jobs[job_id]["total"] = tot
                jobs[job_id]["message"] = msg

            fix_epub_xhtml(
                input_epub_path=input_epub_path,
                output_epub_path=out_path,
                api_key=api_key.strip(),
                model_name=model_name.strip(),
                provider=provider,
                base_url=base_url,
                progress_callback=_cb,
            )
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "XHTML Repair completed!"
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["message"] = f"Error: {e}"

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/translate-epub")
async def api_translate_epub(
    api_key: str = Form(...),
    model_name: str = Form("gemini-2.5-flash"),
    provider: str = Form("gemini"),
    base_url: str = Form(""),
    bilingual_terms: bool = Form(True),
    skip_references: bool = Form(True),
    selected_chapters: str = Form(""),
    custom_notes: str = Form(""),
    source_job_id: str = Form(""),
    file: Optional[UploadFile] = File(None),
):
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="Gemini API Key is required.")

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(WORK_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    if file and file.filename:
        if not file.filename.lower().endswith(".epub"):
            raise HTTPException(status_code=400, detail="Please upload a valid .epub file.")
        input_epub_path = os.path.join(job_dir, file.filename)
        with open(input_epub_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        base_name = os.path.splitext(file.filename)[0]
    elif source_job_id and source_job_id in jobs:
        src = jobs[source_job_id]
        if src.get("status") != "completed" or not os.path.exists(src.get("output_path", "")):
            raise HTTPException(status_code=400, detail="Source EPUB job is not ready.")
        input_epub_path = src["output_path"]
        base_name = os.path.splitext(src["output_file"])[0]
    else:
        raise HTTPException(status_code=400, detail="Provide either an EPUB file or a completed Step 1 job.")

    out_filename = f"{base_name}_myanmar.epub"
    out_path = os.path.join(job_dir, out_filename)
    ckpt_dir = os.path.join(WORK_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, f"{base_name}.json")

    jobs[job_id] = {
        "status": "running",
        "step": "translate_epub",
        "current": 0,
        "total": 1,
        "message": "Starting Myanmar medical translation...",
        "output_file": out_filename,
        "output_path": out_path,
        "error": None,
    }

    def _run():
        try:
            def _cb(cur: int, tot: int, msg: str):
                jobs[job_id]["current"] = cur
                jobs[job_id]["total"] = tot
                jobs[job_id]["message"] = msg

            parsed_chapters = (
                [int(x.strip()) for x in selected_chapters.split(",") if x.strip().isdigit()]
                if selected_chapters.strip()
                else None
            )
            translate_epub(
                input_epub_path=input_epub_path,
                output_epub_path=out_path,
                api_key=api_key.strip(),
                model_name=model_name.strip(),
                provider=provider,
                base_url=base_url,
                bilingual_terms=bilingual_terms,
                skip_references=skip_references,
                selected_chapters=parsed_chapters,
                custom_notes=custom_notes,
                checkpoint_path=checkpoint_path,
                progress_callback=_cb,
            )
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "Myanmar translation completed!"
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["message"] = f"Error: {e}"

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    j = jobs[job_id]
    return {
        "job_id": job_id,
        "status": j["status"],
        "step": j["step"],
        "current": j["current"],
        "total": j["total"],
        "message": j["message"],
        "output_file": j["output_file"],
        "error": j["error"],
    }


@app.get("/api/download/{job_id}")
def api_download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    j = jobs[job_id]
    if j["status"] != "completed" or not os.path.exists(j["output_path"]):
        raise HTTPException(status_code=400, detail="File not ready for download.")
    return FileResponse(
        path=j["output_path"],
        filename=j["output_file"],
        media_type="application/epub+zip",
    )
