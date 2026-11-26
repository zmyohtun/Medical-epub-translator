"""
Step-by-step Medical EPUB Translator (English -> Myanmar)
Preserves HTML structure, tables, figures, annotations, and supports
bilingual medical terminology + resumable chapter checkpoints.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from google import genai
from google.genai import types


SKIP_SECTION_PATTERN = re.compile(
    r"^\s*(references|bibliography|further\s+reading|suggested\s+reading|index|acknowledgments?|glossary\s+of\s+abbreviations)\s*$",
    re.IGNORECASE,
)


def _is_reference_section(body_soup: BeautifulSoup) -> bool:
    """Detect if a chapter/section is a References or Index section that should not be translated."""
    first_heading = body_soup.find(["h1", "h2", "h3"])
    if first_heading:
        heading_text = first_heading.get_text(" ", strip=True)
        if SKIP_SECTION_PATTERN.match(heading_text):
            return True
    return False


def _build_system_prompt(bilingual_terms: bool, custom_notes: str = "") -> str:
    bilingual_rule = (
        "For key medical, anatomical, pharmacological, and clinical terms, "
        "provide the Myanmar translation followed by the original English term in parentheses, "
        "e.g., 'နှလုံးကြွက်သားပုပ်ခြင်း (Myocardial Infarction)'. "
        "Keep drug names, chemical formulas, lab units (mg/dL, mmol/L), and abbreviations (ECG, CT, MRI) in English."
        if bilingual_terms
        else "Translate medical terms naturally into Myanmar, keeping drug names, lab units, and standard abbreviations in English."
    )
    extra = f"\nAdditional user instructions:\n{custom_notes}" if custom_notes.strip() else ""
    return (
        "You are an expert medical textbook translator fluent in both English and Myanmar (Burmese).\n"
        "Your task is to translate the provided XHTML body content from English into clear, accurate, "
        "academic Myanmar suitable for physicians and medical students.\n\n"
        "MEDICAL TEXTBOOK PROTOCOL:\n"
        "1. Preserve ALL HTML tags, attributes, classes, IDs, <table>, <tr>, <th>, <td>, <figure>, <img>, "
        "<div class='annotation'>, <h1>-<h6>, <p>, <ul>, <ol>, <li>, etc. EXACTLY as they are.\n"
        "2. Translate ONLY the human-readable text inside the tags. Do NOT translate tag names, attributes, or image src paths.\n"
        "3. Major Chapter Headings (<h1>, <h2>): Translate clearly and keep the English heading in parentheses if helpful.\n"
        "4. Tables & Fragmented/Flattened Tables: Keep every <tr>, <th>, and <td> cell aligned. Do NOT merge or drop cells. "
        "If a table was flattened into plain <p> lines during PDF extraction (e.g., 'Table X.Y' followed by row lines), "
        "reconstruct it into a clean HTML <table> with <thead>/<tbody>/<tr>/<th>/<td> AND translate it in this single pass.\n"
        "5. Broken PDF Line Wraps: Merge fragmented <p> lines that belong to the same sentence/paragraph.\n"
        "6. Figures & Diagrams: Keep <img> tags untouched; translate <figcaption> and figure descriptions.\n"
        "7. Citations & References: Keep inline citations like [1, 2] or (Smith et al., 2023) untouched.\n"
        "8. " + bilingual_rule + "\n"
        "9. Output ONLY the translated HTML fragment. Do not wrap in markdown code fences (no ```html)."
        + extra
    )


def _chunk_body_elements(body_soup: BeautifulSoup, max_chars: int = 6000) -> list[str]:
    """Split <body> children into manageable HTML chunks without breaking tags."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for child in body_soup.children:
        s = str(child)
        if not s.strip():
            continue
        if current_len + len(s) > max_chars and current:
            chunks.append("".join(current))
            current = [s]
            current_len = len(s)
        else:
            current.append(s)
            current_len += len(s)

    if current:
        chunks.append("".join(current))
    return chunks


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:html|xhtml|xml)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _call_llm(
    prompt_text: str,
    system_prompt: str,
    api_key: str,
    model_name: str,
    provider: str = "gemini",
    base_url: str = "",
) -> str:
    """Call either Gemini or an OpenAI-compatible custom provider with retry on 429/503."""
    last_err = None
    for attempt in range(6):
        try:
            if provider == "openai":
                url = (base_url.strip().rstrip("/") or "https://api.openai.com/v1") + "/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                }
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text},
                    ],
                    "temperature": 0.2,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=180)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                ),
            )
            return response.text or prompt_text
        except Exception as e:
            last_err = e
            err_str = str(e)
            if any(
                code in err_str
                for code in ("503", "502", "504", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "timed out", "Timeout", "ConnectionError", "Too Many Requests")
            ) and attempt < 5:
                time.sleep(min(60, 5 * (2 ** attempt)))
                continue
            raise
    raise last_err


def inspect_epub_chapters(epub_path: str) -> list[dict]:
    """Return a list of chapters and their translatable paragraphs for Sagar-style selection."""
    book = epub.read_epub(epub_path)
    doc_items = [item for item in book.get_items() if item.get_type() == ITEM_DOCUMENT]
    chapters = []
    for idx, item in enumerate(doc_items):
        raw_html = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "lxml")
        body = soup.body
        if not body:
            continue
        heading = body.find(["h1", "h2", "h3"])
        title = heading.get_text(" ", strip=True) if heading else item.get_name()
        paras = [
            p.get_text(" ", strip=True)
            for p in body.find_all(["p", "h1", "h2", "h3", "li"])
            if p.get_text(strip=True)
        ]
        if paras:
            chapters.append({
                "index": idx,
                "name": item.get_name(),
                "title": title[:120],
                "paragraph_count": len(paras),
                "preview": paras[:5],
            })
    return chapters


def fix_epub_xhtml(
    input_epub_path: str,
    output_epub_path: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    provider: str = "gemini",
    base_url: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """
    Sagar's 'Fix with AI': Restructure broken PDF-converted HTML into clean,
    well-formed XHTML (fixing fragmented tables/paragraphs) WITHOUT translating.
    """
    fix_prompt = (
        "You are an expert EPUB/XHTML editor. Restructure the provided HTML body content into clean, "
        "semantic, rich-text-compatible XHTML WITHOUT translating any text.\n"
        "1. Fix broken or fragmented <p> tags caused by PDF line wraps.\n"
        "2. Clean up fragmented <table> structures while keeping all data intact.\n"
        "3. Preserve all <img> tags, src paths, and <figure> elements.\n"
        "4. Output ONLY the cleaned HTML fragment (no ```html fences)."
    )
    book = epub.read_epub(input_epub_path)
    doc_items = [item for item in book.get_items() if item.get_type() == ITEM_DOCUMENT]
    total = len(doc_items)

    for idx, item in enumerate(doc_items):
        item_name = item.get_name()
        if progress_callback:
            progress_callback(idx + 1, total, f"Fixing XHTML: {item_name} ({idx + 1}/{total})...")
        raw_html = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "lxml")
        body = soup.body
        if not body or not body.get_text(strip=True):
            continue
        chunks = _chunk_body_elements(body)
        fixed_parts = []
        for chunk in chunks:
            res = _call_llm(chunk, fix_prompt, api_key, model_name, provider, base_url)
            fixed_parts.append(_strip_code_fences(res))
        new_body = BeautifulSoup(f"<body>{''.join(fixed_parts)}</body>", "lxml")
        body.clear()
        if new_body.body:
            for child in list(new_body.body.children):
                body.append(child)
        item.set_content(str(soup).encode("utf-8"))

    epub.write_epub(output_epub_path, book, {})
    return output_epub_path


def translate_epub(
    input_epub_path: str,
    output_epub_path: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    provider: str = "gemini",
    base_url: str = "",
    bilingual_terms: bool = True,
    skip_references: bool = True,
    selected_chapters: Optional[list[int]] = None,
    custom_notes: str = "",
    checkpoint_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """
    Translate an EPUB file chapter-by-chapter into Myanmar.
    Supports Gemini or OpenAI-compatible providers, chapter selection, and checkpoints.
    """
    system_prompt = _build_system_prompt(bilingual_terms, custom_notes)

    book = epub.read_epub(input_epub_path)
    doc_items = [item for item in book.get_items() if item.get_type() == ITEM_DOCUMENT]
    total_chapters = len(doc_items)

    completed: dict[str, str] = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                completed = json.load(f)
        except Exception:
            completed = {}

    for idx, item in enumerate(doc_items):
        item_name = item.get_name()
        if progress_callback:
            progress_callback(idx + 1, total_chapters, f"Translating {item_name} ({idx + 1}/{total_chapters})...")

        if item_name in completed:
            item.set_content(completed[item_name].encode("utf-8"))
            continue

        raw_html = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "lxml")
        body = soup.body

        if not body or not body.get_text(strip=True):
            continue

        if skip_references and _is_reference_section(body):
            if progress_callback:
                progress_callback(idx + 1, total_chapters, f"Skipping References/Index: {item_name}")
            continue

        chunks = _chunk_body_elements(body)
        translated_parts: list[str] = [""] * len(chunks)

        def _translate_one(c_idx: int, chunk_html: str) -> tuple[int, str]:
            res_text = _call_llm(chunk_html, system_prompt, api_key, model_name, provider, base_url)
            return c_idx, _strip_code_fences(res_text or chunk_html)

        max_workers = min(10, max(1, len(chunks)))
        completed_chunks = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_translate_one, i, ch) for i, ch in enumerate(chunks)]
            for fut in as_completed(futures):
                c_idx, out_html = fut.result()
                translated_parts[c_idx] = out_html
                completed_chunks += 1
                if progress_callback:
                    progress_callback(
                        idx + 1,
                        total_chapters,
                        f"Translating {item_name} ({idx + 1}/{total_chapters}) — chunk {completed_chunks}/{len(chunks)}...",
                    )

        # Replace body contents with translated HTML
        new_body_soup = BeautifulSoup(f"<body>{''.join(translated_parts)}</body>", "lxml")
        body.clear()
        if new_body_soup.body:
            for child in list(new_body_soup.body.children):
                body.append(child)

        final_html = str(soup)
        item.set_content(final_html.encode("utf-8"))
        completed[item_name] = final_html

        if checkpoint_path:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(completed, f, ensure_ascii=False)

    def _ensure_toc_uids(nodes, counter=None):
        if counter is None:
            counter = [0]
        for node in nodes:
            if isinstance(node, (list, tuple)):
                if len(node) == 2 and not isinstance(node[0], (list, tuple)):
                    sec, children = node
                    if getattr(sec, "uid", None) is None:
                        counter[0] += 1
                        sec.uid = f"nav-{counter[0]}"
                    _ensure_toc_uids(children, counter)
                else:
                    _ensure_toc_uids(node, counter)
            else:
                if getattr(node, "uid", None) is None:
                    counter[0] += 1
                    node.uid = f"nav-{counter[0]}"

    _ensure_toc_uids(book.toc)

    for item in book.get_items():
        content = item.get_content()
        if content is None or (item.get_type() == ITEM_DOCUMENT and not content.strip()):
            if item.get_type() == ITEM_DOCUMENT:
                item.set_content(b"<html><body></body></html>")
            else:
                item.set_content(b"")

    book.set_language("my")
    epub.write_epub(output_epub_path, book, {})
    return output_epub_path
