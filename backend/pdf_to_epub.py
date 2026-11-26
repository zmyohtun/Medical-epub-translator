"""
PDF to Structured EPUB Converter
Supports Calibre's `ebook-convert` engine (recommended for complex medical PDFs)
with automatic fallback to PyMuPDF (`fitz`).
"""

import html
import os
import re
import shutil
import subprocess
import uuid
from typing import Callable, Optional

import fitz  # PyMuPDF
from ebooklib import epub


MEDICAL_CSS = """
@charset "utf-8";
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Georgia, serif;
    line-height: 1.7;
    margin: 1.2em;
    color: #1e293b;
    font-size: 1em;
}
h1 {
    font-size: 1.6em;
    color: #0f172a;
    border-bottom: 2px solid #0284c7;
    padding-bottom: 0.3em;
    margin-top: 1.5em;
    margin-bottom: 0.8em;
    page-break-before: always;
}
h2 {
    font-size: 1.3em;
    color: #0369a1;
    margin-top: 1.3em;
    margin-bottom: 0.5em;
}
h3 {
    font-size: 1.1em;
    color: #334155;
    margin-top: 1em;
    margin-bottom: 0.4em;
}
p {
    margin: 0.6em 0;
    text-align: justify;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.2em 0;
    font-size: 0.9em;
    page-break-inside: avoid;
    break-inside: avoid;
}
th, td {
    border: 1px solid #cbd5e1;
    padding: 0.5em 0.7em;
    text-align: left;
    vertical-align: top;
}
th {
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
}
tr:nth-child(even) td {
    background-color: #f8fafc;
}
figure {
    margin: 1.2em 0;
    text-align: center;
    page-break-inside: avoid;
    break-inside: avoid;
}
figure img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
}
figcaption {
    font-size: 0.85em;
    color: #475569;
    margin-top: 0.4em;
    font-style: italic;
}
.annotation, .callout {
    background-color: #f0f9ff;
    border-left: 4px solid #0284c7;
    padding: 0.8em 1em;
    margin: 1em 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.95em;
}
"""


def find_calibre_binary() -> Optional[str]:
    """Locate Calibre's `ebook-convert` binary on macOS or in PATH."""
    mac_path = "/Applications/calibre.app/Contents/MacOS/ebook-convert"
    if os.path.isfile(mac_path) and os.access(mac_path, os.X_OK):
        return mac_path
    return shutil.which("ebook-convert")


def convert_pdf_with_calibre(
    pdf_path: str,
    output_epub_path: str,
    title: Optional[str] = None,
    author: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """
    Convert a complex Medical PDF to EPUB3 using Calibre's `ebook-convert`
    with heuristic processing, chapter detection, and medical CSS.
    """
    calibre_bin = find_calibre_binary()
    if not calibre_bin:
        raise RuntimeError(
            "Calibre (ebook-convert) not found. Install Calibre in /Applications/calibre.app "
            "or switch to the built-in PyMuPDF engine."
        )

    if progress_callback:
        progress_callback(1, 3, "Running Calibre ebook-convert (heuristics & layout analysis)...")

    css_file = output_epub_path + ".medical.css"
    with open(css_file, "w", encoding="utf-8") as f:
        f.write(MEDICAL_CSS)

    cmd = [
        calibre_bin,
        pdf_path,
        output_epub_path,
        "--epub-version", "3",
        "--output-profile", "tablet",
        "--extra-css", css_file,
        "--chapter", "//*[name()='h1' or name()='h2']",
        "--page-breaks-before", "//*[name()='h1']",
    ]
    if title:
        cmd.extend(["--title", title])
    if author:
        cmd.extend(["--authors", author])

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=180)
        if progress_callback:
            progress_callback(3, 3, "Calibre EPUB3 conversion completed!")
        return output_epub_path
    except subprocess.TimeoutExpired:
        if progress_callback:
            progress_callback(1, 3, "Calibre timed out on complex PDF; switching to fast PyMuPDF engine...")
        return ""
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Calibre conversion failed: {e.stderr or e.stdout}")
    finally:
        if os.path.exists(css_file):
            os.remove(css_file)


def _estimate_body_font_size(doc: fitz.Document) -> float:
    """Sample pages to find the most common (body) font size."""
    size_counts: dict[float, int] = {}
    sample_pages = min(len(doc), 15)
    for i in range(sample_pages):
        blocks = doc[i].get_text("dict").get("blocks", [])
        for b in blocks:
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    sz = round(span.get("size", 11.0), 1)
                    text = span.get("text", "").strip()
                    if text:
                        size_counts[sz] = size_counts.get(sz, 0) + len(text)
    if not size_counts:
        return 11.0
    return max(size_counts, key=size_counts.get)


def _table_to_html(table_data: list[list[Optional[str]]]) -> str:
    """Convert extracted PyMuPDF table rows into clean HTML."""
    if not table_data:
        return ""
    rows_html = []
    for idx, row in enumerate(table_data):
        tag = "th" if idx == 0 else "td"
        cells = "".join(
            f"<{tag}>{html.escape((cell or '').strip())}</{tag}>"
            for cell in row
        )
        rows_html.append(f"<tr>{cells}</tr>")
    return f"<table><tbody>{''.join(rows_html)}</tbody></table>"


def _extract_page_html(
    page: fitz.Page,
    body_size: float,
    book: epub.EpubBook,
    seen_xrefs: set[int],
    page_num: int,
) -> str:
    """Extract structured HTML (headings, paragraphs, tables, images) from a page."""
    parts: list[str] = []

    table_bboxes: list[fitz.Rect] = []
    try:
        tables = page.find_tables()
        for tab in tables.tables:
            table_bboxes.append(fitz.Rect(tab.bbox))
            parts.append(_table_to_html(tab.extract()))
    except Exception:
        pass

    blocks = page.get_text("dict").get("blocks", [])
    for b in blocks:
        if b.get("type") != 0:
            continue
        b_rect = fitz.Rect(b["bbox"])
        if any(b_rect.intersects(tb) for tb in table_bboxes):
            continue

        block_text_parts: list[str] = []
        max_size = 0.0
        is_bold = False

        for line in b.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                t = span.get("text", "")
                sz = span.get("size", body_size)
                flags = span.get("flags", 0)
                if t.strip():
                    max_size = max(max_size, sz)
                    if flags & 16:
                        is_bold = True
                line_text += t
            if line_text.strip():
                block_text_parts.append(line_text.strip())

        full_text = " ".join(block_text_parts).strip()
        if not full_text:
            continue

        escaped = html.escape(full_text)

        if max_size >= body_size * 1.45:
            parts.append(f"<h1>{escaped}</h1>")
        elif max_size >= body_size * 1.2 or (is_bold and len(full_text) < 120 and max_size >= body_size * 1.05):
            parts.append(f"<h2>{escaped}</h2>")
        elif re.match(r"^(Note|Clinical Pearl|Warning|Box|Key Point|Annotation)\b", full_text, re.IGNORECASE):
            parts.append(f'<div class="annotation"><p>{escaped}</p></div>')
        else:
            parts.append(f"<p>{escaped}</p>")

    for img_idx, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            base_img = page.parent.extract_image(xref)
            img_bytes = base_img["image"]
            ext = base_img.get("ext", "png")
            if len(img_bytes) < 2048:
                continue
            img_filename = f"images/p{page_num}_img{img_idx}.{ext}"
            epub_img = epub.EpubItem(
                uid=f"img_{page_num}_{img_idx}",
                file_name=img_filename,
                media_type=f"image/{ext}",
                content=img_bytes,
            )
            book.add_item(epub_img)
            parts.append(
                f'<figure><img src="{img_filename}" alt="Figure p.{page_num}"/></figure>'
            )
        except Exception:
            continue

    return "\n".join(parts)


def convert_pdf_to_epub(
    pdf_path: str,
    output_epub_path: str,
    title: Optional[str] = None,
    author: Optional[str] = None,
    pages_per_chapter: int = 15,
    engine: str = "calibre",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """
    Convert a PDF file into a clean, structured EPUB3 file.
    Uses Calibre (`ebook-convert`) by default if available, or PyMuPDF (`fitz`).
    """
    if engine == "calibre" and find_calibre_binary():
        calibre_res = convert_pdf_with_calibre(
            pdf_path=pdf_path,
            output_epub_path=output_epub_path,
            title=title,
            author=author,
            progress_callback=progress_callback,
        )
        if calibre_res:
            return calibre_res

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    body_size = _estimate_body_font_size(doc)

    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book_title = title or doc.metadata.get("title") or os.path.splitext(os.path.basename(pdf_path))[0]
    book.set_title(book_title)
    book.set_language("en")
    book.add_author(author or doc.metadata.get("author") or "Unknown Author")

    css_item = epub.EpubItem(
        uid="style_medical",
        file_name="style/medical.css",
        media_type="text/css",
        content=MEDICAL_CSS.encode("utf-8"),
    )
    book.add_item(css_item)

    toc = doc.get_toc()
    chapter_ranges: list[tuple[str, int, int]] = []
    if toc:
        top_toc = [entry for entry in toc if entry[0] == 1]
        if len(top_toc) >= 2:
            for i, entry in enumerate(top_toc):
                ch_title = entry[1].strip() or f"Chapter {i + 1}"
                start_p = max(0, entry[2] - 1)
                end_p = (top_toc[i + 1][2] - 2) if i + 1 < len(top_toc) else (total_pages - 1)
                if end_p >= start_p:
                    chapter_ranges.append((ch_title, start_p, end_p))

    if not chapter_ranges:
        for i in range(0, total_pages, pages_per_chapter):
            end_p = min(i + pages_per_chapter - 1, total_pages - 1)
            chapter_ranges.append((f"Section {len(chapter_ranges) + 1} (p.{i + 1}-{end_p + 1})", i, end_p))

    epub_chapters: list[epub.EpubHtml] = []
    seen_xrefs: set[int] = set()

    for ch_idx, (ch_title, start_p, end_p) in enumerate(chapter_ranges):
        ch_html_parts = [f"<h1>{html.escape(ch_title)}</h1>"]
        for p_idx in range(start_p, end_p + 1):
            if progress_callback:
                progress_callback(p_idx + 1, total_pages, f"Converting page {p_idx + 1}/{total_pages}...")
            page_html = _extract_page_html(doc[p_idx], body_size, book, seen_xrefs, p_idx + 1)
            ch_html_parts.append(page_html)

        ch_file = f"chap_{ch_idx + 1:03d}.xhtml"
        c = epub.EpubHtml(title=ch_title, file_name=ch_file, lang="en")
        c.content = (
            f'<?xml version="1.0" encoding="utf-8"?>\n'
            f'<html xmlns="http://www.w3.org/1999/xhtml">\n'
            f'<head><title>{html.escape(ch_title)}</title>'
            f'<link rel="stylesheet" type="text/css" href="style/medical.css"/></head>\n'
            f'<body>{"".join(ch_html_parts)}</body>\n</html>'
        ).encode("utf-8")
        c.add_item(css_item)
        book.add_item(c)
        epub_chapters.append(c)

    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + epub_chapters

    epub.write_epub(output_epub_path, book, {})
    doc.close()
    return output_epub_path
