import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT

from backend.epub_translator import _call_llm, _strip_code_fences

MIN_ENGLISH_CHARS_LEFTOVER = 8


def has_leftover_english(text: str) -> bool:
    """Check for untranslated English words outside parenthetical medical terms."""
    cleaned = re.sub(r"\([^)]*\)", "", text)
    english_chunks = re.findall(r"[A-Za-z]{3,}", cleaned)
    return sum(len(w) for w in english_chunks) >= MIN_ENGLISH_CHARS_LEFTOVER


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


def selective_proofread_epub(
    draft_epub_path: str,
    output_epub_path: str,
    api_key: str,
    model_name: str = "google/gemini-2.5-pro",
    provider: str = "openai_compatible",
    base_url: str = "https://openrouter.ai/api/v1",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """Triage flagged paragraphs in a draft Myanmar EPUB and proofread them in batches."""
    book = epub.read_epub(draft_epub_path)
    docs = [item for item in book.get_items() if item.get_type() == ITEM_DOCUMENT]

    soups = {}
    flagged = []

    for doc_idx, item in enumerate(docs):
        raw = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")
        soups[doc_idx] = (item, soup)
        ps = soup.find_all("p")
        num_p = len(ps)
        for p_idx, p in enumerate(ps):
            text = p.get_text(strip=True)
            if not text or len(text) < 5:
                continue
            reasons = []
            if p_idx < 2 or p_idx >= num_p - 2:
                reasons.append("chapter_edge")
            if has_leftover_english(text):
                reasons.append("leftover_english")
            if ";" in text or "—" in text or len(text) > 350:
                reasons.append("complex")
            if reasons:
                flagged.append({
                    "id": f"{doc_idx}::{p_idx}",
                    "doc_idx": doc_idx,
                    "p_idx": p_idx,
                    "my_draft": text,
                })

    if not flagged:
        _ensure_toc_uids(book.toc)
        epub.write_epub(output_epub_path, book, {})
        return output_epub_path

    batch_size = 15
    batches = [flagged[i:i + batch_size] for i in range(0, len(flagged), batch_size)]
    total_batches = len(batches)

    system_prompt = (
        "You are a senior Myanmar medical editor. Polish the provided draft Myanmar medical textbook paragraphs "
        "for natural, accurate, clinical Myanmar prose. Keep English medical terms in parentheses intact. "
        "Return ONLY a valid JSON array of objects with keys 'id' and 'my_proof' (no markdown fences)."
    )

    updates = {}

    def _process_batch(b_idx, batch):
        payload = [{"id": x["id"], "my_draft": x["my_draft"]} for x in batch]
        res = _call_llm(
            json.dumps(payload, ensure_ascii=False),
            system_prompt,
            api_key,
            model_name,
            provider,
            base_url,
        )
        cleaned = _strip_code_fences(res).strip()
        try:
            arr = json.loads(cleaned)
            if isinstance(arr, list):
                return {str(item["id"]): str(item["my_proof"]) for item in arr if "id" in item and "my_proof" in item}
        except Exception:
            pass
        return {}

    completed = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_process_batch, i, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            completed += 1
            updates.update(fut.result())
            if progress_callback:
                progress_callback(
                    completed,
                    total_batches,
                    f"Proofreading flagged paragraphs ({completed}/{total_batches} batches, {len(flagged)} paragraphs)...",
                )

    for item_info in flagged:
        uid = item_info["id"]
        if uid in updates:
            _, soup = soups[item_info["doc_idx"]]
            ps = soup.find_all("p")
            if item_info["p_idx"] < len(ps):
                ps[item_info["p_idx"]].string = updates[uid]

    for doc_idx, (item, soup) in soups.items():
        item.set_content(str(soup).encode("utf-8"))

    _ensure_toc_uids(book.toc)
    for it in book.get_items():
        c = it.get_content()
        if c is None or (it.get_type() == ITEM_DOCUMENT and not c.strip()):
            it.set_content(b"<html><body></body></html>" if it.get_type() == ITEM_DOCUMENT else b"")

    epub.write_epub(output_epub_path, book, {})
    return output_epub_path
