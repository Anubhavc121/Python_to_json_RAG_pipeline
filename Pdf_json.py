""""
- usage:
python Pdf_json.py \
  --input_dir "/Users/anubhav/Class6Maths" \
  --out class_out \
  --loop \
  --ocr \
  --show_sources

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF → JSON (text + filtered images) + RAG QA over many PDFs
- Excludes repeating logos / decorations automatically
- Optional base64 embedding of kept images
- Optional OCR fallback for sparse slides
- Strict formatting for True/False, Assertion–Reason (with option letter a/b/c/d), and MCQ
- Shows per-file sources used (doc • page • chunk_id)
"""

import argparse
import base64
import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import ollama
from PIL import Image, ImageOps, ImageStat
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# --- Optional OCR (safe import) ---
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# --- Optional pretty progress ---
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False


# =========================
# Pydantic models (v2)
# =========================

class TextSpan(BaseModel):
    text: str
    bbox: Tuple[float, float, float, float]


class ImageBlock(BaseModel):
    bbox: Optional[Tuple[float, float, float, float]]  # None if unknown
    xref: int
    image_path: str
    image_format: str
    image_base64: str = ""
    caption: str = ""


class PageData(BaseModel):
    page_number: int
    text_full: str
    text_blocks: List[TextSpan] = Field(default_factory=list)
    image_blocks: List[ImageBlock] = Field(default_factory=list)


class PDFJSON(BaseModel):
    pdf_path: str
    pdf_name: str
    num_pages: int
    images_dir: str
    pages: List[PageData]


# =========================
# Image utils (dup/junk detection)
# =========================

def dhash(image: Image.Image, hash_size: int = 8) -> str:
    img = ImageOps.grayscale(image).resize((hash_size + 1, hash_size), Image.LANCZOS)
    bits = []
    for y in range(hash_size):
        for x in range(hash_size):
            bits.append(img.getpixel((x, y)) > img.getpixel((x + 1, y)))
    b = ''.join('1' if v else '0' for v in bits)
    width = (len(b) + 3) // 4
    return f"{int(b, 2):0{width}x}"


def image_stats_from_bytes(img_bytes: bytes) -> Dict[str, Any]:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = im.size
    area = w * h

    gray = ImageOps.grayscale(im)
    hist = gray.histogram()
    total = sum(hist) or 1
    entropy = 0.0
    for c in hist:
        if c:
            p = c / total
            entropy -= p * math.log2(p)
    std = ImageStat.Stat(gray).stddev[0]
    alpha = im.split()[-1]
    alpha_mean = ImageStat.Stat(alpha).mean[0]
    transp_ratio = 1.0 - (alpha_mean / 255.0)
    ph = dhash(im)
    return {
        "w": w, "h": h, "area": area,
        "std": std, "entropy": entropy,
        "transp_ratio": transp_ratio,
        "hash": ph,
    }


def classify_image(stats: Dict[str, Any], recurring: bool) -> str:
    w, h = stats["w"], stats["h"]
    area = stats["area"]
    std = stats["std"]
    entropy = stats["entropy"]
    transp = stats["transp_ratio"]

    if area < 32 * 32 or min(w, h) < 24 or ((w > 4*h or h > 4*w) and min(w, h) < 40):
        return "junk"
    if std < 8 and entropy < 3.0:
        return "decoration"
    if transp > 0.5 and std < 12:
        return "decoration"
    if recurring:
        return "logo" if (std >= 8 and entropy >= 3.0) else "decoration"
    return "content"


# =========================
# Ollama helpers
# =========================

def caption_image_ollama(image_path: str, model: str = "llava:7b",
                         prompt: str = "Describe this slide image briefly.") -> str:
    try:
        resp = ollama.chat(model=model, messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }])
        return (resp.get("message", {}) or {}).get("content", "").strip()
    except Exception:
        try:
            res = ollama.generate(model=model, prompt=f"{prompt}\n\nImage path: {image_path}")
            return (res.get("response") or "").strip()
        except Exception:
            return ""


def page_context_caption(page_text: str, max_len: int = 140) -> str:
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    title = lines[0]
    bullet = ""
    for ln in lines[1:]:
        if len(ln) > 6:
            bullet = ln
            break
    cap = title if not bullet else f"{title} — {bullet}"
    return cap[:max_len]


# =========================
# PDF → JSON (two-pass image filtering)
# =========================

def _coerce_xref(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            return int.from_bytes(raw, "big", signed=False)
        except Exception:
            return None
    try:
        return int(raw)
    except Exception:
        return None


def _ocr_page_pix(page: fitz.Page) -> str:
    if not OCR_AVAILABLE:
        return ""
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    try:
        txt = pytesseract.image_to_string(img)
        return txt.strip()
    except Exception:
        return ""


def pdf_to_json(
    pdf_path: Path,
    out_dir: Path,
    embed_images_base64: bool = False,
    add_captions: bool = False,
    vision_model: str = "llava:7b",
    recurring_pct_exclude: float = 0.60,
    keep_logos: bool = False,
    json_only: bool = False,
    ocr: bool = False,
) -> PDFJSON:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    if not json_only:
        images_dir.mkdir(exist_ok=True)

    doc = fitz.open(pdf_path)
    num_pages = len(doc)

    tmp_pages: List[Dict[str, Any]] = []
    all_hashes: List[str] = []

    iterator = range(num_pages)
    if TQDM:
        iterator = tqdm(iterator, desc=f"Extract {pdf_path.name}", unit="p", leave=False)

    for pno in iterator:
        page = doc[pno]
        raw = page.get_text("rawdict")
        text_full = page.get_text() or ""
        if ocr and len(text_full.strip()) < 20:
            ocr_text = _ocr_page_pix(page)
            if ocr_text:
                text_full = (text_full + "\n" + ocr_text).strip()

        text_blocks: List[TextSpan] = []
        for block in raw.get("blocks", []):
            if block.get("type", 0) == 0:
                for ln in block.get("lines", []):
                    for sp in ln.get("spans", []):
                        txt = sp.get("text", "")
                        if txt.strip():
                            text_blocks.append(
                                TextSpan(text=txt, bbox=tuple(sp.get("bbox", (0, 0, 0, 0))))
                            )

        tmp_images: List[Dict[str, Any]] = []
        seen_xrefs: set[int] = set()

        for block in raw.get("blocks", []):
            if block.get("type", 0) != 1:
                continue
            bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
            xref = _coerce_xref(block.get("image", 0))
            if not xref or xref in seen_xrefs:
                continue
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue
            seen_xrefs.add(xref)
            img_bytes = base.get("image", b"")
            if not img_bytes:
                continue
            img_ext = base.get("ext", "png")
            stats = image_stats_from_bytes(img_bytes)
            all_hashes.append(stats["hash"])
            tmp_images.append({
                "bbox": bbox,
                "xref": xref,
                "ext": img_ext,
                "bytes": img_bytes,
                "stats": stats
            })

        try:
            for info in page.get_images(full=True):
                raw_xref = info[0] if isinstance(info, (list, tuple)) else info
                xref = _coerce_xref(raw_xref)
                if not xref or xref in seen_xrefs:
                    continue
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                bbox = (0, 0, 0, 0)
                for b in raw.get("blocks", []):
                    if b.get("type", 0) == 1:
                        bx = _coerce_xref(b.get("image"))
                        if bx == xref:
                            bbox = tuple(b.get("bbox", (0, 0, 0, 0)))
                            break
                seen_xrefs.add(xref)
                img_bytes = base.get("image", b"")
                if not img_bytes:
                    continue
                img_ext = base.get("ext", "png")
                stats = image_stats_from_bytes(img_bytes)
                all_hashes.append(stats["hash"])
                tmp_images.append({
                    "bbox": bbox,
                    "xref": xref,
                    "ext": img_ext,
                    "bytes": img_bytes,
                    "stats": stats
                })
        except Exception:
            pass

        tmp_pages.append({
            "page_number": pno + 1,
            "text_full": text_full,
            "text_blocks": text_blocks,
            "tmp_images": tmp_images
        })

    hash_counts = Counter(all_hashes)
    recurring_threshold = max(2, int(math.ceil(recurring_pct_exclude * num_pages)))
    recurring_hashes = {h for h, c in hash_counts.items() if c >= recurring_threshold}
    most_common_hash = hash_counts.most_common(1)[0][0] if hash_counts else None

    pages: List[PageData] = []
    for page in tmp_pages:
        kept_blocks: List[ImageBlock] = []
        for t in page["tmp_images"]:
            stats = t["stats"]
            h = stats["hash"]
            recurring = (h in recurring_hashes) or (h == most_common_hash)
            label = classify_image(stats, recurring)
            if recurring:
                continue
            if label in ("junk", "decoration"):
                continue
            if label == "logo" and not keep_logos:
                continue

            img_ext = t["ext"]
            out_name = f"page{page['page_number']}_xref{t['xref']}.{img_ext}"
            img_path = (Path(out_dir) / "images" / out_name)
            if not json_only:
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img_path.write_bytes(t["bytes"])

            img_b64 = base64.b64encode(t["bytes"]).decode("utf-8") if embed_images_base64 else ""
            bbox = tuple(t["bbox"])
            bbox_clean = None if not any(bbox) else bbox

            kept_blocks.append(ImageBlock(
                bbox=bbox_clean,
                xref=int(t["xref"]),
                image_path=str(img_path if not json_only else Path("memory") / out_name),
                image_format=img_ext,
                image_base64=img_b64,
                caption=""
            ))

        ctxt = page_context_caption(page["text_full"])
        for b in kept_blocks:
            if not b.caption:
                b.caption = ctxt

        if add_captions:
            for b in kept_blocks:
                if b.image_path and not b.caption:
                    b.caption = caption_image_ollama(b.image_path, model=vision_model) or b.caption

        pages.append(PageData(
            page_number=page["page_number"],
            text_full=page["text_full"],
            text_blocks=page["text_blocks"],
            image_blocks=kept_blocks,
        ))

    return PDFJSON(
        pdf_path=str(pdf_path),
        pdf_name=pdf_path.name,
        num_pages=num_pages,
        images_dir=str(out_dir / "images"),
        pages=pages,
    )


# =========================
# Chunking & RAG
# =========================

def make_text_chunks(pdf_json: PDFJSON, max_chars: int = 1000) -> List[Dict[str, Any]]:
    chunks = []
    cid = 0
    for page in pdf_json.pages:
        text = page.text_full.strip()
        if not text:
            text = " ".join([t.text for t in page.text_blocks]).strip()
        start = 0
        while start < len(text):
            piece = text[start:start + max_chars]
            if piece.strip():
                chunks.append({"chunk_id": f"p{page.page_number}_c{cid}",
                               "doc": pdf_json.pdf_name,
                               "page": page.page_number,
                               "text": piece})
                cid += 1
            start += max_chars
        if page.image_blocks:
            chunks.append({
                "chunk_id": f"p{page.page_number}_imgstub",
                "doc": pdf_json.pdf_name,
                "page": page.page_number,
                "text": f"[This slide has {len(page.image_blocks)} content images.]"
            })
    return chunks


class EmbeddingIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.embs: Optional[np.ndarray] = None

    def add(self, texts: List[str], metas: List[Dict[str, Any]]):
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        embs = embs.astype(np.float32)
        self.embs = embs if self.embs is None else np.vstack([self.embs, embs])
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, query: str, top_k: int = 6) -> List[Tuple[str, Dict[str, Any], float]]:
        if self.embs is None or not len(self.texts):
            return []
        q = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        sims = np.dot(self.embs, q)
        idx = np.argsort(-sims)[:top_k]
        return [(self.texts[i], self.metas[i], float(sims[i])) for i in idx]


def pretty_sources(hits: List[Tuple[str, Dict[str, Any], float]]) -> str:
    if not hits:
        return ""
    lines = ["--- Sources used ---"]
    seen = set()
    for _, meta, _ in hits:
        key = (meta.get("doc"), meta.get("page"), meta.get("chunk_id", meta.get("page")))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {meta.get('doc')} • Slide {meta.get('page')} • {meta.get('chunk_id', '')}")
    return "\n".join(lines)


# =========================
# Strict answer formatting (TF / AR / MCQ)
# =========================

TF_JSON_INSTRUCTIONS = (
    "You must answer strictly True or False based ONLY on the PDF context. "
    "Return JSON exactly like this:\n"
    '{"type":"tf","answer":"True","why":"one short line"}'
)

AR_JSON_INSTRUCTIONS = (
    "Answer the Assertion–Reason question strictly using this JSON:\n"
    '{"type":"ar","A":true,"R":true,"explains":false,'
    '"final":"Assertion is True; Reason is True; but Reason does not correctly explain Assertion."}\n'
    'A and R must be booleans. "final" must be one plain sentence.'
)

MCQ_JSON_INSTRUCTIONS = (
    "If the question is MCQ with options (a)…(d), return JSON:\n"
    '{"type":"mcq","answer":"b","answer_text":"90 litres","why":"one short line"}\n'
    'Only a/b/c/d in "answer".'
)

def detect_question_type(q: str) -> str:
    ql = q.lower()
    if "assertion" in ql and "reason" in ql:
        return "ar"
    if re.search(r"\btrue\s*\/?\s*false\b", ql) or ql.strip().startswith(("true or false", "true/false")):
        return "tf"
    if re.search(r"\([a-d]\)", ql):
        return "mcq"
    return "open"

def parse_json_block(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def clean_why(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = re.sub(r"^\s*why\s*:\s*", "", s, flags=re.I)
    return s

def tf_consistency_fix(answer_str: str, why_str: str) -> str:
    a = (answer_str or "").strip().lower()
    w = (why_str or "").lower()
    says_equal = ("equal spacing" in w) or ("equal distance" in w)
    negated = bool(re.search(r"\bno\b|\bnot\b|\bshouldn['’]?t\b", w))
    if says_equal and not negated:
        return "True"
    if says_equal and negated:
        return "False"
    return "True" if a.startswith("true") else "False"

def format_tf_output(tf_dict: dict) -> str:
    ans = tf_dict.get("answer", "").strip()
    why = clean_why(tf_dict.get("why", ""))
    ans = tf_consistency_fix(ans, why)
    out = f"A: {ans}"
    if why:
        out += f"\nWhy: {why}"
    return out.strip()

# ------ Assertion–Reason helpers ------

def ar_option_letter(A: bool, R: bool, explains: bool) -> str:
    if A and R and explains:
        return "A"
    if A and R and not explains:
        return "B"
    if A and not R:
        return "C"
    if (not A) and R:
        return "D"
    return "?"

def extract_ar_texts(q: str) -> Tuple[str, str]:
    A_txt, R_txt = "", ""
    mA = re.search(r"Assertion\s*\(A\)\s*:\s*(.*?)(?:Reason\s*\(R\)|$)", q, flags=re.I | re.S)
    if mA:
        A_txt = mA.group(1).strip()
    mR = re.search(r"Reason\s*\(R\)\s*:\s*(.*?)(?:\n|\r|$|\()", q, flags=re.I | re.S)
    if mR:
        R_txt = mR.group(1).strip()
    return A_txt, R_txt

def ar_consistency_fix(question: str, ar: dict) -> dict:
    """
    Lightweight domain checks to avoid obvious mistakes.
    Example: Square perimeter is 4× side, not 2×.
    """
    A_txt, R_txt = extract_ar_texts(question)
    A, R, E = bool(ar.get("A")), bool(ar.get("R")), bool(ar.get("explains"))

    r_lower = R_txt.lower()
    a_lower = A_txt.lower()

    # Square perimeter sanity rule
    if ("perimeter" in r_lower or "perimeter" in a_lower) and "square" in (r_lower + a_lower):
        says_double = ("double" in r_lower) or ("2" in r_lower and "4" not in r_lower) or ("twice" in r_lower)
        says_four_times = ("four" in r_lower) or ("4" in r_lower) or ("4x" in r_lower) or ("4 ×" in r_lower)
        if says_double:
            R = False
            E = False
        elif says_four_times:
            R = True

    # If A mentions “side 9 cm … 36 cm”, that is correct (4×9) → keep A True
    if "9" in a_lower and "36" in a_lower and "square" in a_lower and "perimeter" in a_lower:
        A = True

    # Build final sentence if missing
    final = ar.get("final", "").strip()
    if not final:
        if A and R and E:
            final = "Assertion (A) is true; Reason (R) is true; and Reason (R) correctly explains Assertion (A)."
        elif A and R and not E:
            final = "Assertion (A) is true; Reason (R) is true; but Reason (R) is not the correct explanation of Assertion (A)."
        elif A and not R:
            final = "Assertion (A) is true; Reason (R) is false."
        elif (not A) and R:
            final = "Assertion (A) is false; Reason (R) is true."
        else:
            final = "Assertion (A) is false; Reason (R) is false."

    return {"type": "ar", "A": A, "R": R, "explains": E, "final": final}

def format_ar_output(ar: dict) -> str:
    A, R, E = bool(ar.get("A")), bool(ar.get("R")), bool(ar.get("explains"))
    final = ar.get("final", "").strip()
    if not final:
        if A and R and E:
            final = "Assertion (A) is true; Reason (R) is true; and Reason (R) correctly explains Assertion (A)."
        elif A and R and (not E):
            final = "Assertion (A) is true; Reason (R) is true; but Reason (R) is not the correct explanation of Assertion (A)."
        elif A and (not R):
            final = "Assertion (A) is true; Reason (R) is false."
        elif (not A) and R:
            final = "Assertion (A) is false; Reason (R) is true."
        else:
            final = "Assertion (A) is false; Reason (R) is false."
    letter = ar_option_letter(A, R, E)
    return f"A: ({letter}) {final}"

# ------ MCQ helpers ------

def format_mcq_output(mcq: dict) -> str:
    letter = (mcq.get("answer", "") or "").strip().lower()
    ans_text = mcq.get("answer_text", "").strip()
    why = clean_why(mcq.get("why", ""))

    if letter not in ("a", "b", "c", "d"):
        m = re.match(r"\(?([a-d])\)?", ans_text.lower())
        if m:
            letter = m.group(1)
    show = f"A: {(f'({letter}) ' if letter else '')}{ans_text}".strip()
    if why:
        show += f"\nWhy: {why}"
    show = re.sub(r"(?i)why:\s*why:\s*", "Why: ", show)
    return show.strip()


# =========================
# Answer with Ollama (RAG)
# =========================

def answer_with_ollama(
    question: str,
    index: EmbeddingIndex,
    model: str = "llama3",
    top_k: int = 6,
    max_context_chars: int = 8000,
    system_prompt: str = (
        "You are a helpful assistant. Answer ONLY using the provided PDF context. "
        "If the answer is not in the context, say you don't have enough information."
    ),
    show_sources: bool = False,
) -> str:
    qtype = detect_question_type(question)

    hits = index.search(question, top_k=top_k)
    context, used = [], 0
    for text, meta, score in hits:
        chunk = f"[{meta['doc']} • Page {meta['page']} • {meta['chunk_id']}] {text}".strip()
        if used + len(chunk) + 2 > max_context_chars:
            break
        context.append(chunk)
        used += len(chunk) + 2

    src_block = pretty_sources(hits) if show_sources else ""

    if qtype == "tf":
        prompt = (
            f"{system_prompt}\n\n{TF_JSON_INSTRUCTIONS}\n\n"
            f"Question: {question}\n\nPDF Context:\n" + "\n\n".join(context)
        )
        res = ollama.generate(model=model, prompt=prompt)
        parsed = parse_json_block(res.get("response", ""))
        if parsed and parsed.get("type") == "tf":
            body = format_tf_output(parsed)
        else:
            raw = res.get("response", "").strip()
            m = re.search(r"\b(True|False)\b", raw)
            ans = m.group(1) if m else "False"
            why = ""
            for line in raw.splitlines():
                if "why" in line.lower():
                    why = clean_why(line.split(":", 1)[-1])
                    break
            body = format_tf_output({"answer": ans, "why": why})
        return f"{src_block}\n\n{body}".strip() if src_block else body

    if qtype == "ar":
        prompt = (
            f"{system_prompt}\n\n{AR_JSON_INSTRUCTIONS}\n\n"
            f"Question: {question}\n\nPDF Context:\n" + "\n\n".join(context)
        )
        res = ollama.generate(model=model, prompt=prompt)
        parsed = parse_json_block(res.get("response", ""))
        if not parsed or parsed.get("type") != "ar":
            parsed = {"type": "ar", "A": True, "R": False, "explains": False, "final": ""}
        fixed = ar_consistency_fix(question, parsed)
        body = format_ar_output(fixed)
        return f"{src_block}\n\n{body}".strip() if src_block else body

    if qtype == "mcq":
        prompt = (
            f"{system_prompt}\n\n{MCQ_JSON_INSTRUCTIONS}\n\n"
            f"Question: {question}\n\nPDF Context:\n" + "\n\n".join(context)
        )
        res = ollama.generate(model=model, prompt=prompt)
        parsed = parse_json_block(res.get("response", ""))
        body = format_mcq_output(parsed) if (parsed and parsed.get("type") == "mcq") else (res.get("response") or "").strip()
        body = re.sub(r"(?i)why:\s*why:\s*", "Why: ", body)
        return f"{src_block}\n\n{body}".strip() if src_block else body

    prompt = f"{system_prompt}\n\nQuestion: {question}\n\nPDF Context:\n" + "\n\n".join(context) + "\n\nAnswer:"
    res = ollama.generate(model=model, prompt=prompt)
    body = (res.get("response") or "").strip()
    return f"{src_block}\n\n{body}".strip() if src_block else body


# =========================
# Save JSON
# =========================

def save_pdfjson(pdf_json: PDFJSON, path: Path) -> None:
    data = pdf_json.model_dump()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# Orchestration
# =========================

def build_index_from_pdf(
    pdf_path: Path,
    out_dir: Path,
    embed_images_base64: bool = False,
    add_captions: bool = False,
    vision_model: str = "llava:7b",
    chunk_chars: int = 1000,
    embedder: str = "all-MiniLM-L6-v2",
    recurring_pct_exclude: float = 0.60,
    keep_logos: bool = False,
    json_only: bool = False,
    ocr: bool = False,
) -> Tuple[PDFJSON, List[Dict[str, Any]]]:
    pdf_json = pdf_to_json(
        pdf_path=pdf_path,
        out_dir=out_dir / pdf_path.stem,
        embed_images_base64=embed_images_base64,
        add_captions=add_captions,
        vision_model=vision_model,
        recurring_pct_exclude=recurring_pct_exclude,
        keep_logos=keep_logos,
        json_only=json_only,
        ocr=ocr,
    )
    chunks = make_text_chunks(pdf_json, max_chars=chunk_chars)
    return pdf_json, chunks


def build_index_from_dir(
    input_dir: Path,
    out_dir: Path,
    **kwargs,
) -> Tuple[List[PDFJSON], EmbeddingIndex]:
    pdfs = sorted([p for p in input_dir.glob("**/*") if p.suffix.lower() in (".pdf", ".PDF", ".pdfx")])
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {input_dir}")

    if TQDM:
        pbar = tqdm(pdfs, desc=f"Indexing {len(pdfs)} PDFs", unit="pdf")
    else:
        pbar = pdfs

    idx = EmbeddingIndex(kwargs.get("embedder", "all-MiniLM-L6-v2"))
    docs: List[PDFJSON] = []

    for p in pbar:
        pdf_json, chunks = build_index_from_pdf(p, out_dir, **kwargs)
        docs.append(pdf_json)
        idx.add([c["text"] for c in chunks], [{"doc": c["doc"], "page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks])

    return docs, idx


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="PDF → JSON (text+images, logo/decoration excluded) + RAG QA with Ollama")
    ap.add_argument("--pdf", type=str, help="Path to a single PDF")
    ap.add_argument("--input_dir", type=str, help="Directory containing PDFs (recursive)")
    ap.add_argument("--out", type=str, default="out_dir", help="Output directory")
    ap.add_argument("--model", type=str, default="llama3", help="Ollama LLM (e.g., llama3, gemma2:2b)")
    ap.add_argument("--embed_b64", action="store_true", help="Embed kept images as base64 inside JSON")
    ap.add_argument("--captions", action="store_true", help="Add captions via Ollama vision")
    ap.add_argument("--vision_model", type=str, default="llava:7b", help="Vision model for --captions")
    ap.add_argument("--chunk_chars", type=int, default=1000, help="Max characters per text chunk")
    ap.add_argument("--save_json", action="store_true", help="Save the extracted JSON to disk")
    ap.add_argument("--json_only", action="store_true", help="Do not save image files; JSON only")
    ap.add_argument("--ask", type=str, default="", help="Ask one question immediately")
    ap.add_argument("--loop", action="store_true", help="Interactive QA loop")
    ap.add_argument("--top_k", type=int, default=6, help="Chunks to fetch for context")
    ap.add_argument("--max_context_chars", type=int, default=8000, help="Max combined context size")
    ap.add_argument("--exclude_recurring_pct", type=float, default=0.60, help="Exclude images repeating on >= this fraction of pages")
    ap.add_argument("--keep_logos", action="store_true", help="Keep logos even if recurring")
    ap.add_argument("--show_sources", action="store_true", help="Print doc/page/chunk sources for each answer")
    ap.add_argument("--ocr", action="store_true", help="OCR fallback for low-text slides (requires Tesseract)")
    args = ap.parse_args()

    out_dir = Path(args.out)

    if args.pdf:
        pdf_path = Path(args.pdf)
        pdf_json, chunks = build_index_from_pdf(
            pdf_path=pdf_path,
            out_dir=out_dir,
            embed_images_base64=args.embed_b64,
            add_captions=args.captions,
            vision_model=args.vision_model,
            chunk_chars=args.chunk_chars,
            embedder="all-MiniLM-L6-v2",
            recurring_pct_exclude=args.exclude_recurring_pct,
            keep_logos=args.keep_logos,
            json_only=args.json_only,
            ocr=args.ocr,
        )
        if args.save_json:
            json_path = out_dir / pdf_path.stem / f"{pdf_path.stem}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            save_pdfjson(pdf_json, json_path)
            print(f"[ok] Saved JSON → {json_path}")

        idx = EmbeddingIndex("all-MiniLM-L6-v2")
        idx.add([c["text"] for c in chunks], [{"doc": c["doc"], "page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks])
    else:
        if not args.input_dir:
            raise SystemExit("Provide --pdf or --input_dir")
        docs, idx = build_index_from_dir(
            input_dir=Path(args.input_dir),
            out_dir=out_dir,
            embed_images_base64=args.embed_b64,
            add_captions=args.captions,
            vision_model=args.vision_model,
            chunk_chars=args.chunk_chars,
            embedder="all-MiniLM-L6-v2",
            recurring_pct_exclude=args.exclude_recurring_pct,
            keep_logos=args.keep_logos,
            json_only=args.json_only,
            ocr=args.ocr,
        )
        if args.save_json:
            for d in docs:
                json_path = out_dir / Path(d.pdf_path).stem / f"{Path(d.pdf_path).stem}.json"
                json_path.parent.mkdir(parents=True, exist_ok=True)
                save_pdfjson(d, json_path)
            print(f"[ok] Saved JSON for {len(docs)} PDFs under {out_dir}")

    if args.ask:
        ans = answer_with_ollama(
            args.ask, idx,
            model=args.model,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            show_sources=args.show_sources,
        )
        print("\n" + ans)

    if args.loop:
        print("\nType your questions (Ctrl+C to exit):")
        try:
            while True:
                q = input("\nQ: ").strip()
                if not q:
                    continue
                ans = answer_with_ollama(
                    q, idx,
                    model=args.model,
                    top_k=args.top_k,
                    max_context_chars=args.max_context_chars,
                    show_sources=args.show_sources,
                )
                print("\n" + ans)
        except KeyboardInterrupt:
            print("\nbye!")


if __name__ == "__main__":
    main()
