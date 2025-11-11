# scripts/pdf_images_with_alt.py
import os, json, base64, time, argparse
from typing import List, Dict, Tuple
from mimetypes import guess_type

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

# ---------- Config ----------
MAX_CONTEXT_CHARS = 700  # text window before/after image for context
MIN_IMAGE_PX = 32        # ignore tiny bullets/icons
VECTOR_MERGE_MARGIN = 4  # how far to grow vector rects when merging
VECTOR_RASTER_SCALE = 2  # render scale for vector snapshots
DETAIL = "high"          # image detail: "auto" | "low" | "high"
REASONING_LEVEL = os.getenv("OPENAI_VISION_REASONING", "medium")
RETRIES = 3

def local_image_to_data_url(image_path: str) -> str:
    mime, _ = guess_type(image_path)
    if mime is None:
        mime = "application/octet-stream"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _get_output_text(resp) -> str:
    # Works with the Responses API; falls back if the SDK version differs
    text = getattr(resp, "output_text", None)
    if text:
        return text
    # Fallbacks for older/newer SDKs:
    try:
        # Some SDKs expose a structured 'output' list
        blocks = getattr(resp, "output", None) or []
        parts = []
        for b in blocks:
            if b.get("type") == "message":
                for c in b.get("content", []):
                    if c.get("type") in ("output_text", "text"):
                        parts.append(c.get("text", ""))
        if parts:
            return "".join(parts)
    except Exception:
        pass
    # As a last resort, try Chat-style choices if present
    try:
        return resp.choices[0].message.content
    except Exception:
        return ""
    
def _extract_text_parts(page) -> List[Dict]:
    """
    Return a list of {'type': 'text', 'text': str, 'bbox': (x0,y0,x1,y1) or None}
    using a robust fallback chain: rawdict -> blocks -> text.
    """
    parts: List[Dict] = []

    # ---- 1) Preferred: rawdict (preserves block structure & bboxes) ----
    try:
        raw = page.get_text("rawdict")
    except Exception:
        raw = None
    rblocks = raw.get("blocks", []) if isinstance(raw, dict) else []

    text_len = 0
    for b in rblocks:
        if b.get("type") == 0:  # text block
            lines = b.get("lines", [])
            buf = []
            for line in lines:
                spans = line.get("spans", [])
                # join spans; add a space if style switches to avoid word-joins
                acc, prev_style = [], None
                for s in spans:
                    t = s.get("text", "")
                    style = (s.get("font"), s.get("size"))
                    if acc and style != prev_style and not acc[-1].endswith((" ", "\n")) and not t.startswith(" "):
                        acc.append(" ")
                    acc.append(t)
                    prev_style = style
                buf.append("".join(acc))
            txt = "\n".join(buf).rstrip() + "\n"
            text_len += len(txt)
            parts.append({"type": "text", "text": txt, "bbox": tuple(b.get("bbox", (0,0,0,0)))})

    if text_len > 0:
        return parts

    # ---- 2) Fallback: blocks() gives (x0,y0,x1,y1, text, ...) ----
    try:
        bks = page.get_text("blocks") or []
        # keep only entries that have non-empty text
        bks = [b for b in bks if isinstance(b, (list, tuple)) and len(b) >= 5 and (b[4] or "").strip()]
        # sort roughly top-to-bottom, then left-to-right
        bks.sort(key=lambda t: (((t[1] + t[3]) / 2.0), t[0]))
        for (x0, y0, x1, y1, txt, *_rest) in bks:
            parts.append({"type": "text", "text": (txt.strip() + "\n"), "bbox": (x0, y0, x1, y1)})
        if parts:
            return parts
    except Exception:
        pass

    # ---- 3) Last resort: plain text (no positions) ----
    try:
        txt = page.get_text("text") or ""
        if txt.strip():
            parts.append({"type": "text", "text": txt, "bbox": None})
    except Exception:
        pass

    return parts


def ask_vision_for_alt(client: OpenAI, model: str, image_path: str, pre: str, post: str) -> Dict:
    """
    Uses Responses API if model looks like GPT-5, otherwise uses Chat Completions.
    Returns a dict with keys: alt_short, alt_long, purpose, suggested_caption
    """
    reasoning_effort = os.getenv("OPENAI_VISION_REASONING", "high")
    data_url = local_image_to_data_url(image_path)

    # Prompt text (we'll reuse in both APIs)
    system_instr = (
        "You are an accessibility expert. Follow WCAG guidance.\n"
        "Return ONLY valid JSON with the keys: alt_short, alt_long, purpose, suggested_caption.\n"
        "alt_short: ≤160 chars, no 'image of', no duplication of nearby text.\n"
        "alt_long: multi-sentence, include salient visual details relevant to the context "
        "(axes, legends, labels, values, color encodings, trends, annotations), be specific.\n"
        "purpose: why this image is present (instructional, decorative, evidence, data viz, etc.).\n"
        "suggested_caption: a short figure caption/title if appropriate.\n"
        "Be certain to describe the full content in depth, you must identify every data point covered in a chart or graph."
    )
    user_context = f"BEFORE:\n{pre}\n\nAFTER:\n{post}"

    for attempt in range(RETRIES):
        try:
            # --- GPT-5 path: Responses API ---
            if model.startswith("gpt-5"):
                resp = client.responses.create(
                    model=model,
                    input=[
                        {
                            "role": "developer",
                            "content": [
                                {"type": "input_text", "text": system_instr}
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_context},
                                {"type": "input_image", "image_url": data_url}
                            ]
                        }
                    ],
                    text={"format": {"type": "text"}},     # we want text JSON back
                    reasoning={"effort": reasoning_effort},
                )
                text = _get_output_text(resp)

            # --- non-GPT-5 path: Chat Completions ---
            else:
                msg_content = [
                    {"type": "text", "text": system_instr + "\n\n" + user_context},
                    {"type": "image_url", "image_url": {"url": data_url}}  # 'detail' not needed here
                ]
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Return only a JSON object."},
                        {"role": "user", "content": msg_content},
                    ],
                    max_tokens=900,
                    temperature=0.2,
                )
                text = resp.choices[0].message.content.strip()

            # extract JSON safely
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1 and e > s:
                text = text[s:e+1]
            return json.loads(text)

        except Exception as e:
            # Optional: set DEBUG=1 in your env to see the reason
            if os.getenv("DEBUG") == "1":
                print(f"[ask_vision_for_alt] attempt {attempt+1} failed: {repr(e)}")
            if attempt == RETRIES - 1:
                return {
                    "alt_short": "Figure supporting the surrounding discussion.",
                    "alt_long": "Detailed description unavailable; describe the chart, axes, legends, and data trends manually to preserve accessibility context.",
                    "purpose": "contextual illustration",
                    "suggested_caption": ""
                }
            time.sleep(0.6 * (attempt + 1))


def _concat_prev(parts: List[Dict], idx: int, limit: int) -> str:
    acc, remaining = [], limit
    for j in range(idx - 1, -1, -1):
        if parts[j]["type"] == "text":
            t = parts[j]["text"]
            if len(t) >= remaining:
                acc.append(t[-remaining:])
                break
            acc.append(t); remaining -= len(t)
    return "".join(reversed(acc)).strip()

def _concat_next(parts: List[Dict], idx: int, limit: int) -> str:
    acc, taken = [], 0
    for j in range(idx + 1, len(parts)):
        if parts[j]["type"] == "text":
            t = parts[j]["text"]; need = min(limit - taken, len(t))
            acc.append(t[:need]); taken += need
            if taken >= limit: break
    return "".join(acc).strip()


def _merge_rectangles(rects: List[fitz.Rect], margin: float) -> List[fitz.Rect]:
    clusters: List[fitz.Rect] = []
    expanded: List[fitz.Rect] = []

    for rect in rects:
        base = fitz.Rect(rect)
        grow = base + (-margin, -margin, margin, margin)

        merged = False
        for idx in range(len(clusters)):
            if grow.intersects(expanded[idx]):
                clusters[idx] = clusters[idx] | base
                expanded[idx] = expanded[idx] | grow
                merged = True
                break
        if merged:
            continue

        clusters.append(base)
        expanded.append(grow)

    # second-pass merge to catch transitive overlaps
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        for i in range(len(clusters)):
            if merged:
                break
            for j in range(i + 1, len(clusters)):
                if expanded[i].intersects(expanded[j]):
                    clusters[i] = clusters[i] | clusters[j]
                    expanded[i] = expanded[i] | expanded[j]
                    del clusters[j]
                    del expanded[j]
                    merged = True
                    break
    return clusters


def _insert_part_by_position(parts: List[Dict], new_part: Dict) -> None:
    bbox = new_part.get("bbox")
    if not bbox:
        parts.append(new_part)
        return

    y_center = (bbox[1] + bbox[3]) / 2
    for idx, existing in enumerate(parts):
        eb = existing.get("bbox")
        if not eb:
            continue
        e_center = (eb[1] + eb[3]) / 2
        if y_center < e_center:
            parts.insert(idx, new_part)
            return
    parts.append(new_part)

# def pdf_to_text_with_image_labels_and_alts(pdf_path: str, out_dir: str, doc_id: str,
#                                            do_alts: bool = True, model: str = "gpt-5") -> Tuple[str, List[Dict]]:
#     os.makedirs(out_dir, exist_ok=True)
#     images_dir = os.path.join(out_dir, "images")
#     os.makedirs(images_dir, exist_ok=True)

#     client = OpenAI() if do_alts else None

#     doc = fitz.open(pdf_path)
#     manifest: List[Dict] = []
#     pages_text, pages_text_with_alt = [], []

#     for pno, page in enumerate(doc, start=1):
#         raw = page.get_text("rawdict")  # includes text & image blocks with positions
#         blocks = raw.get("blocks", [])  # see PyMuPDF block dictionary docs
#         parts: List[Dict] = []

#         for b in blocks:
#             if b["type"] == 0:
#                 lines = b.get("lines", [])
#                 buf = []
#                 for line in lines:
#                     spans = line.get("spans", [])
#                     buf.append("".join(s.get("text", "") for s in spans))
#                 parts.append({
#                     "type": "text",
#                     "text": ("\n".join(buf) + "\n"),
#                     "bbox": tuple(b.get("bbox", (0, 0, 0, 0))),
#                     "page": pno
#                 })

#             elif b["type"] == 1:
#                 bbox = b.get("bbox", [0,0,0,0])
#                 w = int(bbox[2] - bbox[0]); h = int(bbox[3] - bbox[1])
#                 if w < MIN_IMAGE_PX or h < MIN_IMAGE_PX:
#                     continue
#                 xref = b.get("image")

#                 # direct extract by xref; fallback = rasterize the bbox
#                 try:
#                     img = doc.extract_image(xref)
#                     ext = img.get("ext", "png"); data = img["image"]
#                 except Exception:
#                     pix = page.get_pixmap(clip=fitz.Rect(*bbox), alpha=False)
#                     ext = "png"; data = pix.tobytes()

#                 parts.append({
#                     "type": "img",
#                     "bbox": tuple(bbox),
#                     "page": pno,
#                     "ext": ext,
#                     "image_bytes": data,
#                     "origin": "raster"
#                 })

#         # synthesize raster snapshots from vector drawings (charts, diagrams, etc.)
#         vector_rects = []
#         for drawing in page.get_drawings():
#             rect = drawing.get("rect")
#             if not rect:
#                 continue
#             r = fitz.Rect(rect)
#             vector_rects.append(r)

#         merged_vectors = _merge_rectangles(vector_rects, VECTOR_MERGE_MARGIN) if vector_rects else []
#         for rect in merged_vectors:
#             if rect.width < MIN_IMAGE_PX or rect.height < MIN_IMAGE_PX:
#                 continue

#             # skip if we already have a raster image covering this area
#             overlaps_existing = False
#             for existing in parts:
#                 if existing.get("type") != "img":
#                     continue
#                 eb = existing.get("bbox")
#                 if not eb:
#                     continue
#                 existing_rect = fitz.Rect(*eb)
#                 intersection = rect & existing_rect
#                 if intersection.is_empty:
#                     continue
#                 overlap_ratio = intersection.get_area() / max(rect.get_area(), 1)
#                 if overlap_ratio > 0.8:
#                     overlaps_existing = True
#                     break
#             if overlaps_existing:
#                 continue

#             pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(VECTOR_RASTER_SCALE, VECTOR_RASTER_SCALE), alpha=False)
#             data = pix.tobytes("png")

#             vector_part = {
#                 "type": "img",
#                 "bbox": tuple(rect),
#                 "page": pno,
#                 "ext": "png",
#                 "image_bytes": data,
#                 "origin": "vector"
#             }
#             _insert_part_by_position(parts, vector_part)

#         # Build page text + (optional) inline short alts for quick QA
#         plain_chunks, with_alt_chunks = [], []
#         image_index = 0
#         for part in parts:
#             if part.get("type") == "img":
#                 image_index += 1
#                 label = f"[[IMG:p{pno}-i{image_index}]]"
#                 part["label"] = label
#                 ext = part.get("ext", "png")
#                 img_path = os.path.join(images_dir, f"{doc_id}_p{pno}_i{image_index}.{ext}")
#                 with open(img_path, "wb") as f:
#                     f.write(part.get("image_bytes", b""))
#                 part["path"] = img_path
#                 if "image_bytes" in part:
#                     del part["image_bytes"]

#         for idx, part in enumerate(parts):
#             if part["type"] == "text":
#                 plain_chunks.append(part["text"])
#                 with_alt_chunks.append(part["text"])
#             else:
#                 label = part["label"]
#                 plain_chunks.append(label + "\n")

#                 if do_alts:
#                     pre = _concat_prev(parts, idx, MAX_CONTEXT_CHARS)
#                     post = _concat_next(parts, idx, MAX_CONTEXT_CHARS)
#                     alts = ask_vision_for_alt(client, model, part["path"], pre, post)
#                 else:
#                     alts = {"alt_short": "", "alt_long": "", "purpose": "", "suggested_caption": ""}

#                 detailed_alt = alts.get("alt_long") or alts.get("alt_short", "")

#                 manifest.append({
#                     "label": label, "page": part["page"], "bbox": part["bbox"], "path": part["path"],
#                     "alt_short": alts.get("alt_short", ""), "alt_long": alts.get("alt_long", ""),
#                     "purpose": alts.get("purpose", ""), "suggested_caption": alts.get("suggested_caption", ""),
#                     "source": part.get("origin", "raster")
#                 })

#                 if do_alts and detailed_alt:
#                     with_alt_chunks.append(f"{label}\nALT_TEXT: {detailed_alt}\n")
#                 else:
#                     with_alt_chunks.append(label + "\n")

#         pages_text.append("".join(plain_chunks).strip())
#         pages_text_with_alt.append("".join(with_alt_chunks).strip())

#     plain_text = "\n\n".join(pages_text)
#     text_with_alt = "\n\n".join(pages_text_with_alt)

#     with open(os.path.join(out_dir, f"{doc_id}_text.txt"), "w", encoding="utf-8") as f:
#         f.write(plain_text)
#     with open(os.path.join(out_dir, f"{doc_id}_images.json"), "w", encoding="utf-8") as f:
#         json.dump(manifest, f, ensure_ascii=False, indent=2)
#     if do_alts:
#         with open(os.path.join(out_dir, f"{doc_id}_text_with_alt.txt"), "w", encoding="utf-8") as f:
#             f.write(text_with_alt)
#         with open(os.path.join(out_dir, f"{doc_id}_text_for_rag.txt"), "w", encoding="utf-8") as f:
#             f.write(text_with_alt)

#     return plain_text, manifest

def pdf_to_text_with_image_labels_and_alts(pdf_path: str, out_dir: str, doc_id: str,
                                           do_alts: bool = True, model: str = "gpt-5") -> Tuple[str, List[Dict]]:
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    client = OpenAI() if do_alts else None

    doc = fitz.open(pdf_path)
    manifest: List[Dict] = []
    pages_text, pages_text_with_alt = [], []

    for pno, page in enumerate(doc, start=1):
        parts: List[Dict] = []

        # ---- Text parts (robust fallback chain) ----
        text_parts = _extract_text_parts(page)
        for tp in text_parts:
            tp.update({"page": pno})
            parts.append(tp)

        # ---- Raster image blocks discovered via rawdict ----
        try:
            raw = page.get_text("rawdict")
            blocks = raw.get("blocks", []) if isinstance(raw, dict) else []
        except Exception:
            blocks = []

        for b in blocks:
            if b.get("type") != 1:
                continue
            bbox = b.get("bbox", [0, 0, 0, 0])
            w = int(bbox[2] - bbox[0]); h = int(bbox[3] - bbox[1])
            if w < MIN_IMAGE_PX or h < MIN_IMAGE_PX:
                continue

            xref = b.get("image")
            # direct extract; fallback = crop render
            try:
                img = doc.extract_image(xref)
                ext = img.get("ext", "png")
                data = img["image"]
            except Exception:
                pix = page.get_pixmap(clip=fitz.Rect(*bbox), alpha=False)
                ext = "png"
                data = pix.tobytes()

            raster_part = {
                "type": "img",
                "bbox": tuple(bbox),
                "page": pno,
                "ext": ext,
                "image_bytes": data,
                "origin": "raster"
            }
            # insert by vertical position among existing parts
            _insert_part_by_position(parts, raster_part)

        # ---- Vector drawings → raster snapshots (charts/diagrams) ----
        vector_rects = []
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect:
                vector_rects.append(fitz.Rect(rect))

        merged_vectors = _merge_rectangles(vector_rects, VECTOR_MERGE_MARGIN) if vector_rects else []
        for rect in merged_vectors:
            if rect.width < MIN_IMAGE_PX or rect.height < MIN_IMAGE_PX:
                continue

            # skip if largely overlapping a raster image we already added
            overlaps_existing = False
            for existing in parts:
                if existing.get("type") != "img":
                    continue
                eb = existing.get("bbox")
                if not eb:
                    continue
                existing_rect = fitz.Rect(*eb)
                inter = rect & existing_rect
                if not inter.is_empty and (inter.get_area() / max(rect.get_area(), 1)) > 0.8:
                    overlaps_existing = True
                    break
            if overlaps_existing:
                continue

            pix = page.get_pixmap(
                clip=rect,
                matrix=fitz.Matrix(VECTOR_RASTER_SCALE, VECTOR_RASTER_SCALE),
                alpha=False
            )
            data = pix.tobytes("png")

            _insert_part_by_position(parts, {
                "type": "img",
                "bbox": tuple(rect),
                "page": pno,
                "ext": "png",
                "image_bytes": data,
                "origin": "vector"
            })

        # ---- Assign labels, write image files ----
        plain_chunks, with_alt_chunks = [], []
        image_index = 0

        # write images & attach labels first
        for part in parts:
            if part.get("type") == "img":
                image_index += 1
                label = f"[[IMG:p{pno}-i{image_index}]]"
                part["label"] = label
                ext = part.get("ext", "png")
                img_path = os.path.join(images_dir, f"{doc_id}_p{pno}_i{image_index}.{ext}")
                with open(img_path, "wb") as f:
                    f.write(part.get("image_bytes", b""))
                part["path"] = img_path
                if "image_bytes" in part:
                    del part["image_bytes"]

        # ---- Build page text (+ inline ALT for QA) & call the model if enabled ----
        for idx, part in enumerate(parts):
            if part["type"] == "text":
                plain_chunks.append(part["text"])
                with_alt_chunks.append(part["text"])
            else:
                label = part["label"]
                plain_chunks.append(label + "\n")

                if do_alts:
                    pre = _concat_prev(parts, idx, MAX_CONTEXT_CHARS)
                    post = _concat_next(parts, idx, MAX_CONTEXT_CHARS)
                    alts = ask_vision_for_alt(client, model, part["path"], pre, post)
                else:
                    alts = {"alt_short": "", "alt_long": "", "purpose": "", "suggested_caption": ""}

                detailed_alt = alts.get("alt_long") or alts.get("alt_short", "")

                manifest.append({
                    "label": label,
                    "page": part["page"],
                    "bbox": part["bbox"],
                    "path": part["path"],
                    "alt_short": alts.get("alt_short", ""),
                    "alt_long": alts.get("alt_long", ""),
                    "purpose": alts.get("purpose", ""),
                    "suggested_caption": alts.get("suggested_caption", ""),
                    "source": part.get("origin", "raster")
                })

                if do_alts and detailed_alt:
                    with_alt_chunks.append(f"{label}\nALT_TEXT: {detailed_alt}\n")
                else:
                    with_alt_chunks.append(label + "\n")

        pages_text.append("".join(plain_chunks).strip())
        pages_text_with_alt.append("".join(with_alt_chunks).strip())

    plain_text = "\n\n".join(pages_text)
    text_with_alt = "\n\n".join(pages_text_with_alt)

    with open(os.path.join(out_dir, f"{doc_id}_text.txt"), "w", encoding="utf-8") as f:
        f.write(plain_text)
    with open(os.path.join(out_dir, f"{doc_id}_images.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if do_alts:
        with open(os.path.join(out_dir, f"{doc_id}_text_with_alt.txt"), "w", encoding="utf-8") as f:
            f.write(text_with_alt)
        with open(os.path.join(out_dir, f"{doc_id}_text_for_rag.txt"), "w", encoding="utf-8") as f:
            f.write(text_with_alt)

    return plain_text, manifest


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Extract images, label them in text, and generate alt descriptions.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", required=True, help="Output folder (will be created)")
    parser.add_argument("--doc-id", default="doc", help="Identifier for filenames / labels")
    parser.add_argument("--no-alts", action="store_true", help="Skip calling the vision model")
    args = parser.parse_args()

    model = os.getenv("OPENAI_VISION_MODEL", "gpt-5")
    do_alts = not args.no_alts

    text, manifest = pdf_to_text_with_image_labels_and_alts(
        pdf_path=args.pdf,
        out_dir=args.out,
        doc_id=args.doc_id,
        do_alts=do_alts,
        model=model
    )
    written = [
        f"{args.out}/{args.doc_id}_text.txt",
        f"{args.out}/{args.doc_id}_images.json"
    ]
    if do_alts:
        written.append(f"{args.out}/{args.doc_id}_text_with_alt.txt")
        written.append(f"{args.out}/{args.doc_id}_text_for_rag.txt")
    print(f"[ok] Wrote {'; '.join(written)}")

if __name__ == "__main__":
    main()
