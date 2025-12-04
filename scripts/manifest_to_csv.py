import json, csv, re, sys
from datetime import date
from pathlib import Path

COLUMNS = [
    "Date",
    "DisplayUserLink",
    "Priority",
    "Title",
    "UserLink",
    "ContentLink",
    "ContentText",
    "ContentDecorator",
    "ContentType",
]

LABEL_RE = re.compile(r"\[\[IMG:p(\d+)-i(\d+)\]\]")

def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()

def make_title(label: str, page_value) -> str:
    m = LABEL_RE.fullmatch(label or "")
    if m:
        p, i = int(m.group(1)), int(m.group(2))
        return f"Image - Page {p}, image {i}"
    # fallback if label missing/odd: try the page field
    try:
        p = int(page_value) if page_value is not None else "?"
    except Exception:
        p = "?"
    return f"Image - Page {p}, image ?"

def build_content_text(title: str, *, label: str, suggested_caption: str,
                       alt_short: str, alt_long: str, purpose: str) -> str:
    sections = [title] if title else []

    def add_section(name: str, value: str):
        cleaned = _clean(value)
        if cleaned:
            sections.append(f"{name}: {cleaned}")
        else:
            sections.append(f"{name}: (not provided)")

    add_section("Label", label)
    add_section("Suggested caption", suggested_caption)
    add_section("Alt short", alt_short)
    add_section("Alt long", alt_long)
    add_section("Purpose", purpose)

    return "\n\n".join(sections).strip()

def main(in_path: str, out_path: str):
    today = date.today().isoformat()

    with open(in_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    rows = []
    for obj in items:
        label = _clean(obj.get("label"))
        page = obj.get("page")
        title = make_title(label, page)

        suggested_caption = _clean(obj.get("suggested_caption"))
        alt_short = _clean(obj.get("alt_short"))
        alt_long = _clean(obj.get("alt_long"))
        purpose = _clean(obj.get("purpose"))

        content_text = build_content_text(
            title,
            label=label,
            suggested_caption=suggested_caption,
            alt_short=alt_short,
            alt_long=alt_long,
            purpose=purpose,
        )

        row = {
            "Date": today,
            "DisplayUserLink": 1,
            "Priority": 50,
            "Title": title,
            "UserLink": "",
            "ContentLink": "",
            "ContentText": content_text,
            "ContentDecorator": "",
            "ContentType": "",
        }
        rows.append(row)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] Wrote {out_path} ({len(rows)} rows)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/manifest_to_csv.py <input_json> <output_csv>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
