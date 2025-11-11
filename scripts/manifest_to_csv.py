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

def main(in_path: str, out_path: str):
    today = date.today().isoformat()

    with open(in_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    rows = []
    for obj in items:
        label = obj.get("label", "")
        page = obj.get("page")
        title = make_title(label, page)

        suggested_caption = (obj.get("suggested_caption") or "").strip()
        alt_short = (obj.get("alt_short") or "").strip()

        content_text = f"{suggested_caption}\n\n{title}\n\n{alt_short}\n\n{alt_short}"

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
