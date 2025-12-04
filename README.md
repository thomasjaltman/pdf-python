# PDF Image Extractor + Alt Text Generator

Small toolkit to extract text and images from a PDF, label images inline, optionally generate accessibility alt text with OpenAI Vision, and export an image manifest to CSV.

## Install
- Python 3.10+ recommended. Optional: use a venv (e.g., `python -m venv .venv && source .venv/bin/activate`).
- Install dependencies:
  - `pip install PyMuPDF python-dotenv openai`

## Environment
- Set your OpenAI API key (shell or `.env`):
  - `export OPENAI_API_KEY=sk-...`
- Optional model tuning via env vars:
  - `OPENAI_VISION_MODEL` (default: `gpt-5`). Set to a vision-capable model ID (e.g., `gpt-4o-mini`) if desired.
  - `OPENAI_VISION_REASONING` (default: `high`). One of `low|medium|high`.
  - `DEBUG=1` to print errors from vision calls when retries happen.

## Usage: Extract text/images (+ optional alts)
Run the main script to process a PDF.

- Command:
  - `python scripts/pdf_images_with_alt.py --pdf <path/to.pdf> --out <out_dir> [--doc-id DOC] [--no-alts]`

- Flags (scripts/pdf_images_with_alt.py):
  - `--pdf` (required): Path to input PDF.
  - `--out` (required): Output directory (created if missing).
  - `--doc-id` (optional, default `doc`): Identifier used in output filenames and labels.
  - `--no-alts` (optional): Skip OpenAI Vision calls; still extracts text, images, labels, and manifest.

- Outputs written to `--out`:
  - `<DOC>_text.txt` — PDF text with `[[IMG:pX-iY]]` labels inserted.
  - `<DOC>_images.json` — image manifest with page, bbox, file path, and alt text (if generated).
  - `<DOC>_text_with_alt.txt` and `<DOC>_text_for_rag.txt` — only when alt generation is enabled.
  - `images/` — extracted images as files named `<DOC>_p{page}_i{index}.<ext>`.

Example (using the sample PDF):
- `python scripts/pdf_images_with_alt.py --pdf data/nhra-judgingdiagrams.pdf --out out --doc-id nhra`
- To avoid model calls: `python scripts/pdf_images_with_alt.py --pdf data/nhra-judgingdiagrams.pdf --out out --doc-id nhra --no-alts`

## Usage: Manifest → CSV
Convert the image manifest JSON to a CSV suitable for downstream workflows.

- Command:
  - `python scripts/manifest_to_csv.py <input_manifest_json> <output_csv>`

Example:
- `python scripts/manifest_to_csv.py out/nhra_images.json out/nhra_images.csv`

## Notes
- Alt text generation requires a valid `OPENAI_API_KEY` and a vision-capable model.
- The tool deduplicates vector drawings from raster images and labels images in reading order.
- Images in text are denoted as `[[IMG:p{page}-i{index}]]` and referenced in the manifest.

