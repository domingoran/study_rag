"""
Build a filename <-> element-tag map from a MinerU content_list.json.

The multimodal pipeline assigns sequential tags (fig-N / tbl-N / eq-N) to
images, tables, and equations in reading order at ingest time. Those tags are
never written to disk — this utility reconstructs the same numbering so you can
map an image hash filename back to its fig-N tag (and vice versa) without
ingesting.

Usage:
    python tag_map.py output/2412.02458v1
    python tag_map.py output/2412.02458v1 --lookup 0d82ef91   # partial hash ok
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_content_list(paper_dir: Path) -> Path:
    auto = paper_dir / "auto"
    base = auto if auto.is_dir() else paper_dir
    # Prefer the v1 file over *_v2.json (mirrors mineru_parser.find_content_list)
    cands = sorted(base.glob("*_content_list.json"))
    v1 = [c for c in cands if not c.name.endswith("_v2.json")]
    if not (v1 or cands):
        raise FileNotFoundError(f"No *_content_list.json under {base}")
    return (v1 or cands)[0]


def build_map(content_list_path: Path) -> list[dict]:
    items = json.loads(content_list_path.read_text(encoding="utf-8"))
    counters = {"figure": 0, "table": 0, "equation": 0}
    rows: list[dict] = []
    for item in items:
        t = item.get("type")
        if t in ("image", "chart"):
            kind, prefix = "figure", "fig"
        elif t == "table":
            kind, prefix = "table", "tbl"
        elif t == "equation":
            kind, prefix = "equation", "eq"
        else:
            continue
        counters[kind] += 1
        tag = f"{prefix}-{counters[kind]}"
        img = item.get("img_path") or ""
        caption = " ".join(
            item.get("img_caption") or item.get("table_caption") or []
        ).strip()
        rows.append({
            "tag": tag,
            "type": kind,
            "page": item.get("page_idx"),
            "img_path": img,
            "filename": Path(img).name if img else "",
            "caption": caption,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="MinerU filename <-> element-tag map")
    ap.add_argument("paper_dir", help="output/<paper_id> (or its auto/ dir)")
    ap.add_argument("--lookup", help="print only rows whose filename contains this string")
    ap.add_argument("--no-write", action="store_true", help="don't write tag_map.json")
    args = ap.parse_args()

    paper_dir = Path(args.paper_dir)
    cl = find_content_list(paper_dir)
    rows = build_map(cl)

    shown = rows
    if args.lookup:
        shown = [r for r in rows if args.lookup in r["filename"]]
        if not shown:
            print(f"No element whose filename contains '{args.lookup}'")

    for r in shown:
        cap = (r["caption"][:70] + "…") if len(r["caption"]) > 70 else r["caption"]
        print(f"{r['tag']:<7} p{r['page']:<3} {r['filename'] or '—':<70} {cap}")

    if not args.no_write:
        out = cl.parent / "tag_map.json"
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nWrote {out}  ({len(rows)} elements)")


if __name__ == "__main__":
    main()
