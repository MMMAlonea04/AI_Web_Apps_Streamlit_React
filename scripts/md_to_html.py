"""Chuyển một file Markdown (có bảng) thành file HTML độc lập, tự chứa CSS — mở/ in được ngay.

    python scripts/md_to_html.py docs/cach-lam-tong-hop.md          # -> docs/cach-lam-tong-hop.html
    python scripts/md_to_html.py README.md --out dist/README.html
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.colab_utils import use_utf8_console  # noqa: E402

use_utf8_console()

CSS = """
:root { --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --code-bg:#f6f8fa; --link:#0969da; }
* { box-sizing: border-box; }
body { margin:0; background:#f6f8fa; color:var(--fg); font:16px/1.65 -apple-system,"Segoe UI",Roboto,"Noto Sans",Arial,sans-serif; }
main { max-width:940px; margin:0 auto; padding:36px 28px 72px; background:#fff; }
h1 { font-size:1.9em; border-bottom:1px solid var(--line); padding-bottom:.3em; }
h2 { font-size:1.4em; margin-top:1.9em; border-bottom:1px solid var(--line); padding-bottom:.2em; }
h3, h4 { font-size:1.12em; margin-top:1.5em; }
a { color:var(--link); }
code { background:var(--code-bg); padding:.15em .35em; border-radius:5px; font-size:.9em; font-family:ui-monospace,"Cascadia Mono",Consolas,monospace; }
pre { background:var(--code-bg); border:1px solid var(--line); border-radius:8px; padding:12px 14px; overflow-x:auto; }
pre code { background:none; padding:0; }
table { border-collapse:collapse; width:100%; margin:1.1em 0; font-size:.95em; display:block; overflow-x:auto; }
th, td { border:1px solid var(--line); padding:7px 10px; text-align:left; vertical-align:top; }
th { background:#f0f3f6; white-space:nowrap; }
tbody tr:nth-child(even) { background:#fafbfc; }
blockquote { margin:1.1em 0; padding:2px 16px; border-left:4px solid var(--line); color:var(--muted); }
hr { border:0; border-top:1px solid var(--line); margin:2.2em 0; }
ul, ol { padding-left:1.5em; }
@media print { body { background:#fff; } main { max-width:none; padding:0; } table { display:table; font-size:.85em; } pre { white-space:pre-wrap; } }
"""

PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def render(md_path: Path) -> str:
    from markdown_it import MarkdownIt

    # "commonmark" không bật bảng nên phải enable("table"); preset "gfm-like" cần linkify-it-py.
    md = MarkdownIt("commonmark").enable("table")
    text = md_path.read_text(encoding="utf-8")
    body = md.render(text)
    title = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return PAGE.format(title=title.group(1).strip() if title else md_path.stem, css=CSS.strip(), body=body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("md", help="file Markdown nguồn")
    parser.add_argument("--out", default=None, help="file HTML đích (mặc định: cùng tên, đổi đuôi .html)")
    args = parser.parse_args()

    src = Path(args.md)
    if not src.is_file():
        print(f"❌ Không thấy {src}", flush=True)
        return 1
    out = Path(args.out) if args.out else src.with_suffix(".html")
    body = render(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(
        f"✅ {out} · {len(body) / 1024:.0f} KB · {body.count('<table>')} bảng · "
        f"{len(re.findall(r'<h[23]>', body))} mục · {body.count('<pre>')} khối code",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
