#!/usr/bin/env python3
"""Simple markdown -> docx converter with fallbacks.

Usage:
    python tools/md_to_docx.py input.md [output.docx]

If `pypandoc` + `pandoc` are available they will be used. Otherwise a basic
fallback using `markdown` + `python-docx` will be used to produce a simple
DOCX file.
"""
from pathlib import Path
import sys


def convert(md_path: str, out_path: str):
    md_path = Path(md_path)
    out_path = Path(out_path)

    if not md_path.exists():
        raise FileNotFoundError(md_path)

    # Try pypandoc first (best quality) -------------------------------------------------
    try:
        import pypandoc  # type: ignore
        pypandoc.convert_file(str(md_path), 'docx', outputfile=str(out_path))
        print(f"Converted with pypandoc -> {out_path}")
        return
    except Exception:
        pass

    # Fallback: convert Markdown -> HTML -> DOCX using html2docx (preserves
    # headings, lists, tables, code blocks, links, and images)
    try:
        from markdown import markdown  # type: ignore
        from html2docx import html2docx as h2d  # type: ignore
        import re

        md = md_path.read_text(encoding='utf-8')
        # Use common extensions to keep structure and code blocks
        html = markdown(md, extensions=['extra', 'sane_lists', 'fenced_code', 'codehilite', 'tables'])

        # Convert relative image sources to absolute file URIs so html2docx can
        # embed them correctly. Skip remote URLs.
        def _fix_src(match):
            src = match.group(1)
            if src.startswith(('http://', 'https://', 'file:')):
                return f'src="{src}"'
            abs_path = (md_path.parent / src).resolve()
            return f'src="file://{abs_path}"'

        html = re.sub(r'src=[\'\"]([^\'\"]+)[\'\"]', _fix_src, html)

        # html2docx returns a BytesIO object containing the .docx data
        buf = h2d(html, title=md_path.stem)
        with open(out_path, 'wb') as fh:
            fh.write(buf.getbuffer())

        print(f"Converted with html2docx fallback -> {out_path}")
        return
    except Exception:
        # As a last resort, fall back to plain-text paragraphs via python-docx so
        # conversion never silently fails.
        try:
            from docx import Document  # type: ignore
            import re

            md = md_path.read_text(encoding='utf-8')
            text = re.sub(r'<[^>]+>', '', md)

            doc = Document()
            for para in text.split('\n\n'):
                para = para.strip()
                if not para:
                    continue
                doc.add_paragraph(para)

            doc.save(str(out_path))
            print(f"Converted with minimal python-docx fallback -> {out_path}")
            return
        except Exception as e:
            print("Failed to convert markdown -> docx:", e, file=sys.stderr)
            raise


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: md_to_docx.py input.md [output.docx]")
        sys.exit(2)

    md_file = sys.argv[1]
    if len(sys.argv) >= 3:
        out_file = sys.argv[2]
    else:
        out_file = str(Path.home() / 'Downloads' / (Path(md_file).stem + '.docx'))

    convert(md_file, out_file)
    print('Saved:', out_file)
