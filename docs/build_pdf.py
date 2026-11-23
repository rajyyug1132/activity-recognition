"""Export docs/HLD.md to docs/HLD.pdf with headless Chrome/Edge. Run: python docs/build_pdf.py"""
import os, re, subprocess, sys, tempfile, markdown
from pathlib import Path

DOCS = Path(__file__).parent
TEAM = [("Yyug", "Lead and integration"), ("Nihal", "UCI data pipeline"), ("Shravya", "Own data and validation"),
        ("Yash", "Models and training"), ("Yashaswini", "App and presentation")]
USN = {}  # ponytail: fill e.g. {"Yyug": "1BM..."}; blanks print as an empty line to write in

md = (DOCS / "HLD.md").read_text(encoding="utf-8")
body_md = md[md.index("## 1."):]  # title block is rebuilt as the cover page
body = markdown.markdown(body_md, extensions=["fenced_code", "tables"])
body = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
              lambda m: f'<div class="mermaid">{m.group(1)}</div>', body, flags=re.S)

rows = "".join(f"<tr><td>{n}</td><td>{USN.get(n, '')}</td><td>{r}</td></tr>" for n, r in TEAM)
html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 20mm 18mm; }}
body {{ font: 11pt/1.5 "Segoe UI", Arial, sans-serif; color: #111; }}
.cover {{ height: 250mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }}
.cover h1 {{ font-size: 28pt; margin: 0 0 6mm; }} .cover p {{ font-size: 13pt; margin: 2mm 0; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 2px; margin-top: 8mm; page-break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 10pt; page-break-inside: avoid; }}
th, td {{ border: 1px solid #bbb; padding: 4px 8px; text-align: left; vertical-align: top; }}
.cover td:nth-child(2) {{ min-width: 35mm; }} th {{ background: #f0f0f0; }}
code {{ font-size: 9.5pt; background: #f4f4f4; padding: 0 3px; }}
.mermaid {{ text-align: center; margin: 4mm 0; page-break-inside: avoid; }}
</style></head><body>
<div class="cover"><p>IBM Skill Lab</p><h1>High-Level Design<br>Activity Recognition System</h1>
<p>Review 2 &middot; 28 November 2026</p><br>
<table><tr><th>Name</th><th>USN</th><th>Role</th></tr>{rows}</table></div>
{body}
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad: true, theme: "neutral"}});</script>
</body></html>"""

browser = next(p for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                           r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"] if os.path.exists(p))
with tempfile.TemporaryDirectory() as tmp:
    page = Path(tmp) / "HLD.html"
    page.write_text(html, encoding="utf-8")
    out = DOCS / "HLD.pdf"
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--user-data-dir={tmp}/profile", f"--print-to-pdf={out}", page.as_uri()], check=True)
print("wrote", out) if out.exists() else sys.exit("PDF not written")
