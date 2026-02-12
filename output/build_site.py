"""Convert predictions markdown to a styled HTML site for GitHub Pages."""

import os
import re
import sys

from config import PROCESSED_DIR, PREDICTION_SEASON, PROJECT_ROOT

SITE_DIR = os.path.join(PROJECT_ROOT, "_site")


def md_to_html(md_path: str) -> str:
    """Convert the predictions markdown to styled HTML."""
    with open(md_path, "r") as f:
        md = f.read()

    # Extract sections
    title_match = re.search(r"^# (.+)$", md, re.MULTILINE)
    title = title_match.group(1) if title_match else "NCAA Tournament Predictions"

    timestamp_match = re.search(r"^\*Last updated: (.+)\*$", md, re.MULTILINE)
    timestamp = timestamp_match.group(1) if timestamp_match else ""

    # Parse seed list table
    seed_rows = re.findall(r"^\| (\d+) \| (.+?) \|$", md, re.MULTILINE)

    # Parse First Four
    first_four = re.findall(r"^- (.+)$", md, re.MULTILINE)

    # Parse bracket code blocks
    brackets = re.findall(r"## (\w+) Region\n\n```text\n(.+?)```", md, re.DOTALL)

    # Build HTML
    seed_table_rows = ""
    for seed, teams in seed_rows:
        team_list = teams.split(", ")
        styled = []
        for t in team_list:
            if t.endswith("*"):
                styled.append(f'<span class="play-in">{t[:-1]}</span>')
            else:
                styled.append(t)
        seed_table_rows += f"<tr><td class='seed-num'>{seed}</td><td>{', '.join(styled)}</td></tr>\n"

    first_four_html = ""
    for game in first_four:
        first_four_html += f"<li>{game}</li>\n"

    brackets_html = ""
    for region, art in brackets:
        brackets_html += f"""
        <div class="bracket-region">
            <h3>{region} Region</h3>
            <pre>{art.rstrip()}</pre>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>68bracket — {title}</title>
    <style>
        :root {{
            --bg: #0f1117;
            --surface: #1a1d27;
            --border: #2a2d3a;
            --text: #e4e4e7;
            --text-muted: #8b8d98;
            --accent: #f97316;
            --accent-dim: #c2410c;
            --seed-1: #fbbf24;
            --seed-2: #a3a3a3;
            --seed-3: #d97706;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }}

        .container {{
            max-width: 960px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }}

        h1 {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}

        h1 .logo {{
            color: var(--accent);
        }}

        .subtitle {{
            font-size: 1.1rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .timestamp {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        h2 {{
            font-size: 1.4rem;
            font-weight: 600;
            margin: 2.5rem 0 1rem;
            color: var(--accent);
        }}

        h3 {{
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }}

        /* Seed list table */
        .seed-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}

        .seed-table th {{
            text-align: left;
            padding: 0.6rem 1rem;
            border-bottom: 2px solid var(--accent);
            color: var(--accent);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .seed-table td {{
            padding: 0.5rem 1rem;
            border-bottom: 1px solid var(--border);
        }}

        .seed-table tr:hover {{
            background: var(--surface);
        }}

        .seed-num {{
            font-weight: 700;
            width: 3rem;
            text-align: center;
            color: var(--accent);
        }}

        .play-in {{
            color: var(--text-muted);
            font-style: italic;
        }}

        .play-in::after {{
            content: '*';
        }}

        /* First Four */
        .first-four-list {{
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 0.5rem;
            margin: 1rem 0;
        }}

        .first-four-list li {{
            background: var(--surface);
            padding: 0.6rem 1rem;
            border-radius: 6px;
            border-left: 3px solid var(--accent);
            font-size: 0.9rem;
        }}

        /* Brackets */
        .brackets-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
            gap: 1.5rem;
            margin: 1rem 0;
        }}

        .bracket-region {{
            background: var(--surface);
            border-radius: 8px;
            padding: 1.25rem;
            border: 1px solid var(--border);
        }}

        .bracket-region h3 {{
            color: var(--accent);
            margin-bottom: 0.5rem;
        }}

        .bracket-region pre {{
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            font-size: 0.7rem;
            line-height: 1.35;
            overflow-x: auto;
            color: var(--text);
            white-space: pre;
        }}

        .footnote {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
        }}

        .footnote a {{
            color: var(--accent);
            text-decoration: none;
        }}

        @media (max-width: 600px) {{
            h1 {{ font-size: 1.5rem; }}
            .brackets-grid {{ grid-template-columns: 1fr; }}
            .bracket-region pre {{ font-size: 0.55rem; }}
            .container {{ padding: 1rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span class="logo">68</span>bracket</h1>
            <div class="subtitle">{title}</div>
            <div class="timestamp">{timestamp}</div>
        </header>

        <h2>Seed List</h2>
        <table class="seed-table">
            <thead>
                <tr><th>Seed</th><th>Teams</th></tr>
            </thead>
            <tbody>
                {seed_table_rows}
            </tbody>
        </table>

        <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">
            * First Four play-in game
        </p>

        <h2>First Four</h2>
        <ul class="first-four-list">
            {first_four_html}
        </ul>

        <h2>Regional Brackets</h2>
        <div class="brackets-grid">
            {brackets_html}
        </div>

        <div class="footnote">
            Predictions generated by <a href="https://github.com/hunterwalklin/68bracket">68bracket</a>
            — updated daily
        </div>
    </div>
</body>
</html>"""

    return html


def build():
    """Build the site from the latest predictions."""
    md_path = os.path.join(PROCESSED_DIR, f"predictions_{PREDICTION_SEASON}.md")
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found. Run the predict command first.")
        sys.exit(1)

    os.makedirs(SITE_DIR, exist_ok=True)

    html = md_to_html(md_path)

    out_path = os.path.join(SITE_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Site built: {out_path}")


if __name__ == "__main__":
    build()
