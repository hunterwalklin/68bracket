"""Convert predictions markdown to a styled HTML site for GitHub Pages."""

import json
import os
import re
import sys
from html import escape

import pandas as pd

from config import PROCESSED_DIR, PREDICTION_SEASON, PROJECT_ROOT

SITE_DIR = os.path.join(PROJECT_ROOT, "_site")


def _style_team(name: str, changes: dict, is_play_in: bool = False) -> str:
    """Wrap a team name with change indicator if they moved after playing."""
    display = escape(name)
    change = changes.get(name)

    if is_play_in and not change:
        return f'<span class="play-in">{display}</span>'

    if change:
        direction = change["direction"]
        prev = change["prev_seed"]
        new = change["new_seed"]
        css = "move-up" if direction == "up" else "move-down"
        arrow = "\u25b2" if direction == "up" else "\u25bc"  # ▲ ▼
        if prev is None:
            tooltip = f"NEW to field at {new} seed"
        elif direction == "up":
            tooltip = f"Moved up from {prev} to {new} seed after playing"
        else:
            tooltip = f"Dropped from {prev} to {new} seed after playing"

        inner = f'<span class="{css}" title="{tooltip}">{arrow}</span> {display}'
        if is_play_in:
            return f'<span class="play-in">{inner}</span>'
        return inner

    if is_play_in:
        return f'<span class="play-in">{display}</span>'
    return display


def _build_stats_table(stats_df: pd.DataFrame) -> str:
    """Generate an HTML table of all team stats, sorted by NET ranking."""
    df = stats_df[stats_df["net_ranking"] > 0].sort_values("net_ranking").reset_index(drop=True)
    rows = ""
    for i, r in df.iterrows():
        wab = r.get("wab")
        if pd.notna(wab):
            wab_val = float(wab)
            wab_cls = "wab-pos" if wab_val > 0 else "wab-neg" if wab_val < 0 else ""
            wab_str = f"{wab_val:+.1f}"
        else:
            wab_cls = ""
            wab_str = "—"

        def _int(col):
            v = r.get(col)
            return str(int(v)) if pd.notna(v) else "—"

        def _rec(w_col, l_col):
            w, l = r.get(w_col), r.get(l_col)
            if pd.notna(w) and pd.notna(l):
                return f"{int(w)}-{int(l)}"
            return "—"

        def _sv(w_col, l_col):
            """Sort value for W-L records: wins * 100 - losses."""
            w, l = r.get(w_col), r.get(l_col)
            if pd.notna(w) and pd.notna(l):
                return int(w) * 100 - int(l)
            return -9999

        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td class='stats-team'>{escape(str(r.get('team', '')))}</td>"
            f"<td>{escape(str(r.get('conference', '')))}</td>"
            f"<td data-sv='{_sv('wins', 'losses')}'>{_rec('wins', 'losses')}</td>"
            f"<td>{_int('net_ranking')}</td>"
            f"<td>{_int('kpi')}</td>"
            f"<td>{_int('sor')}</td>"
            f"<td>{_int('bpi')}</td>"
            f"<td>{_int('pom')}</td>"
            f"<td>{_int('trank_rank')}</td>"
            f"<td class='{wab_cls}'>{wab_str}</td>"
            f"<td data-sv='{_sv('q1_wins', 'q1_losses')}'>{_rec('q1_wins', 'q1_losses')}</td>"
            f"<td data-sv='{_sv('q2_wins', 'q2_losses')}'>{_rec('q2_wins', 'q2_losses')}</td>"
            f"<td data-sv='{_sv('q3_wins', 'q3_losses')}'>{_rec('q3_wins', 'q3_losses')}</td>"
            f"<td data-sv='{_sv('q4_wins', 'q4_losses')}'>{_rec('q4_wins', 'q4_losses')}</td>"
            f"</tr>\n"
        )

    return f"""<div class="stats-scroll"><table class="stats-table" id="stats-table">
<thead><tr>
    <th data-sort="num">#</th><th data-sort="str">Team</th><th data-sort="str">Conf</th><th data-sort="sv">Record</th>
    <th data-sort="num">NET</th><th data-sort="num">KPI</th><th data-sort="num">SOR</th><th data-sort="num">BPI</th>
    <th data-sort="num">KenPom</th><th data-sort="num">Torvik</th><th data-sort="num">WAB</th>
    <th data-sort="sv">Q1</th><th data-sort="sv">Q2</th><th data-sort="sv">Q3</th><th data-sort="sv">Q4</th>
</tr></thead>
<tbody>
{rows}</tbody>
</table></div>"""


def md_to_html(md_path: str, changes: dict | None = None, stats_html: str = "") -> str:
    """Convert the predictions markdown to styled HTML."""
    if changes is None:
        changes = {}

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

    # Parse bubble
    last_4_in = re.findall(r"^\*\*Last 4 In:\*\* (.+)$", md, re.MULTILINE)
    first_4_out = re.findall(r"^\*\*First 4 Out:\*\* (.+)$", md, re.MULTILINE)
    next_4_out = re.findall(r"^\*\*Next 4 Out:\*\* (.+)$", md, re.MULTILINE)

    # Parse bracket code blocks
    brackets = re.findall(r"## (\w+) Region\n\n```text\n(.+?)```", md, re.DOTALL)

    # Build bubble HTML
    bubble_html = ""
    if last_4_in:
        def _bubble_row(label: str, teams_str: str, css_class: str) -> str:
            teams = [t.strip() for t in teams_str.split(", ")]
            items = "".join(f'<span class="bubble-team">{escape(t)}</span>' for t in teams)
            return f'<div class="bubble-row {css_class}"><div class="bubble-label">{label}</div><div class="bubble-teams">{items}</div></div>'

        bubble_html = '<div class="bubble-section"><h2>Bubble Watch</h2>'
        bubble_html += _bubble_row("Last 4 In", last_4_in[0], "bubble-in")
        if first_4_out:
            bubble_html += _bubble_row("First 4 Out", first_4_out[0], "bubble-out")
        if next_4_out:
            bubble_html += _bubble_row("Next 4 Out", next_4_out[0], "bubble-far")
        bubble_html += "</div>"

    # Build seed table rows with change indicators
    seed_table_rows = ""
    for seed, teams in seed_rows:
        team_list = teams.split(", ")
        styled = []
        for t in team_list:
            is_play_in = t.endswith("*")
            clean_name = t[:-1] if is_play_in else t
            styled.append(_style_team(clean_name, changes, is_play_in))
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
            --green: #22c55e;
            --green-dim: #16a34a;
            --green-bg: rgba(34, 197, 94, 0.1);
            --red: #ef4444;
            --red-dim: #dc2626;
            --red-bg: rgba(239, 68, 68, 0.1);
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

        /* Change indicators */
        .move-up {{
            color: var(--green);
            font-size: 0.7em;
            margin-right: 2px;
        }}

        .move-down {{
            color: var(--red);
            font-size: 0.7em;
            margin-right: 2px;
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

        /* Bubble Watch */
        .bubble-section {{
            margin: 2rem 0;
        }}

        .bubble-row {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.6rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.4rem;
        }}

        .bubble-in {{
            background: var(--green-bg);
            border-left: 3px solid var(--green);
        }}

        .bubble-out {{
            background: var(--red-bg);
            border-left: 3px solid var(--red);
        }}

        .bubble-far {{
            background: var(--surface);
            border-left: 3px solid var(--text-muted);
        }}

        .bubble-label {{
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            min-width: 100px;
        }}

        .bubble-in .bubble-label {{ color: var(--green); }}
        .bubble-out .bubble-label {{ color: var(--red); }}
        .bubble-far .bubble-label {{ color: var(--text-muted); }}

        .bubble-teams {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }}

        .bubble-team {{
            background: rgba(255, 255, 255, 0.05);
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }}

        /* First Four */
        .first-four-list {{
            list-style: none;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.5rem;
            margin: 1rem 0;
        }}

        .first-four-list li {{
            background: var(--surface);
            padding: 0.6rem 0.75rem;
            border-radius: 6px;
            border-left: 3px solid var(--accent);
            font-size: 0.8rem;
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

        /* Tab navigation */
        .tab-radio {{ display: none; }}
        .tab-bar {{
            display: flex;
            gap: 0;
            border-bottom: 2px solid var(--border);
            margin-bottom: 2rem;
            position: sticky;
            top: 0;
            background: var(--bg);
            z-index: 10;
        }}
        .tab-bar label {{
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-muted);
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: color 0.15s, border-color 0.15s;
        }}
        .tab-bar label:hover {{
            color: var(--text);
        }}
        .tab-panel {{ display: none; }}
        #tab-bracket:checked ~ .tab-bar label[for="tab-bracket"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-stats:checked ~ .tab-bar label[for="tab-stats"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-bracket:checked ~ #panel-bracket {{ display: block; }}
        #tab-stats:checked ~ #panel-stats {{ display: block; }}

        /* Stats table */
        .stats-scroll {{
            overflow-x: auto;
            margin: 1rem 0;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            white-space: nowrap;
        }}
        .stats-table th {{
            text-align: left;
            padding: 0.5rem 0.6rem;
            border-bottom: 2px solid var(--accent);
            color: var(--accent);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            position: sticky;
            top: 0;
            background: var(--bg);
            cursor: pointer;
            user-select: none;
        }}
        .stats-table th:hover {{
            color: var(--text);
        }}
        .stats-table th::after {{
            content: '';
            display: inline-block;
            width: 0.6em;
            margin-left: 0.3em;
        }}
        .stats-table th.sort-asc::after {{
            content: '\\25b2';
        }}
        .stats-table th.sort-desc::after {{
            content: '\\25bc';
        }}
        .stats-table td {{
            padding: 0.4rem 0.6rem;
            border-bottom: 1px solid var(--border);
        }}
        .stats-table tr:hover {{
            background: var(--surface);
        }}
        .stats-team {{
            font-weight: 600;
        }}
        .wab-pos {{ color: var(--green); }}
        .wab-neg {{ color: var(--red); }}

        @media (max-width: 600px) {{
            h1 {{ font-size: 1.5rem; }}
            .first-four-list {{ grid-template-columns: repeat(2, 1fr); }}
            .brackets-grid {{ grid-template-columns: 1fr; }}
            .bracket-region pre {{ font-size: 0.55rem; }}
            .container {{ padding: 1rem; }}
            .tab-bar label {{ padding: 0.6rem 1rem; font-size: 0.85rem; }}
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

        <input type="radio" name="tabs" id="tab-bracket" class="tab-radio" checked>
        <input type="radio" name="tabs" id="tab-stats" class="tab-radio">

        <div class="tab-bar">
            <label for="tab-bracket">Bracket</label>
            <label for="tab-stats">Team Stats</label>
        </div>

        <div id="panel-bracket" class="tab-panel">
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

            {bubble_html}

            <h2>First Four</h2>
            <ul class="first-four-list">
                {first_four_html}
            </ul>

            <h2>Regional Brackets</h2>
            <div class="brackets-grid">
                {brackets_html}
            </div>
        </div>

        <div id="panel-stats" class="tab-panel">
            <h2>Team Stats</h2>
            {stats_html if stats_html else '<p style="color: var(--text-muted);">Stats data not available. Run the predict command to generate.</p>'}
        </div>

        <div class="footnote">
            Predictions generated by <a href="https://github.com/hunterwalklin/68bracket">68bracket</a>
            — updated daily
        </div>
    </div>
    <script>
    (function(){{
        var table=document.getElementById('stats-table');
        if(!table)return;
        var thead=table.tHead, tbody=table.tBodies[0];
        var ths=thead.rows[0].cells;
        var curCol=-1, curAsc=true;
        for(var i=0;i<ths.length;i++)(function(col){{
            ths[col].addEventListener('click',function(){{
                var asc=(col===curCol)?!curAsc:true;
                var type=this.dataset.sort||'str';
                var rows=Array.from(tbody.rows);
                rows.sort(function(a,b){{
                    var av,bv;
                    if(type==='sv'){{
                        av=parseFloat(a.cells[col].dataset.sv)||0;
                        bv=parseFloat(b.cells[col].dataset.sv)||0;
                    }}else if(type==='num'){{
                        av=parseFloat(a.cells[col].textContent)||9999;
                        bv=parseFloat(b.cells[col].textContent)||9999;
                    }}else{{
                        av=a.cells[col].textContent.toLowerCase();
                        bv=b.cells[col].textContent.toLowerCase();
                        return asc?av.localeCompare(bv):bv.localeCompare(av);
                    }}
                    return asc?av-bv:bv-av;
                }});
                for(var j=0;j<rows.length;j++)tbody.appendChild(rows[j]);
                for(var k=0;k<ths.length;k++)ths[k].classList.remove('sort-asc','sort-desc');
                this.classList.add(asc?'sort-asc':'sort-desc');
                curCol=col;curAsc=asc;
            }});
        }})(i);
    }})();
    </script>
</body>
</html>"""

    return html


def build(changes: dict | None = None, stats_df=None):
    """Build the site from the latest predictions.

    Args:
        changes: dict mapping team_name -> {"direction": "up"|"down", "prev_seed": N, "new_seed": N}
        stats_df: DataFrame of all teams with stats columns.
    """
    md_path = os.path.join(PROCESSED_DIR, f"predictions_{PREDICTION_SEASON}.md")
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found. Run the predict command first.")
        sys.exit(1)

    # Also try loading changes from disk (for standalone builds)
    if changes is None:
        changes_path = os.path.join(PROCESSED_DIR, "daily_changes.json")
        if os.path.exists(changes_path):
            with open(changes_path, "r") as f:
                changes = json.load(f)
        else:
            changes = {}

    # Persist / load stats snapshot (same pattern as daily_changes)
    stats_path = os.path.join(PROCESSED_DIR, "stats_snapshot.parquet")
    if stats_df is not None:
        stats_df.to_parquet(stats_path, index=False)
    elif os.path.exists(stats_path):
        stats_df = pd.read_parquet(stats_path)

    stats_html = ""
    if stats_df is not None:
        stats_html = _build_stats_table(stats_df)

    os.makedirs(SITE_DIR, exist_ok=True)

    html = md_to_html(md_path, changes=changes, stats_html=stats_html)

    out_path = os.path.join(SITE_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(html)

    # Persist changes so standalone site rebuilds can use them
    changes_path = os.path.join(PROCESSED_DIR, "daily_changes.json")
    with open(changes_path, "w") as f:
        json.dump(changes, f)

    n_changes = len(changes)
    print(f"Site built: {out_path}" + (f" ({n_changes} movers highlighted)" if n_changes else ""))


if __name__ == "__main__":
    build()
