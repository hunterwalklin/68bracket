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


def _build_conf_tab(stats_df) -> str:
    """Aggregate per-conference stats and generate an HTML rankings table."""
    if stats_df is None:
        return '<p style="color: var(--text-muted);">Conference data not available. Run the predict command to generate.</p>'

    df = stats_df[stats_df["net_ranking"] > 0].copy()
    grouped = df.groupby("conference")

    agg = pd.DataFrame({
        "teams": grouped.size(),
        "bids": grouped["selection_prob"].apply(lambda s: (s > 0.5).sum()),
        "avg_net": grouped["net_ranking"].mean(),
        "avg_kpi": grouped["kpi"].mean(),
        "avg_sor": grouped["sor"].mean(),
        "avg_bpi": grouped["bpi"].mean(),
        "wab": grouped["wab"].sum(),
        "q1w": grouped["q1_wins"].sum(),
        "q1l": grouped["q1_losses"].sum(),
        "q2w": grouped["q2_wins"].sum(),
        "q2l": grouped["q2_losses"].sum(),
        "q3w": grouped["q3_wins"].sum(),
        "q3l": grouped["q3_losses"].sum(),
        "q4w": grouped["q4_wins"].sum(),
        "q4l": grouped["q4_losses"].sum(),
    }).sort_values("avg_net").reset_index()

    def _sv(w, l):
        return int(w) * 100 - int(l)

    rows = ""
    for i, r in agg.iterrows():
        wab_val = float(r["wab"])
        wab_cls = "wab-pos" if wab_val > 0 else "wab-neg" if wab_val < 0 else ""
        wab_str = f"{wab_val:+.1f}"

        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td class='stats-team'>{escape(str(r['conference']))}</td>"
            f"<td>{int(r['teams'])}</td>"
            f"<td data-sv='{int(r['bids']) if int(r['bids']) > 0 else -1}'>{int(r['bids'])}</td>"
            f"<td>{r['avg_net']:.1f}</td>"
            f"<td>{r['avg_kpi']:.1f}</td>"
            f"<td>{r['avg_sor']:.1f}</td>"
            f"<td>{r['avg_bpi']:.1f}</td>"
            f"<td class='{wab_cls}'>{wab_str}</td>"
            f"<td data-sv='{_sv(r['q1w'], r['q1l'])}'>{int(r['q1w'])}-{int(r['q1l'])}</td>"
            f"<td data-sv='{_sv(r['q2w'], r['q2l'])}'>{int(r['q2w'])}-{int(r['q2l'])}</td>"
            f"<td data-sv='{_sv(r['q3w'], r['q3l'])}'>{int(r['q3w'])}-{int(r['q3l'])}</td>"
            f"<td data-sv='{_sv(r['q4w'], r['q4l'])}'>{int(r['q4w'])}-{int(r['q4l'])}</td>"
            f"</tr>\n"
        )

    return f"""<div class="stats-scroll"><table class="stats-table" id="conf-table">
<thead><tr>
    <th data-sort="num">#</th><th data-sort="str">Conference</th><th data-sort="num">Teams</th><th data-sort="sv">Bids</th>
    <th data-sort="num">Avg NET</th><th data-sort="num">Avg KPI</th><th data-sort="num">Avg SOR</th><th data-sort="num">Avg BPI</th>
    <th data-sort="num">WAB</th>
    <th data-sort="sv">Q1</th><th data-sort="sv">Q2</th><th data-sort="sv">Q3</th><th data-sort="sv">Q4</th>
</tr></thead>
<tbody>
{rows}</tbody>
</table></div>"""


def _build_autobid_tab(stats_df) -> str:
    """Generate an HTML table showing projected conference auto-bid winners."""
    if stats_df is None:
        return '<p style="color: var(--text-muted);">Auto-bid data not available. Run the predict command to generate.</p>'

    df = stats_df[stats_df["net_ranking"] > 0].copy()
    conferences = []
    for conf, grp in df.groupby("conference"):
        # Projected winner: best conf_win_pct, tiebreak by lowest net_ranking
        grp_sorted = grp.sort_values(["conf_win_pct", "net_ranking"], ascending=[False, True])
        winner = grp_sorted.iloc[0]
        runner_up = grp_sorted.iloc[1] if len(grp_sorted) > 1 else None

        prob = float(winner["selection_prob"]) if pd.notna(winner.get("selection_prob")) else 0.0
        at_large = int((grp[grp["team"] != winner["team"]]["selection_prob"] > 0.5).sum())

        conferences.append({
            "conf": conf,
            "winner": winner,
            "runner_up": runner_up,
            "prob": prob,
            "at_large": at_large,
        })

    # Sort by winner's NET ranking
    conferences.sort(key=lambda c: int(c["winner"]["net_ranking"]))

    rows = ""
    for i, c in enumerate(conferences):
        w = c["winner"]
        ru = c["runner_up"]
        prob_val = c["prob"] * 100

        if prob_val >= 70:
            prob_cls = "prob-high"
        elif prob_val >= 40:
            prob_cls = "prob-mid"
        else:
            prob_cls = "prob-low"
        prob_str = f"{prob_val:.1f}%"

        if c["prob"] >= 0.9:
            status = "Lock"
            status_cls = "status-lock"
        elif c["prob"] >= 0.5:
            status = "Likely In"
            status_cls = "status-likely"
        else:
            status = "Needs Bid"
            status_cls = "status-needs"

        conf_wins = int(w["conf_wins"]) if pd.notna(w.get("conf_wins")) else 0
        conf_losses = int(w["conf_losses"]) if pd.notna(w.get("conf_losses")) else 0
        conf_rec = f"{conf_wins}-{conf_losses}"
        conf_sv = conf_wins * 100 - conf_losses

        net = int(w["net_ranking"])
        al = c["at_large"]
        al_sv = al if al > 0 else -1

        ru_name = escape(str(ru["team"])) if ru is not None else "—"
        ru_net = int(ru["net_ranking"]) if ru is not None and pd.notna(ru.get("net_ranking")) else "—"

        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td class='stats-team'>{escape(str(c['conf']))}</td>"
            f"<td class='stats-team'>{escape(str(w['team']))}</td>"
            f"<td data-sv='{conf_sv}'>{conf_rec}</td>"
            f"<td>{net}</td>"
            f"<td class='{prob_cls}'>{prob_str}</td>"
            f"<td class='{status_cls}'>{status}</td>"
            f"<td data-sv='{al_sv}'>{al}</td>"
            f"<td>{ru_name}</td>"
            f"<td>{ru_net}</td>"
            f"</tr>\n"
        )

    return f"""<div class="stats-scroll"><table class="stats-table" id="autobid-table">
<thead><tr>
    <th data-sort="num">#</th><th data-sort="str">Conference</th><th data-sort="str">Projected Winner</th><th data-sort="sv">Conf Record</th>
    <th data-sort="num">NET</th><th data-sort="num">Prob</th><th data-sort="str">Status</th><th data-sort="sv">At-Large</th>
    <th data-sort="str">Runner-Up</th><th data-sort="num">Runner-Up NET</th>
</tr></thead>
<tbody>
{rows}</tbody>
</table></div>"""


def _build_bubble_tab(bubble: dict | None, stats_df) -> str:
    """Generate HTML for the Bubble Watch tab with detailed résumé data."""
    if bubble is None or stats_df is None:
        return '<p style="color: var(--text-muted);">Bubble data not available. Run the predict command to generate.</p>'

    def _int(row, col):
        v = row.get(col)
        return str(int(v)) if pd.notna(v) else "—"

    def _rec(row, w_col, l_col):
        w, l = row.get(w_col), row.get(l_col)
        if pd.notna(w) and pd.notna(l):
            return f"{int(w)}-{int(l)}"
        return "—"

    def _build_section(label, team_names, css_class):
        header_html = f'<div class="bubble-tab-header {css_class}"><span class="bubble-tab-label">{label}</span></div>'
        rows = ""
        for name in team_names:
            match = stats_df[stats_df["team"] == name]
            if match.empty:
                rows += f"<tr><td class='stats-team'>{escape(name)}</td>" + "<td>—</td>" * 12 + "</tr>\n"
                continue
            r = match.iloc[0]

            # Probability
            prob = r.get("selection_prob")
            if pd.notna(prob):
                prob_val = float(prob) * 100
                if prob_val >= 70:
                    prob_cls = "prob-high"
                elif prob_val >= 40:
                    prob_cls = "prob-mid"
                else:
                    prob_cls = "prob-low"
                prob_str = f"{prob_val:.1f}%"
            else:
                prob_cls = ""
                prob_str = "—"

            # WAB
            wab = r.get("wab")
            if pd.notna(wab):
                wab_val = float(wab)
                wab_cls = "wab-pos" if wab_val > 0 else "wab-neg" if wab_val < 0 else ""
                wab_str = f"{wab_val:+.1f}"
            else:
                wab_cls = ""
                wab_str = "—"

            rows += (
                f"<tr>"
                f"<td class='stats-team'>{escape(str(r.get('team', '')))}</td>"
                f"<td>{escape(str(r.get('conference', '')))}</td>"
                f"<td>{_rec(r, 'wins', 'losses')}</td>"
                f"<td class='{prob_cls}'>{prob_str}</td>"
                f"<td>{_int(r, 'net_ranking')}</td>"
                f"<td>{_int(r, 'kpi')}</td>"
                f"<td>{_int(r, 'sor')}</td>"
                f"<td>{_int(r, 'bpi')}</td>"
                f"<td class='{wab_cls}'>{wab_str}</td>"
                f"<td>{_rec(r, 'q1_wins', 'q1_losses')}</td>"
                f"<td>{_rec(r, 'q2_wins', 'q2_losses')}</td>"
                f"<td>{_rec(r, 'q3_wins', 'q3_losses')}</td>"
                f"<td>{_rec(r, 'q4_wins', 'q4_losses')}</td>"
                f"</tr>\n"
            )

        table = f"""<table class="stats-table bubble-tab-table">
<thead><tr>
    <th>Team</th><th>Conf</th><th>Record</th><th>Prob</th>
    <th>NET</th><th>KPI</th><th>SOR</th><th>BPI</th><th>WAB</th>
    <th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th>
</tr></thead>
<tbody>
{rows}</tbody>
</table>"""
        return f'<div class="bubble-tab-section">{header_html}{table}</div>'

    html = ""
    html += _build_section("Last 4 In", bubble.get("last_4_in", []), "bubble-in")
    html += _build_section("First 4 Out", bubble.get("first_4_out", []), "bubble-out")
    html += _build_section("Next 4 Out", bubble.get("next_4_out", []), "bubble-far")
    return html


def _build_matrix_tab(seed_rows: list[tuple[str, str]], bracketology: dict | None, stats_df) -> str:
    """Compare our seed predictions against CBS bracketology projections.

    Args:
        seed_rows: List of (seed, teams_str) from the predictions markdown.
        bracketology: Loaded bracketology.json data or None.
        stats_df: DataFrame with NET rankings for all teams.

    Returns:
        HTML string for the matrix table.
    """
    if not bracketology or not seed_rows:
        return '<p style="color: var(--text-muted);">Bracket Matrix data not available. Run scrape and predict to generate.</p>'

    # Parse our seed predictions, preserving S-curve order
    our_teams = {}
    our_order = {}  # team_name -> overall position (0-based)
    pos = 0
    for seed_str, teams_str in seed_rows:
        seed = int(seed_str)
        for t in teams_str.split(", "):
            clean = t.rstrip("*").strip()
            our_teams[clean] = {"seed": seed, "region": ""}
            our_order[clean] = pos
            pos += 1

    # Parse our regions from bracket data (will be filled by md_to_html caller)
    # For now, region comes from bracketology or stays empty

    # Parse CBS teams from bracketology JSON
    cbs_teams = {}  # team_name -> {"seed": N, "region": "X", "school_id": "..."}
    source_name = ""
    for src, src_data in bracketology.get("sources", {}).items():
        source_name = src
        for entry in src_data.get("teams", []):
            cbs_teams[entry["team"]] = {
                "seed": entry["seed"],
                "region": entry.get("region", ""),
                "school_id": entry.get("school_id", ""),
            }

    if not cbs_teams:
        return '<p style="color: var(--text-muted);">No CBS bracketology data found.</p>'

    # Build a unified list of all teams from both sources
    all_team_names = set(our_teams.keys()) | set(cbs_teams.keys())

    # Also try matching by normalized name for teams that appear under different names
    # Build a school_id lookup for our teams using stats_df
    our_id_to_name = {}
    if stats_df is not None:
        for name in our_teams:
            match = stats_df[stats_df["team"] == name]
            if not match.empty:
                sid = match.iloc[0].get("school_id", "")
                if sid:
                    our_id_to_name[sid] = name

    cbs_id_to_name = {v["school_id"]: k for k, v in cbs_teams.items() if v.get("school_id")}

    # Match CBS teams to our teams by school_id
    cbs_matched = {}  # our_team_name -> cbs_data
    unmatched_cbs = dict(cbs_teams)

    for cbs_name, cbs_data in cbs_teams.items():
        # Direct name match
        if cbs_name in our_teams:
            cbs_matched[cbs_name] = cbs_data
            unmatched_cbs.pop(cbs_name, None)
            continue

        # school_id match
        sid = cbs_data.get("school_id", "")
        if sid and sid in our_id_to_name:
            our_name = our_id_to_name[sid]
            cbs_matched[our_name] = cbs_data
            unmatched_cbs.pop(cbs_name, None)

    # Build rows
    matrix_rows = []

    # Teams in our bracket (matched or only-ours)
    for name, our_data in our_teams.items():
        our_seed = our_data["seed"]
        net = ""
        if stats_df is not None:
            match = stats_df[stats_df["team"] == name]
            if not match.empty:
                net_val = match.iloc[0].get("net_ranking")
                if pd.notna(net_val):
                    net = str(int(net_val))

        if name in cbs_matched:
            cbs_data = cbs_matched[name]
            cbs_seed = cbs_data["seed"]
            diff = our_seed - cbs_seed
            abs_diff = abs(diff)
            cbs_region = cbs_data.get("region", "")

            if diff == 0:
                status = "Agreement"
                status_cls = "diff-zero"
            elif diff < 0:
                status = "Higher"      # We rate higher (lower seed number)
                status_cls = "diff-pos"
            else:
                status = "Lower"       # We rate lower (higher seed number)
                status_cls = "diff-neg"

            diff_str = f"{diff:+d}" if diff != 0 else "0"
            diff_cls = "diff-pos" if diff < 0 else "diff-neg" if diff > 0 else "diff-zero"

            matrix_rows.append({
                "team": name, "our_seed": our_seed, "cbs_seed": cbs_seed,
                "diff": diff, "abs_diff": abs_diff, "diff_str": diff_str,
                "diff_cls": diff_cls, "our_region": our_data.get("region", ""),
                "cbs_region": cbs_region, "net": net,
                "status": status, "status_cls": status_cls,
            })
        else:
            matrix_rows.append({
                "team": name, "our_seed": our_seed, "cbs_seed": None,
                "diff": None, "abs_diff": 99, "diff_str": "",
                "diff_cls": "", "our_region": our_data.get("region", ""),
                "cbs_region": "", "net": net,
                "status": "Only Ours", "status_cls": "status-only-ours",
            })

    # Teams only in CBS bracket
    for cbs_name, cbs_data in unmatched_cbs.items():
        net = ""
        if stats_df is not None:
            match = stats_df[stats_df["team"] == cbs_name]
            if not match.empty:
                net_val = match.iloc[0].get("net_ranking")
                if pd.notna(net_val):
                    net = str(int(net_val))

        matrix_rows.append({
            "team": cbs_name, "our_seed": None, "cbs_seed": cbs_data["seed"],
            "diff": None, "abs_diff": 99, "diff_str": "",
            "diff_cls": "", "our_region": "",
            "cbs_region": cbs_data.get("region", ""), "net": net,
            "status": "Only CBS", "status_cls": "status-only-cbs",
        })

    # Sort by S-curve order (teams not in our bracket go to the bottom)
    matrix_rows.sort(key=lambda r: our_order.get(r["team"], 9999))

    rows_html = ""
    for i, r in enumerate(matrix_rows):
        our_seed_str = str(r["our_seed"]) if r["our_seed"] is not None else "\u2014"
        cbs_seed_str = str(r["cbs_seed"]) if r["cbs_seed"] is not None else "\u2014"
        diff_display = r["diff_str"] if r["diff_str"] else "\u2014"
        our_seed_sv = r["our_seed"] if r["our_seed"] is not None else 99
        cbs_seed_sv = r["cbs_seed"] if r["cbs_seed"] is not None else 99
        diff_sv = r["abs_diff"]
        net_sv = int(r["net"]) if r["net"] else 9999

        rows_html += (
            f"<tr>"
            f"<td class='stats-team'>{escape(r['team'])}</td>"
            f"<td data-sv='{our_seed_sv}'>{our_seed_str}</td>"
            f"<td data-sv='{cbs_seed_sv}'>{cbs_seed_str}</td>"
            f"<td class='{r['diff_cls']}' data-sv='{diff_sv}'>{diff_display}</td>"
            f"<td data-sv='{net_sv}'>{r['net'] or '\u2014'}</td>"
            f"<td class='{r['status_cls']}'>{r['status']}</td>"
            f"</tr>\n"
        )

    return f"""<div class="stats-scroll"><table class="stats-table" id="matrix-table">
<thead><tr>
    <th data-sort="str">Team</th><th data-sort="sv">Our Seed</th><th data-sort="sv">CBS Seed</th>
    <th data-sort="sv">Diff</th><th data-sort="sv">NET</th><th data-sort="str">Status</th>
</tr></thead>
<tbody>
{rows_html}</tbody>
</table></div>"""


def md_to_html(md_path: str, changes: dict | None = None, stats_html: str = "", bubble_tab_html: str = "", conf_tab_html: str = "", autobid_tab_html: str = "", matrix_tab_html: str = "") -> str:
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
        #tab-bubble:checked ~ .tab-bar label[for="tab-bubble"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-conf:checked ~ .tab-bar label[for="tab-conf"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-autobid:checked ~ .tab-bar label[for="tab-autobid"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-matrix:checked ~ .tab-bar label[for="tab-matrix"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-bracket:checked ~ #panel-bracket {{ display: block; }}
        #tab-stats:checked ~ #panel-stats {{ display: block; }}
        #tab-bubble:checked ~ #panel-bubble {{ display: block; }}
        #tab-conf:checked ~ #panel-conf {{ display: block; }}
        #tab-autobid:checked ~ #panel-autobid {{ display: block; }}
        #tab-matrix:checked ~ #panel-matrix {{ display: block; }}

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

        /* Bubble Watch tab */
        .bubble-tab-section {{
            margin-bottom: 2rem;
        }}
        .bubble-tab-header {{
            padding: 0.6rem 1rem;
            border-radius: 6px 6px 0 0;
            margin-bottom: 0;
        }}
        .bubble-tab-header.bubble-in {{
            background: var(--green-bg);
            border-left: 3px solid var(--green);
        }}
        .bubble-tab-header.bubble-out {{
            background: var(--red-bg);
            border-left: 3px solid var(--red);
        }}
        .bubble-tab-header.bubble-far {{
            background: var(--surface);
            border-left: 3px solid var(--text-muted);
        }}
        .bubble-tab-label {{
            font-weight: 700;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .bubble-in .bubble-tab-label {{ color: var(--green); }}
        .bubble-out .bubble-tab-label {{ color: var(--red); }}
        .bubble-far .bubble-tab-label {{ color: var(--text-muted); }}
        .bubble-tab-table {{
            border-radius: 0 0 6px 6px;
            overflow: hidden;
        }}
        .prob-high {{ color: var(--green); font-weight: 600; }}
        .prob-mid {{ color: var(--accent); }}
        .prob-low {{ color: var(--red); font-weight: 600; }}
        .status-lock {{ color: var(--green); font-weight: 600; }}
        .status-likely {{ color: var(--accent); }}
        .status-needs {{ color: var(--red); font-weight: 600; }}

        /* Bracket Matrix */
        .diff-pos {{ color: var(--green); font-weight: 600; }}
        .diff-neg {{ color: var(--red); font-weight: 600; }}
        .diff-zero {{ color: var(--text-muted); }}
        .status-only-ours {{ color: var(--accent); }}
        .status-only-cbs {{ color: var(--text-muted); font-style: italic; }}

        @media (max-width: 600px) {{
            h1 {{ font-size: 1.5rem; }}
            .first-four-list {{ grid-template-columns: repeat(2, 1fr); }}
            .brackets-grid {{ grid-template-columns: 1fr; }}
            .bracket-region pre {{ font-size: 0.55rem; }}
            .container {{ padding: 1rem 0.5rem; }}
            .tab-bar {{
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: none;
            }}
            .tab-bar::-webkit-scrollbar {{ display: none; }}
            .tab-bar label {{
                padding: 0.6rem 0.75rem;
                font-size: 0.8rem;
                white-space: nowrap;
                flex-shrink: 0;
            }}
            .stats-table {{ font-size: 0.75rem; }}
            .stats-table th {{ font-size: 0.68rem; padding: 0.4rem; }}
            .stats-table td {{ padding: 0.35rem 0.4rem; }}
            .seed-table td {{ padding: 0.4rem 0.5rem; font-size: 0.85rem; }}
            .bubble-row {{ flex-direction: column; gap: 0.4rem; }}
            .bubble-label {{ min-width: auto; }}
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
        <input type="radio" name="tabs" id="tab-bubble" class="tab-radio">
        <input type="radio" name="tabs" id="tab-conf" class="tab-radio">
        <input type="radio" name="tabs" id="tab-autobid" class="tab-radio">
        <input type="radio" name="tabs" id="tab-matrix" class="tab-radio">

        <div class="tab-bar">
            <label for="tab-bracket">Bracket</label>
            <label for="tab-stats">Team Stats</label>
            <label for="tab-bubble">Bubble Watch</label>
            <label for="tab-conf">Conferences</label>
            <label for="tab-autobid">Auto Bids</label>
            <label for="tab-matrix">Bracket Matrix</label>
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

        <div id="panel-bubble" class="tab-panel">
            <h2>Bubble Watch</h2>
            {bubble_tab_html if bubble_tab_html else '<p style="color: var(--text-muted);">Bubble data not available. Run the predict command to generate.</p>'}
        </div>

        <div id="panel-conf" class="tab-panel">
            <h2>Conference Rankings</h2>
            {conf_tab_html if conf_tab_html else '<p style="color: var(--text-muted);">Conference data not available. Run the predict command to generate.</p>'}
        </div>

        <div id="panel-autobid" class="tab-panel">
            <h2>Auto Bids</h2>
            {autobid_tab_html if autobid_tab_html else '<p style="color: var(--text-muted);">Auto-bid data not available. Run the predict command to generate.</p>'}
        </div>

        <div id="panel-matrix" class="tab-panel">
            <h2>Bracket Matrix</h2>
            {matrix_tab_html if matrix_tab_html else '<p style="color: var(--text-muted);">Bracket Matrix data not available. Run scrape and predict to generate.</p>'}
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
    (function(){{
        var table=document.getElementById('conf-table');
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
    (function(){{
        var table=document.getElementById('autobid-table');
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
    (function(){{
        var table=document.getElementById('matrix-table');
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


def build(changes: dict | None = None, stats_df=None, bubble: dict | None = None):
    """Build the site from the latest predictions.

    Args:
        changes: dict mapping team_name -> {"direction": "up"|"down", "prev_seed": N, "new_seed": N}
        stats_df: DataFrame of all teams with stats columns.
        bubble: dict with "last_4_in", "first_4_out", "next_4_out" team name lists.
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

    # Persist / load bubble data
    bubble_path = os.path.join(PROCESSED_DIR, "bubble.json")
    if bubble is not None:
        with open(bubble_path, "w") as f:
            json.dump(bubble, f)
    elif os.path.exists(bubble_path):
        with open(bubble_path, "r") as f:
            bubble = json.load(f)

    stats_html = ""
    conf_tab_html = ""
    autobid_tab_html = ""
    if stats_df is not None:
        stats_html = _build_stats_table(stats_df)
        conf_tab_html = _build_conf_tab(stats_df)
        autobid_tab_html = _build_autobid_tab(stats_df)

    bubble_tab_html = _build_bubble_tab(bubble, stats_df)

    # Load bracketology data and build matrix tab
    bracketology = None
    brack_path = os.path.join(PROCESSED_DIR, "bracketology.json")
    if os.path.exists(brack_path):
        with open(brack_path, "r") as f:
            bracketology = json.load(f)

    # Parse seed rows from the markdown for the matrix tab
    with open(md_path, "r") as f:
        md_text = f.read()
    seed_rows_for_matrix = re.findall(r"^\| (\d+) \| (.+?) \|$", md_text, re.MULTILINE)

    matrix_tab_html = _build_matrix_tab(seed_rows_for_matrix, bracketology, stats_df)

    os.makedirs(SITE_DIR, exist_ok=True)

    html = md_to_html(md_path, changes=changes, stats_html=stats_html, bubble_tab_html=bubble_tab_html, conf_tab_html=conf_tab_html, autobid_tab_html=autobid_tab_html, matrix_tab_html=matrix_tab_html)

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
