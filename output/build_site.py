"""Convert predictions markdown to a styled HTML site for GitHub Pages."""

import json
import os
import re
import sys
from html import escape

import pandas as pd

from config import PROCESSED_DIR, PREDICTION_SEASON, PROJECT_ROOT, DATA_DIR, POWER_RANKING_WEIGHTS

SITE_DIR = os.path.join(PROJECT_ROOT, "_site")

# ESPN logo mapping: school_id -> espn_id
_ESPN_LOGOS = {}
_espn_logos_path = os.path.join(DATA_DIR, "espn_logos.json")
if os.path.exists(_espn_logos_path):
    with open(_espn_logos_path, "r") as _f:
        _ESPN_LOGOS = json.load(_f)

# ESPN conference logo mapping: conf_name -> slug
_ESPN_CONF_LOGOS = {}
_espn_conf_logos_path = os.path.join(DATA_DIR, "espn_conf_logos.json")
if os.path.exists(_espn_conf_logos_path):
    with open(_espn_conf_logos_path, "r") as _f:
        _ESPN_CONF_LOGOS = json.load(_f)


def _team_logo(school_id: str) -> str:
    """Return an <img> tag for a team's ESPN logo, or empty string if unknown."""
    espn_id = _ESPN_LOGOS.get(school_id)
    if not espn_id:
        return ""
    return f'<img class="team-logo" src="https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/{espn_id}.png&h=40&w=40" alt="" loading="lazy">'


def _conf_logo(conf_name: str) -> str:
    """Return an <img> tag for a conference's ESPN logo, or empty string if unknown."""
    slug = _ESPN_CONF_LOGOS.get(conf_name)
    if not slug:
        return ""
    return f'<img class="conf-logo" src="https://a.espncdn.com/i/teamlogos/ncaa_conf/sml/trans/{slug}.gif" alt="" loading="lazy">'


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

        logo = _team_logo(str(r.get('school_id', '')))
        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td class='stats-team'>{logo}{escape(str(r.get('team', '')))}</td>"
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
    }).sort_values("bids", ascending=False).reset_index()

    def _sv(w, l):
        return int(w) * 100 - int(l)

    rows = ""
    for i, r in agg.iterrows():
        wab_val = float(r["wab"])
        wab_cls = "wab-pos" if wab_val > 0 else "wab-neg" if wab_val < 0 else ""
        wab_str = f"{wab_val:+.1f}"

        clogo = _conf_logo(str(r['conference']))
        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td class='stats-team'>{clogo}{escape(str(r['conference']))}</td>"
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


def _build_standings_tab(stats_df) -> str:
    """Generate HTML for conference standings with collapsible groups."""
    if stats_df is None:
        return '<p style="color: var(--text-muted);">Standings data not available. Run the predict command to generate.</p>'

    df = stats_df[stats_df["net_ranking"] > 0].copy()

    # Count projected bids per conference to determine sort order and expand state
    conf_bids = df.groupby("conference")["selection_prob"].apply(lambda s: (s > 0.5).sum())

    # Group by conference, sort conferences by bids descending
    conferences = sorted(df["conference"].unique(), key=lambda c: -conf_bids.get(c, 0))

    html = ""
    for conf in conferences:
        grp = df[df["conference"] == conf].copy()
        bids = int(conf_bids.get(conf, 0))

        # Rank by conf_win_pct desc, tiebreak by net_ranking asc
        grp = grp.sort_values(["conf_win_pct", "net_ranking"], ascending=[False, True]).reset_index(drop=True)

        clogo = _conf_logo(conf)
        bids_label = f"{bids} bid{'s' if bids != 1 else ''}" if bids > 0 else "0 bids"
        open_attr = " open"

        rows = ""
        for i, r in grp.iterrows():
            # Conference record
            cw = r.get("conf_wins")
            cl = r.get("conf_losses")
            if pd.notna(cw) and pd.notna(cl):
                conf_rec = f"{int(cw)}-{int(cl)}"
            else:
                conf_rec = "\u2014"

            # Overall record
            w, l = r.get("wins"), r.get("losses")
            if pd.notna(w) and pd.notna(l):
                overall_rec = f"{int(w)}-{int(l)}"
            else:
                overall_rec = "\u2014"

            net = int(r["net_ranking"]) if pd.notna(r.get("net_ranking")) else "\u2014"

            # Tournament status indicator
            prob = r.get("selection_prob")
            if pd.notna(prob):
                prob_val = float(prob)
                if prob_val > 0.5:
                    status = '<span class="standing-status standing-in" title="Projected In">\u25cf</span>'
                elif prob_val > 0.15:
                    status = '<span class="standing-status standing-bubble" title="Bubble">\u25cf</span>'
                else:
                    status = ''
            else:
                status = ''

            logo = _team_logo(str(r.get('school_id', '')))
            team_name = str(r.get('team', ''))
            rows += (
                f"<tr data-team=\"{escape(team_name)}\">"
                f"<td>{i + 1}</td>"
                f"<td class='stats-team'>{logo}{escape(team_name)}</td>"
                f"<td>{conf_rec}</td>"
                f"<td>{overall_rec}</td>"
                f"<td>{net}</td>"
                f"<td>{status}</td>"
                f"<td class='title-pct'>\u2014</td>"
                f"</tr>\n"
            )

        table = f"""<table class="stats-table standings-table">
<thead><tr>
    <th>#</th><th>Team</th><th>Conf</th><th>Overall</th><th>NET</th><th></th><th>Title %</th>
</tr></thead>
<tbody>
{rows}</tbody>
</table>"""

        html += f"""<details class="standings-conf" data-conf="{escape(conf)}"{open_attr}>
<summary class="standings-summary">{clogo}<span class="standings-conf-name">{escape(conf)}</span><span class="standings-conf-meta">{bids_label}</span></summary>
{table}
</details>\n"""

    return html


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

        w_logo = _team_logo(str(w.get('school_id', '')))
        ru_logo = _team_logo(str(ru.get('school_id', ''))) if ru is not None else ""
        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td class='stats-team'>{escape(str(c['conf']))}</td>"
            f"<td class='stats-team'>{w_logo}{escape(str(w['team']))}</td>"
            f"<td data-sv='{conf_sv}'>{conf_rec}</td>"
            f"<td>{net}</td>"
            f"<td class='{prob_cls}'>{prob_str}</td>"
            f"<td class='{status_cls}'>{status}</td>"
            f"<td data-sv='{al_sv}'>{al}</td>"
            f"<td>{ru_logo}{ru_name}</td>"
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

            b_logo = _team_logo(str(r.get('school_id', '')))
            rows += (
                f"<tr>"
                f"<td class='stats-team'>{b_logo}{escape(str(r.get('team', '')))}</td>"
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
    html += _build_section("Last 4 Byes", bubble.get("last_4_byes", []), "bubble-bye")
    html += _build_section("Last 4 In", bubble.get("last_4_in", []), "bubble-in")
    html += _build_section("First 4 Out", bubble.get("first_4_out", []), "bubble-out")
    html += _build_section("Next 4 Out", bubble.get("next_4_out", []), "bubble-far")
    return html


def _build_matrix_tab(seed_rows: list[tuple[str, str]], bracketology: dict | None, stats_df) -> str:
    """Compare our seed predictions against bracketology sources.

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
            our_teams[clean] = {"seed": seed}
            our_order[clean] = pos
            pos += 1

    # Parse each bracketology source separately
    sources = {}  # source_label -> {team_name -> {"seed": N, "school_id": "..."}}
    for src, src_data in bracketology.get("sources", {}).items():
        src_teams = {}
        for entry in src_data.get("teams", []):
            src_teams[entry["team"]] = {
                "seed": entry["seed"],
                "school_id": entry.get("school_id", ""),
            }
        sources[src] = src_teams

    if not sources:
        return '<p style="color: var(--text-muted);">No bracketology data found.</p>'

    # Short labels for column headers — separate BM Avg from individual sources
    avg_label = "BM Avg" if "BM Avg" in sources else None
    source_labels = [s for s in sources if s != avg_label]

    # Build school_id lookup for our teams using stats_df
    our_id_to_name = {}
    if stats_df is not None:
        for name in our_teams:
            match = stats_df[stats_df["team"] == name]
            if not match.empty:
                sid = match.iloc[0].get("school_id", "")
                if sid:
                    our_id_to_name[sid] = name

    # Match each source's teams to our teams (by name or school_id)
    matched = {}  # source_label -> {our_team_name -> src_data}
    unmatched = {}  # source_label -> {src_team_name -> src_data}
    for src_label, src_teams in sources.items():
        matched[src_label] = {}
        unmatched[src_label] = dict(src_teams)

        for src_name, src_data in src_teams.items():
            if src_name in our_teams:
                matched[src_label][src_name] = src_data
                unmatched[src_label].pop(src_name, None)
                continue

            sid = src_data.get("school_id", "")
            if sid and sid in our_id_to_name:
                our_name = our_id_to_name[sid]
                matched[src_label][our_name] = src_data
                unmatched[src_label].pop(src_name, None)

    # Collect all teams only in external sources (not in ours)
    only_external = {}  # team_name -> {source_label -> seed}
    for src_label, src_unmatched in unmatched.items():
        for src_name, src_data in src_unmatched.items():
            if src_name not in only_external:
                only_external[src_name] = {"net": ""}
            only_external[src_name][src_label] = src_data["seed"]

    # Get NET for external-only teams
    if stats_df is not None:
        for name in only_external:
            m = stats_df[stats_df["team"] == name]
            if not m.empty:
                net_val = m.iloc[0].get("net_ranking")
                if pd.notna(net_val):
                    only_external[name]["net"] = str(int(net_val))

    # Build rows for our teams
    matrix_rows = []
    for name, our_data in our_teams.items():
        our_seed = our_data["seed"]
        net = ""
        sid = ""
        if stats_df is not None:
            m = stats_df[stats_df["team"] == name]
            if not m.empty:
                net_val = m.iloc[0].get("net_ranking")
                if pd.notna(net_val):
                    net = str(int(net_val))
                sid = str(m.iloc[0].get("school_id", ""))

        row = {"team": name, "school_id": sid, "our_seed": our_seed, "net": net, "src_seeds": {}}
        for src_label in source_labels:
            if name in matched[src_label]:
                row["src_seeds"][src_label] = matched[src_label][name]["seed"]
        if avg_label and name in matched.get(avg_label, {}):
            row["src_seeds"][avg_label] = matched[avg_label][name]["seed"]

        matrix_rows.append(row)

    # Add external-only teams
    for name, ext_data in only_external.items():
        ext_sid = ""
        if stats_df is not None:
            m = stats_df[stats_df["team"] == name]
            if not m.empty:
                ext_sid = str(m.iloc[0].get("school_id", ""))
        row = {"team": name, "school_id": ext_sid, "our_seed": None, "net": ext_data.get("net", ""), "src_seeds": {}}
        for src_label in source_labels:
            if src_label in ext_data and src_label != "net":
                row["src_seeds"][src_label] = ext_data[src_label]
        if avg_label and avg_label in ext_data and avg_label != "net":
            row["src_seeds"][avg_label] = ext_data[avg_label]
        matrix_rows.append(row)

    # Sort by S-curve order (teams not in our bracket go to the bottom)
    matrix_rows.sort(key=lambda r: our_order.get(r["team"], 9999))

    # Build HTML
    rows_html = ""
    for r in matrix_rows:
        our_seed = r["our_seed"]
        our_seed_str = str(our_seed) if our_seed is not None else "\u2014"
        our_seed_sv = our_seed if our_seed is not None else 99
        net_sv = int(r["net"]) if r["net"] else 9999

        src_cells = ""
        for src_label in source_labels:
            src_seed = r["src_seeds"].get(src_label)
            if src_seed is not None:
                src_sv = src_seed
                src_str = str(src_seed)
                if our_seed is not None:
                    diff = our_seed - src_seed
                    if diff < 0:
                        cls = "diff-pos"
                    elif diff > 0:
                        cls = "diff-neg"
                    else:
                        cls = "diff-zero"
                else:
                    cls = "status-only-cbs"
            else:
                src_sv = 99
                src_str = "\u2014"
                cls = ""
            src_cells += f"<td class='{cls}' data-sv='{src_sv}'>{src_str}</td>"

        # Status: compare our seed to BM Avg (0.5 threshold)
        avg_seed_val = r["src_seeds"].get(avg_label) if avg_label else None
        if our_seed is None:
            status = "Not in Ours"
            status_cls = "status-only-cbs"
        elif avg_seed_val is None:
            status = "Only Ours"
            status_cls = "status-only-ours"
        else:
            diff = our_seed - avg_seed_val
            if diff <= -0.5:
                status = "Higher"
                status_cls = "diff-pos"
            elif diff >= 0.5:
                status = "Lower"
                status_cls = "diff-neg"
            else:
                status = "Agreement"
                status_cls = "diff-zero"

        # BM Avg cell (highlighted)
        avg_cell = ""
        if avg_label:
            avg_seed = r["src_seeds"].get(avg_label)
            if avg_seed is not None:
                avg_sv = avg_seed
                avg_str = f"{avg_seed:.2f}"
                if our_seed is not None:
                    diff = our_seed - avg_seed
                    avg_cls = "diff-pos" if diff < 0 else "diff-neg" if diff > 0 else "diff-zero"
                else:
                    avg_cls = "status-only-cbs"
            else:
                avg_sv = 99
                avg_str = "\u2014"
                avg_cls = ""
            avg_cell = f"<td class='col-avg {avg_cls}' data-sv='{avg_sv}'>{avg_str}</td>"

        m_logo = _team_logo(r.get('school_id', ''))
        rows_html += (
            f"<tr>"
            f"<td class='stats-team'>{m_logo}{escape(r['team'])}</td>"
            f"<td data-sv='{our_seed_sv}'>{our_seed_str}</td>"
            f"<td class='{status_cls}'>{status}</td>"
            f"{avg_cell}"
            f"{src_cells}"
            f"<td data-sv='{net_sv}'>{r['net'] or chr(0x2014)}</td>"
            f"</tr>\n"
        )

    # Build header with dynamic source columns
    avg_header = f"<th class='col-avg' data-sort='sv'>Avg</th>" if avg_label else ""
    src_headers = ""
    for src_label in source_labels:
        short = src_label.split(" - ")[0].strip()
        src_headers += f"<th data-sort='sv'>{escape(short)}</th>"

    return f"""<div class="stats-scroll"><table class="stats-table" id="matrix-table">
<thead><tr>
    <th data-sort="str">Team</th><th data-sort="sv">Ours</th>
    <th data-sort="str">Status</th>{avg_header}{src_headers}
    <th data-sort="sv">NET</th>
</tr></thead>
<tbody>
{rows_html}</tbody>
</table></div>"""


def _build_scores_tab(stats_df: pd.DataFrame) -> str:
    """Build an interactive matchup predictor with two searchable team dropdowns.

    Embeds team stats as JSON; spread/win% computed client-side.
    Also embeds extended team profile data for the modal.
    """
    if stats_df is None:
        return '<p style="color: var(--text-muted);">Scores data not available. Run the predict command to generate.</p>'

    has_kenpom = "hca_score" in stats_df.columns and stats_df["hca_score"].sum() > 0

    # Load AP poll ranks for current week
    ap_rank_lookup = {}
    _AP_ALIASES = {
        "UConn": "Connecticut",
        "BYU": "Brigham Young",
        "St. John's": "St. John's (NY)",
        "UCF": "Central Florida",
        "SMU": "Southern Methodist",
        "LSU": "Louisiana State",
        "USC": "Southern California",
        "UNLV": "Nevada-Las Vegas",
        "Ole Miss": "Mississippi",
        "Pitt": "Pittsburgh",
        "UNC": "North Carolina",
    }
    poll_path = os.path.join(PROCESSED_DIR, "ap_poll.json")
    if os.path.exists(poll_path):
        with open(poll_path, "r") as f:
            poll = json.load(f)
        cw = str(poll.get("current_week", ""))
        week_data = poll.get("weeks", {}).get(cw, {})
        for entry in week_data.get("ranks", []):
            tn = entry.get("team_name", "")
            if tn:
                ap_rank_lookup[_AP_ALIASES.get(tn, tn)] = int(entry["rank"])

    # Build JSON blob of all teams with valid data
    df = stats_df[stats_df["net_ranking"] > 0].copy()
    teams_json = []
    for _, r in df.iterrows():
        adj_oe = pd.to_numeric(r.get("adj_oe"), errors="coerce")
        adj_de = pd.to_numeric(r.get("adj_de"), errors="coerce")
        barthag = pd.to_numeric(r.get("barthag"), errors="coerce")
        pace = pd.to_numeric(r.get("pace"), errors="coerce")
        net = pd.to_numeric(r.get("net_ranking"), errors="coerce")
        wins = pd.to_numeric(r.get("wins"), errors="coerce")
        losses = pd.to_numeric(r.get("losses"), errors="coerce")
        road_wins = pd.to_numeric(r.get("road_wins"), errors="coerce")
        road_losses = pd.to_numeric(r.get("road_losses"), errors="coerce")
        if pd.isna(adj_oe) or pd.isna(adj_de) or pd.isna(barthag):
            continue

        # Use KenPom HCA if available, else fallback to legacy formula
        if has_kenpom:
            hca_val = float(r.get("hca_score", 0) or 0)
        else:
            hca_val = 0.0
            if pd.notna(wins) and pd.notna(road_wins) and pd.notna(losses) and pd.notna(road_losses):
                hw = wins - road_wins
                hl = losses - road_losses
                hg = hw + hl
                rg = road_wins + road_losses
                if hg >= 3 and rg >= 3:
                    home_wp = hw / hg
                    road_wp = road_wins / rg
                    dominance = home_wp - road_wp
                    quality = max(0.0, 1 - float(net) / 200) if pd.notna(net) else 0.0
                    hca_val = dominance * 0.65 + quality * 0.35

        espn_id = _ESPN_LOGOS.get(str(r.get("school_id", "")), "")

        # Home/road record for profile — prefer ESPN data, fallback to WarrenNolan
        _ehw = r.get("espn_home_wins", 0)
        _ehl = r.get("espn_home_losses", 0)
        _erw = r.get("espn_road_wins", 0)
        _erl = r.get("espn_road_losses", 0)
        espn_hw = int(_ehw) if pd.notna(_ehw) else 0
        espn_hl = int(_ehl) if pd.notna(_ehl) else 0
        espn_rw = int(_erw) if pd.notna(_erw) else 0
        espn_rl = int(_erl) if pd.notna(_erl) else 0
        if espn_hw + espn_hl + espn_rw + espn_rl > 0:
            hw, hl, rw, rl = espn_hw, espn_hl, espn_rw, espn_rl
        else:
            hw = int(wins - road_wins) if pd.notna(wins) and pd.notna(road_wins) else 0
            hl = int(losses - road_losses) if pd.notna(losses) and pd.notna(road_losses) else 0
            rw = int(road_wins) if pd.notna(road_wins) else 0
            rl = int(road_losses) if pd.notna(road_losses) else 0

        team_data = {
            "name": str(r.get("team", "")),
            "conf": str(r.get("conference", "")),
            "espn": espn_id,
            "oe": round(float(adj_oe), 1),
            "de": round(float(adj_de), 1),
            "bar": round(float(barthag), 4),
            "pace": round(float(pace), 1) if pd.notna(pace) else None,
            "net": int(net) if pd.notna(net) else 999,
            "rec": f"{int(wins)}-{int(losses)}" if pd.notna(wins) and pd.notna(losses) else "",
            "hca": round(hca_val, 3),
            # Extended profile fields
            "homeRec": f"{hw}-{hl}",
            "roadRec": f"{rw}-{rl}",
            "srs": round(float(r.get("srs", 0) or 0), 2),
            "sor": int(r.get("sor", 0) or 0),
            "kpi": int(r.get("kpi", 0) or 0),
            "bpi": int(r.get("bpi", 0) or 0),
            "pom": int(r.get("pom", 0) or 0),
            "wab": round(float(r.get("wab", 0) or 0), 1),
            "confW": int(r.get("conf_wins", 0) or 0),
            "confL": int(r.get("conf_losses", 0) or 0),
            "apRank": ap_rank_lookup.get(str(r.get("team", "")), 0),
        }

        # Quadrant records
        for q in ["q1", "q2", "q3", "q4"]:
            qw = int(r.get(f"{q}_wins", 0) or 0)
            ql = int(r.get(f"{q}_losses", 0) or 0)
            team_data[q] = f"{qw}-{ql}"

        # HCA breakdown (KenPom)
        if has_kenpom:
            team_data["hcaPts"] = round(float(r.get("hca_points", 3.5) or 3.5), 1)
            team_data["hcaFoul"] = round(float(r.get("foul_advantage", 0) or 0), 2)
            team_data["hcaScoring"] = round(float(r.get("scoring_advantage", 0) or 0), 1)
            team_data["hcaTO"] = round(float(r.get("turnover_advantage", 0) or 0), 2)
            team_data["hcaBlk"] = round(float(r.get("block_advantage", 0) or 0), 2)
            team_data["homePtsM"] = round(float(r.get("home_pts_margin", 0) or 0), 1)
            team_data["roadPtsM"] = round(float(r.get("road_pts_margin", 0) or 0), 1)
            team_data["homeFoulM"] = round(float(r.get("home_foul_margin", 0) or 0), 1)
            team_data["roadFoulM"] = round(float(r.get("road_foul_margin", 0) or 0), 1)

        teams_json.append(team_data)

    teams_json.sort(key=lambda t: t["net"])
    import json as _json
    blob = _json.dumps(teams_json, separators=(",", ":"))

    # Save teams data for the Twitter bot
    teams_data_path = os.path.join(PROCESSED_DIR, "teams_data.json")
    with open(teams_data_path, "w") as f:
        _json.dump(teams_json, f, separators=(",", ":"))

    return f'<div id="scores-app"></div><script>window.__SCORES_TEAMS__={blob};</script>'


def _build_ap_poll_tab() -> str:
    """Build the AP Top 25 poll tab with client-side week selector."""
    poll_path = os.path.join(PROCESSED_DIR, "ap_poll.json")
    if not os.path.exists(poll_path):
        return ""

    with open(poll_path, "r") as f:
        poll = json.load(f)

    weeks = poll.get("weeks", {})
    if not weeks:
        # Backwards compat: old single-week format
        ranks = poll.get("ranks", [])
        if not ranks:
            return ""
        weeks = {"1": {"week_label": "", "updated": poll.get("updated", ""), "ranks": ranks, "others": poll.get("others", [])}}
        poll = {"season": PREDICTION_SEASON, "current_week": 1, "weeks": weeks}

    import json as _json
    blob = _json.dumps(poll, separators=(",", ":"))
    logos_blob = _json.dumps(_ESPN_LOGOS, separators=(",", ":"))

    return (
        f'<div id="ap-poll-app"></div>'
        f'<script>window.__AP_POLL_DATA__={blob};'
        f'window.__ESPN_LOGOS__={logos_blob};</script>'
    )


def _build_summary_tab(stats_df: pd.DataFrame, changes: dict, bubble: dict | None, seed_rows: list | None = None) -> str:
    """Build the Summary tab container. Embeds daily changes, bubble, and seed data for client-side JS."""
    if stats_df is None:
        return '<p style="color: var(--text-muted);">Summary data not available. Run the predict command to generate.</p>'

    import json as _json
    changes_blob = _json.dumps(changes or {}, separators=(",", ":"))
    bubble_blob = _json.dumps(bubble or {}, separators=(",", ":"))

    # Build seed lookup: team_name -> seed_number
    seed_lookup = {}
    if seed_rows:
        for seed_num, teams_str in seed_rows:
            for team in teams_str.split(", "):
                team = team.strip().rstrip("*")
                if team:
                    seed_lookup[team] = int(seed_num)
    seed_blob = _json.dumps(seed_lookup, separators=(",", ":"))

    return (
        '<div id="summary-app"></div>'
        f'<script>window.__DAILY_CHANGES__={changes_blob};'
        f'window.__BUBBLE_DATA__={bubble_blob};'
        f'window.__SEED_LIST__={seed_blob};</script>'
    )


def _build_schedule_tab(stats_df: pd.DataFrame) -> str:
    """Build the Schedule tab container. All logic is client-side JS."""
    if stats_df is None:
        return '<p style="color: var(--text-muted);">Schedule data not available. Run the predict command to generate.</p>'
    return '<div id="schedule-app"></div>'


def _build_homecourt_tab(stats_df: pd.DataFrame) -> str:
    """Rank teams by KenPom-style home court advantage score.

    Uses ESPN box score data (fouls, turnovers, blocks, scoring margin)
    when available, falls back to the old win% formula otherwise.
    """
    if stats_df is None:
        return '<p style="color: var(--text-muted);">Home Court data not available. Run the predict command to generate.</p>'

    has_kenpom = "hca_score" in stats_df.columns and stats_df["hca_score"].sum() > 0

    if has_kenpom:
        return _build_homecourt_tab_kenpom(stats_df)
    else:
        return _build_homecourt_tab_legacy(stats_df)


def _build_homecourt_tab_kenpom(stats_df: pd.DataFrame) -> str:
    """KenPom-style HCA table with component columns."""
    df = stats_df[stats_df["net_ranking"] > 0].copy()
    df = df[df["hca_score"] > 0].copy()

    if df.empty:
        return _build_homecourt_tab_legacy(stats_df)

    df = df.sort_values("hca_score", ascending=False).reset_index(drop=True)

    def _color_val(val, invert=False):
        """Color a component value green (good) or red (bad)."""
        v = float(val)
        if invert:
            v = -v
        if v > 0.5:
            return "color: var(--green);"
        elif v < -0.5:
            return "color: var(--red);"
        return ""

    rows = ""
    for i, r in df.iterrows():
        score = float(r.get("hca_score", 0))
        hca_pts = float(r.get("hca_points", 3.5))
        foul = float(r.get("foul_advantage", 0))
        scoring = float(r.get("scoring_advantage", 0))
        to_adv = float(r.get("turnover_advantage", 0))
        blk = float(r.get("block_advantage", 0))

        # Color the HCA score percentile
        hue = int(score * 120)
        score_style = f"color: hsl({hue}, 70%, 50%);"

        # Color HCA points
        pts_diff = hca_pts - 3.5
        if pts_diff > 0.5:
            pts_style = "color: var(--green);"
        elif pts_diff < -0.5:
            pts_style = "color: var(--red);"
        else:
            pts_style = ""

        # Use ESPN-derived records (available for all teams) with WarrenNolan fallback
        _ehw = r.get("espn_home_wins", 0)
        _ehl = r.get("espn_home_losses", 0)
        _erw = r.get("espn_road_wins", 0)
        _erl = r.get("espn_road_losses", 0)
        hw = int(_ehw) if pd.notna(_ehw) else 0
        hl = int(_ehl) if pd.notna(_ehl) else 0
        rw = int(_erw) if pd.notna(_erw) else 0
        rl = int(_erl) if pd.notna(_erl) else 0
        if hw == 0 and hl == 0 and rw == 0 and rl == 0:
            # Fallback to WarrenNolan-derived records
            _w = r.get("wins", 0)
            _l = r.get("losses", 0)
            _rw = r.get("road_wins", 0)
            _rl = r.get("road_losses", 0)
            hw = int(_w) - int(_rw) if pd.notna(_w) and pd.notna(_rw) else 0
            hl = int(_l) - int(_rl) if pd.notna(_l) and pd.notna(_rl) else 0
            rw = int(_rw) if pd.notna(_rw) else 0
            rl = int(_rl) if pd.notna(_rl) else 0

        logo = _team_logo(str(r.get("school_id", "")))
        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td class='stats-team'>{logo}{escape(str(r.get('team', '')))}</td>"
            f"<td>{escape(str(r.get('conference', '')))}</td>"
            f"<td>{hw}-{hl}</td>"
            f"<td>{rw}-{rl}</td>"
            f"<td style='{score_style} font-weight:600;'>{score:.2f}</td>"
            f"<td style='{pts_style} font-weight:600;'>{hca_pts:+.1f}</td>"
            f"<td style='{_color_val(foul)}'>{foul:+.2f}</td>"
            f"<td style='{_color_val(scoring)}'>{scoring:+.1f}</td>"
            f"<td style='{_color_val(to_adv)}'>{to_adv:+.2f}</td>"
            f"<td style='{_color_val(blk)}'>{blk:+.2f}</td>"
            f"<td>{int(r['net_ranking'])}</td>"
            f"</tr>\n"
        )

    return f"""<div class="stats-scroll"><table class="stats-table" id="homecourt-table">
<thead><tr>
    <th data-sort="num">#</th><th data-sort="str">Team</th><th data-sort="str">Conf</th>
    <th data-sort="str">Home</th><th data-sort="str">Road</th>
    <th data-sort="num">HCA Score</th><th data-sort="num">HCA Pts</th>
    <th data-sort="num">Fouls</th><th data-sort="num">Scoring</th>
    <th data-sort="num">TOs</th><th data-sort="num">Blocks</th><th data-sort="num">NET</th>
</tr></thead>
<tbody>
{rows}</tbody>
</table></div>"""


def _build_homecourt_tab_legacy(stats_df: pd.DataFrame) -> str:
    """Fallback HCA table using win% formula (no ESPN data)."""
    df = stats_df[stats_df["net_ranking"] > 0].copy()

    df["home_wins"] = df["wins"] - df["road_wins"]
    df["home_losses"] = df["losses"] - df["road_losses"]
    df["home_games"] = df["home_wins"] + df["home_losses"]
    df["road_games"] = df["road_wins"] + df["road_losses"]

    df = df[(df["home_games"] >= 3) & (df["road_games"] >= 3)].copy()

    df["home_wp"] = df["home_wins"] / df["home_games"]
    df["road_wp"] = df["road_wins"] / df["road_games"]
    df["dominance"] = df["home_wp"] - df["road_wp"]
    df["quality"] = (1 - df["net_ranking"] / 200).clip(lower=0)
    df["hca_score_legacy"] = df["dominance"] * 0.65 + df["quality"] * 0.35

    df = df.sort_values("hca_score_legacy", ascending=False).reset_index(drop=True)

    score_min = df["hca_score_legacy"].min()
    score_max = df["hca_score_legacy"].max()

    rows = ""
    for i, r in df.iterrows():
        score = float(r["hca_score_legacy"])
        if score_max != score_min:
            t = (score - score_min) / (score_max - score_min)
        else:
            t = 0.5
        hue = int(t * 120)
        score_style = f"color: hsl({hue}, 70%, 50%);"

        hw, hl = int(r["home_wins"]), int(r["home_losses"])
        rw, rl = int(r["road_wins"]), int(r["road_losses"])

        logo = _team_logo(str(r.get("school_id", "")))
        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td class='stats-team'>{logo}{escape(str(r.get('team', '')))}</td>"
            f"<td>{escape(str(r.get('conference', '')))}</td>"
            f"<td>{hw}-{hl}</td>"
            f"<td>{rw}-{rl}</td>"
            f"<td>{r['home_wp']:.3f}</td>"
            f"<td>{r['road_wp']:.3f}</td>"
            f"<td>{r['dominance']:.3f}</td>"
            f"<td style='{score_style} font-weight:600;'>{score:.3f}</td>"
            f"<td>{int(r['net_ranking'])}</td>"
            f"</tr>\n"
        )

    return f"""<div class="stats-scroll"><table class="stats-table" id="homecourt-table">
<thead><tr>
    <th data-sort="num">#</th><th data-sort="str">Team</th><th data-sort="str">Conf</th>
    <th data-sort="str">Home</th><th data-sort="str">Road</th>
    <th data-sort="num">Home W%</th><th data-sort="num">Road W%</th>
    <th data-sort="num">Dominance</th><th data-sort="num">HCA Score</th><th data-sort="num">NET</th>
</tr></thead>
<tbody>
{rows}</tbody>
</table></div>"""


def _build_power_rankings_tab(stats_df: pd.DataFrame) -> str:
    """Compute a weighted composite score and build an HTML rankings table."""
    if stats_df is None:
        return '<p style="color: var(--text-muted);">Power Rankings data not available. Run the predict command to generate.</p>'

    weights = POWER_RANKING_WEIGHTS
    cols = list(weights.keys())

    # Only keep teams with a valid NET ranking and at least some data
    df = stats_df[stats_df["net_ranking"] > 0].copy()

    # Z-score normalize each column; handle missing values
    for col in cols:
        if col not in df.columns:
            df[col] = float("nan")
        vals = pd.to_numeric(df[col], errors="coerce")
        mean, std = vals.mean(), vals.std()
        if pd.notna(mean) and pd.notna(std) and std > 0:
            z = (vals - mean) / std
        else:
            z = pd.Series(0.0, index=df.index)
        # For negative weights (lower raw = better), flip sign so positive z = good
        if weights[col] < 0:
            z = -z
        df[f"_z_{col}"] = z

    # Composite score = sum(|weight| * z-score)
    df["_composite"] = 0.0
    for col in cols:
        w = abs(weights[col])
        df["_composite"] += w * df[f"_z_{col}"].fillna(0.0)

    df = df.sort_values("_composite", ascending=False).reset_index(drop=True)

    def _int(row, col):
        v = row.get(col)
        return str(int(v)) if pd.notna(v) else "\u2014"

    def _rec(row, w_col, l_col):
        w, l = row.get(w_col), row.get(l_col)
        if pd.notna(w) and pd.notna(l):
            return f"{int(w)}-{int(l)}"
        return "\u2014"

    rows = ""
    for i, r in df.iterrows():
        comp = float(r["_composite"])
        # Color-code composite: green (high) → red (low) via hue interpolation
        # Normalize composite within this set for coloring
        comp_min = df["_composite"].min()
        comp_max = df["_composite"].max()
        if comp_max != comp_min:
            t = (comp - comp_min) / (comp_max - comp_min)
        else:
            t = 0.5
        # HSL: 0=red, 120=green
        hue = int(t * 120)
        comp_style = f"color: hsl({hue}, 70%, 50%);"

        wab = r.get("wab")
        if pd.notna(wab):
            wab_val = float(wab)
            wab_cls = "wab-pos" if wab_val > 0 else "wab-neg" if wab_val < 0 else ""
            wab_str = f"{wab_val:+.1f}"
        else:
            wab_cls = ""
            wab_str = "\u2014"

        def _float(col, fmt=".1f"):
            v = r.get(col)
            return f"{float(v):{fmt}}" if pd.notna(v) else "\u2014"

        # Bad losses (Q3 + Q4)
        q3l = r.get("q3_losses")
        q4l = r.get("q4_losses")
        if pd.notna(q3l) and pd.notna(q4l):
            bad_l = int(q3l) + int(q4l)
            bad_cls = "wab-neg" if bad_l > 0 else ""
            bad_str = str(bad_l)
        else:
            bad_cls = ""
            bad_str = "\u2014"

        logo = _team_logo(str(r.get("school_id", "")))
        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td class='stats-team'>{logo}{escape(str(r.get('team', '')))}</td>"
            f"<td>{escape(str(r.get('conference', '')))}</td>"
            f"<td>{_rec(r, 'wins', 'losses')}</td>"
            f"<td style='{comp_style} font-weight:600;'>{comp:.3f}</td>"
            f"<td>{_int(r, 'net_ranking')}</td>"
            f"<td>{_int(r, 'sor')}</td>"
            f"<td>{_int(r, 'pom')}</td>"
            f"<td>{_int(r, 'net_sos')}</td>"
            f"<td class='{wab_cls}'>{wab_str}</td>"
            f"<td>{_int(r, 'q1_wins')}</td>"
            f"<td class='{bad_cls}'>{bad_str}</td>"
            f"<td>{_int(r, 'road_wins')}</td>"
            f"<td>{_float('adj_oe')}</td>"
            f"<td>{_float('adj_de')}</td>"
            f"</tr>\n"
        )

    return f"""<div class="stats-scroll"><table class="stats-table" id="ranking-table">
<thead><tr>
    <th data-sort="num">#</th><th data-sort="str">Team</th><th data-sort="str">Conf</th><th data-sort="str">Record</th>
    <th data-sort="num">Score</th><th data-sort="num">NET</th><th data-sort="num">SOR</th>
    <th data-sort="num">KenPom</th><th data-sort="num">SOS</th><th data-sort="num">WAB</th>
    <th data-sort="num">Q1 W</th><th data-sort="num">Bad L</th><th data-sort="num">Road W</th>
    <th data-sort="num">AdjOE</th><th data-sort="num">AdjDE</th>
</tr></thead>
<tbody>
{rows}</tbody>
</table></div>"""


def md_to_html(md_path: str, changes: dict | None = None, stats_html: str = "", bubble_tab_html: str = "", conf_tab_html: str = "", standings_tab_html: str = "", autobid_tab_html: str = "", matrix_tab_html: str = "", ranking_tab_html: str = "", scores_tab_html: str = "", schedule_tab_html: str = "", homecourt_tab_html: str = "", summary_tab_html: str = "", appoll_tab_html: str = "", bubble: dict | None = None, stats_df=None, model_type: str = "rf") -> str:
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

    model_label = {"rf": "Random Forest", "xgb": "XGBoost", "ensemble": "Ensemble (RF + XGB)"}[model_type]

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

    # Build name -> school_id lookup for logos
    _name_to_sid = {}
    if stats_df is not None:
        for _, _r in stats_df.iterrows():
            _name_to_sid[str(_r.get("team", ""))] = str(_r.get("school_id", ""))

    # Build bubble HTML
    bubble_html = ""
    if last_4_in:
        def _bubble_row(label: str, teams_str: str, css_class: str) -> str:
            teams = [t.strip() for t in teams_str.split(", ")]
            items = "".join(
                f'<span class="bubble-team stats-team" data-team="{escape(t)}">{_team_logo(_name_to_sid.get(t, ""))}{escape(t)}</span>'
                for t in teams
            )
            return f'<div class="bubble-row {css_class}"><div class="bubble-label">{label}</div><div class="bubble-teams">{items}</div></div>'

        bubble_html = '<div class="bubble-section"><h2>Bubble Watch</h2>'
        last_4_byes = bubble.get("last_4_byes", []) if bubble else []
        if last_4_byes:
            bubble_html += _bubble_row("Last 4 Byes", ", ".join(last_4_byes), "bubble-bye")
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
            logo = _team_logo(_name_to_sid.get(clean_name, ""))
            styled.append(f'<span class="stats-team" data-team="{escape(clean_name)}">{logo}{_style_team(clean_name, changes, is_play_in)}</span>')
        seed_table_rows += f"<tr><td class='seed-num'>{seed}</td><td>{', '.join(styled)}</td></tr>\n"

    first_four_html = ""
    for game in first_four:
        # Add logos to First Four: "(seed) Team1 vs Team2 [Region]"
        ff_match = re.match(r'\((\d+)\) (.+?) vs (.+?) \[(.+?)\]', game)
        if ff_match:
            ff_seed, ff_t1, ff_t2, ff_region = ff_match.groups()
            ff_logo1 = _team_logo(_name_to_sid.get(ff_t1.strip(), ""))
            ff_logo2 = _team_logo(_name_to_sid.get(ff_t2.strip(), ""))
            first_four_html += f'<li>({ff_seed}) <span class="stats-team" data-team="{escape(ff_t1)}">{ff_logo1}{escape(ff_t1)}</span> vs <span class="stats-team" data-team="{escape(ff_t2)}">{ff_logo2}{escape(ff_t2)}</span> [{ff_region}]</li>\n'
        else:
            first_four_html += f"<li>{escape(game)}</li>\n"

    brackets_html = ""
    for region, art in brackets:
        # Add team logos to bracket art lines
        def _bracket_logo_sub(m):
            seed = m.group(1)
            team_name = m.group(2)
            # Handle play-in "Team1/Team2" — use first team's logo
            first_team = team_name.split("/")[0]
            sid = _name_to_sid.get(first_team, "")
            espn_id = _ESPN_LOGOS.get(sid)
            if espn_id:
                logo = f'<img class="bracket-logo" src="https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/{espn_id}.png&h=40&w=40" alt="" loading="lazy">'
            else:
                logo = ""
            # Escape & in team names for HTML
            safe_name = team_name.replace("&", "&amp;")
            data_name = team_name.replace("&", "&amp;").replace('"', "&quot;")
            return f'<span class="stats-team" data-team="{data_name}">{logo}({seed}) {safe_name}</span> '
        processed_art = re.sub(
            r'\((\d+)\) (.+?) (?=─)',
            _bracket_logo_sub,
            art.rstrip(),
        )
        brackets_html += f"""
        <div class="bracket-region">
            <h3>{region} Region</h3>
            <pre>{processed_art}</pre>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <meta name="description" content="Machine learning NCAA Tournament bracket predictions updated daily. Seed projections, bubble watch, matchup predictor, and game-by-game analysis for the {PREDICTION_SEASON} season.">
    <meta name="author" content="68bracket">
    <meta property="og:title" content="68bracket — {title}">
    <meta property="og:description" content="ML-powered NCAA Tournament bracket predictions updated daily. Seed projections, bubble watch, matchup predictor, and more.">
    <meta property="og:type" content="website">
    <meta name="twitter:card" content="summary">
    <meta name="twitter:title" content="68bracket — {title}">
    <meta name="twitter:description" content="ML-powered NCAA Tournament bracket predictions updated daily.">
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#127936;</text></svg>">
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
            margin-bottom: 0.35rem;
        }}

        .tagline {{
            font-size: 0.88rem;
            color: var(--text-muted);
            opacity: 0.75;
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

        .bubble-bye {{
            background: rgba(59, 130, 246, 0.1);
            border-left: 3px solid #3b82f6;
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

        .bubble-bye .bubble-label {{ color: #3b82f6; }}
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
            padding-left: 22px;
        }}
        .bracket-logo {{
            width: 16px;
            height: 16px;
            vertical-align: middle;
            margin-left: -20px;
            margin-right: 4px;
        }}

        .footnote {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            line-height: 1.7;
        }}

        .footnote a {{
            color: var(--accent);
            text-decoration: none;
        }}

        .footnote-sources {{
            font-size: 0.72rem;
            opacity: 0.7;
            margin-top: 0.4rem;
        }}

        .footnote-disclaimer {{
            font-size: 0.7rem;
            opacity: 0.55;
            margin-top: 0.4rem;
        }}

        /* Tab navigation */
        .tab-radio {{ display: none; }}
        .tab-nav {{
            display: flex;
            align-items: stretch;
            border-bottom: 2px solid var(--border);
            margin-bottom: 2rem;
            position: sticky;
            top: 0;
            background: var(--bg);
            z-index: 10;
        }}
        .tab-bar {{
            display: flex;
            gap: 0;
        }}
        .tab-bar label {{
            padding: 0.75rem 0.6rem;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-muted);
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: color 0.15s, border-color 0.15s;
            white-space: nowrap;
        }}
        .tab-bar label:hover {{
            color: var(--text);
        }}
        .tab-more {{
            position: relative;
            flex-shrink: 0;
            align-self: stretch;
            display: flex;
            align-items: stretch;
        }}
        .tab-more-btn {{
            background: none;
            border: none;
            padding: 0.75rem 0.6rem;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-muted);
            font-family: inherit;
            white-space: nowrap;
        }}
        .tab-more-btn:hover {{
            color: var(--text);
        }}
        .tab-more-btn.more-active {{
            color: var(--accent);
            border-bottom: 2px solid var(--accent);
            margin-bottom: -2px;
        }}
        .tab-more-menu {{
            display: none;
            position: absolute;
            top: 100%;
            right: 0;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 20;
            min-width: 180px;
            padding: 0.25rem 0;
        }}
        .tab-more-menu.open {{
            display: block;
        }}
        .tab-more-menu label {{
            display: block;
            padding: 0.6rem 1rem;
            border-bottom: none;
            margin-bottom: 0;
            font-size: 0.85rem;
        }}
        .tab-more-menu label:hover {{
            background: var(--bg);
        }}
        .tab-panel {{ display: none; }}
        #tab-summary:checked ~ .tab-nav .tab-bar label[for="tab-summary"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-bracket:checked ~ .tab-nav .tab-bar label[for="tab-bracket"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-stats:checked ~ .tab-nav .tab-bar label[for="tab-stats"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-bubble:checked ~ .tab-nav .tab-bar label[for="tab-bubble"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-matrix:checked ~ .tab-nav .tab-bar label[for="tab-matrix"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-ranking:checked ~ .tab-nav .tab-bar label[for="tab-ranking"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-scores:checked ~ .tab-nav .tab-bar label[for="tab-scores"],
        #tab-homecourt:checked ~ .tab-nav .tab-bar label[for="tab-homecourt"],
        #tab-autobid:checked ~ .tab-nav .tab-bar label[for="tab-autobid"],
        #tab-conf:checked ~ .tab-nav .tab-bar label[for="tab-conf"],
        #tab-appoll:checked ~ .tab-nav .tab-bar label[for="tab-appoll"] {{
            color: var(--accent);
            font-weight: 700;
        }}
        #tab-scores:checked ~ .tab-nav .tab-more-btn,
        #tab-homecourt:checked ~ .tab-nav .tab-more-btn,
        #tab-autobid:checked ~ .tab-nav .tab-more-btn,
        #tab-conf:checked ~ .tab-nav .tab-more-btn,
        #tab-appoll:checked ~ .tab-nav .tab-more-btn {{
            color: var(--accent);
            border-bottom: 2px solid var(--accent);
            margin-bottom: -2px;
        }}
        #tab-schedule:checked ~ .tab-nav .tab-bar label[for="tab-schedule"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-standings:checked ~ .tab-nav .tab-bar label[for="tab-standings"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-summary:checked ~ #panel-summary {{ display: block; }}
        #tab-bracket:checked ~ #panel-bracket {{ display: block; }}
        #tab-stats:checked ~ #panel-stats {{ display: block; }}
        #tab-bubble:checked ~ #panel-bubble {{ display: block; }}
        #tab-conf:checked ~ #panel-conf {{ display: block; }}
        #tab-autobid:checked ~ #panel-autobid {{ display: block; }}
        #tab-matrix:checked ~ #panel-matrix {{ display: block; }}
        #tab-ranking:checked ~ #panel-ranking {{ display: block; }}
        #tab-scores:checked ~ #panel-scores {{ display: block; }}
        #tab-schedule:checked ~ #panel-schedule {{ display: block; }}
        #tab-homecourt:checked ~ #panel-homecourt {{ display: block; }}
        #tab-standings:checked ~ #panel-standings {{ display: block; }}
        #tab-appoll:checked ~ #panel-appoll {{ display: block; }}

        /* AP Poll tab styling */
        .ap-week-select {{
            display: inline-block;
            padding: 0.35rem 0.7rem;
            font-size: 0.88rem;
            font-family: inherit;
            color: var(--text);
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 0.75rem;
        }}
        .ap-week-select:focus {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
        .ap-subtitle {{ color: var(--text-muted); font-size: 0.82rem; margin: 0 0 0.75rem; }}
        .ap-table td:first-child {{ font-weight: 700; width: 2.5rem; text-align: center; }}
        .ap-fpv {{ color: var(--text-muted); font-size: 0.8em; }}
        .ap-trend {{ font-size: 0.85em; font-weight: 600; }}
        .ap-up {{ color: #22c55e; }}
        .ap-down {{ color: #ef4444; }}
        .ap-flat {{ color: var(--text-muted); }}
        .ap-new {{ color: var(--accent); font-size: 0.75em; font-weight: 700; }}
        .ap-others {{
            margin-top: 1.5rem;
            padding: 1rem;
            background: var(--surface);
            border-radius: 8px;
            border: 1px solid var(--border);
            font-size: 0.88rem;
            line-height: 1.8;
        }}
        .ap-others .team-logo {{ height: 20px; width: 20px; margin-right: 2px; vertical-align: middle; }}

        /* Standings tab — collapsible conference groups */
        .standings-conf {{
            margin-bottom: 0.5rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        .standings-summary {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.7rem 1rem;
            background: var(--surface);
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
            list-style: none;
            user-select: none;
        }}
        .standings-summary::-webkit-details-marker {{ display: none; }}
        .standings-summary::before {{
            content: '\\25b6';
            font-size: 0.6rem;
            color: var(--text-muted);
            transition: transform 0.15s;
        }}
        details.standings-conf[open] > .standings-summary::before {{
            transform: rotate(90deg);
        }}
        .standings-conf-name {{
            flex: 1;
        }}
        .standings-conf-meta {{
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 400;
        }}
        .standings-table {{
            border-radius: 0;
        }}
        .standings-table thead th {{
            background: var(--surface);
            border-bottom-color: var(--border);
        }}
        .standing-status {{
            font-size: 0.7rem;
        }}
        .standing-in {{
            color: var(--green);
        }}
        .standing-bubble {{
            color: var(--accent);
        }}
        .title-pct {{
            font-weight: 600;
            font-size: 0.8rem;
            text-align: right;
            min-width: 50px;
        }}
        .title-pct.simulated {{
            color: var(--text);
        }}
        .title-pct.title-high {{
            color: var(--green);
        }}
        .title-pct.title-lock {{
            color: var(--green);
            font-weight: 700;
        }}

        /* Scores tab — interactive picker */
        .scores-picker {{
            display: flex;
            gap: 2rem;
            align-items: flex-start;
            justify-content: center;
            margin-bottom: 2rem;
        }}
        .scores-side {{
            flex: 1;
            max-width: 340px;
        }}
        .scores-side label {{
            display: block;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }}
        .scores-search {{
            position: relative;
        }}
        .scores-search input {{
            width: 100%;
            padding: 0.6rem 0.75rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            font-size: 0.9rem;
            outline: none;
        }}
        .scores-search input:focus {{
            border-color: var(--accent);
        }}
        .scores-dropdown {{
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            max-height: 260px;
            overflow-y: auto;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0 0 6px 6px;
            z-index: 20;
        }}
        .scores-dropdown.open {{
            display: block;
        }}
        .scores-option {{
            padding: 0.45rem 0.75rem;
            cursor: pointer;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .scores-option:hover, .scores-option.active {{
            background: rgba(249, 115, 22, 0.15);
        }}
        .scores-option img {{
            width: 20px;
            height: 20px;
        }}
        .scores-option .opt-meta {{
            color: var(--text-muted);
            font-size: 0.75rem;
            margin-left: auto;
        }}
        .scores-vs {{
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text-muted);
            padding-top: 1.8rem;
            flex-shrink: 0;
        }}
        .scores-result {{
            text-align: center;
            padding: 1.5rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            display: none;
        }}
        .scores-result.visible {{
            display: block;
        }}
        .scores-matchup {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .scores-team-card {{
            text-align: center;
            min-width: 140px;
        }}
        .scores-team-card img {{
            width: 48px;
            height: 48px;
            display: block;
            margin: 0 auto 0.4rem;
        }}
        .scores-team-card .tc-name {{
            font-weight: 700;
            font-size: 1rem;
        }}
        .scores-team-card .tc-meta {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        .scores-team-card .tc-pct {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }}
        .scores-spread {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}
        .scores-detail {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            font-size: 0.82rem;
            color: var(--text-muted);
        }}
        .scores-detail span {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .scores-detail .sd-val {{
            font-weight: 600;
            color: var(--text);
            font-size: 0.9rem;
        }}
        .venue-toggle {{
            display: flex;
            justify-content: center;
            gap: 0;
            margin-bottom: 1.5rem;
        }}
        .venue-btn {{
            padding: 0.45rem 1.1rem;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s, color 0.15s;
        }}
        .venue-btn:first-child {{ border-radius: 6px 0 0 6px; }}
        .venue-btn:last-child {{ border-radius: 0 6px 6px 0; }}
        .venue-btn:not(:first-child) {{ border-left: none; }}
        .venue-btn.active {{
            background: var(--accent);
            color: var(--bg);
            border-color: var(--accent);
        }}
        .venue-btn:hover:not(.active) {{
            color: var(--text);
        }}
        .scores-venue {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.3rem;
        }}
        /* Schedule tab */
        .sched-nav {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }}
        .sched-nav button {{
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            transition: background 0.15s, border-color 0.15s;
        }}
        .sched-nav button:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        .sched-nav .sched-date {{
            font-size: 1.1rem;
            font-weight: 700;
            min-width: 200px;
            text-align: center;
        }}
        .sched-filter {{
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.4rem 0.6rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: border-color 0.15s;
        }}
        .sched-filter:hover, .sched-filter:focus {{
            border-color: var(--accent);
            outline: none;
        }}
        .sched-summary {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 1.5rem;
        }}
        .sched-game {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 1rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.15s;
        }}
        .sched-game:hover {{
            border-color: var(--accent);
        }}
        .sched-live {{
            border-color: var(--green);
            border-width: 2px;
        }}
        .sched-team {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}
        .sched-team.away {{
            justify-content: flex-end;
            text-align: right;
        }}
        .sched-team.home {{
            justify-content: flex-start;
            text-align: left;
        }}
        .sched-team img {{
            width: 32px;
            height: 32px;
            flex-shrink: 0;
        }}
        .sched-team-info {{
            display: flex;
            flex-direction: column;
        }}
        .sched-team-name {{
            font-weight: 700;
            font-size: 0.95rem;
        }}
        .sched-ap-rank {{
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--accent);
            margin-right: 0.2rem;
        }}
        .sched-team-meta {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}
        .sched-venue-tag {{
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-muted);
            opacity: 0.7;
        }}
        .sched-center {{
            text-align: center;
            min-width: 100px;
        }}
        .sched-score {{
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .sched-score .sched-winner {{
            color: var(--accent);
        }}
        .sched-time {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.15rem;
        }}
        .sched-pred {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        .sched-live-badge {{
            display: inline-block;
            background: var(--green);
            color: var(--bg);
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
            letter-spacing: 0.05em;
        }}
        .sched-watch {{
            position: absolute;
            top: 0.5rem;
            right: 0.65rem;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .sched-game {{
            position: relative;
        }}
        .sched-loading {{
            text-align: center;
            color: var(--text-muted);
            padding: 3rem 1rem;
            font-size: 0.95rem;
        }}

        /* Summary tab */
        .summary-section {{
            margin-bottom: 2.5rem;
        }}
        .summary-section-header {{
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 1rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid var(--accent);
        }}
        .summary-sub-header {{
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--text);
            margin: 1.25rem 0 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .summary-narrative {{
            line-height: 1.75;
            font-size: 0.9rem;
        }}
        .summary-narrative p {{
            margin: 0.6rem 0;
        }}
        .summary-narrative .team-logo {{
            width: 18px;
            height: 18px;
            vertical-align: middle;
            margin: 0 2px;
        }}
        .summary-team-ref {{
            font-weight: 600;
            cursor: pointer;
        }}
        .summary-team-ref:hover {{
            text-decoration: underline;
        }}
        .summary-ctx {{
            color: var(--text-muted);
            font-size: 0.82rem;
            font-weight: 400;
        }}
        .summary-score {{
            font-weight: 700;
        }}
        .summary-impact {{
            color: var(--text-muted);
            font-style: italic;
            font-size: 0.85rem;
        }}
        .summary-movers {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.75rem 0 1rem;
        }}
        .summary-mover {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.35rem 0.65rem;
            font-size: 0.8rem;
            border-left: 3px solid var(--text-muted);
        }}
        .summary-mover .team-logo {{
            width: 22px;
            height: 22px;
        }}
        .mover-up {{ border-left-color: var(--green); }}
        .mover-up .mover-arrow {{ color: var(--green); }}
        .mover-down {{ border-left-color: var(--red); }}
        .mover-down .mover-arrow {{ color: var(--red); }}
        .mover-new {{ border-left-color: var(--accent); }}
        .mover-new .mover-arrow {{ color: var(--accent); }}
        .mover-arrow {{
            font-weight: 700;
            font-size: 0.7rem;
        }}
        .mover-seeds {{
            color: var(--text-muted);
            font-size: 0.72rem;
        }}
        .summary-preview-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.6rem;
        }}
        .summary-preview-card .summary-matchup-line {{
            display: flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.88rem;
            margin-bottom: 0.3rem;
        }}
        .summary-preview-card .summary-matchup-line .team-logo {{
            width: 20px;
            height: 20px;
        }}
        .summary-preview-card .summary-stakes {{
            font-size: 0.82rem;
            color: var(--text-muted);
            line-height: 1.5;
            margin-top: 0.2rem;
        }}
        .summary-preview-card .summary-pred-line {{
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        .summary-loading {{
            text-align: center;
            color: var(--text-muted);
            padding: 2rem 1rem;
            font-size: 0.95rem;
        }}
        .summary-empty {{
            color: var(--text-muted);
            font-size: 0.85rem;
            font-style: italic;
            padding: 0.5rem 0;
        }}
        .summary-live-section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.6rem 0.85rem;
            margin-bottom: 0.75rem;
        }}
        .summary-live-game {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.85rem;
            padding: 0.25rem 0;
        }}
        .summary-live-game .team-logo {{
            width: 18px;
            height: 18px;
        }}
        @media (max-width: 600px) {{
            .summary-narrative {{ font-size: 0.85rem; }}
            .summary-mover {{ padding: 0.25rem 0.5rem; font-size: 0.75rem; }}
            .summary-preview-card {{ padding: 0.5rem 0.65rem; }}
        }}

        @media (max-width: 600px) {{
            .scores-picker {{
                flex-direction: column;
                align-items: stretch;
                gap: 0.5rem;
            }}
            .scores-side {{ max-width: none; }}
            .scores-vs {{
                text-align: center;
                padding: 0;
            }}
            .scores-matchup {{ flex-direction: column; gap: 1rem; }}
        }}

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
            cursor: pointer;
        }}
        .stats-team:hover {{
            color: var(--accent);
        }}
        .team-logo {{
            width: 20px;
            height: 20px;
            vertical-align: middle;
            margin-right: 4px;
        }}
        .conf-logo {{
            width: 18px;
            height: 18px;
            vertical-align: middle;
            margin-right: 4px;
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
        .bubble-tab-header.bubble-bye {{
            background: rgba(59, 130, 246, 0.1);
            border-left: 3px solid #3b82f6;
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
        .bubble-bye .bubble-tab-label {{ color: #3b82f6; }}
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
        .col-avg {{ background: rgba(99, 102, 241, 0.12); font-weight: 700; }}

        /* Frozen columns for Bracket Matrix horizontal scroll */
        #matrix-table th:nth-child(-n+4),
        #matrix-table td:nth-child(-n+4) {{
            position: sticky;
            z-index: 2;
            background: var(--bg);
        }}
        #matrix-table tr:hover td:nth-child(-n+4) {{
            background: var(--surface);
        }}
        #matrix-table th:nth-child(-n+4) {{
            z-index: 3;
        }}
        #matrix-table th:nth-child(1),
        #matrix-table td:nth-child(1) {{ left: 0; min-width: 140px; }}
        #matrix-table th:nth-child(2),
        #matrix-table td:nth-child(2) {{ left: 140px; min-width: 44px; }}
        #matrix-table th:nth-child(3),
        #matrix-table td:nth-child(3) {{ left: 184px; min-width: 80px; }}
        #matrix-table th:nth-child(4),
        #matrix-table td:nth-child(4) {{ left: 264px; min-width: 50px; border-right: 2px solid var(--border); }}
        /* Keep col-avg purple tint on the frozen avg column */
        #matrix-table td.col-avg {{ background: color-mix(in srgb, rgba(99, 102, 241, 0.12), var(--bg)); }}
        #matrix-table tr:hover td.col-avg {{ background: color-mix(in srgb, rgba(99, 102, 241, 0.12), var(--surface)); }}
        #matrix-table th.col-avg {{ background: color-mix(in srgb, rgba(99, 102, 241, 0.12), var(--bg)); }}

        @media (max-width: 600px) {{
            h1 {{ font-size: 1.5rem; }}
            .first-four-list {{ grid-template-columns: repeat(2, 1fr); }}
            .brackets-grid {{ grid-template-columns: 1fr; }}
            .bracket-region pre {{ font-size: 0.55rem; padding-left: 16px; }}
            .bracket-logo {{ width: 12px; height: 12px; margin-left: -14px; margin-right: 2px; }}
            .container {{ padding: 1rem 0.5rem; }}
            .tab-bar {{
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: none;
            }}
            .tab-bar::-webkit-scrollbar {{ display: none; }}
            .tab-bar label {{
                padding: 0.5rem 0.5rem;
                font-size: 0.72rem;
                white-space: nowrap;
                flex-shrink: 0;
            }}
            .tab-more-btn {{
                padding: 0.5rem 0.5rem;
                font-size: 0.72rem;
            }}
            .stats-table {{ font-size: 0.75rem; }}
            .stats-table th {{ font-size: 0.68rem; padding: 0.4rem; }}
            .stats-table td {{ padding: 0.35rem 0.4rem; }}
            .seed-table td {{ padding: 0.4rem 0.5rem; font-size: 0.85rem; }}
            .bubble-row {{ flex-direction: column; gap: 0.4rem; }}
            .bubble-label {{ min-width: auto; }}
            .sched-game {{
                grid-template-columns: 1fr;
                gap: 0.5rem;
                text-align: center;
            }}
            .sched-team.away, .sched-team.home {{
                justify-content: center;
                text-align: center;
            }}
            .sched-team img {{ width: 24px; height: 24px; }}
            .sched-nav .sched-date {{ min-width: 160px; font-size: 0.95rem; }}
        }}

        /* ── Team Profile (Full Page) ── */
        .modal-overlay {{
            display: none;
            position: fixed;
            inset: 0;
            z-index: 1000;
            background: var(--bg);
            overflow-y: auto;
        }}
        .modal-overlay.active {{ display: block; }}
        .modal-box {{
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            padding: 1.5rem 1.5rem 3rem;
            position: relative;
            min-height: 100vh;
        }}
        .modal-box.compare-mode {{ max-width: 900px; }}
        .modal-close {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: none;
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-muted);
            font-size: 0.85rem;
            cursor: pointer;
            padding: 0.35rem 0.75rem;
            margin-bottom: 1rem;
        }}
        .modal-close:hover {{ color: var(--text); border-color: var(--text-muted); }}
        .modal-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }}
        .modal-header img {{ width: 64px; height: 64px; }}
        .modal-header .team-info h2 {{ margin: 0; font-size: 1.5rem; color: var(--text); }}
        .modal-header .team-info .meta {{ color: var(--text-muted); font-size: 0.9rem; }}
        .modal-stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .modal-stat {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem;
            text-align: center;
        }}
        .modal-stat .label {{ font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}
        .modal-stat .value {{ font-size: 1rem; font-weight: 600; color: var(--text); margin-top: 2px; }}
        .modal-section {{ margin-bottom: 1rem; }}
        .modal-section h3 {{ font-size: 0.9rem; color: var(--text-muted); margin: 0 0 0.5rem; text-transform: uppercase; letter-spacing: 0.5px; }}
        .quad-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
            gap: 0.4rem;
        }}
        .quad-cell {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.35rem 0.5rem;
            text-align: center;
            font-size: 0.85rem;
        }}
        .quad-cell .qlabel {{ font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; }}
        .quad-cell .qval {{ font-weight: 600; color: var(--text); }}
        .hca-bars {{ display: flex; flex-direction: column; gap: 0.4rem; }}
        .hca-bar-row {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .hca-bar-label {{ width: 70px; font-size: 0.8rem; color: var(--text-muted); text-align: right; flex-shrink: 0; }}
        .hca-bar-track {{
            flex: 1;
            height: 18px;
            background: var(--bg);
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }}
        .hca-bar-fill {{
            position: absolute;
            top: 0;
            height: 100%;
            border-radius: 4px;
            min-width: 2px;
        }}
        .hca-bar-fill.positive {{ background: var(--green); left: 50%; }}
        .hca-bar-fill.negative {{ background: var(--red); right: 50%; }}
        .hca-bar-center {{
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 1px;
            background: var(--border);
        }}
        .hca-bar-val {{ width: 50px; font-size: 0.8rem; color: var(--text); font-weight: 600; }}
        .modal-btn {{
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg);
            color: var(--text);
            font-size: 0.85rem;
            cursor: pointer;
        }}
        .modal-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
        .compare-search {{
            position: relative;
            margin-top: 0.5rem;
        }}
        .compare-search input {{
            width: 100%;
            padding: 0.5rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg);
            color: var(--text);
            font-size: 0.9rem;
            box-sizing: border-box;
        }}
        .compare-search input::placeholder {{ color: var(--text-muted); }}
        .compare-search .results {{
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0 0 6px 6px;
            max-height: 200px;
            overflow-y: auto;
            z-index: 10;
            display: none;
        }}
        .compare-search .results.show {{ display: block; }}
        .compare-search .results div {{
            padding: 0.4rem 0.6rem;
            cursor: pointer;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .compare-search .results div:hover {{ background: var(--bg); }}
        .compare-search .results img {{ width: 20px; height: 20px; }}
        .compare-layout {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}
        .compare-layout .compare-col {{ min-width: 0; }}
        .compare-stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.85rem;
        }}
        .compare-stat-row .stat-label {{ color: var(--text-muted); flex: 1; text-align: center; }}
        .compare-stat-row .stat-val {{ flex: 1; font-weight: 600; }}
        .compare-stat-row .stat-val.left {{ text-align: right; padding-right: 0.5rem; }}
        .compare-stat-row .stat-val.right {{ text-align: left; padding-left: 0.5rem; }}
        .compare-stat-row .stat-val.better {{ color: var(--green); }}
        .compare-matchup {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 0.75rem;
        }}
        .compare-matchup h4 {{ margin: 0 0 0.5rem; font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; }}
        .venue-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            font-size: 0.85rem;
        }}
        .venue-row .venue {{ color: var(--text-muted); flex: 1; }}
        .venue-row .spread {{ flex: 1; text-align: center; font-weight: 600; color: var(--accent); }}
        .venue-row .winpct {{ flex: 1; text-align: right; color: var(--text); }}
        /* ── Team Schedule ── */
        .team-sched-row {{ border-bottom: 1px solid var(--border); }}
        .team-sched-row:last-child {{ border-bottom: none; }}
        .team-sched-game {{
            display: flex;
            align-items: center;
            padding: 0.6rem 0;
            gap: 0.75rem;
            font-size: 0.85rem;
            cursor: pointer;
        }}
        .team-sched-game:hover {{ opacity: 0.8; }}
        .team-sched-date {{
            color: var(--text-muted);
            font-size: 0.75rem;
            width: 70px;
            flex-shrink: 0;
        }}
        .team-sched-opp {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex: 1;
            min-width: 0;
        }}
        .team-sched-opp img {{ width: 24px; height: 24px; flex-shrink: 0; }}
        .team-sched-opp .opp-name {{
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .team-sched-venue {{
            color: var(--text-muted);
            font-size: 0.75rem;
            width: 32px;
            text-align: center;
            flex-shrink: 0;
        }}
        .team-sched-result {{
            width: 80px;
            text-align: right;
            font-weight: 600;
            flex-shrink: 0;
        }}
        .team-sched-result.win {{ color: var(--green); }}
        .team-sched-result.loss {{ color: var(--red); }}
        .team-sched-loading {{
            color: var(--text-muted);
            text-align: center;
            padding: 1.5rem;
            font-size: 0.85rem;
        }}
        /* ── Box Score ── */
        .box-score {{
            display: none;
            padding: 0.5rem 0 0.75rem;
        }}
        .box-score.open {{ display: block; }}
        .box-score-team {{
            margin-bottom: 0.75rem;
        }}
        .box-score-team:last-child {{ margin-bottom: 0; }}
        .box-team-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            font-size: 0.8rem;
            margin-bottom: 0.35rem;
            color: var(--text);
        }}
        .box-team-header img {{ width: 20px; height: 20px; }}
        .box-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.7rem;
        }}
        .box-table th {{
            text-align: right;
            padding: 0.2rem 0.3rem;
            color: var(--text-muted);
            font-weight: 500;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}
        .box-table th:first-child {{ text-align: left; }}
        .box-table td {{
            text-align: right;
            padding: 0.2rem 0.3rem;
            color: var(--text);
            white-space: nowrap;
        }}
        .box-table td:first-child {{
            text-align: left;
            font-weight: 500;
            max-width: 110px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .box-table tr.box-totals td {{
            border-top: 1px solid var(--border);
            font-weight: 600;
        }}
        .box-table tr.box-bench td:first-child {{
            color: var(--text-muted);
            font-style: italic;
        }}
        .box-score-loading {{
            color: var(--text-muted);
            font-size: 0.75rem;
            padding: 0.5rem 0;
        }}
        @media (max-width: 640px) {{
            .modal-box {{ padding: 1rem 1rem 2rem; }}
            .modal-stats-grid {{ grid-template-columns: repeat(3, 1fr); }}
            .compare-layout {{ grid-template-columns: 1fr; }}
            .team-sched-date {{ width: 55px; font-size: 0.7rem; }}
            .team-sched-result {{ width: 65px; }}
            .box-table {{ font-size: 0.6rem; }}
            .box-table td:first-child {{ max-width: 80px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span class="logo">68</span>bracket</h1>
            <div class="subtitle">{title}</div>
            <div class="tagline">Machine-learning bracket projections, updated daily with the latest team metrics and game results.</div>
            <div class="timestamp">Model: {model_label} &middot; Data updated: {timestamp}</div>
        </header>

        <input type="radio" name="tabs" id="tab-summary" class="tab-radio" checked>
        <input type="radio" name="tabs" id="tab-bracket" class="tab-radio">
        <input type="radio" name="tabs" id="tab-stats" class="tab-radio">
        <input type="radio" name="tabs" id="tab-bubble" class="tab-radio">
        <input type="radio" name="tabs" id="tab-conf" class="tab-radio">
        <input type="radio" name="tabs" id="tab-autobid" class="tab-radio">
        <input type="radio" name="tabs" id="tab-matrix" class="tab-radio">
        <input type="radio" name="tabs" id="tab-ranking" class="tab-radio">
        <input type="radio" name="tabs" id="tab-scores" class="tab-radio">
        <input type="radio" name="tabs" id="tab-schedule" class="tab-radio">
        <input type="radio" name="tabs" id="tab-homecourt" class="tab-radio">
        <input type="radio" name="tabs" id="tab-standings" class="tab-radio">
        <input type="radio" name="tabs" id="tab-appoll" class="tab-radio">

        <div class="tab-nav">
            <div class="tab-bar">
                <label for="tab-summary">Summary</label>
                <label for="tab-bracket">Bracket</label>
                <label for="tab-bubble">Bubble Watch</label>
                <label for="tab-schedule">Schedule</label>
                <label for="tab-standings">Standings</label>
                <label for="tab-matrix">Bracket Matrix</label>
                <label for="tab-stats">Team Stats</label>
                <label for="tab-ranking">Power Rankings</label>
            </div>
            <div class="tab-more">
                <button class="tab-more-btn" type="button">More &#9662;</button>
                <div class="tab-more-menu">
                    <label for="tab-appoll">AP Poll</label>
                    <label for="tab-scores">Matchup Predictor</label>
                    <label for="tab-homecourt">Home Court</label>
                    <label for="tab-autobid">Auto Bids</label>
                    <label for="tab-conf">Conferences</label>
                </div>
            </div>
        </div>

        <div id="panel-summary" class="tab-panel">
            <h2>Daily Summary</h2>
            {summary_tab_html if summary_tab_html else '<p style="color: var(--text-muted);">Summary data not available. Run the predict command to generate.</p>'}
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

        <div id="panel-standings" class="tab-panel">
            <h2>Conference Standings</h2>
            <p style="color: var(--text-muted); font-size: 0.82rem; margin-top: 0;">Title % shows each team's chance to win or share the regular season conference title based on 1,000 simulations of remaining games. Percentages can add up to over 100% because multiple teams can share a title in the same simulation.</p>
            {standings_tab_html if standings_tab_html else '<p style="color: var(--text-muted);">Standings data not available. Run the predict command to generate.</p>'}
        </div>

        <div id="panel-autobid" class="tab-panel">
            <h2>Auto Bids</h2>
            {autobid_tab_html if autobid_tab_html else '<p style="color: var(--text-muted);">Auto-bid data not available. Run the predict command to generate.</p>'}
        </div>

        <div id="panel-matrix" class="tab-panel">
            <h2>Bracket Matrix</h2>
            {matrix_tab_html if matrix_tab_html else '<p style="color: var(--text-muted);">Bracket Matrix data not available. Run scrape and predict to generate.</p>'}
        </div>

        <div id="panel-ranking" class="tab-panel">
            <h2>Power Rankings</h2>
            {ranking_tab_html if ranking_tab_html else '<p style="color: var(--text-muted);">Power Rankings data not available. Run the predict command to generate.</p>'}
        </div>

        <div id="panel-scores" class="tab-panel">
            <h2>Matchup Predictor</h2>
            {scores_tab_html if scores_tab_html else '<p style="color: var(--text-muted);">Scores data not available. Run the predict command to generate.</p>'}
        </div>

        <div id="panel-schedule" class="tab-panel">
            <h2>Schedule</h2>
            {schedule_tab_html if schedule_tab_html else '<p style="color: var(--text-muted);">Schedule data not available. Run the predict command to generate.</p>'}
        </div>

        <div id="panel-homecourt" class="tab-panel">
            <h2>Home Court Rankings</h2>
            {homecourt_tab_html if homecourt_tab_html else '<p style="color: var(--text-muted);">Home Court data not available. Run the predict command to generate.</p>'}
        </div>

        <div id="panel-appoll" class="tab-panel">
            <h2>AP Top 25 Poll</h2>
            {appoll_tab_html if appoll_tab_html else '<p style="color: var(--text-muted);">AP Poll data not available. Run scrape to fetch latest rankings.</p>'}
        </div>

        <!-- Team Profile Modal -->
        <div id="team-modal" class="modal-overlay">
            <div class="modal-box" id="modal-content"></div>
        </div>

        <div class="footnote">
            Built by <a href="https://github.com/hunterwalklin/68bracket">68bracket</a> &middot; Updated daily during the season
            <div class="footnote-sources">Data from ESPN, KenPom, Bart Torvik, WarrenNolan, Sports Reference, AP &middot; Bracketology via Bracket Matrix</div>
            <div class="footnote-disclaimer">Predictions are for informational and entertainment purposes only. Not affiliated with the NCAA.</div>
        </div>
    </div>
    <script>
    /* Tab persistence via URL hash */
    (function(){{
        var radios=document.querySelectorAll('.tab-radio');
        function setTab(){{
            var h=location.hash.replace('#','');
            if(!h)return;
            var r=document.getElementById('tab-'+h);
            if(r)r.checked=true;
        }}
        setTab();
        window.addEventListener('hashchange',setTab);
        radios.forEach(function(r){{
            r.addEventListener('change',function(){{
                if(this.checked)location.hash=this.id.replace('tab-','');
            }});
        }});
        /* Tab bar scroll fade: hide right fade when scrolled to end */
        var bar=document.querySelector('.tab-bar');
        function checkScroll(){{
            if(!bar)return;
            var atEnd=bar.scrollLeft+bar.clientWidth>=bar.scrollWidth-5;
            bar.classList.toggle('scrolled-end',atEnd);
        }}
        if(bar){{
            bar.addEventListener('scroll',checkScroll);
            checkScroll();
            /* Scroll active tab into view on load and tab change */
            function scrollActiveTab(){{
                var active=bar.querySelector('label[for="tab-'+location.hash.replace('#','')+'"]');
                if(active)active.scrollIntoView({{behavior:'smooth',block:'nearest',inline:'center'}});
            }}
            scrollActiveTab();
            window.addEventListener('hashchange',scrollActiveTab);
        }}
    }})();
    /* More dropdown menu */
    (function(){{
        var btn=document.querySelector('.tab-more-btn');
        var menu=document.querySelector('.tab-more-menu');
        if(!btn||!menu)return;
        btn.addEventListener('click',function(e){{
            e.stopPropagation();
            menu.classList.toggle('open');
        }});
        /* Close when clicking a menu item */
        menu.querySelectorAll('label').forEach(function(label){{
            label.addEventListener('click',function(){{
                menu.classList.remove('open');
            }});
        }});
        /* Close when clicking outside */
        document.addEventListener('click',function(){{
            menu.classList.remove('open');
        }});
        menu.addEventListener('click',function(e){{e.stopPropagation();}});
    }})();
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
                        av=parseFloat(a.cells[col].textContent);if(isNaN(av))av=9999;
                        bv=parseFloat(b.cells[col].textContent);if(isNaN(bv))bv=9999;
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
                        av=parseFloat(a.cells[col].textContent);if(isNaN(av))av=9999;
                        bv=parseFloat(b.cells[col].textContent);if(isNaN(bv))bv=9999;
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
                        av=parseFloat(a.cells[col].textContent);if(isNaN(av))av=9999;
                        bv=parseFloat(b.cells[col].textContent);if(isNaN(bv))bv=9999;
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
        var table=document.getElementById('homecourt-table');
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
                        av=parseFloat(a.cells[col].textContent);if(isNaN(av))av=9999;
                        bv=parseFloat(b.cells[col].textContent);if(isNaN(bv))bv=9999;
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
                        av=parseFloat(a.cells[col].textContent);if(isNaN(av))av=9999;
                        bv=parseFloat(b.cells[col].textContent);if(isNaN(bv))bv=9999;
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
        var table=document.getElementById('ranking-table');
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
                        av=parseFloat(a.cells[col].textContent);if(isNaN(av))av=9999;
                        bv=parseFloat(b.cells[col].textContent);if(isNaN(bv))bv=9999;
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
    /* Matchup predictor */
    (function(){{
        var app=document.getElementById('scores-app');
        if(!app||!window.__SCORES_TEAMS__)return;
        var teams=window.__SCORES_TEAMS__;
        var sel=[null,null];
        var venue='neutral'; /* 'homeA', 'neutral', 'homeB' */
        var CDN='https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/';

        function logoUrl(espn){{return espn?CDN+espn+'.png&h=40&w=40':'';}}

        /* Build DOM */
        app.innerHTML='<div class="scores-picker">'
            +'<div class="scores-side" id="ss-0"><label>Team A</label><div class="scores-search"><input type="text" placeholder="Search teams..." autocomplete="off"><div class="scores-dropdown"></div></div></div>'
            +'<div class="scores-vs">vs</div>'
            +'<div class="scores-side" id="ss-1"><label>Team B</label><div class="scores-search"><input type="text" placeholder="Search teams..." autocomplete="off"><div class="scores-dropdown"></div></div></div>'
            +'</div>'
            +'<div class="venue-toggle" id="venue-toggle">'
            +'<button class="venue-btn" data-v="homeA">Home A</button>'
            +'<button class="venue-btn active" data-v="neutral">Neutral</button>'
            +'<button class="venue-btn" data-v="homeB">Home B</button>'
            +'</div>'
            +'<div class="scores-result" id="scores-result"></div>';

        /* Venue toggle handler */
        document.getElementById('venue-toggle').addEventListener('click',function(e){{
            var btn=e.target.closest('.venue-btn');
            if(!btn)return;
            venue=btn.dataset.v;
            this.querySelectorAll('.venue-btn').forEach(function(b){{b.classList.remove('active');}});
            btn.classList.add('active');
            updateResult();
        }});

        function setupSide(idx){{
            var side=document.getElementById('ss-'+idx);
            var input=side.querySelector('input');
            var dd=side.querySelector('.scores-dropdown');

            function render(query){{
                var q=query.toLowerCase();
                var html='';
                var count=0;
                for(var i=0;i<teams.length&&count<50;i++){{
                    var t=teams[i];
                    if(q&&t.name.toLowerCase().indexOf(q)===-1&&t.conf.toLowerCase().indexOf(q)===-1)continue;
                    var img=t.espn?'<img src="'+logoUrl(t.espn)+'" alt="" loading="lazy">':'';
                    html+='<div class="scores-option" data-idx="'+i+'">'+img+'<span>'+t.name+'</span><span class="opt-meta">'+t.conf+' &middot; '+t.rec+' &middot; NET '+t.net+'</span></div>';
                    count++;
                }}
                dd.innerHTML=html||'<div class="scores-option" style="color:var(--text-muted)">No teams found</div>';
            }}

            input.addEventListener('focus',function(){{
                render(input.value);
                dd.classList.add('open');
            }});
            input.addEventListener('input',function(){{
                sel[idx]=null;
                render(input.value);
                dd.classList.add('open');
                updateResult();
            }});
            dd.addEventListener('mousedown',function(e){{
                var opt=e.target.closest('.scores-option');
                if(!opt||!opt.dataset.idx)return;
                e.preventDefault();
                var t=teams[parseInt(opt.dataset.idx)];
                sel[idx]=t;
                input.value=t.name;
                dd.classList.remove('open');
                updateResult();
            }});
            input.addEventListener('blur',function(){{
                setTimeout(function(){{dd.classList.remove('open');}},150);
            }});
        }}

        setupSide(0);
        setupSide(1);

        function updateResult(){{
            var res=document.getElementById('scores-result');
            if(!sel[0]||!sel[1]){{res.classList.remove('visible');return;}}
            var a=sel[0],b=sel[1];

            /* AdjEM spread — use pace if available */
            var posFactor=(a.pace&&b.pace)?(a.pace+b.pace)/2/100:0.68;
            var emA=a.oe-a.de, emB=b.oe-b.de;
            var spread=(emA-emB)*posFactor;

            /* Apply team-specific home court advantage */
            if(venue==='homeA'){{
                var hcaA=a.hcaPts!==undefined?a.hcaPts:(a.hca*6+1);
                if(hcaA<0.5)hcaA=0.5;
                spread+=hcaA;
            }}else if(venue==='homeB'){{
                var hcaB=b.hcaPts!==undefined?b.hcaPts:(b.hca*6+1);
                if(hcaB<0.5)hcaB=0.5;
                spread-=hcaB;
            }}

            /* Convert spread to win probability using logistic function */
            var winA=1/(1+Math.pow(10,-spread/(11*posFactor)));
            var winB=1-winA;

            var pctA=(winA*100).toFixed(1);
            var pctB=(winB*100).toFixed(1);
            var spreadAbs=Math.abs(spread).toFixed(1);
            var favName=spread>=0?a.name:b.name;

            var hueA=Math.round(winA*120);
            var hueB=Math.round(winB*120);

            var imgA=a.espn?'<img src="'+logoUrl(a.espn)+'" alt="">':'';
            var imgB=b.espn?'<img src="'+logoUrl(b.espn)+'" alt="">':'';

            var venueLabel=venue==='homeA'?'@ '+a.name:venue==='homeB'?'@ '+b.name:'Neutral Court';

            res.innerHTML='<div class="scores-matchup">'
                +'<div class="scores-team-card">'+imgA+'<div class="tc-name stats-team" data-team="'+a.name+'">'+a.name+'</div><div class="tc-meta">'+a.conf+' &middot; '+a.rec+'</div><div class="tc-pct" style="color:hsl('+hueA+',70%,50%)">'+pctA+'%</div></div>'
                +'<div class="scores-vs">vs</div>'
                +'<div class="scores-team-card">'+imgB+'<div class="tc-name stats-team" data-team="'+b.name+'">'+b.name+'</div><div class="tc-meta">'+b.conf+' &middot; '+b.rec+'</div><div class="tc-pct" style="color:hsl('+hueB+',70%,50%)">'+pctB+'%</div></div>'
                +'</div>'
                +'<div class="scores-venue">'+venueLabel+'</div>'
                +'<div class="scores-spread">'+favName+' by '+spreadAbs+'</div>'
                +'<div class="scores-detail">'
                +'<span>AdjOE<span class="sd-val">'+a.oe+' / '+b.oe+'</span></span>'
                +'<span>AdjDE<span class="sd-val">'+a.de+' / '+b.de+'</span></span>'
                +'<span>AdjEM<span class="sd-val">'+emA.toFixed(1)+' / '+emB.toFixed(1)+'</span></span>'
                +'<span>Barthag<span class="sd-val">'+a.bar.toFixed(4)+' / '+b.bar.toFixed(4)+'</span></span>'
                +'<span>Pace<span class="sd-val">'+(a.pace||'—')+' / '+(b.pace||'—')+'</span></span>'
                +'<span>NET<span class="sd-val">'+a.net+' / '+b.net+'</span></span>'
                +'</div>';
            res.classList.add('visible');
        }}
    }})();
    /* Schedule tab */
    (function(){{
        var app=document.getElementById('schedule-app');
        if(!app||!window.__SCORES_TEAMS__)return;
        var teams=window.__SCORES_TEAMS__;
        var CDN='https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/';
        var API='https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard';
        var cache={{}};
        var loaded=false;
        var currentDate=null;
        var currentFilter='all';
        var POWER_CONFS=['ACC','Big 12','Big East','Big Ten','SEC'];

        /* Build ESPN ID lookup + conference list */
        var byEspn={{}};
        var confSet={{}};
        teams.forEach(function(t){{
            if(t.espn)byEspn[String(t.espn)]=t;
            if(t.conf)confSet[t.conf]=true;
        }});
        var confList=Object.keys(confSet).sort();

        function pad(n){{return n<10?'0'+n:''+n;}}
        function fmtDate(d){{return d.getFullYear()+pad(d.getMonth()+1)+pad(d.getDate());}}
        function displayDate(d){{
            var days=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
            var months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
            return days[d.getDay()]+', '+months[d.getMonth()]+' '+d.getDate()+', '+d.getFullYear();
        }}

        function predict(a,b,homeId){{
            if(!a||!b)return null;
            /* Use average pace if both teams have it, else fall back to 0.68 constant */
            var posFactor=(a.pace&&b.pace)?(a.pace+b.pace)/2/100:0.68;
            var ptsA=(a.oe+b.de)/2*posFactor;
            var ptsB=(b.oe+a.de)/2*posFactor;
            /* Home court advantage: use KenPom hcaPts if available, else legacy formula */
            if(homeId){{
                var homeTeam=String(homeId)===String(a.espn)?a:(String(homeId)===String(b.espn)?b:null);
                if(homeTeam){{
                    var pts=homeTeam.hcaPts!==undefined?homeTeam.hcaPts:(homeTeam.hca*6+1);
                    if(pts<0.5)pts=0.5;
                    var half=pts/2;
                    if(String(homeId)===String(a.espn)){{ptsA+=half;ptsB-=half;}}
                    else{{ptsB+=half;ptsA-=half;}}
                }}
            }}
            var spread=ptsA-ptsB;
            var winA=1/(1+Math.pow(10,-spread/(11*posFactor)));
            /* Watchability: competitiveness (60%) + team quality (40%) */
            var comp=Math.max(0,1-Math.abs(spread)/20);
            var avgNet=(a.net+b.net)/2;
            var qual=Math.max(0,1-avgNet/200);
            var watch=Math.round((comp*0.6+qual*0.4)*100);
            var rA=Math.round(ptsA),rB=Math.round(ptsB);
            if(rA===rB){{if(spread>=0)rA+=1;else rB+=1;}}
            return {{ptsA:rA,ptsB:rB,spread:spread,winA:winA,watch:watch}};
        }}

        function logoUrl(espn){{return espn?CDN+espn+'.png&h=40&w=40':'';}}

        function filterGames(games){{
            if(currentFilter==='all')return games;
            return games.filter(function(g){{
                if(currentFilter==='top25'){{
                    return (g.awayTeam&&g.awayTeam.net<=25)||(g.homeTeam&&g.homeTeam.net<=25);
                }}
                if(currentFilter==='power'){{
                    return (g.awayTeam&&POWER_CONFS.indexOf(g.awayTeam.conf)!==-1)||(g.homeTeam&&POWER_CONFS.indexOf(g.homeTeam.conf)!==-1);
                }}
                /* Conference filter */
                return (g.awayTeam&&g.awayTeam.conf===currentFilter)||(g.homeTeam&&g.homeTeam.conf===currentFilter);
            }});
        }}

        function buildFilterSelect(){{
            var s='<select class="sched-filter" id="sched-filter">';
            s+='<option value="all"'+(currentFilter==='all'?' selected':'')+'>All Games</option>';
            s+='<option value="top25"'+(currentFilter==='top25'?' selected':'')+'>Top 25</option>';
            s+='<option value="power"'+(currentFilter==='power'?' selected':'')+'>Power Conferences</option>';
            s+='<optgroup label="Conferences">';
            confList.forEach(function(c){{
                s+='<option value="'+c+'"'+(currentFilter===c?' selected':'')+'>'+c+'</option>';
            }});
            s+='</optgroup></select>';
            return s;
        }}

        function renderGames(data){{
            var events=data.events||[];
            var games=[];
            events.forEach(function(ev){{
                var comp=ev.competitions&&ev.competitions[0];
                if(!comp)return;
                var away=null,home=null;
                (comp.competitors||[]).forEach(function(c){{
                    if(c.homeAway==='away')away=c;
                    else home=c;
                }});
                if(!away||!home)return;
                var awayId=away.team&&away.team.id?String(away.team.id):'';
                var homeId=home.team&&home.team.id?String(home.team.id):'';
                var awayTeam=byEspn[awayId]||null;
                var homeTeam=byEspn[homeId]||null;
                var awayName=away.team&&away.team.displayName?away.team.displayName:(awayTeam?awayTeam.name:'TBD');
                var homeName=home.team&&home.team.displayName?home.team.displayName:(homeTeam?homeTeam.name:'TBD');
                var awayScore=away.score?parseInt(away.score):0;
                var homeScore=home.score?parseInt(home.score):0;
                var status=comp.status||{{}};
                var statusType=status.type||{{}};
                var state=statusType.state||'pre'; /* pre, in, post */
                var detail=statusType.shortDetail||statusType.detail||'';
                var startDate=comp.date||ev.date||'';
                var pred=predict(awayTeam,homeTeam,homeId);
                games.push({{
                    awayName:awayName,homeName:homeName,
                    awayId:awayId,homeId:homeId,
                    awayTeam:awayTeam,homeTeam:homeTeam,
                    awayLogo:away.team&&away.team.logo?away.team.logo:logoUrl(awayId),
                    homeLogo:home.team&&home.team.logo?home.team.logo:logoUrl(homeId),
                    awayScore:awayScore,homeScore:homeScore,
                    state:state,detail:detail,
                    startDate:startDate,
                    pred:pred
                }});
            }});

            /* Sort: live first, then by watchability (high to low), then unpredicted by time */
            games.sort(function(a,b){{
                var aOrd=a.state==='in'?0:(a.pred?1:2);
                var bOrd=b.state==='in'?0:(b.pred?1:2);
                if(aOrd!==bOrd)return aOrd-bOrd;
                /* Within same tier, sort by watchability desc */
                var aw=a.pred?a.pred.watch:0;
                var bw=b.pred?b.pred.watch:0;
                if(aw!==bw)return bw-aw;
                return (a.startDate||'').localeCompare(b.startDate||'');
            }});

            var totalGames=games.length;
            games=filterGames(games);

            var withPred=games.filter(function(g){{return g.pred!==null;}}).length;
            var html='<div class="sched-summary">'+(currentFilter!=='all'?games.length+' of '+totalGames+' games':games.length+' games')+
                (withPred>0?' &middot; '+withPred+' with predictions':'')+
                '</div>';

            games.forEach(function(g){{
                var liveCls=g.state==='in'?' sched-live':'';
                var awayImg=g.awayLogo?'<img src="'+g.awayLogo+'" alt="" loading="lazy">':'';
                var homeImg=g.homeLogo?'<img src="'+g.homeLogo+'" alt="" loading="lazy">':'';
                var awayMeta='',homeMeta='';
                if(g.awayTeam)awayMeta=g.awayTeam.conf+' &middot; '+g.awayTeam.rec+' &middot; NET '+g.awayTeam.net;
                if(g.homeTeam)homeMeta=g.homeTeam.conf+' &middot; '+g.homeTeam.rec+' &middot; NET '+g.homeTeam.net;

                var centerHtml='';
                if(g.state==='post'){{
                    var awScoreCls=g.awayScore>g.homeScore?' sched-winner':'';
                    var hmScoreCls=g.homeScore>g.awayScore?' sched-winner':'';
                    centerHtml='<div class="sched-score"><span class="'+awScoreCls+'">'+g.awayScore+'</span> - <span class="'+hmScoreCls+'">'+g.homeScore+'</span></div>';
                    centerHtml+='<div class="sched-time">Final</div>';
                    if(g.pred){{
                        var diff=Math.abs(g.pred.spread).toFixed(1);
                        var favName=g.pred.spread>=0?g.awayName:g.homeName;
                        var pct=(Math.max(g.pred.winA,1-g.pred.winA)*100).toFixed(0);
                        centerHtml+='<div class="sched-pred">Pred: '+g.pred.ptsA+'-'+g.pred.ptsB+' ('+favName+' '+pct+'%)</div>';
                    }}
                }}else if(g.state==='in'){{
                    centerHtml='<div class="sched-score"><span>'+g.awayScore+'</span> - <span>'+g.homeScore+'</span></div>';
                    centerHtml+='<div class="sched-time"><span class="sched-live-badge">LIVE</span> '+g.detail+'</div>';
                    if(g.pred){{
                        var diff2=Math.abs(g.pred.spread).toFixed(1);
                        var favName2=g.pred.spread>=0?g.awayName:g.homeName;
                        var pct2=(Math.max(g.pred.winA,1-g.pred.winA)*100).toFixed(0);
                        centerHtml+='<div class="sched-pred">Pred: '+g.pred.ptsA+'-'+g.pred.ptsB+' ('+favName2+' '+pct2+'%)</div>';
                    }}
                }}else{{
                    /* Pre-game */
                    if(g.pred){{
                        centerHtml='<div class="sched-score">'+g.pred.ptsA+' - '+g.pred.ptsB+'</div>';
                        var diff3=Math.abs(g.pred.spread).toFixed(1);
                        var favName3=g.pred.spread>=0?g.awayName:g.homeName;
                        var pct3=(Math.max(g.pred.winA,1-g.pred.winA)*100).toFixed(0);
                        centerHtml+='<div class="sched-pred">'+favName3+' '+pct3+'% &middot; '+favName3+' by '+diff3+'</div>';
                    }}
                    /* Show game time */
                    if(g.detail){{
                        centerHtml+='<div class="sched-time">'+g.detail+'</div>';
                    }}else if(g.startDate){{
                        try{{
                            var d=new Date(g.startDate);
                            var h=d.getHours(),m=d.getMinutes();
                            var ampm=h>=12?'PM':'AM';
                            h=h%12;if(h===0)h=12;
                            centerHtml+='<div class="sched-time">'+h+':'+pad(m)+' '+ampm+'</div>';
                        }}catch(e){{}}
                    }}
                }}

                /* Watchability badge */
                var watchHtml='';
                if(g.pred&&g.pred.watch!=null){{
                    var w=g.pred.watch;
                    var hue=Math.round(w*1.2); /* 0=red, 60=yellow, 100+=green */
                    watchHtml='<div class="sched-watch" style="color:hsl('+hue+',70%,50%)" title="Watchability: '+w+'/100">'+w+'</div>';
                }}

                var awayDataTeam=g.awayTeam?g.awayTeam.name:'';
                var homeDataTeam=g.homeTeam?g.homeTeam.name:'';
                var awayTeamCls=awayDataTeam?'sched-team-name stats-team':'sched-team-name';
                var homeTeamCls=homeDataTeam?'sched-team-name stats-team':'sched-team-name';
                var awayRank=g.awayTeam&&g.awayTeam.apRank?'<span class="sched-ap-rank">#'+g.awayTeam.apRank+'</span>':'';
                var homeRank=g.homeTeam&&g.homeTeam.apRank?'<span class="sched-ap-rank">#'+g.homeTeam.apRank+'</span>':'';
                html+='<div class="sched-game'+liveCls+'">'
                    +watchHtml
                    +'<div class="sched-team away">'
                        +'<div class="sched-team-info"><div class="'+awayTeamCls+'"'+(awayDataTeam?' data-team="'+awayDataTeam+'"':'')+'>'+awayRank+g.awayName+'</div><div class="sched-venue-tag">Away</div>'+(awayMeta?'<div class="sched-team-meta">'+awayMeta+'</div>':'')+'</div>'
                        +awayImg
                    +'</div>'
                    +'<div class="sched-center">'+centerHtml+'</div>'
                    +'<div class="sched-team home">'
                        +homeImg
                        +'<div class="sched-team-info"><div class="'+homeTeamCls+'"'+(homeDataTeam?' data-team="'+homeDataTeam+'"':'')+'>'+homeRank+g.homeName+'</div><div class="sched-venue-tag">Home</div>'+(homeMeta?'<div class="sched-team-meta">'+homeMeta+'</div>':'')+'</div>'
                    +'</div>'
                    +'</div>';
            }});

            if(games.length===0){{
                html='<div class="sched-summary">'+(currentFilter!=='all'&&totalGames>0?'No games match the current filter ('+totalGames+' total).':'No games scheduled for this date.')+'</div>';
            }}

            return html;
        }}

        function loadDate(dateStr){{
            currentDate=dateStr;
            var d=new Date(dateStr.slice(0,4)+'-'+dateStr.slice(4,6)+'-'+dateStr.slice(6,8)+'T12:00:00');
            var navHtml='<div class="sched-nav">'
                +'<button id="sched-prev">&larr; Prev</button>'
                +'<button id="sched-today">Today</button>'
                +'<div class="sched-date">'+displayDate(d)+'</div>'
                +'<button id="sched-next">Next &rarr;</button>'
                +buildFilterSelect()
                +'</div>';

            if(cache[dateStr]){{
                app.innerHTML=navHtml+renderGames(cache[dateStr]);
                bindNav();
                startAutoRefresh();
                return;
            }}

            app.innerHTML=navHtml+'<div class="sched-loading">Loading games...</div>';
            bindNav();

            fetch(API+'?dates='+dateStr+'&limit=200&groups=50')
                .then(function(r){{return r.json();}})
                .then(function(data){{
                    cache[dateStr]=data;
                    if(currentDate===dateStr){{
                        app.innerHTML=navHtml+renderGames(data);
                        bindNav();
                        startAutoRefresh();
                    }}
                }})
                .catch(function(){{
                    if(currentDate===dateStr){{
                        app.innerHTML=navHtml+'<div class="sched-loading">Failed to load games. Try again later.</div>';
                        bindNav();
                    }}
                }});
        }}

        function shiftDate(offset){{
            var d=new Date(currentDate.slice(0,4)+'-'+currentDate.slice(4,6)+'-'+currentDate.slice(6,8)+'T12:00:00');
            d.setDate(d.getDate()+offset);
            loadDate(fmtDate(d));
        }}

        function bindNav(){{
            var prev=document.getElementById('sched-prev');
            var next=document.getElementById('sched-next');
            var today=document.getElementById('sched-today');
            var filter=document.getElementById('sched-filter');
            if(prev)prev.onclick=function(){{shiftDate(-1);}};
            if(next)next.onclick=function(){{shiftDate(1);}};
            if(today)today.onclick=function(){{loadDate(fmtDate(new Date()));}};
            if(filter)filter.onchange=function(){{
                currentFilter=this.value;
                if(cache[currentDate])loadDate(currentDate);
            }};
        }}

        /* Auto-refresh for live games — 20s polling */
        var refreshTimer=null;
        var REFRESH_INTERVAL=20000;

        function hasLiveGames(){{
            var cached=cache[currentDate];
            if(!cached||!cached.events)return false;
            return cached.events.some(function(ev){{
                var s=ev.competitions&&ev.competitions[0]&&ev.competitions[0].status;
                return s&&s.type&&s.type.state==='in';
            }});
        }}

        function refreshLive(){{
            if(!currentDate)return;
            var dateStr=currentDate;
            delete cache[dateStr];
            loadDate(dateStr);
        }}

        function startAutoRefresh(){{
            stopAutoRefresh();
            if(hasLiveGames()){{
                refreshTimer=setInterval(refreshLive,REFRESH_INTERVAL);
            }}
        }}

        function stopAutoRefresh(){{
            if(refreshTimer){{clearInterval(refreshTimer);refreshTimer=null;}}
        }}

        /* Lazy load — fetch when Schedule tab is activated */
        var schedRadio=document.getElementById('tab-schedule');
        function initSchedule(){{
            if(!loaded){{
                loaded=true;
                loadDate(fmtDate(new Date()));
            }}else{{
                startAutoRefresh();
            }}
        }}
        if(schedRadio){{
            schedRadio.addEventListener('change',function(){{
                if(this.checked)initSchedule();
                else stopAutoRefresh();
            }});
            /* If tab is already active on page load (hash restore) */
            if(schedRadio.checked)initSchedule();
        }}

        /* Stop polling when switching to another tab */
        document.querySelectorAll('.tab-radio').forEach(function(r){{
            if(r.id!=='tab-schedule'){{
                r.addEventListener('change',function(){{stopAutoRefresh();}});
            }}
        }});

        /* Keyboard navigation */
        document.addEventListener('keydown',function(e){{
            if(!schedRadio||!schedRadio.checked)return;
            if(e.key==='ArrowLeft')shiftDate(-1);
            else if(e.key==='ArrowRight')shiftDate(1);
        }});
    }})();

    /* ── Conference Title Simulation ── */
    (function(){{
        if(!window.__SCORES_TEAMS__)return;
        var teams=window.__SCORES_TEAMS__;
        var SCHED_API='https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/';
        var schedCache={{}};
        var SIMS=1000;

        /* Build conference -> teams lookup */
        var confTeams={{}};
        var byEspn={{}};
        teams.forEach(function(t){{
            if(t.conf){{
                if(!confTeams[t.conf])confTeams[t.conf]=[];
                confTeams[t.conf].push(t);
            }}
            if(t.espn)byEspn[String(t.espn)]=t;
        }});

        function predict(a,b,homeId){{
            if(!a||!b)return 0.5;
            var posFactor=(a.pace&&b.pace)?(a.pace+b.pace)/2/100:0.68;
            var ptsA=(a.oe+b.de)/2*posFactor;
            var ptsB=(b.oe+a.de)/2*posFactor;
            if(homeId){{
                var ht=String(homeId)===String(a.espn)?a:(String(homeId)===String(b.espn)?b:null);
                if(ht){{
                    var pts=ht.hcaPts!==undefined?ht.hcaPts:(ht.hca*6+1);
                    if(pts<0.5)pts=0.5;
                    var half=pts/2;
                    if(String(homeId)===String(a.espn)){{ptsA+=half;ptsB-=half;}}
                    else{{ptsB+=half;ptsA-=half;}}
                }}
            }}
            var spread=ptsA-ptsB;
            return 1/(1+Math.pow(10,-spread/(11*posFactor)));
        }}

        function fetchSchedule(espnId){{
            if(schedCache[espnId])return Promise.resolve(schedCache[espnId]);
            var now=new Date();
            var yr=now.getMonth()>=7?now.getFullYear()+1:now.getFullYear();
            return fetch(SCHED_API+espnId+'/schedule?season='+yr)
                .then(function(r){{return r.json();}})
                .then(function(data){{schedCache[espnId]=data;return data;}})
                .catch(function(){{return null;}});
        }}

        function simulateConference(confName,details){{
            var ct=confTeams[confName];
            if(!ct||ct.length===0)return;

            /* Collect ESPN IDs for this conference */
            var confEspnIds={{}};
            ct.forEach(function(t){{if(t.espn)confEspnIds[String(t.espn)]=t;}});

            /* Update cells to show loading */
            var rows=details.querySelectorAll('tr[data-team]');
            rows.forEach(function(row){{
                var cell=row.querySelector('.title-pct');
                if(cell)cell.textContent='...';
            }});

            /* Fetch all team schedules in parallel */
            var teamsWithEspn=ct.filter(function(t){{return t.espn;}});
            var fetches=teamsWithEspn.map(function(t){{return fetchSchedule(t.espn);}});

            Promise.all(fetches).then(function(schedules){{
                /* Find remaining conference games */
                var remainingGames=[];
                var seen={{}};

                schedules.forEach(function(sched,i){{
                    if(!sched)return;
                    var team=teamsWithEspn[i];
                    var events=[];
                    if(sched.events)events=sched.events;
                    else if(sched.team&&sched.team.events)events=sched.team.events;

                    events.forEach(function(ev){{
                        var comp=ev.competitions&&ev.competitions[0];
                        if(!comp)return;
                        var st=(comp.status||{{}}).type||{{}};
                        if(st.state!=='pre')return;

                        var competitors=comp.competitors||[];
                        if(competitors.length<2)return;
                        var home=null,away=null;
                        competitors.forEach(function(c){{
                            if(c.homeAway==='home')home=c;
                            else away=c;
                        }});
                        if(!home||!away)return;

                        var homeId=String(home.team&&home.team.id||'');
                        var awayId=String(away.team&&away.team.id||'');

                        /* Both teams must be in the conference */
                        if(!confEspnIds[homeId]||!confEspnIds[awayId])return;

                        var key=[homeId,awayId].sort().join('-');
                        if(seen[key])return;
                        seen[key]=true;

                        var homeTeam=confEspnIds[homeId];
                        var awayTeam=confEspnIds[awayId];
                        var winProb=predict(awayTeam,homeTeam,homeId);
                        remainingGames.push({{
                            homeName:homeTeam.name,
                            awayName:awayTeam.name,
                            awayWinProb:winProb
                        }});
                    }});
                }});

                /* Run Monte Carlo */
                var titleCounts={{}};
                ct.forEach(function(t){{titleCounts[t.name]=0;}});

                for(var sim=0;sim<SIMS;sim++){{
                    /* Start with current conference records */
                    var records={{}};
                    ct.forEach(function(t){{
                        records[t.name]={{w:t.confW||0,l:t.confL||0}};
                    }});

                    /* Simulate remaining games */
                    remainingGames.forEach(function(g){{
                        if(Math.random()<g.awayWinProb){{
                            records[g.awayName].w++;
                            records[g.homeName].l++;
                        }}else{{
                            records[g.homeName].w++;
                            records[g.awayName].l++;
                        }}
                    }});

                    /* Find best record(s) — share of title */
                    var bestWins=-1;
                    ct.forEach(function(t){{
                        var r=records[t.name];
                        if(r.w>bestWins)bestWins=r.w;
                    }});
                    ct.forEach(function(t){{
                        if(records[t.name].w===bestWins)titleCounts[t.name]++;
                    }});
                }}

                /* Update DOM */
                rows.forEach(function(row){{
                    var name=row.dataset.team;
                    var cell=row.querySelector('.title-pct');
                    if(!cell)return;
                    var count=titleCounts[name]||0;
                    var pct=Math.round(count/SIMS*100);
                    if(pct===0&&count>0)pct=1; /* show <1% as 1% */
                    cell.textContent=pct>0?pct+'%':'\u2014';
                    cell.classList.add('simulated');
                    if(pct>=90)cell.classList.add('title-lock');
                    else if(pct>=25)cell.classList.add('title-high');
                }});
            }});
        }}

        /* Simulate all conferences when standings tab is activated */
        var standingsRadio=document.getElementById('tab-standings');
        var simulated=false;
        function initStandings(){{
            if(simulated)return;
            simulated=true;
            var allConfs=document.querySelectorAll('.standings-conf[data-conf]');
            /* Stagger fetches — process 3 conferences at a time */
            var queue=Array.from(allConfs);
            function processNext(){{
                var batch=queue.splice(0,3);
                if(batch.length===0)return;
                batch.forEach(function(d){{simulateConference(d.dataset.conf,d);}});
                setTimeout(processNext,500);
            }}
            processNext();
        }}
        if(standingsRadio){{
            standingsRadio.addEventListener('change',function(){{if(this.checked)initStandings();}});
            if(standingsRadio.checked)initStandings();
        }}
    }})();

    /* ── Summary tab ── */
    (function(){{
        var app=document.getElementById('summary-app');
        if(!app||!window.__SCORES_TEAMS__)return;
        var teams=window.__SCORES_TEAMS__;
        var changes=window.__DAILY_CHANGES__||{{}};
        var bubbleData=window.__BUBBLE_DATA__||{{}};
        var seeds=window.__SEED_LIST__||{{}};
        var CDN='https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/';
        var API='https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard';
        var loaded=false;

        var byEspn={{}},byName={{}};
        teams.forEach(function(t){{
            if(t.espn)byEspn[String(t.espn)]=t;
            if(t.name)byName[t.name]=t;
        }});

        var bubbleTeams={{}};
        var bubbleLabels={{last_4_in:'Last Four In',first_4_out:'First Four Out',next_4_out:'Next Four Out',last_4_byes:'Last Four Byes'}};
        ['last_4_in','first_4_out','next_4_out','last_4_byes'].forEach(function(k){{
            (bubbleData[k]||[]).forEach(function(n){{bubbleTeams[n]=k;}});
        }});

        function pad(n){{return n<10?'0'+n:''+n;}}
        function fmtDate(d){{return d.getFullYear()+pad(d.getMonth()+1)+pad(d.getDate());}}
        function displayDate(d){{
            var days=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
            var months=['January','February','March','April','May','June','July','August','September','October','November','December'];
            return days[d.getDay()]+', '+months[d.getMonth()]+' '+d.getDate();
        }}

        function predict(a,b,homeId){{
            if(!a||!b)return null;
            var posFactor=(a.pace&&b.pace)?(a.pace+b.pace)/2/100:0.68;
            var ptsA=(a.oe+b.de)/2*posFactor;
            var ptsB=(b.oe+a.de)/2*posFactor;
            if(homeId){{
                var ht=String(homeId)===String(a.espn)?a:(String(homeId)===String(b.espn)?b:null);
                if(ht){{
                    var pts=ht.hcaPts!==undefined?ht.hcaPts:(ht.hca*6+1);
                    if(pts<0.5)pts=0.5;
                    var half=pts/2;
                    if(String(homeId)===String(a.espn)){{ptsA+=half;ptsB-=half;}}
                    else{{ptsB+=half;ptsA-=half;}}
                }}
            }}
            var spread=ptsA-ptsB;
            var winA=1/(1+Math.pow(10,-spread/(11*posFactor)));
            var comp=Math.max(0,1-Math.abs(spread)/20);
            var avgNet=(a.net+b.net)/2;
            var qual=Math.max(0,1-avgNet/200);
            var watch=Math.round((comp*0.6+qual*0.4)*100);
            var rA=Math.round(ptsA),rB=Math.round(ptsB);
            if(rA===rB){{if(spread>=0)rA+=1;else rB+=1;}}
            return {{ptsA:rA,ptsB:rB,spread:spread,winA:winA,watch:watch}};
        }}

        function parseGames(data){{
            var events=data.events||[];
            var games=[];
            events.forEach(function(ev){{
                var comp=ev.competitions&&ev.competitions[0];
                if(!comp)return;
                var away=null,home=null;
                (comp.competitors||[]).forEach(function(c){{
                    if(c.homeAway==='away')away=c;else home=c;
                }});
                if(!away||!home)return;
                var awayId=away.team&&away.team.id?String(away.team.id):'';
                var homeId=home.team&&home.team.id?String(home.team.id):'';
                var awayTeam=byEspn[awayId]||null;
                var homeTeam=byEspn[homeId]||null;
                var awayName=awayTeam?awayTeam.name:(away.team&&away.team.displayName?away.team.displayName:'TBD');
                var homeName=homeTeam?homeTeam.name:(home.team&&home.team.displayName?home.team.displayName:'TBD');
                games.push({{
                    awayName:awayName,homeName:homeName,
                    awayId:awayId,homeId:homeId,
                    awayTeam:awayTeam,homeTeam:homeTeam,
                    awayScore:away.score?parseInt(away.score):0,
                    homeScore:home.score?parseInt(home.score):0,
                    state:(comp.status||{{}}).type?((comp.status||{{}}).type.state||'pre'):'pre',
                    detail:((comp.status||{{}}).type||{{}}).shortDetail||'',
                    pred:predict(awayTeam,homeTeam,homeId)
                }});
            }});
            return games;
        }}

        /* ── Helpers ── */
        function ref(name,team){{
            var logo=team&&team.espn?'<img class="team-logo" src="'+CDN+team.espn+'.png&h=20&w=20" alt="" loading="lazy">':'';
            var dt=team?team.name:name;
            return logo+'<span class="summary-team-ref stats-team" data-team="'+dt+'">'+name+'</span>';
        }}

        function ctx(name){{
            if(seeds[name])return ' <span class="summary-ctx">('+seeds[name]+'-seed)</span>';
            var bk=bubbleTeams[name];
            if(bk)return ' <span class="summary-ctx">('+bubbleLabels[bk]+')</span>';
            var t=byName[name];
            if(t&&t.net<=60)return ' <span class="summary-ctx">(NET '+t.net+')</span>';
            return '';
        }}

        function isFieldTeam(name){{
            return !!seeds[name];
        }}
        function isBubble(name){{
            return !!bubbleTeams[name];
        }}
        function isRelevant(name){{
            return isFieldTeam(name)||isBubble(name);
        }}
        function isRelevantGame(g){{
            return isRelevant(g.awayName)||isRelevant(g.homeName);
        }}

        function winVerb(margin){{
            if(margin>=20)return 'dominated';
            if(margin>=12)return 'cruised past';
            if(margin>=6)return 'beat';
            if(margin>=3)return 'held off';
            return 'edged';
        }}

        function venueWord(g,teamName){{
            /* Did this team play at home or on the road? */
            var t=byName[teamName];
            if(!t)return '';
            if(String(g.homeId)===String(t.espn))return ' at home';
            return ' on the road';
        }}

        /* ── Yesterday's Recap ── */
        function renderRecap(data){{
            var games=parseGames(data);
            var completed=games.filter(function(g){{return g.state==='post';}});
            var html='<div class="summary-narrative">';

            /* 1. Seed movers */
            var changesArr=[];
            for(var name in changes){{
                if(!changes.hasOwnProperty(name))continue;
                changesArr.push({{name:name,dir:changes[name].direction,prev:changes[name].prev_seed,ns:changes[name].new_seed}});
            }}

            if(changesArr.length>0){{
                changesArr.sort(function(a,b){{
                    if(a.dir==='up'&&b.dir!=='up')return -1;
                    if(a.dir!=='up'&&b.dir==='up')return 1;
                    return (a.ns||99)-(b.ns||99);
                }});

                html+='<div class="summary-sub-header">Bracket Movers</div>';

                /* Chip strip */
                html+='<div class="summary-movers">';
                changesArr.forEach(function(c){{
                    var t=byName[c.name];
                    var logo=t&&t.espn?'<img class="team-logo" src="'+CDN+t.espn+'.png&h=24&w=24" alt="" loading="lazy">':'';
                    var cls='summary-mover';
                    var arrow='',seedTxt='';
                    if(c.prev===null){{cls+=' mover-new';arrow='NEW';seedTxt='&rarr; '+c.ns;}}
                    else if(c.dir==='up'){{cls+=' mover-up';arrow='&#9650;';seedTxt=c.prev+' &rarr; '+c.ns;}}
                    else{{cls+=' mover-down';arrow='&#9660;';seedTxt=c.prev+' &rarr; '+c.ns;}}
                    html+='<div class="'+cls+'">'+logo
                        +'<span class="stats-team" data-team="'+(t?t.name:c.name)+'" style="cursor:pointer">'+c.name+'</span>'
                        +'<span class="mover-arrow">'+arrow+'</span>'
                        +'<span class="mover-seeds">'+seedTxt+'</span></div>';
                }});
                html+='</div>';

                /* Narrative for each mover, connecting to their game if found */
                changesArr.forEach(function(c){{
                    var t=byName[c.name];
                    var game=null;
                    completed.forEach(function(g){{
                        if(g.awayName===c.name||g.homeName===c.name||(g.awayTeam&&g.awayTeam.name===c.name)||(g.homeTeam&&g.homeTeam.name===c.name))game=g;
                    }});

                    if(!game){{
                        /* Moved without playing — other results shifted them */
                        if(c.prev===null){{
                            html+='<p>'+ref(c.name,t)+' enters the projected field as a new <strong>'+c.ns+'-seed</strong>.</p>';
                        }}else if(c.dir==='up'){{
                            html+='<p>'+ref(c.name,t)+' moves up from a '+c.prev+' to a <strong>'+c.ns+'-seed</strong> without playing — other results shifted the field in their favor.</p>';
                        }}else{{
                            html+='<p>'+ref(c.name,t)+' drops from a '+c.prev+' to a <strong>'+c.ns+'-seed</strong> despite not playing — other results pushed them down.</p>';
                        }}
                        return;
                    }}

                    var awayWon=game.awayScore>game.homeScore;
                    var margin=Math.abs(game.awayScore-game.homeScore);
                    var won=(game.awayName===c.name&&awayWon)||(game.homeName===c.name&&!awayWon)||
                        (game.awayTeam&&game.awayTeam.name===c.name&&awayWon)||(game.homeTeam&&game.homeTeam.name===c.name&&!awayWon);
                    var oppName=won?(awayWon?game.homeName:game.awayName):(awayWon?game.awayName:game.homeName);
                    var oppTeam=byName[oppName];
                    var score=game.awayScore>game.homeScore?game.awayScore+'-'+game.homeScore:game.homeScore+'-'+game.awayScore;
                    var venue=venueWord(game,c.name);

                    if(c.prev===null){{
                        html+='<p>'+ref(c.name,t)+' enters the projected field as a new <strong>'+c.ns+'-seed</strong> after '
                            +(won?'a '+score+' win over':'falling to')+' '+ref(oppName,oppTeam)+ctx(oppName)+venue+'.</p>';
                    }}else if(c.dir==='up'){{
                        html+='<p>'+ref(c.name,t)+' climbed from a '+c.prev+' to a <strong>'+c.ns+'-seed</strong> after '
                            +(won?winVerb(margin)+'ing':('a '+score+' loss to'))+' '+ref(oppName,oppTeam)+ctx(oppName)
                            +(won?', '+score+venue:venue)+'.</p>';
                    }}else{{
                        html+='<p>'+ref(c.name,t)+' dropped from a '+c.prev+' to a <strong>'+c.ns+'-seed</strong> after '
                            +(won?'a narrow '+score+' win over':'falling to')+' '+ref(oppName,oppTeam)+ctx(oppName)
                            +(won?venue:', '+score+venue)+'.</p>';
                    }}
                }});
            }}

            /* 2. Relevant completed games not already covered by movers */
            var coveredTeams={{}};
            changesArr.forEach(function(c){{coveredTeams[c.name]=true;}});

            var relevantGames=completed.filter(function(g){{
                if(!isRelevantGame(g))return false;
                var aCov=coveredTeams[g.awayName]||(g.awayTeam&&coveredTeams[g.awayTeam.name]);
                var bCov=coveredTeams[g.homeName]||(g.homeTeam&&coveredTeams[g.homeTeam.name]);
                if(aCov&&bCov)return false; /* both teams already narrated as movers */
                return true;
            }});

            /* Categorize */
            var upsetGames=[],bubbleGames=[],bracketGames=[];
            relevantGames.forEach(function(g){{
                var awayWon=g.awayScore>g.homeScore;
                var winTeam=awayWon?g.awayTeam:g.homeTeam;
                var loseTeam=awayWon?g.homeTeam:g.awayTeam;
                var isUpset=winTeam&&loseTeam&&(winTeam.net-loseTeam.net>=30);
                var bub=(isBubble(g.awayName)||isBubble(g.homeName)||(g.awayTeam&&isBubble(g.awayTeam.name))||(g.homeTeam&&isBubble(g.homeTeam.name)));

                if(isUpset)upsetGames.push(g);
                else if(bub)bubbleGames.push(g);
                else bracketGames.push(g);
            }});

            /* Upsets */
            if(upsetGames.length>0){{
                html+='<div class="summary-sub-header">Upsets</div>';
                upsetGames.forEach(function(g){{
                    var awayWon=g.awayScore>g.homeScore;
                    var winName=awayWon?g.awayName:g.homeName;
                    var loseName=awayWon?g.homeName:g.awayName;
                    var winTeam=byName[winName];
                    var loseTeam=byName[loseName];
                    var score=Math.max(g.awayScore,g.homeScore)+'-'+Math.min(g.awayScore,g.homeScore);
                    var venue=venueWord(g,winName);
                    html+='<p>'+ref(winName,winTeam)+ctx(winName)+' knocked off '
                        +ref(loseName,loseTeam)+ctx(loseName)+', '+score+venue
                        +'. <span class="summary-impact">'
                        +(loseTeam?'The '+loseName+' loss is a hit to their seed — '+(seeds[loseName]?'currently a '+seeds[loseName]+'-seed.':'watch for movement.'):'')
                        +'</span></p>';
                }});
            }}

            /* Bubble results */
            if(bubbleGames.length>0){{
                html+='<div class="summary-sub-header">Bubble Watch</div>';
                bubbleGames.forEach(function(g){{
                    var awayWon=g.awayScore>g.homeScore;
                    var winName=awayWon?g.awayName:g.homeName;
                    var loseName=awayWon?g.homeName:g.awayName;
                    var winTeam=byName[winName];
                    var loseTeam=byName[loseName];
                    var score=Math.max(g.awayScore,g.homeScore)+'-'+Math.min(g.awayScore,g.homeScore);
                    var venue=venueWord(g,winName);

                    var winBub=bubbleTeams[winName];
                    var loseBub=bubbleTeams[loseName];

                    if(winBub){{
                        var label=bubbleLabels[winBub];
                        var oppCtx=ctx(loseName);
                        html+='<p>'+ref(winName,winTeam)+' <span class="summary-ctx">('+label+')</span> picked up a '
                            +(loseTeam&&loseTeam.net<=50?'quality ':'')+'win over '+ref(loseName,loseTeam)+oppCtx
                            +', '+score+venue+'. <span class="summary-impact">'
                            +(winBub==='first_4_out'||winBub==='next_4_out'?'A much-needed result for their at-large case.':'A resume-builder that should help their standing.')
                            +'</span></p>';
                    }}else if(loseBub){{
                        var label2=bubbleLabels[loseBub];
                        html+='<p>'+ref(loseName,loseTeam)+' <span class="summary-ctx">('+label2+')</span> took a tough loss to '
                            +ref(winName,winTeam)+ctx(winName)+', '+score
                            +'. <span class="summary-impact">'
                            +(loseBub==='last_4_in'?'They could be in danger of slipping out of the field.':'That makes their path to an at-large bid even steeper.')
                            +'</span></p>';
                    }}
                }});
            }}

            /* Other bracket results */
            if(bracketGames.length>0){{
                html+='<div class="summary-sub-header">Around the Bracket</div>';
                /* Sort by winner seed (most important first) */
                bracketGames.sort(function(a,b){{
                    var awA=a.awayScore>a.homeScore?a.awayName:a.homeName;
                    var awB=b.awayScore>b.homeScore?b.awayName:b.homeName;
                    return (seeds[awA]||99)-(seeds[awB]||99);
                }});
                bracketGames.slice(0,8).forEach(function(g){{
                    var awayWon=g.awayScore>g.homeScore;
                    var winName=awayWon?g.awayName:g.homeName;
                    var loseName=awayWon?g.homeName:g.awayName;
                    var winTeam=byName[winName];
                    var loseTeam=byName[loseName];
                    var margin=Math.abs(g.awayScore-g.homeScore);
                    var score=Math.max(g.awayScore,g.homeScore)+'-'+Math.min(g.awayScore,g.homeScore);
                    var venue=venueWord(g,winName);
                    html+='<p>'+ref(winName,winTeam)+ctx(winName)+' '+winVerb(margin)+' '
                        +ref(loseName,loseTeam)+ctx(loseName)+', '+score+venue+'.</p>';
                }});
            }}

            if(completed.length===0||relevantGames.length===0&&changesArr.length===0){{
                html+='<p class="summary-empty">No games of note yesterday.</p>';
            }}

            html+='</div>';
            return html;
        }}

        /* ── Today's Preview ── */
        function renderPreview(data){{
            var games=parseGames(data);
            var html='<div class="summary-narrative">';

            /* Separate by state */
            var live=games.filter(function(g){{return g.state==='in'&&isRelevantGame(g);}});
            var upcoming=games.filter(function(g){{return g.state==='pre'&&isRelevantGame(g)&&g.pred;}});
            var done=games.filter(function(g){{return g.state==='post'&&isRelevantGame(g);}});

            /* Sort upcoming by watchability */
            upcoming.sort(function(a,b){{return (b.pred?b.pred.watch:0)-(a.pred?a.pred.watch:0);}});

            /* ── Live games ── */
            if(live.length>0){{
                html+='<div class="summary-sub-header">Live Now</div>';
                live.forEach(function(g){{
                    var awayLead=g.awayScore>g.homeScore;
                    var leadName=awayLead?g.awayName:g.homeName;
                    var trailName=awayLead?g.homeName:g.awayName;
                    var leadTeam=byName[leadName];
                    var trailTeam=byName[trailName];
                    var score=Math.max(g.awayScore,g.homeScore)+'-'+Math.min(g.awayScore,g.homeScore);
                    var det=g.detail?' ('+g.detail+')':'';
                    html+='<p>'+ref(leadName,leadTeam)+ctx(leadName)+' leads '
                        +ref(trailName,trailTeam)+ctx(trailName)+', '+score+det+'.';
                    if(g.pred){{
                        var favName=g.pred.spread>=0?g.awayName:g.homeName;
                        var pct=(Math.max(g.pred.winA,1-g.pred.winA)*100).toFixed(0);
                        html+=' <span class="summary-impact">Pre-game pick: '+favName+' '+pct+'%.</span>';
                    }}
                    html+='</p>';
                }});
            }}

            /* ── Coming up ── */
            if(upcoming.length>0){{
                html+='<div class="summary-sub-header">Games to Watch</div>';

                upcoming.forEach(function(g){{
                    var p=g.pred;
                    var favName=p.spread>=0?g.awayName:g.homeName;
                    var spreadAbs=Math.abs(p.spread).toFixed(1);
                    var pct=(Math.max(p.winA,1-p.winA)*100).toFixed(0);
                    var awayTeam=g.awayTeam;
                    var homeTeam=g.homeTeam;
                    var w=p.watch;
                    var hue=Math.round(w*1.2);

                    /* Build stakes description */
                    var stakes=[];
                    if(seeds[g.awayName]&&seeds[g.homeName]){{
                        stakes.push('A matchup between two projected tournament teams.');
                    }}
                    if(isBubble(g.awayName)){{
                        var bl=bubbleLabels[bubbleTeams[g.awayName]];
                        stakes.push(g.awayName+' ('+bl+') '+(seeds[g.homeName]?'has a chance at a signature win.':'needs this one.'));
                    }}
                    if(isBubble(g.homeName)){{
                        var bl2=bubbleLabels[bubbleTeams[g.homeName]];
                        stakes.push(g.homeName+' ('+bl2+') '+(seeds[g.awayName]?'has a chance at a signature win at home.':'needs this one at home.'));
                    }}

                    html+='<div class="summary-preview-card">'
                        +'<div class="summary-matchup-line">'
                            +ref(g.awayName,awayTeam)+ctx(g.awayName)
                            +' <span class="summary-ctx">at</span> '
                            +ref(g.homeName,homeTeam)+ctx(g.homeName)
                        +'</div>';
                    if(stakes.length>0){{
                        html+='<div class="summary-stakes">'+stakes.join(' ')+'</div>';
                    }}
                    html+='<div class="summary-pred-line">'
                        +'Prediction: '+favName+' by '+spreadAbs+' ('+pct+'%)'
                        +' &middot; <span style="color:hsl('+hue+',70%,50%);font-weight:600">Watchability: '+w+'</span>'
                        +'</div></div>';
                }});
            }}

            /* ── Completed today ── */
            if(done.length>0){{
                html+='<div class="summary-sub-header">Earlier Today</div>';
                done.forEach(function(g){{
                    var awayWon=g.awayScore>g.homeScore;
                    var winName=awayWon?g.awayName:g.homeName;
                    var loseName=awayWon?g.homeName:g.awayName;
                    var winTeam=byName[winName];
                    var loseTeam=byName[loseName];
                    var margin=Math.abs(g.awayScore-g.homeScore);
                    var score=Math.max(g.awayScore,g.homeScore)+'-'+Math.min(g.awayScore,g.homeScore);
                    var venue=venueWord(g,winName);

                    var winBub=bubbleTeams[winName];
                    var loseBub=bubbleTeams[loseName];
                    var impact='';
                    if(winBub)impact=' <span class="summary-impact">Good result for the '+bubbleLabels[winBub]+'.</span>';
                    else if(loseBub)impact=' <span class="summary-impact">Tough loss for the '+bubbleLabels[loseBub]+'.</span>';

                    html+='<p>'+ref(winName,winTeam)+ctx(winName)+' '+winVerb(margin)+' '
                        +ref(loseName,loseTeam)+ctx(loseName)+', '+score+venue+'.'+impact+'</p>';
                }});
            }}

            if(live.length===0&&upcoming.length===0&&done.length===0){{
                html+='<p class="summary-empty">No games of note on the schedule today.</p>';
            }}

            html+='</div>';
            return html;
        }}

        function renderLookahead(data){{
            var games=parseGames(data);
            var html='<div class="summary-narrative">';
            var upcoming=games.filter(function(g){{return isRelevantGame(g)&&g.pred;}});
            upcoming.sort(function(a,b){{return (b.pred?b.pred.watch:0)-(a.pred?a.pred.watch:0);}});

            if(upcoming.length>0){{
                html+='<div class="summary-sub-header">Games to Watch</div>';
                upcoming.forEach(function(g){{
                    var p=g.pred;
                    var favName=p.spread>=0?g.awayName:g.homeName;
                    var spreadAbs=Math.abs(p.spread).toFixed(1);
                    var pct=(Math.max(p.winA,1-p.winA)*100).toFixed(0);
                    var w=p.watch;
                    var hue=Math.round(w*1.2);

                    var stakes=[];
                    if(seeds[g.awayName]&&seeds[g.homeName]){{
                        stakes.push('A matchup between two projected tournament teams.');
                    }}
                    if(isBubble(g.awayName)){{
                        var bl=bubbleLabels[bubbleTeams[g.awayName]];
                        stakes.push(g.awayName+' ('+bl+') '+(seeds[g.homeName]?'has a chance at a signature win.':'needs this one.'));
                    }}
                    if(isBubble(g.homeName)){{
                        var bl2=bubbleLabels[bubbleTeams[g.homeName]];
                        stakes.push(g.homeName+' ('+bl2+') '+(seeds[g.awayName]?'has a chance at a signature win at home.':'needs this one at home.'));
                    }}

                    html+='<div class="summary-preview-card">'
                        +'<div class="summary-matchup-line">'
                            +ref(g.awayName,g.awayTeam)+ctx(g.awayName)
                            +' <span class="summary-ctx">at</span> '
                            +ref(g.homeName,g.homeTeam)+ctx(g.homeName)
                        +'</div>';
                    if(stakes.length>0){{
                        html+='<div class="summary-stakes">'+stakes.join(' ')+'</div>';
                    }}
                    html+='<div class="summary-pred-line">'
                        +'Prediction: '+favName+' by '+spreadAbs+' ('+pct+'%)'
                        +' &middot; <span style="color:hsl('+hue+',70%,50%);font-weight:600">Watchability: '+w+'</span>'
                        +'</div></div>';
                }});
            }} else {{
                html+='<p class="summary-empty">No notable games on the schedule.</p>';
            }}

            html+='</div>';
            return html;
        }}

        function fetchAndRender(){{
            var today=new Date();
            var yesterday=new Date(today);
            yesterday.setDate(yesterday.getDate()-1);
            var tomorrow=new Date(today);
            tomorrow.setDate(tomorrow.getDate()+1);
            var todayStr=fmtDate(today);
            var yesterdayStr=fmtDate(yesterday);
            var tomorrowStr=fmtDate(tomorrow);

            app.innerHTML='<div class="summary-loading">Loading daily summary...</div>';

            Promise.all([
                fetch(API+'?dates='+yesterdayStr+'&limit=200&groups=50').then(function(r){{return r.json();}}),
                fetch(API+'?dates='+todayStr+'&limit=200&groups=50').then(function(r){{return r.json();}}),
                fetch(API+'?dates='+tomorrowStr+'&limit=200&groups=50').then(function(r){{return r.json();}})
            ]).then(function(results){{
                var html='';
                html+='<div class="summary-section">';
                html+='<div class="summary-section-header">Yesterday &middot; '+displayDate(yesterday)+'</div>';
                html+=renderRecap(results[0]);
                html+='</div>';
                html+='<div class="summary-section">';
                html+='<div class="summary-section-header">Today &middot; '+displayDate(today)+'</div>';
                html+=renderPreview(results[1]);
                html+='</div>';
                html+='<div class="summary-section">';
                html+='<div class="summary-section-header">Tomorrow &middot; '+displayDate(tomorrow)+'</div>';
                html+=renderLookahead(results[2]);
                html+='</div>';
                app.innerHTML=html;
            }}).catch(function(){{
                app.innerHTML='<div class="summary-loading">Failed to load summary. Try refreshing.</div>';
            }});
        }}

        var summaryRadio=document.getElementById('tab-summary');
        function initSummary(){{
            if(!loaded){{loaded=true;fetchAndRender();}}
        }}
        if(summaryRadio){{
            summaryRadio.addEventListener('change',function(){{if(this.checked)initSummary();}});
            if(summaryRadio.checked)initSummary();
        }}
    }})();

    /* ── Team Profile Modal ── */
    (function(){{
        var overlay=document.getElementById('team-modal');
        var box=document.getElementById('modal-content');
        if(!overlay||!box)return;
        var teams=window.__SCORES_TEAMS__||[];

        function findTeam(name){{
            for(var i=0;i<teams.length;i++){{
                if(teams[i].name===name)return teams[i];
            }}
            return null;
        }}

        function espnLogo(id,sz){{
            if(!id)return '';
            sz=sz||40;
            return '<img src="https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/'+id+'.png&h='+sz+'&w='+sz+'" alt="" loading="lazy">';
        }}

        function statCard(label,val){{
            return '<div class="modal-stat"><div class="label">'+label+'</div><div class="value">'+val+'</div></div>';
        }}

        function hcaBar(label,val,maxAbs){{
            maxAbs=maxAbs||5;
            var pct=Math.min(Math.abs(val)/maxAbs*50,50);
            var cls=val>=0?'positive':'negative';
            var fill='<div class="hca-bar-fill '+cls+'" style="width:'+pct+'%"></div>';
            return '<div class="hca-bar-row">'
                +'<div class="hca-bar-label">'+label+'</div>'
                +'<div class="hca-bar-track"><div class="hca-bar-center"></div>'+fill+'</div>'
                +'<div class="hca-bar-val">'+(val>=0?'+':'')+val.toFixed(2)+'</div>'
                +'</div>';
        }}

        var schedCache={{}};
        var SCHED_CDN='https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/';
        var SCHED_API='https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/';

        function fetchTeamSchedule(t){{
            var container=document.getElementById('team-schedule-container');
            if(!container||!t.espn){{
                if(container)container.innerHTML='<div class="team-sched-loading">Schedule not available.</div>';
                return;
            }}
            if(schedCache[t.espn]){{
                container.innerHTML=renderSchedule(schedCache[t.espn],t);
                return;
            }}
            var now=new Date();
            var yr=now.getMonth()>=7?now.getFullYear()+1:now.getFullYear();
            fetch(SCHED_API+t.espn+'/schedule?season='+yr)
                .then(function(r){{return r.json();}})
                .then(function(data){{
                    schedCache[t.espn]=data;
                    var c=document.getElementById('team-schedule-container');
                    if(c)c.innerHTML=renderSchedule(data,t);
                }})
                .catch(function(){{
                    var c=document.getElementById('team-schedule-container');
                    if(c)c.innerHTML='<div class="team-sched-loading">Failed to load schedule.</div>';
                }});
        }}

        function renderSchedule(data,team){{
            var events=data.events||[];
            if(events.length===0)return '<div class="team-sched-loading">No games found.</div>';
            var months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
            var html='';
            events.forEach(function(ev){{
                var comp=ev.competitions&&ev.competitions[0];
                if(!comp)return;
                var competitors=comp.competitors||[];
                var us=null,them=null;
                var ourId=String(team.espn);
                competitors.forEach(function(c){{
                    var cId=c.team?String(c.team.id):String(c.id||'');
                    if(cId===ourId)us=c;
                    else them=c;
                }});
                if(!us||!them)return;
                var isHome=us.homeAway==='home';
                var oppName=them.team?(them.team.shortDisplayName||them.team.displayName||'TBD'):'TBD';
                var oppId=them.team?String(them.team.id):'';
                var oppLogo=oppId?'<img src="'+SCHED_CDN+oppId+'.png&h=24&w=24" alt="" loading="lazy">':'';
                var dateStr='';
                try{{
                    var d=new Date(ev.date||comp.date);
                    dateStr=months[d.getMonth()]+' '+d.getDate();
                }}catch(e){{}}
                var status=comp.status||{{}};
                var statusType=status.type||{{}};
                var state=statusType.state||'pre';
                var resultHtml='',resultCls='';
                function getScore(c){{
                    if(!c.score)return 0;
                    if(typeof c.score==='object')return parseInt(c.score.displayValue||c.score.value)||0;
                    return parseInt(c.score)||0;
                }}
                if(state==='post'){{
                    var usScore=getScore(us);
                    var themScore=getScore(them);
                    var won=us.winner===true||(us.winner===undefined&&usScore>themScore);
                    resultCls=won?'win':'loss';
                    resultHtml=(won?'W':'L')+' '+usScore+'-'+themScore;
                }}else if(state==='in'){{
                    resultHtml='<span style="color:var(--accent)">LIVE</span> '+getScore(us)+'-'+getScore(them);
                }}else{{
                    var detail=statusType.shortDetail||'';
                    resultHtml='<span style="color:var(--text-muted)">'+detail+'</span>';
                }}
                var venue=isHome?'vs':'@';
                var evId=ev.id||'';
                var canExpand=state==='post'||state==='in';
                html+='<div class="team-sched-row" data-eid="'+evId+'">';
                html+='<div class="team-sched-game">';
                html+='<div class="team-sched-date">'+dateStr+'</div>';
                html+='<div class="team-sched-venue">'+venue+'</div>';
                html+='<div class="team-sched-opp">'+oppLogo+'<span class="opp-name">'+oppName+'</span></div>';
                html+='<div class="team-sched-result '+resultCls+'">'+resultHtml+'</div>';
                html+='</div>';
                if(canExpand)html+='<div class="box-score" id="box-'+evId+'"></div>';
                html+='</div>';
            }});
            return html||'<div class="team-sched-loading">No games found.</div>';
        }}

        var boxCache={{}};
        var BOX_API='https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event=';
        var BOX_COLS=['MIN','PTS','FG','3PT','FT','REB','AST','STL','BLK','TO','PF'];
        var BOX_IDX=null;

        function toggleBoxScore(eid){{
            var el=document.getElementById('box-'+eid);
            if(!el)return;
            if(el.classList.contains('open')){{
                el.classList.remove('open');
                return;
            }}
            el.classList.add('open');
            if(boxCache[eid]){{
                el.innerHTML=renderBoxScore(boxCache[eid]);
                return;
            }}
            el.innerHTML='<div class="box-score-loading">Loading box score...</div>';
            fetch(BOX_API+eid)
                .then(function(r){{return r.json();}})
                .then(function(data){{
                    boxCache[eid]=data;
                    var b=document.getElementById('box-'+eid);
                    if(b)b.innerHTML=renderBoxScore(data);
                }})
                .catch(function(){{
                    var b=document.getElementById('box-'+eid);
                    if(b)b.innerHTML='<div class="box-score-loading">Failed to load box score.</div>';
                }});
        }}

        function renderBoxScore(data){{
            var bs=data.boxscore;
            if(!bs||!bs.players)return '<div class="box-score-loading">No box score available.</div>';
            var html='';
            bs.players.forEach(function(pt){{
                var tname=pt.team?(pt.team.shortDisplayName||pt.team.displayName||''):'';
                var tid=pt.team?String(pt.team.id):'';
                var tlogo=tid?'<img src="'+SCHED_CDN+tid+'.png&h=20&w=20" alt="" loading="lazy">':'';
                var stats=pt.statistics&&pt.statistics[0];
                if(!stats)return;
                var labels=stats.labels||[];
                /* Map our display columns to API label indices */
                if(!BOX_IDX){{
                    BOX_IDX={{}};
                    for(var i=0;i<labels.length;i++)BOX_IDX[labels[i]]=i;
                }}
                html+='<div class="box-score-team">';
                html+='<div class="box-team-header">'+tlogo+tname+'</div>';
                html+='<table class="box-table"><thead><tr><th>Player</th>';
                for(var c=0;c<BOX_COLS.length;c++)html+='<th>'+BOX_COLS[c]+'</th>';
                html+='</tr></thead><tbody>';
                var athletes=stats.athletes||[];
                var sawBench=false;
                athletes.forEach(function(a){{
                    var pname=a.athlete?(a.athlete.shortName||a.athlete.displayName||''):'';
                    var pstats=a.stats||[];
                    var isBench=a.starter===false;
                    var trCls=isBench?'box-bench':'';
                    if(isBench&&!sawBench){{
                        sawBench=true;
                        html+='<tr class="box-bench"><td colspan="'+(BOX_COLS.length+1)+'" style="color:var(--text-muted);font-size:0.65rem;padding-top:0.3rem;">BENCH</td></tr>';
                    }}
                    html+='<tr class="'+trCls+'"><td>'+pname+'</td>';
                    for(var c=0;c<BOX_COLS.length;c++){{
                        var idx=BOX_IDX[BOX_COLS[c]];
                        html+='<td>'+(idx!==undefined&&pstats[idx]!==undefined?pstats[idx]:'—')+'</td>';
                    }}
                    html+='</tr>';
                }});
                /* Totals row */
                var totals=stats.totals||[];
                if(totals.length){{
                    html+='<tr class="box-totals"><td>TOTAL</td>';
                    for(var c=0;c<BOX_COLS.length;c++){{
                        var idx=BOX_IDX[BOX_COLS[c]];
                        html+='<td>'+(idx!==undefined&&totals[idx]!==undefined?totals[idx]:'—')+'</td>';
                    }}
                    html+='</tr>';
                }}
                html+='</tbody></table></div>';
            }});
            return html||'<div class="box-score-loading">No box score available.</div>';
        }}

        function renderProfile(t){{
            var h='<button class="modal-close" id="modal-close-btn">&larr; Back</button>';
            h+='<div class="modal-header">';
            h+=espnLogo(t.espn,64);
            h+='<div class="team-info"><h2>'+t.name+'</h2>';
            h+='<div class="meta">'+t.conf+' &middot; '+t.rec+'</div></div></div>';

            h+='<div class="modal-stats-grid">';
            h+=statCard('NET',t.net);
            h+=statCard('KenPom',t.pom||'—');
            h+=statCard('SOR',t.sor||'—');
            h+=statCard('KPI',t.kpi||'—');
            h+=statCard('BPI',t.bpi||'—');
            h+=statCard('WAB',t.wab!==undefined?t.wab.toFixed(1):'—');
            h+=statCard('AdjOE',t.oe);
            h+=statCard('AdjDE',t.de);
            h+=statCard('Barthag',t.bar);
            h+='</div>';

            h+='<div class="modal-section"><h3>Records</h3>';
            h+='<div class="quad-grid">';
            h+='<div class="quad-cell"><div class="qlabel">Q1</div><div class="qval">'+(t.q1||'—')+'</div></div>';
            h+='<div class="quad-cell"><div class="qlabel">Q2</div><div class="qval">'+(t.q2||'—')+'</div></div>';
            h+='<div class="quad-cell"><div class="qlabel">Q3</div><div class="qval">'+(t.q3||'—')+'</div></div>';
            h+='<div class="quad-cell"><div class="qlabel">Q4</div><div class="qval">'+(t.q4||'—')+'</div></div>';
            h+='<div class="quad-cell"><div class="qlabel">Home</div><div class="qval">'+(t.homeRec||'—')+'</div></div>';
            h+='<div class="quad-cell"><div class="qlabel">Road</div><div class="qval">'+(t.roadRec||'—')+'</div></div>';
            h+='</div></div>';

            if(t.hcaPts!==undefined){{
                h+='<div class="modal-section"><h3>Home Court Advantage</h3>';
                h+='<div style="display:flex;gap:1rem;margin-bottom:0.5rem;">';
                h+=statCard('HCA Score',(t.hca*100).toFixed(0)+'%');
                h+=statCard('Point Adv',(t.hcaPts>=0?'+':'')+t.hcaPts.toFixed(1));
                h+='</div>';
                h+='<div class="hca-bars">';
                h+=hcaBar('Fouls',t.hcaFoul||0,4);
                h+=hcaBar('Scoring',t.hcaScoring||0,15);
                h+=hcaBar('TOs',t.hcaTO||0,4);
                h+=hcaBar('Blocks',t.hcaBlk||0,3);
                h+='</div>';
                if(t.homePtsM!==undefined){{
                    h+='<div style="display:flex;gap:1rem;margin-top:0.5rem;">';
                    h+=statCard('Home Pts Margin',(t.homePtsM>=0?'+':'')+t.homePtsM.toFixed(1));
                    h+=statCard('Road Pts Margin',(t.roadPtsM>=0?'+':'')+t.roadPtsM.toFixed(1));
                    h+='</div>';
                }}
                h+='</div>';
            }}

            h+='<div style="margin-top:1rem;">';
            h+='<button class="modal-btn" id="compare-btn" data-team="'+t.name+'">Compare with...</button>';
            h+='<div class="compare-search" id="compare-search" style="display:none;">';
            h+='<input type="text" id="compare-input" placeholder="Search team...">';
            h+='<div class="results" id="compare-results"></div>';
            h+='</div></div>';

            h+='<div class="modal-section" style="margin-top:1.5rem;"><h3>Schedule &amp; Results</h3>';
            h+='<div id="team-schedule-container"><div class="team-sched-loading">Loading schedule...</div></div>';
            h+='</div>';

            return h;
        }}

        function compareVal(a,b,lower){{
            /* Return 'better' class for the better value */
            if(a===b||a===undefined||b===undefined)return ['',''];
            var aBetter=lower?(a<b):(a>b);
            return aBetter?['better','']:['','better'];
        }}

        function cmpRow(label,aVal,bVal,lower){{
            var cls=compareVal(aVal,bVal,lower);
            var av=aVal!==undefined&&aVal!==null?String(aVal):'—';
            var bv=bVal!==undefined&&bVal!==null?String(bVal):'—';
            return '<div class="compare-stat-row">'
                +'<div class="stat-val left '+cls[0]+'">'+av+'</div>'
                +'<div class="stat-label">'+label+'</div>'
                +'<div class="stat-val right '+cls[1]+'">'+bv+'</div></div>';
        }}

        function renderCompare(a,b){{
            var h='<button class="modal-close" id="modal-close-btn">&larr; Back</button>';
            box.classList.add('compare-mode');

            h+='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">';
            h+='<div class="modal-header" style="margin-bottom:0;">'+espnLogo(a.espn,36)+'<div class="team-info"><h2 style="font-size:1rem;">'+a.name+'</h2><div class="meta">'+a.conf+' '+a.rec+'</div></div></div>';
            h+='<div style="font-size:1.2rem;color:var(--text-muted);font-weight:600;">vs</div>';
            h+='<div class="modal-header" style="margin-bottom:0;flex-direction:row-reverse;">'+espnLogo(b.espn,36)+'<div class="team-info" style="text-align:right;"><h2 style="font-size:1rem;">'+b.name+'</h2><div class="meta">'+b.conf+' '+b.rec+'</div></div></div>';
            h+='</div>';

            h+='<div class="modal-section"><h3>Rankings</h3>';
            h+=cmpRow('NET',a.net,b.net,true);
            h+=cmpRow('KenPom',a.pom,b.pom,true);
            h+=cmpRow('SOR',a.sor,b.sor,true);
            h+=cmpRow('KPI',a.kpi,b.kpi,true);
            h+=cmpRow('BPI',a.bpi,b.bpi,true);
            h+='</div>';

            h+='<div class="modal-section"><h3>Efficiency</h3>';
            h+=cmpRow('AdjOE',a.oe,b.oe,false);
            h+=cmpRow('AdjDE',a.de,b.de,true);
            h+=cmpRow('Barthag',a.bar,b.bar,false);
            h+=cmpRow('WAB',a.wab,b.wab,false);
            h+='</div>';

            h+='<div class="modal-section"><h3>Records</h3>';
            h+=cmpRow('Overall',a.rec,b.rec);
            h+=cmpRow('Q1',a.q1||'—',b.q1||'—');
            h+=cmpRow('Q2',a.q2||'—',b.q2||'—');
            h+=cmpRow('Home',a.homeRec||'—',b.homeRec||'—');
            h+=cmpRow('Road',a.roadRec||'—',b.roadRec||'—');
            h+='</div>';

            if(a.hcaPts!==undefined&&b.hcaPts!==undefined){{
                h+='<div class="modal-section"><h3>Home Court Advantage</h3>';
                h+=cmpRow('HCA Score',(a.hca*100).toFixed(0)+'%',(b.hca*100).toFixed(0)+'%',false);
                h+=cmpRow('Point Adv',a.hcaPts.toFixed(1),b.hcaPts.toFixed(1),false);
                h+=cmpRow('Fouls',a.hcaFoul!==undefined?a.hcaFoul.toFixed(2):'—',b.hcaFoul!==undefined?b.hcaFoul.toFixed(2):'—',false);
                h+=cmpRow('Scoring',a.hcaScoring!==undefined?a.hcaScoring.toFixed(1):'—',b.hcaScoring!==undefined?b.hcaScoring.toFixed(1):'—',false);
                h+='</div>';
            }}

            /* Matchup prediction at 3 venues */
            h+='<div class="compare-matchup"><h4>Matchup Prediction</h4>';
            var venues=[
                {{label:a.name+' Home',home:a}},
                {{label:'Neutral',home:null}},
                {{label:b.name+' Home',home:b}}
            ];
            var avgT=70;
            for(var v=0;v<venues.length;v++){{
                var venue=venues[v];
                var aOE=a.oe,aDE=a.de,bOE=b.oe,bDE=b.de;
                var hcaPts=0;
                if(venue.home===a)hcaPts=a.hcaPts!==undefined?a.hcaPts:(a.hca*6+1);
                else if(venue.home===b)hcaPts=-(b.hcaPts!==undefined?b.hcaPts:(b.hca*6+1));
                var aExpOE=(aOE+bDE)/2, bExpOE=(bOE+aDE)/2;
                var margin=aExpOE-bExpOE+hcaPts;
                var aScore=Math.round(avgT+margin/2);
                var bScore=Math.round(avgT-margin/2);
                var spread=margin>=0?a.name+' -'+Math.abs(margin).toFixed(1):b.name+' -'+Math.abs(margin).toFixed(1);
                h+='<div class="venue-row">';
                h+='<div class="venue">'+venue.label+'</div>';
                h+='<div class="spread">'+spread+'</div>';
                h+='<div class="winpct">'+aScore+'-'+bScore+'</div>';
                h+='</div>';
            }}
            h+='</div>';

            h+='<div style="margin-top:0.75rem;text-align:center;">';
            h+='<button class="modal-btn" id="back-btn" data-team="'+a.name+'">Back to '+a.name+'</button>';
            h+='</div>';

            return h;
        }}

        function closeModal(){{
            overlay.classList.remove('active');
            box.classList.remove('compare-mode');
            box.innerHTML='';
            document.body.style.overflow='';
        }}

        overlay.addEventListener('click',function(e){{
            if(e.target===overlay)closeModal();
        }});

        document.addEventListener('keydown',function(e){{
            if(e.key==='Escape')closeModal();
        }});

        box.addEventListener('click',function(e){{
            if(e.target.id==='modal-close-btn')closeModal();
            if(e.target.id==='back-btn'){{
                var name=e.target.dataset.team;
                if(name)openProfile(name);
            }}
            /* Box score toggle */
            var gameRow=e.target.closest('.team-sched-game');
            if(gameRow){{
                var row=gameRow.closest('.team-sched-row');
                if(row&&row.dataset.eid)toggleBoxScore(row.dataset.eid);
            }}
        }});

        function openProfile(name){{
            var t=findTeam(name);
            if(!t)return;
            box.classList.remove('compare-mode');
            box.innerHTML=renderProfile(t);
            overlay.classList.add('active');
            overlay.scrollTop=0;
            document.body.style.overflow='hidden';
            fetchTeamSchedule(t);

            var cmpBtn=document.getElementById('compare-btn');
            var cmpSearch=document.getElementById('compare-search');
            var cmpInput=document.getElementById('compare-input');
            var cmpResults=document.getElementById('compare-results');

            if(cmpBtn)cmpBtn.addEventListener('click',function(){{
                cmpSearch.style.display='block';
                cmpInput.focus();
            }});

            if(cmpInput)cmpInput.addEventListener('input',function(){{
                var q=this.value.toLowerCase();
                if(q.length<2){{cmpResults.classList.remove('show');return;}}
                var html='';
                for(var i=0;i<teams.length;i++){{
                    if(teams[i].name===name)continue;
                    if(teams[i].name.toLowerCase().indexOf(q)!==-1){{
                        html+='<div data-name="'+teams[i].name+'">'+espnLogo(teams[i].espn,20)+teams[i].name+'</div>';
                    }}
                }}
                cmpResults.innerHTML=html||'<div style="color:var(--text-muted)">No matches</div>';
                cmpResults.classList.add('show');
            }});

            if(cmpResults)cmpResults.addEventListener('click',function(e){{
                var el=e.target.closest('[data-name]');
                if(!el)return;
                var b=findTeam(el.dataset.name);
                if(!b)return;
                box.innerHTML=renderCompare(t,b);
            }});
        }}

        window.__openProfile=openProfile;

        /* Click delegation on .stats-team cells across all tabs */
        document.addEventListener('click',function(e){{
            var cell=e.target.closest('.stats-team');
            if(!cell)return;
            var name=cell.dataset.team||cell.textContent.trim();
            if(!name)return;
            e.preventDefault();
            openProfile(name);
        }});
    }})();
    /* AP Poll week selector */
    (function(){{
        var app=document.getElementById('ap-poll-app');
        if(!app||!window.__AP_POLL_DATA__)return;
        var D=window.__AP_POLL_DATA__;
        var weeks=D.weeks||{{}};
        var curWeek=D.current_week||1;
        var keys=Object.keys(weeks).map(Number).sort(function(a,b){{return a-b;}});
        if(!keys.length)return;

        function esc(s){{var d=document.createElement('div');d.textContent=s;return d.innerHTML;}}
        function logoImg(sid){{
            if(!sid||!window.__ESPN_LOGOS__)return '';
            var eid=window.__ESPN_LOGOS__[sid];
            if(!eid)return '';
            return '<img class="team-logo" src="https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/'+eid+'.png&h=40&w=40" alt="" loading="lazy">';
        }}

        function render(weekNum){{
            var w=weeks[String(weekNum)];
            if(!w)return;
            var ranks=w.ranks||[];
            var others=w.others||[];
            var label=w.week_label||('Week '+weekNum);
            var updated=w.updated||'';

            var subtitle=updated?'<p class="ap-subtitle">'+esc(label)+' &mdash; '+esc(updated)+'. Points are AP voter ballots (first-place votes in parentheses).</p>':'';

            var rows='';
            for(var i=0;i<ranks.length;i++){{
                var e=ranks[i];
                var rank=e.rank;
                var prev=e.previous||0;
                var logo=logoImg(e.school_id);
                var name=esc(e.team_name||'');
                var record=esc(e.record||'');
                var pts=parseInt(e.points)||0;
                var fpv=parseInt(e.first_place_votes)||0;
                var ptsStr=String(pts);
                if(fpv)ptsStr+=' <span class="ap-fpv">('+fpv+')</span>';

                var trend;
                if(prev===0)trend='<span class="ap-trend ap-new">NEW</span>';
                else if(rank<prev)trend='<span class="ap-trend ap-up">\u25b2'+(prev-rank)+'</span>';
                else if(rank>prev)trend='<span class="ap-trend ap-down">\u25bc'+(rank-prev)+'</span>';
                else trend='<span class="ap-trend ap-flat">\u2014</span>';

                var teamCell='<span class="stats-team" data-team="'+name+'">'+logo+name+'</span>';
                rows+='<tr><td>'+rank+'</td><td>'+teamCell+'</td><td>'+record+'</td><td>'+ptsStr+'</td><td>'+trend+'</td></tr>';
            }}

            var table='<div class="stats-scroll"><table class="stats-table ap-table">'
                +'<thead><tr><th data-sort="num">#</th><th data-sort="str">Team</th><th data-sort="str">Record</th>'
                +'<th data-sort="num">Points</th><th data-sort="num">Trend</th></tr></thead>'
                +'<tbody>'+rows+'</tbody></table></div>';

            var othersHtml='';
            if(others.length){{
                others.sort(function(a,b){{return (b.points||0)-(a.points||0);}});
                var items=[];
                for(var j=0;j<others.length;j++){{
                    var o=others[j];
                    var oLogo=logoImg(o.school_id);
                    var oName=esc(o.team_name||'');
                    items.push(oLogo+'<span class="stats-team" data-team="'+oName+'">'+oName+'</span> '+parseInt(o.points||0));
                }}
                othersHtml='<div class="ap-others"><strong>Others receiving votes:</strong> '+items.join(', ')+'</div>';
            }}

            var container=document.getElementById('ap-poll-content');
            if(container)container.innerHTML=subtitle+table+othersHtml;
        }}

        /* Build select dropdown */
        var sel=document.createElement('select');
        sel.className='ap-week-select';
        for(var k=keys.length-1;k>=0;k--){{
            var opt=document.createElement('option');
            opt.value=keys[k];
            var wData=weeks[String(keys[k])];
            opt.textContent=wData.week_label||('Week '+keys[k]);
            if(keys[k]===curWeek)opt.selected=true;
            sel.appendChild(opt);
        }}
        sel.addEventListener('change',function(){{render(parseInt(this.value));}});

        var content=document.createElement('div');
        content.id='ap-poll-content';
        app.appendChild(sel);
        app.appendChild(content);
        render(curWeek);
    }})();
    </script>
</body>
</html>"""

    return html


def build(changes: dict | None = None, stats_df=None, bubble: dict | None = None, model_type: str = "rf"):
    """Build the site from the latest predictions.

    Args:
        changes: dict mapping team_name -> {"direction": "up"|"down", "prev_seed": N, "new_seed": N}
        stats_df: DataFrame of all teams with stats columns.
        bubble: dict with "last_4_in", "first_4_out", "next_4_out" team name lists.
        model_type: "rf" or "xgb".
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
    standings_tab_html = ""
    autobid_tab_html = ""
    if stats_df is not None:
        stats_html = _build_stats_table(stats_df)
        conf_tab_html = _build_conf_tab(stats_df)
        standings_tab_html = _build_standings_tab(stats_df)
        autobid_tab_html = _build_autobid_tab(stats_df)

    # Compute last_4_byes if missing (older bubble.json files lack this field)
    if bubble and "last_4_byes" not in bubble and stats_df is not None:
        last_4_in_names = set(bubble.get("last_4_in", []))
        first_4_out_names = set(bubble.get("first_4_out", []))
        next_4_out_names = set(bubble.get("next_4_out", []))
        exclude = last_4_in_names | first_4_out_names | next_4_out_names
        # At-large teams that are in the field but not on the bubble edges
        candidates = stats_df[
            (stats_df["selection_prob"] > 0.5)
            & (~stats_df["team"].isin(exclude))
        ].sort_values("selection_prob")
        bubble["last_4_byes"] = candidates.head(4)["team"].tolist()

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

    ranking_tab_html = ""
    if stats_df is not None:
        ranking_tab_html = _build_power_rankings_tab(stats_df)

    # Build scores tab (interactive team picker)
    scores_tab_html = ""
    if stats_df is not None:
        scores_tab_html = _build_scores_tab(stats_df)

    # Build schedule tab
    schedule_tab_html = ""
    if stats_df is not None:
        schedule_tab_html = _build_schedule_tab(stats_df)

    # Build home court tab
    homecourt_tab_html = ""
    if stats_df is not None:
        homecourt_tab_html = _build_homecourt_tab(stats_df)

    # Build summary tab
    summary_tab_html = ""
    if stats_df is not None:
        summary_tab_html = _build_summary_tab(stats_df, changes, bubble, seed_rows_for_matrix)

    # Build AP Poll tab
    appoll_tab_html = _build_ap_poll_tab()

    os.makedirs(SITE_DIR, exist_ok=True)

    html = md_to_html(md_path, changes=changes, stats_html=stats_html, bubble_tab_html=bubble_tab_html, conf_tab_html=conf_tab_html, standings_tab_html=standings_tab_html, autobid_tab_html=autobid_tab_html, matrix_tab_html=matrix_tab_html, ranking_tab_html=ranking_tab_html, scores_tab_html=scores_tab_html, schedule_tab_html=schedule_tab_html, homecourt_tab_html=homecourt_tab_html, summary_tab_html=summary_tab_html, appoll_tab_html=appoll_tab_html, bubble=bubble, stats_df=stats_df, model_type=model_type)

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
