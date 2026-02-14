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
    """
    if stats_df is None:
        return '<p style="color: var(--text-muted);">Scores data not available. Run the predict command to generate.</p>'

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
        if pd.isna(adj_oe) or pd.isna(adj_de) or pd.isna(barthag):
            continue
        espn_id = _ESPN_LOGOS.get(str(r.get("school_id", "")), "")
        teams_json.append({
            "name": str(r.get("team", "")),
            "conf": str(r.get("conference", "")),
            "espn": espn_id,
            "oe": round(float(adj_oe), 1),
            "de": round(float(adj_de), 1),
            "bar": round(float(barthag), 4),
            "pace": round(float(pace), 1) if pd.notna(pace) else None,
            "net": int(net) if pd.notna(net) else 999,
            "rec": f"{int(wins)}-{int(losses)}" if pd.notna(wins) and pd.notna(losses) else "",
        })

    teams_json.sort(key=lambda t: t["net"])
    import json as _json
    blob = _json.dumps(teams_json, separators=(",", ":"))

    return f'<div id="scores-app"></div><script>window.__SCORES_TEAMS__={blob};</script>'


def _build_schedule_tab(stats_df: pd.DataFrame) -> str:
    """Build the Schedule tab container. All logic is client-side JS."""
    if stats_df is None:
        return '<p style="color: var(--text-muted);">Schedule data not available. Run the predict command to generate.</p>'
    return '<div id="schedule-app"></div>'


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


def md_to_html(md_path: str, changes: dict | None = None, stats_html: str = "", bubble_tab_html: str = "", conf_tab_html: str = "", autobid_tab_html: str = "", matrix_tab_html: str = "", ranking_tab_html: str = "", scores_tab_html: str = "", schedule_tab_html: str = "", bubble: dict | None = None, stats_df=None, model_type: str = "rf") -> str:
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
                f'<span class="bubble-team">{_team_logo(_name_to_sid.get(t, ""))}{escape(t)}</span>'
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
            styled.append(logo + _style_team(clean_name, changes, is_play_in))
        seed_table_rows += f"<tr><td class='seed-num'>{seed}</td><td>{', '.join(styled)}</td></tr>\n"

    first_four_html = ""
    for game in first_four:
        # Add logos to First Four: "(seed) Team1 vs Team2 [Region]"
        ff_match = re.match(r'\((\d+)\) (.+?) vs (.+?) \[(.+?)\]', game)
        if ff_match:
            ff_seed, ff_t1, ff_t2, ff_region = ff_match.groups()
            ff_logo1 = _team_logo(_name_to_sid.get(ff_t1.strip(), ""))
            ff_logo2 = _team_logo(_name_to_sid.get(ff_t2.strip(), ""))
            first_four_html += f"<li>({ff_seed}) {ff_logo1}{escape(ff_t1)} vs {ff_logo2}{escape(ff_t2)} [{ff_region}]</li>\n"
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
            return f'{logo}({seed}) {safe_name} '
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
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
        }}
        .tab-bar::-webkit-scrollbar {{ display: none; }}
        .tab-bar label {{
            padding: 0.75rem 1rem;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-muted);
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: color 0.15s, border-color 0.15s;
            white-space: nowrap;
            flex-shrink: 0;
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
        #tab-ranking:checked ~ .tab-bar label[for="tab-ranking"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-scores:checked ~ .tab-bar label[for="tab-scores"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-schedule:checked ~ .tab-bar label[for="tab-schedule"] {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        #tab-bracket:checked ~ #panel-bracket {{ display: block; }}
        #tab-stats:checked ~ #panel-stats {{ display: block; }}
        #tab-bubble:checked ~ #panel-bubble {{ display: block; }}
        #tab-conf:checked ~ #panel-conf {{ display: block; }}
        #tab-autobid:checked ~ #panel-autobid {{ display: block; }}
        #tab-matrix:checked ~ #panel-matrix {{ display: block; }}
        #tab-ranking:checked ~ #panel-ranking {{ display: block; }}
        #tab-scores:checked ~ #panel-scores {{ display: block; }}
        #tab-schedule:checked ~ #panel-schedule {{ display: block; }}

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
        .sched-team-meta {{
            font-size: 0.75rem;
            color: var(--text-muted);
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
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span class="logo">68</span>bracket</h1>
            <div class="subtitle">{title}</div>
            <div class="timestamp">Model: {model_label} &middot; Last Updated: {timestamp}</div>
        </header>

        <input type="radio" name="tabs" id="tab-bracket" class="tab-radio" checked>
        <input type="radio" name="tabs" id="tab-stats" class="tab-radio">
        <input type="radio" name="tabs" id="tab-bubble" class="tab-radio">
        <input type="radio" name="tabs" id="tab-conf" class="tab-radio">
        <input type="radio" name="tabs" id="tab-autobid" class="tab-radio">
        <input type="radio" name="tabs" id="tab-matrix" class="tab-radio">
        <input type="radio" name="tabs" id="tab-ranking" class="tab-radio">
        <input type="radio" name="tabs" id="tab-scores" class="tab-radio">
        <input type="radio" name="tabs" id="tab-schedule" class="tab-radio">

        <div class="tab-bar">
            <label for="tab-bracket">Bracket</label>
            <label for="tab-ranking">Power Rankings</label>
            <label for="tab-scores">Scores</label>
            <label for="tab-schedule">Schedule</label>
            <label for="tab-bubble">Bubble Watch</label>
            <label for="tab-autobid">Auto Bids</label>
            <label for="tab-matrix">Bracket Matrix</label>
            <label for="tab-conf">Conferences</label>
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

        <div class="footnote">
            Predictions generated by <a href="https://github.com/hunterwalklin/68bracket">68bracket</a>
            — updated daily
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
        var HCA=3.5; /* home court advantage in points */
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

            /* Apply home court advantage */
            if(venue==='homeA')spread+=HCA;
            else if(venue==='homeB')spread-=HCA;

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
                +'<div class="scores-team-card">'+imgA+'<div class="tc-name">'+a.name+'</div><div class="tc-meta">'+a.conf+' &middot; '+a.rec+'</div><div class="tc-pct" style="color:hsl('+hueA+',70%,50%)">'+pctA+'%</div></div>'
                +'<div class="scores-vs">vs</div>'
                +'<div class="scores-team-card">'+imgB+'<div class="tc-name">'+b.name+'</div><div class="tc-meta">'+b.conf+' &middot; '+b.rec+'</div><div class="tc-pct" style="color:hsl('+hueB+',70%,50%)">'+pctB+'%</div></div>'
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

        /* Build ESPN ID lookup */
        var byEspn={{}};
        teams.forEach(function(t){{if(t.espn)byEspn[String(t.espn)]=t;}});

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
            /* Home court advantage */
            if(homeId){{
                if(String(homeId)===String(a.espn)){{ptsA+=1.75;ptsB-=1.75;}}
                else if(String(homeId)===String(b.espn)){{ptsB+=1.75;ptsA-=1.75;}}
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

            var withPred=games.filter(function(g){{return g.pred!==null;}}).length;
            var html='<div class="sched-summary">'+games.length+' games'+
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

                html+='<div class="sched-game'+liveCls+'">'
                    +watchHtml
                    +'<div class="sched-team away">'
                        +'<div class="sched-team-info"><div class="sched-team-name">'+g.awayName+'</div>'+(awayMeta?'<div class="sched-team-meta">'+awayMeta+'</div>':'')+'</div>'
                        +awayImg
                    +'</div>'
                    +'<div class="sched-center">'+centerHtml+'</div>'
                    +'<div class="sched-team home">'
                        +homeImg
                        +'<div class="sched-team-info"><div class="sched-team-name">'+g.homeName+'</div>'+(homeMeta?'<div class="sched-team-meta">'+homeMeta+'</div>':'')+'</div>'
                    +'</div>'
                    +'</div>';
            }});

            if(games.length===0){{
                html='<div class="sched-summary">No games scheduled for this date.</div>';
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
            if(prev)prev.onclick=function(){{shiftDate(-1);}};
            if(next)next.onclick=function(){{shiftDate(1);}};
            if(today)today.onclick=function(){{loadDate(fmtDate(new Date()));}};
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
    autobid_tab_html = ""
    if stats_df is not None:
        stats_html = _build_stats_table(stats_df)
        conf_tab_html = _build_conf_tab(stats_df)
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

    os.makedirs(SITE_DIR, exist_ok=True)

    html = md_to_html(md_path, changes=changes, stats_html=stats_html, bubble_tab_html=bubble_tab_html, conf_tab_html=conf_tab_html, autobid_tab_html=autobid_tab_html, matrix_tab_html=matrix_tab_html, ranking_tab_html=ranking_tab_html, scores_tab_html=scores_tab_html, schedule_tab_html=schedule_tab_html, bubble=bubble, stats_df=stats_df, model_type=model_type)

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
