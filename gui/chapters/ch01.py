import streamlit as st
import json
import os
import math
import pandas as pd
import plotly.graph_objects as go
import altair as alt

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
def _tx():
    return {
        "title":          "Chapter 01 — MDP Baseline & ε-Greedy Dispatch",
        "subtitle":       "Warsaw ASP · 5 Technicians · Up to 20 Work Orders · Q=0 Baseline",
        "engine_missing": "❌ Rust engine not found. Run: cd rlvr-py && maturin develop --release",
        "sidebar_title":  "Ch01 Settings",
        "n_tech":         "Technicians",
        "n_orders":       "Work Orders",
        "n_episodes":     "Episodes (Learning Curve)",
        "epsilon":        "ε (exploration rate)",
        "gamma":          "γ (discount factor)",
        "seed":           "Random seed",
        "run_btn":        "▶ Run All Episodes",
        "map_title":      "🗺️ Warsaw Dispatch Map",
        "map_caption":    "Blue = Technicians · Red markers = Work Orders · Green line = SLA met · Red line = SLA breach",
        "step_slider":    "🔍 Highlight step on map",
        "glass_title":    "🔬 Glass-Box — MDP Trace",
        "curve_title":    "📈 Learning Curve — Gₜ over Episodes (MA-5)",
        "summary_title":  "📋 Episode Summary Table",
        "metric_gt":      "Total Gₜ",
        "metric_sla":     "SLA Rate",
        "metric_skill":   "Skill Match",
        "metric_explore": "Exploration Rate",
        "metric_dist":    "Avg Distance",
        "col_step":       "Step",
        "col_tech":       "Technician",
        "col_order":      "Work Order",
        "col_reward":     "Reward",
        "col_gt":         "Gₜ",
        "col_sla":        "SLA Met",
        "col_skill":      "Skill Match",
        "col_dist":       "Distance",
        "col_action":     "Action (ε-greedy)",
    }


# ---------------------------------------------------------------------------
# Handbook tab
# ---------------------------------------------------------------------------
def _render_handbook():
    col1, col2 = st.columns([8, 1])
    with col1:
        st.subheader("Hands-On Guide — Chapter 01 (EN)")
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "..", "..", "docs", "handson_ch01_en.html")
    with open(path, encoding="utf-8") as f:
        html = f.read()
    with col2:
        st.download_button("💾 Save", data=html,
                           file_name="handson_ch01_en.html", mime="text/html")
    st.components.v1.html(html, height=4000, scrolling=True)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
def _render_map(steps, sel, tx):
    # Orders: fixed positions (all visible)
    orders = {}
    for s in steps:
        orders[s["order_idx"]] = (s["order_x"], s["order_y"])

    # Technician positions at step `sel`:
    # - for steps 0..sel: technician is at the work order location (after dispatch)
    # - for steps not yet dispatched: technician is at initial position (step 0)
    # Initial positions = position at step 0 for each technician
    techs = {}
    for s in steps:
        if s["tech_idx"] not in techs:
            techs[s["tech_idx"]] = (s["tech_x"], s["tech_y"])  # initial position

    # Update positions for all steps up to (but not including) sel
    for s in steps[:sel]:
        # After step s, technician moves to order location
        techs[s["tech_idx"]] = (s["order_x"], s["order_y"])

    fig = go.Figure()

    # Technicians
    fig.add_trace(go.Scattermapbox(
        lat=[v[1] for v in techs.values()],
        lon=[v[0] for v in techs.values()],
        mode="markers+text",
        marker=dict(size=17, color="#0082F0"),
        text=[f"T{k}" for k in techs.keys()],
        textposition="top right",
        textfont=dict(size=12, color="#0082F0"),
        name="Technicians",
    ))

    # Work orders — color depends on completion status at step sel
    completed  = {}
    dispatched = {}
    for s in steps[:sel + 1]:
        completed[s["order_idx"]]  = s.get("sla_met", False)
        dispatched[s["order_idx"]] = s["tech_idx"]

    for k, v in orders.items():
        if k not in completed:
            # Not yet dispatched — white circle with black border
            marker_color = "white"
            border_color = "#000000"
            text_color   = "#000000"
            label        = f"W{k}"
        elif completed[k]:
            # SLA met — dark green, no border
            marker_color = "#006400"
            border_color = "#006400"
            text_color   = "#006400"
            label        = f"W{k} (T{dispatched[k]} ✅)"
        else:
            # SLA breach — red, no border
            marker_color = "#FF4B4B"
            border_color = "#FF4B4B"
            text_color   = "#FF4B4B"
            label        = f"W{k} (T{dispatched[k]} ❌)"

        fig.add_trace(go.Scattermapbox(
            lat=[v[1]], lon=[v[0]],
            mode="markers+text",
            marker=dict(size=12, color=marker_color,
                       allowoverlap=True),
            text=[label], textposition="top right",
            textfont=dict(size=12, color=text_color),
            name=f"W{k}", showlegend=False,
        ))

    # Highlight selected step — green=SLA met, red=SLA breach
    # sel==len(steps) means "show all delivered, no travel line"
    if sel < len(steps):
        s = steps[sel]
        line_color = "#0FC373" if s.get("sla_met") else "#FF4B4B"
        label = "✅ SLA met" if s.get("sla_met") else "❌ SLA breach"
        fig.add_trace(go.Scattermapbox(
            lat=[s["tech_y"], s["order_y"]],
            lon=[s["tech_x"], s["order_x"]],
            mode="lines+markers",
            line=dict(width=3, color=line_color),
            marker=dict(size=12, color=line_color),
            name=f"Step {sel}: T{s['tech_idx']}→W{s['order_idx']} ({label})",
        ))

    # Auto-fit zoom — 40% padding so all markers are always visible
    all_lats = [v[1] for v in techs.values()] + [v[1] for v in orders.values()]
    all_lons = [v[0] for v in techs.values()] + [v[0] for v in orders.values()]
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_c = (lat_min + lat_max) / 2
    lon_c = (lon_min + lon_max) / 2
    lat_span = (lat_max - lat_min) * 1.2 or 0.05
    lon_span = (lon_max - lon_min) * 1.2 or 0.05
    span = max(lat_span, lon_span)
    zoom = max(9, min(13, round(8.5 - math.log2(span * 111))))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=lat_c, lon=lon_c), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Reward per Step Chart
# ---------------------------------------------------------------------------
def _render_reward_per_step(steps, sel):
    rewards = [s["reward"] for s in steps]
    colors  = ["#0FC373" if r >= 0 else "#FF4B4B" for r in rewards]
    # Highlight selected step with brighter color (clamp to valid range)
    if sel < len(rewards):
        colors[sel] = "#FFD700" if rewards[sel] >= 0 else "#FF8C00"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(steps))),
        y=rewards,
        marker_color=colors,
        hovertemplate="Step %{x}<br>Reward: %{y:+.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    if sel < len(rewards):
        fig.add_vline(x=sel, line_dash="dot", line_color="#FFD700", line_width=2)
    else:
        pass
    fig.update_layout(
        xaxis_title="Step", yaxis_title="Reward R",
        height=250, margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    if sel < len(rewards):
        st.caption(f"Step {sel}: R = {rewards[sel]:+.3f} · Green = SLA met · Red = SLA breach · Yellow = selected step")
    else:
        st.caption("All steps completed · Green = SLA met · Red = SLA breach")


# ---------------------------------------------------------------------------
# Discounted Return Gt per Step Chart
# ---------------------------------------------------------------------------
def _render_gt_per_step(steps, sel):
    gts    = [s["gt"] for s in steps]
    eff_sel = min(sel, len(steps) - 1)
    colors = ["#FFD700" if i == eff_sel else "#8B5CF6" for i in range(len(steps))]
    sizes  = [12 if i == eff_sel else 6 for i in range(len(steps))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(steps))),
        y=gts,
        mode="lines+markers",
        line=dict(color="#8B5CF6", width=2),
        marker=dict(size=sizes, color=colors),
        hovertemplate="Step %{x}<br>Gₜ = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    if sel < len(rewards):
        if sel < len(gts):
        fig.add_vline(x=sel, line_dash="dot", line_color="#FFD700", line_width=2)
    fig.update_layout(
        xaxis_title="Step t", yaxis_title="Discounted Return Gₜ",
        height=250, margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    if sel < len(gts):
        st.caption(f"Step {sel}: Gₜ = {gts[sel]:.3f} · G₀ = {gts[0]:.3f} · Yellow = selected step · Gₜ = Rₜ + γRₜ₊₁ + …")
    else:
        st.caption(f"All steps completed · G₀ = {gts[0]:.3f} · Gₜ = Rₜ + γRₜ₊₁ + …")


# ---------------------------------------------------------------------------
# Glass-Box — MDP Trace
# ---------------------------------------------------------------------------
def _render_glass_box(steps, sel, tx, gamma):
    rows = []
    for i, s in enumerate(steps):
        tech_skill  = s.get("tech_skill", "?")
        order_skill = s.get("order_skill", "?")
        rows.append({
            tx["col_step"]:   i,
            tx["col_tech"]:   f"T{s['tech_idx']}",
            tx["col_order"]:  f"W{s['order_idx']}",
            tx["col_action"]: "explore" if s.get("explored") else "greedy",
            tx["col_reward"]: round(s["reward"], 3),
            tx["col_gt"]:     round(s["gt"], 3),
            tx["col_sla"]:    "✅" if s.get("sla_met") else "❌",
            "Task vs Tech Skill": f"{order_skill} vs {tech_skill}",
            tx["col_skill"]:  "✅" if s.get("skill_match") else "❌",
            tx["col_dist"]:   f"{s.get('distance_km', 0):.1f} km",
        })
    df = pd.DataFrame(rows)

    def highlight_row(row):
        return ["background-color: #252840"] * len(row) if row[tx["col_step"]] == sel else [""] * len(row)

    st.dataframe(df.style.apply(highlight_row, axis=1),
                 use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# Episode Summary Table
# ---------------------------------------------------------------------------
def _render_summary(steps, total_gt, tx):
    n = len(steps)
    sla_rate   = sum(1 for s in steps if s.get("sla_met"))     / max(n, 1)
    skill_rate = sum(1 for s in steps if s.get("skill_match")) / max(n, 1)
    exp_rate   = sum(1 for s in steps if s.get("explored"))    / max(n, 1)
    avg_dist   = sum(s.get("distance_km", 0) for s in steps)      / max(n, 1)
    avg_reward = sum(s.get("reward", 0) for s in steps)        / max(n, 1)

    df = pd.DataFrame([
        {"Metric": tx["metric_gt"],      "Value": f"{total_gt:.3f}"},
        {"Metric": tx["metric_sla"],     "Value": f"{sla_rate*100:.1f}% ({int(sla_rate*n)}/{n})"},
        {"Metric": tx["metric_skill"],   "Value": f"{skill_rate*100:.1f}%"},
        {"Metric": tx["metric_explore"], "Value": f"{exp_rate*100:.1f}%"},
        {"Metric": tx["metric_dist"],    "Value": f"{avg_dist:.1f} km"},
        {"Metric": "Avg Step Reward",    "Value": f"{avg_reward:.3f}"},
        {"Metric": "Steps",              "Value": str(n)},
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Learning Curve (MA-5)
# ---------------------------------------------------------------------------
def _render_curve(curve, tx):
    ma5 = []
    for i in range(len(curve)):
        window = curve[max(0, i - 4):i + 1]
        ma5.append(sum(window) / len(window))

    mean_gt = sum(curve) / len(curve)

    df = pd.DataFrame({
        "Episode": list(range(len(curve))),
        "Gₜ":      curve,
        "MA-5":    ma5,
    })

    raw_line = alt.Chart(df).mark_line(opacity=0.3, color="#0082F0").encode(
        x="Episode:Q",
        y=alt.Y("Gₜ:Q", title="Total Discounted Return Gₜ"),
        tooltip=["Episode", alt.Tooltip("Gₜ:Q", format=".3f")],
    )
    ma_line = alt.Chart(df).mark_line(color="#0082F0", strokeWidth=2).encode(
        x="Episode:Q",
        y="MA-5:Q",
        tooltip=["Episode", alt.Tooltip("MA-5:Q", format=".3f")],
    )
    mean_line = alt.Chart(
        pd.DataFrame({"mean": [mean_gt]})
    ).mark_rule(color="#FF8C0A", strokeDash=[6, 3]).encode(y="mean:Q")

    st.altair_chart(
        (raw_line + ma_line + mean_line).properties(height=280),
        use_container_width=True
    )
    st.caption(
        f"Blue = raw Gₜ · Bold blue = MA-5 · Orange dashed = mean G₀ = {mean_gt:.3f} "
        f"(Ch01 baseline for Ch02–Ch18). "
        f"📌 **Expected: no upward trend** — Q=0, agent never learns."
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render():
    tx = _tx()
    st.title(tx["title"])
    st.caption(tx["subtitle"])

    tab_main, tab_handbook = st.tabs(["🔬 Interactive Lab", "📘 Hands-On Guide EN"])
    with tab_handbook:
        _render_handbook()
    with tab_main:
        try:
            import rlvr_py
        except ImportError:
            st.error(tx["engine_missing"])
            return

        # ── Sidebar ───────────────────────────────────────────────────────
        st.sidebar.header(tx["sidebar_title"])
        n_tech   = st.sidebar.slider(tx["n_tech"],     2, 10, 5)
        n_orders = st.sidebar.slider(tx["n_orders"],   4, 20, 10)
        n_ep     = st.sidebar.slider(tx["n_episodes"], 5, 100, 30)
        # Ch01: ε=1.0 fixed — pure random policy, no learning
        st.sidebar.slider(tx["epsilon"], 0.0, 1.0, 1.0, 0.05, disabled=True)
        epsilon = 1.0
        st.sidebar.caption("ε is fixed at 1.0 in Ch01 — pure random baseline. "
                           "Q-table learning starts in Ch06.")
        gamma    = st.sidebar.slider(tx["gamma"],      0.5, 1.0, 0.95, 0.01)
        seed     = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

        # ── Run button ────────────────────────────────────────────────────
        if st.button(tx["run_btn"], type="primary"):
            all_eps    = []
            curve_data = []
            with st.spinner(f"Running {n_ep} episodes..."):
                for ep in range(n_ep):
                    raw = rlvr_py.run_ch01_episode(
                        int(seed) + ep,   # seed+ep = same env, different policy
                        int(n_tech), int(n_orders),
                        float(epsilon), float(gamma)
                    )
                    ep_data = json.loads(raw) if isinstance(raw, str) else raw
                    all_eps.append(ep_data.get("steps", []))
                    curve_data.append(ep_data.get("total_gt", 0.0))
            st.session_state["ch01_all_episodes"] = all_eps
            st.session_state["ch01_curve"]        = curve_data

        # ── Check if data available ───────────────────────────────────────
        if "ch01_all_episodes" not in st.session_state:
            st.info("Configure settings and click **▶ Run All Episodes**.")
            return

        all_episodes = st.session_state["ch01_all_episodes"]
        curve        = st.session_state["ch01_curve"]
        n_eps        = len(all_episodes)

        # ── No-learning notice ────────────────────────────────────────────
        mean_g0 = sum(curve) / len(curve)
        st.info(
            "**Ch01 — No learning across episodes.** "
            "Q=0 throughout: the agent never updates its Q-table. "
            "Each episode uses the same environment (fixed positions) but a different "
            "random policy seed — expect **no upward trend** in the Learning Curve. "
            f"Baseline G₀ = {mean_g0:.2f} — reference for Ch02–Ch18."
        )

        # ── Episode selector ──────────────────────────────────────────────
        ep_sel   = st.slider("🎬 Select episode to inspect",
                             0, n_eps - 1, 0, key="ep_sel")
        ep_steps = all_episodes[ep_sel]
        ep_gt    = curve[ep_sel]
        st.caption(
            f"Episode {ep_sel + 1}/{n_eps} — "
            f"Total Gₜ = **{ep_gt:.3f}** · Mean G₀ = {mean_g0:.3f}"
        )

        # ── Step selector ─────────────────────────────────────────────────
        n_steps = len(ep_steps)
        # Extra step at end (sel==n_steps) shows all delivered, no travel line
        sel = st.slider(tx["step_slider"], 0, n_steps, 0, key="step_sel")

        # ── Warsaw Dispatch Map ───────────────────────────────────────────
        st.subheader(tx["map_title"])
        _render_map(ep_steps, sel, tx)
        st.caption(tx["map_caption"])

        # ── Reward per Step Chart ─────────────────────────────────────────
        st.subheader("📊 Reward per Step Chart")
        _render_reward_per_step(ep_steps, sel)

        # ── Discounted Return Gt Chart ────────────────────────────────────
        st.subheader("📈 Discounted Return Gₜ Chart")
        _render_gt_per_step(ep_steps, sel)

        # ── Glass-Box — MDP Trace ─────────────────────────────────────────
        st.subheader(tx["glass_title"])
        _render_glass_box(ep_steps, sel, tx, gamma)

        # ── Episode Summary Table ─────────────────────────────────────────
        st.subheader(tx["summary_title"])
        _render_summary(ep_steps, ep_gt, tx)

        # ── Learning Curve (MA-5) — always at the bottom ──────────────────
        st.subheader(tx["curve_title"])
        _render_curve(curve, tx)
