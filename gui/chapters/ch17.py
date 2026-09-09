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
        "title":          "Chapter 17 — Multi-Agent RL: Independent Q-Learning (IQL)",
        "subtitle":       "Warsaw ASP · 5 Independent Agents · Individual Rewards · No Communication",
        "engine_missing": "❌ Rust engine not found. Run: cd rlvr-py && maturin develop --release",
        "sidebar_title":  "Ch17 Settings",
        "n_tech":         "Technicians (Agents)",
        "n_orders":       "Work Orders",
        "n_episodes":     "Episodes",
        "alpha":          "α (learning rate)",
        "gamma":          "γ (discount factor)",
        "epsilon_start":  "ε start",
        "epsilon_end":    "ε end",
        "seed":           "Random seed",
        "run_btn":        "▶ Run IQL Training",
        "map_title":      "🗺️ Warsaw Dispatch Map",
        "map_caption":    "Blue = Technicians · White = Pending · Dark green = SLA met · Red = SLA breach",
        "step_slider":    "🔍 Highlight step on map",
        "glass_title":    "🔬 Glass-Box — IQL Step Trace",
        "curve_title":    "📈 Team Learning Curve — Gₜ over Episodes (MA-5)",
        "agent_curve_title": "📊 Per-Agent Learning Curves",
        "summary_title":  "📋 Episode Summary",
        "agent_title":    "🤖 Agent Stats",
        "qtable_title":   "🧮 Q-Tables (Final)",
    }


# ---------------------------------------------------------------------------
# Handbook
# ---------------------------------------------------------------------------
def _render_handbook():
    col1, col2 = st.columns([8, 1])
    with col1:
        st.subheader("Hands-On Guide — Chapter 17 (EN)")
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "..", "..", "docs", "handson_ch17_en.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            html = f.read()
        with col2:
            st.download_button("💾 Save", data=html,
                               file_name="handson_ch17_en.html", mime="text/html")
        st.components.v1.html(html, height=4000, scrolling=True)
    else:
        st.info("📘 Hands-On Guide for Ch17 coming soon.")


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
def _render_map(steps, sel, tx):
    techs = {}
    for s in steps:
        if s["tech_idx"] not in techs:
            techs[s["tech_idx"]] = (s["tech_x"], s["tech_y"])
    for s in steps[:sel]:
        techs[s["tech_idx"]] = (s["order_x"], s["order_y"])

    orders = {}
    for s in steps:
        orders[s["order_idx"]] = (s["order_x"], s["order_y"])

    # Agent colors
    agent_colors = ["#0082F0", "#FF8C0A", "#8B5CF6", "#0FC373", "#FF4B4B",
                    "#00BCD4", "#E91E63", "#9C27B0", "#4CAF50", "#FF5722"]

    fig = go.Figure()

    # Technicians — each agent has its own color
    for k, v in techs.items():
        color = agent_colors[k % len(agent_colors)]
        fig.add_trace(go.Scattermapbox(
            lat=[v[1]], lon=[v[0]],
            mode="markers+text",
            marker=dict(size=17, color=color),
            text=[f"T{k}"], textposition="top right",
            textfont=dict(size=12, color=color),
            name=f"T{k}",
        ))

    # Work orders
    completed  = {}
    dispatched = {}
    for s in steps[:sel + 1]:
        completed[s["order_idx"]]  = s.get("sla_met", False)
        dispatched[s["order_idx"]] = s["tech_idx"]

    for k, v in orders.items():
        if k not in completed:
            # Black outer + white inner
            fig.add_trace(go.Scattermapbox(
                lat=[v[1]], lon=[v[0]], mode="markers",
                marker=dict(size=14, color="#000000"),
                name=f"W{k}_border", showlegend=False,
            ))
            fig.add_trace(go.Scattermapbox(
                lat=[v[1]], lon=[v[0]], mode="markers+text",
                marker=dict(size=10, color="white"),
                text=[f"W{k}"], textposition="top right",
                textfont=dict(size=12, color="#000000"),
                name=f"W{k}", showlegend=False,
            ))
        elif completed[k]:
            tech_color = agent_colors[dispatched[k] % len(agent_colors)]
            fig.add_trace(go.Scattermapbox(
                lat=[v[1]], lon=[v[0]], mode="markers+text",
                marker=dict(size=12, color="#006400"),
                text=[f"W{k} (T{dispatched[k]} ✅)"], textposition="top right",
                textfont=dict(size=12, color="#006400"),
                name=f"W{k}", showlegend=False,
            ))
        else:
            fig.add_trace(go.Scattermapbox(
                lat=[v[1]], lon=[v[0]], mode="markers+text",
                marker=dict(size=12, color="#FF4B4B"),
                text=[f"W{k} (T{dispatched[k]} ❌)"], textposition="top right",
                textfont=dict(size=12, color="#FF4B4B"),
                name=f"W{k}", showlegend=False,
            ))

    # Travel line
    if sel < len(steps):
        s = steps[sel]
        line_color = "#0FC373" if s.get("sla_met") else "#FF4B4B"
        label = "✅ SLA met" if s.get("sla_met") else "❌ SLA breach"
        fig.add_trace(go.Scattermapbox(
            lat=[s["tech_y"], s["order_y"]],
            lon=[s["tech_x"], s["order_x"]],
            mode="lines",
            line=dict(width=3, color=line_color),
            name=f"Step {sel}: T{s['tech_idx']}→W{s['order_idx']} ({label})",
        ))

    # Auto-fit zoom
    all_lats = [v[1] for v in techs.values()] + [v[1] for v in orders.values()]
    all_lons = [v[0] for v in techs.values()] + [v[0] for v in orders.values()]
    lat_c = (min(all_lats) + max(all_lats)) / 2
    lon_c = (min(all_lons) + max(all_lons)) / 2
    span  = max((max(all_lats) - min(all_lats)) * 1.2,
                (max(all_lons) - min(all_lons)) * 1.2, 0.05)
    zoom  = max(9, min(13, round(8.5 - math.log2(span * 111))))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=lat_c, lon=lon_c), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Reward per Step
# ---------------------------------------------------------------------------
def _render_reward_per_step(steps, sel):
    rewards = [s["reward"] for s in steps]
    eff_sel = sel if sel < len(rewards) else None
    colors  = ["#0FC373" if r >= 0 else "#FF4B4B" for r in rewards]
    if eff_sel is not None:
        colors[eff_sel] = "#FFD700"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(steps))),
        y=rewards,
        marker_color=colors,
        hovertemplate="Step %{x}<br>Reward: %{y:+.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    if eff_sel is not None:
        fig.add_vline(x=eff_sel, line_dash="dot", line_color="#FFD700", line_width=2)
    fig.update_layout(
        xaxis_title="Step", yaxis_title="Reward R",
        height=220, margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# TD Error per Step
# ---------------------------------------------------------------------------
def _render_td_error(steps, sel):
    td_errors = [s["td_error"] for s in steps]
    eff_sel   = sel if sel < len(td_errors) else None
    colors    = ["#8B5CF6"] * len(td_errors)
    if eff_sel is not None:
        colors[eff_sel] = "#FFD700"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(steps))),
        y=td_errors,
        marker_color=colors,
        hovertemplate="Step %{x}<br>TD Error: %{y:+.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    if eff_sel is not None:
        fig.add_vline(x=eff_sel, line_dash="dot", line_color="#FFD700", line_width=2)
    fig.update_layout(
        xaxis_title="Step", yaxis_title="TD Error δ",
        height=220, margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("TD Error δ = R + γ·max Q(s',a') − Q(s,a) — converges to 0 as agent learns")


# ---------------------------------------------------------------------------
# Glass-Box
# ---------------------------------------------------------------------------
def _render_glass_box(steps, sel, tx):
    rows = []
    for i, s in enumerate(steps):
        rows.append({
            "Step":            i,
            "Agent":           f"T{s['tech_idx']}",
            "Work Order":      f"W{s['order_idx']}",
            "Action":          "explore" if s.get("explored") else "exploit",
            "ε":               f"{s.get('epsilon', 0):.3f}",
            "Reward":          round(s["reward"], 3),
            "Gₜ":              round(s["gt"], 3),
            "SLA":             "✅" if s.get("sla_met") else "❌",
            "Task vs Agent":   f"{s.get('order_skill','?')} vs {s.get('tech_skill','?')}",
            "Skill Match":     "✅" if s.get("skill_match") else "❌",
            "Distance":        f"{s.get('distance_km', 0):.1f} km",
            "Q before":        round(s.get("q_before", 0), 4),
            "Q after":         round(s.get("q_after", 0), 4),
            "TD Error":        round(s.get("td_error", 0), 4),
        })
    df = pd.DataFrame(rows)
    eff_sel = sel if sel < len(rows) else len(rows) - 1

    def highlight_row(row):
        return ["background-color: #252840"] * len(row) if row["Step"] == eff_sel else [""] * len(row)

    st.dataframe(df.style.apply(highlight_row, axis=1),
                 use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# Team Learning Curve (MA-5)
# ---------------------------------------------------------------------------
def _render_team_curve(curve, tx):
    ma5 = []
    for i in range(len(curve)):
        w = curve[max(0, i - 4):i + 1]
        ma5.append(sum(w) / len(w))
    mean_gt = sum(curve) / len(curve)
    df = pd.DataFrame({"Episode": list(range(len(curve))), "Gₜ": curve, "MA-5": ma5})
    raw  = alt.Chart(df).mark_line(opacity=0.3, color="#0082F0").encode(
        x="Episode:Q", y=alt.Y("Gₜ:Q", title="Team Gₜ"),
        tooltip=["Episode", alt.Tooltip("Gₜ:Q", format=".3f")])
    ma   = alt.Chart(df).mark_line(color="#0082F0", strokeWidth=2).encode(
        x="Episode:Q", y="MA-5:Q",
        tooltip=["Episode", alt.Tooltip("MA-5:Q", format=".3f")])
    mean = alt.Chart(pd.DataFrame({"m": [mean_gt]})).mark_rule(
        color="#FF8C0A", strokeDash=[6, 3]).encode(y="m:Q")
    st.altair_chart((raw + ma + mean).properties(height=260), use_container_width=True)
    st.caption(
        f"Blue = raw Gₜ · Bold = MA-5 · Orange = mean {mean_gt:.2f} · "
        f"📌 IQL: expect upward trend as each agent learns independently"
    )


# ---------------------------------------------------------------------------
# Per-Agent Learning Curves
# ---------------------------------------------------------------------------
def _render_agent_curves(agent_curves, n_tech):
    agent_colors = ["#0082F0", "#FF8C0A", "#8B5CF6", "#0FC373", "#FF4B4B",
                    "#00BCD4", "#E91E63", "#9C27B0", "#4CAF50", "#FF5722"]
    fig = go.Figure()
    for t in range(n_tech):
        if t < len(agent_curves) and agent_curves[t]:
            curve = agent_curves[t]
            ma5   = []
            for i in range(len(curve)):
                w = curve[max(0, i - 4):i + 1]
                ma5.append(sum(w) / len(w))
            color = agent_colors[t % len(agent_colors)]
            fig.add_trace(go.Scatter(
                x=list(range(len(curve))), y=ma5,
                mode="lines", name=f"T{t} (MA-5)",
                line=dict(color=color, width=2),
                hovertemplate=f"T{t} Ep %{{x}}<br>Gₜ=%{{y:.3f}}<extra></extra>",
            ))
    fig.update_layout(
        xaxis_title="Episode", yaxis_title="Agent Gₜ (MA-5)",
        height=280, margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each line = one independent agent (technician). IQL: agents learn without coordination.")


# ---------------------------------------------------------------------------
# Agent Stats Table
# ---------------------------------------------------------------------------
def _render_agent_stats(agent_stats):
    rows = []
    for a in agent_stats:
        rows.append({
            "Agent":         f"T{a['tech_idx']}",
            "Skill":         a["tech_skill"],
            "Orders Served": a["orders_served"],
            "Total Gₜ":      round(a["total_gt"], 3),
            "SLA Rate":      f"{a['sla_rate']*100:.1f}%",
            "Skill Match":   f"{a['skill_rate']*100:.1f}%",
            "Avg Distance":  f"{a['avg_distance']:.1f} km",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Q-Table Heatmap
# ---------------------------------------------------------------------------
def _render_qtable(final_q_tables, n_tech, n_orders):
    # Reshape flat array back to [n_tech][n_orders]
    q = []
    for t in range(n_tech):
        row = final_q_tables[t * n_orders:(t + 1) * n_orders]
        q.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=q,
        x=[f"W{i}" for i in range(n_orders)],
        y=[f"T{i}" for i in range(n_tech)],
        colorscale="RdYlGn",
        zmid=0,
        hovertemplate="Agent T%{y}<br>Order W%{x}<br>Q = %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Work Order",
        yaxis_title="Agent (Technician)",
        height=300,
        margin=dict(l=60, r=20, t=20, b=40),
        paper_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = high Q-value (agent learned this order is profitable) · Red = low Q-value · Each row = one independent agent's Q-table")


# ---------------------------------------------------------------------------
# Episode Summary
# ---------------------------------------------------------------------------
def _render_summary(steps, total_gt, team_sla):
    n = len(steps)
    skill_rate = sum(1 for s in steps if s.get("skill_match")) / max(n, 1)
    avg_dist   = sum(s.get("distance_km", 0) for s in steps) / max(n, 1)
    avg_reward = sum(s.get("reward", 0) for s in steps) / max(n, 1)
    exp_rate   = sum(1 for s in steps if s.get("explored")) / max(n, 1)

    df = pd.DataFrame([
        {"Metric": "Team Total Gₜ",    "Value": f"{total_gt:.3f}"},
        {"Metric": "Team SLA Rate",    "Value": f"{team_sla*100:.1f}%"},
        {"Metric": "Skill Match Rate", "Value": f"{skill_rate*100:.1f}%"},
        {"Metric": "Exploration Rate", "Value": f"{exp_rate*100:.1f}%"},
        {"Metric": "Avg Distance",     "Value": f"{avg_dist:.1f} km"},
        {"Metric": "Avg Step Reward",  "Value": f"{avg_reward:.3f}"},
        {"Metric": "Steps",            "Value": str(n)},
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


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
        n_tech         = st.sidebar.slider(tx["n_tech"],        2, 10,  5)
        n_orders       = st.sidebar.slider(tx["n_orders"],      4, 20, 10)
        n_ep           = st.sidebar.slider(tx["n_episodes"],    5, 200, 50)
        alpha          = st.sidebar.slider(tx["alpha"],         0.01, 1.0, 0.1, 0.01)
        gamma          = st.sidebar.slider(tx["gamma"],         0.5,  1.0, 0.95, 0.01)
        epsilon_start  = st.sidebar.slider(tx["epsilon_start"], 0.1,  1.0, 1.0, 0.05)
        epsilon_end    = st.sidebar.slider(tx["epsilon_end"],   0.0,  0.5, 0.05, 0.01)
        seed           = st.sidebar.number_input(tx["seed"],    0, 9999, 42)

        st.sidebar.caption(
            "IQL: each technician is an independent agent. "
            "ε decays from ε_start to ε_end over all episodes. "
            "Each agent updates only its own Q-table."
        )

        # ── Run button ────────────────────────────────────────────────────
        if st.button(tx["run_btn"], type="primary"):
            with st.spinner(f"Running {n_ep} IQL episodes..."):
                raw = rlvr_py.run_ch17_episode(
                    int(seed), int(n_tech), int(n_orders), int(n_ep),
                    float(alpha), float(gamma),
                    float(epsilon_start), float(epsilon_end)
                )
            data = json.loads(raw) if isinstance(raw, str) else raw
            st.session_state["ch17_data"]    = data
            st.session_state["ch17_n_tech"]  = n_tech
            st.session_state["ch17_n_orders"] = n_orders

        if "ch17_data" not in st.session_state:
            st.info("Configure settings and click **▶ Run IQL Training**.")
            return

        data     = st.session_state["ch17_data"]
        n_tech   = st.session_state["ch17_n_tech"]
        n_orders = st.session_state["ch17_n_orders"]

        episodes       = data["episodes"]
        curve          = data["curve"]
        agent_curves   = data["agent_curves"]
        final_q_tables = data["final_q_tables"]
        n_eps          = len(episodes)

        # ── IQL info ──────────────────────────────────────────────────────
        mean_g0 = sum(curve) / len(curve)
        st.info(
            f"**Ch17 — Independent Q-Learning (IQL).** "
            f"Each of the {n_tech} technicians is an independent agent with its own Q-table. "
            f"Agents learn without communication. "
            f"ε decays from {epsilon_start:.2f} → {epsilon_end:.2f} over {n_eps} episodes. "
            f"Mean team Gₜ = {mean_g0:.2f}"
        )

        # ── Episode selector ──────────────────────────────────────────────
        ep_sel   = st.slider("🎬 Select episode to inspect", 0, n_eps - 1, n_eps - 1, key="ep_sel")
        ep_data  = episodes[ep_sel]
        ep_steps = ep_data["steps"]
        ep_gt    = ep_data["total_gt"]
        ep_sla   = ep_data["team_sla_rate"]
        ep_eps   = ep_steps[0]["epsilon"] if ep_steps else 0.0

        st.caption(
            f"Episode {ep_sel + 1}/{n_eps} — "
            f"Team Gₜ = **{ep_gt:.3f}** · "
            f"SLA = {ep_sla*100:.1f}% · "
            f"ε = {ep_eps:.3f}"
        )

        # ── Step selector ─────────────────────────────────────────────────
        n_steps = len(ep_steps)
        sel = st.slider(tx["step_slider"], 0, n_steps, 0, key="step_sel")
        if sel == n_steps:
            st.caption("📍 All work orders dispatched — no travel line shown.")

        # ── Map ───────────────────────────────────────────────────────────
        st.subheader(tx["map_title"])
        _render_map(ep_steps, sel, tx)
        st.caption(tx["map_caption"])

        # ── Reward per Step ───────────────────────────────────────────────
        st.subheader("📊 Reward per Step")
        _render_reward_per_step(ep_steps, sel)

        # ── TD Error ──────────────────────────────────────────────────────
        st.subheader("📉 TD Error per Step")
        _render_td_error(ep_steps, sel)

        # ── Glass-Box ─────────────────────────────────────────────────────
        st.subheader(tx["glass_title"])
        _render_glass_box(ep_steps, sel, tx)

        # ── Episode Summary ───────────────────────────────────────────────
        st.subheader(tx["summary_title"])
        _render_summary(ep_steps, ep_gt, ep_sla)

        # ── Agent Stats ───────────────────────────────────────────────────
        st.subheader(tx["agent_title"])
        _render_agent_stats(ep_data["agent_stats"])

        # ── Per-Agent Learning Curves ─────────────────────────────────────
        st.subheader(tx["agent_curve_title"])
        _render_agent_curves(agent_curves, n_tech)

        # ── Team Learning Curve ───────────────────────────────────────────
        st.subheader(tx["curve_title"])
        _render_team_curve(curve, tx)

        # ── Q-Table Heatmap ───────────────────────────────────────────────
        st.subheader(tx["qtable_title"])
        _render_qtable(final_q_tables, n_tech, n_orders)
