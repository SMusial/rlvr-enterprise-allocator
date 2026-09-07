import streamlit as st
import json
import os

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
        "run_episode":    "▶ Run Single Episode",
        "run_curve":      "📈 Run Learning Curve",
        "map_title":      "🗺️ Warsaw Dispatch Map",
        "map_caption":    "Blue = Technicians · Red = Work Orders · Green line = assigned dispatch",
        "step_slider":    "Step to highlight on map",
        "glass_title":    "🔬 Glass-Box — Step Trace",
        "curve_title":    "📈 Learning Curve — Gₜ over Episodes",
        "summary_title":  "📋 Episode Summary",
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
    import plotly.graph_objects as go

    techs  = {}
    orders = {}
    for s in steps:
        techs[s["tech_idx"]]   = (s["tech_x"],  s["tech_y"])
        orders[s["order_idx"]] = (s["order_x"], s["order_y"])

    t_lats = [v[1] for v in techs.values()]
    t_lons = [v[0] for v in techs.values()]
    t_text = [f"T{k}" for k in techs.keys()]

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=t_lats, lon=t_lons, mode="markers+text",
        marker=dict(size=14, color="#0082F0"),
        text=t_text, textposition="top right",
        name="Technicians",
    ))

    # work orders
    o_lats  = [v[1] for v in orders.values()]
    o_lons  = [v[0] for v in orders.values()]
    o_text  = [f"W{k}" for k in orders.keys()]

    for k in orders.keys():
        color = "#FF4B4B"
        fig.add_trace(go.Scattermapbox(
            lat=[orders[k][1]], lon=[orders[k][0]],
            mode="markers+text",
            marker=dict(size=10, color=color),
            text=[f"W{k}"], textposition="top right",
            name=f"W{k}", showlegend=False,
        ))

    # highlight selected step
    if sel < len(steps):
        s = steps[sel]
        fig.add_trace(go.Scattermapbox(
            lat=[s["tech_y"], s["order_y"]],
            lon=[s["tech_x"], s["order_x"]],
            mode="lines+markers",
            line=dict(width=3, color="#0FC373"),
            marker=dict(size=10, color="#0FC373"),
            name=f"Step {sel}: T{s['tech_idx']}→W{s['order_idx']}",
        ))

    # Auto-fit: center + zoom from all points with padding
    all_lats = [v[1] for v in techs.values()] + [v[1] for v in orders.values()]
    all_lons = [v[0] for v in techs.values()] + [v[0] for v in orders.values()]
    lat_c = (min(all_lats) + max(all_lats)) / 2
    lon_c = (min(all_lons) + max(all_lons)) / 2
    import math as _math
    span  = max(max(all_lats) - min(all_lats), max(all_lons) - min(all_lons), 0.001) * 2.5
    zoom  = max(10, min(13, round(7.0 - _math.log2(span))))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat_c, lon=lon_c),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Reward per Step Chart
# ---------------------------------------------------------------------------
def _render_reward_per_step(steps):
    import plotly.graph_objects as go
    rewards = [s["reward"] for s in steps]
    colors  = ["#0FC373" if r >= 0 else "#FF4B4B" for r in rewards]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(steps))),
        y=rewards,
        marker_color=colors,
        hovertemplate="Step %{x}<br>Reward: %{y:+.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Reward R",
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = positive reward (SLA met, skill match) · Red = penalty (SLA breach, skill mismatch, distance)")


# ---------------------------------------------------------------------------
# Discounted Return Gt per Step Chart
# ---------------------------------------------------------------------------
def _render_gt_per_step(steps, tx):
    import plotly.graph_objects as go
    gts = [s["gt"] for s in steps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(steps))),
        y=gts,
        mode="lines+markers",
        line=dict(color="#8B5CF6", width=2),
        marker=dict(size=6, color="#8B5CF6"),
        hovertemplate="Step %{x}<br>Gₜ = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        xaxis_title="Step t",
        yaxis_title="Discounted Return Gₜ",
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Gₜ = Rₜ + γRₜ₊₁ + γ²Rₜ₊₂ + … — computed backward from episode end. Step 0 has the highest Gₜ (most future rewards ahead).")


# ---------------------------------------------------------------------------
# Glass-Box
# ---------------------------------------------------------------------------
def _render_glass_box(steps, sel, tx, gamma):
    st.caption(f"Showing episode step trace — {len(steps)} steps total.")
    rows = []
    for i, s in enumerate(steps):
        rows.append({
            tx["col_step"]:    i,
            tx["col_tech"]:    f"T{s['tech_idx']}",
            tx["col_order"]:   f"W{s['order_idx']}",
            tx["col_action"]:  "explore" if s.get("explored") else "greedy",
            tx["col_reward"]:  round(s["reward"], 3),
            tx["col_gt"]:      round(s["gt"], 3),
            tx["col_sla"]:     "✅" if s.get("sla_met") else "❌",
            tx["col_skill"]:   "✅" if s.get("skill_match") else "❌",
            tx["col_dist"]:    f"{s.get('distance', 0):.1f} km",
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# Learning Curve
# ---------------------------------------------------------------------------
def _render_curve(curve, tx):
    import altair as alt
    import pandas as pd
    df = pd.DataFrame({
        "Episode": list(range(len(curve))),
        "Gₜ":      curve,
    })
    mean_gt = sum(curve) / len(curve)
    chart = alt.Chart(df).mark_line(color="#0082F0", opacity=0.8).encode(
        x=alt.X("Episode:Q"),
        y=alt.Y("Gₜ:Q", title="Total Discounted Return Gₜ"),
        tooltip=["Episode", alt.Tooltip("Gₜ:Q", format=".3f")],
    )
    mean_line = alt.Chart(pd.DataFrame({"mean": [mean_gt]})).mark_rule(
        color="#FF8C0A", strokeDash=[6, 3]
    ).encode(y="mean:Q")
    st.altair_chart((chart + mean_line).properties(height=280), use_container_width=True)
    st.caption(f"Orange dashed line = mean Gₜ = {mean_gt:.3f} — this is the Ch01 baseline G₀ for all future chapters.")


# ---------------------------------------------------------------------------
# Episode Summary
# ---------------------------------------------------------------------------
def _render_summary(steps, result, sla_rate, skill_rate, exp_rate,
                    avg_dist, avg_reward, sla_saved, total_gt, tx):
    import pandas as pd
    n = len(steps)
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| {tx['metric_gt']} | **{total_gt:.3f}** |
| {tx['metric_sla']} | **{sla_rate*100:.1f}%** ({int(sla_rate*n)}/{n} orders) |
| {tx['metric_skill']} | **{skill_rate*100:.1f}%** |
| {tx['metric_explore']} | **{exp_rate*100:.1f}%** |
| {tx['metric_dist']} | **{avg_dist:.1f} km** |
| Avg Step Reward | **{avg_reward:.3f}** |
""")


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
        n_tech    = st.sidebar.slider(tx["n_tech"],     2, 10, 5)
        n_orders  = st.sidebar.slider(tx["n_orders"],   4, 20, 10)
        n_ep      = st.sidebar.slider(tx["n_episodes"], 5, 100, 30)
        epsilon   = st.sidebar.slider(tx["epsilon"],    0.0, 1.0, 1.0, 0.05)
        gamma     = st.sidebar.slider(tx["gamma"],      0.5, 1.0, 0.95, 0.01)
        seed      = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

        col_run, col_curve = st.columns(2)
        run_ep    = col_run.button(tx["run_episode"],  type="primary")
        run_curve = col_curve.button(tx["run_curve"],  type="primary")

        # ── Run single episode ────────────────────────────────────────────
        if run_ep:
            with st.spinner("Running episode..."):
                raw = rlvr_py.run_ch01_episode(
                    int(seed), int(n_tech), int(n_orders),
                    float(epsilon), float(gamma)
                )
            result = json.loads(raw) if isinstance(raw, str) else raw
            st.session_state["ch01_result"]   = result
            st.session_state["ch01_steps"]    = result.get("steps", [])
            st.session_state["ch01_seed"]     = seed
            st.session_state["ch01_n_tech"]   = n_tech
            st.session_state["ch01_n_orders"] = n_orders
            st.session_state["ch01_gamma"]    = gamma
            st.session_state["ch01_epsilon"]  = epsilon

        # ── Run learning curve ────────────────────────────────────────────
        if run_curve:
            with st.spinner(f"Running {n_ep} episodes..."):
                curve_data = []
                for ep in range(n_ep):
                    raw = rlvr_py.run_ch01_episode(
                        int(seed), int(n_tech), int(n_orders),
                        float(epsilon), float(gamma)
                    )
                    ep_res = json.loads(raw) if isinstance(raw, str) else raw
                    curve_data.append(ep_res["total_gt"])
            st.session_state["ch01_curve"]    = curve_data
            st.session_state["ch01_seed"]     = seed
            st.session_state["ch01_n_tech"]   = n_tech
            st.session_state["ch01_n_orders"] = n_orders
            st.session_state["ch01_gamma"]    = gamma
            st.session_state["ch01_epsilon"]  = epsilon
            # Also store first episode steps for display
            raw = rlvr_py.run_ch01_episode(
                int(seed), int(n_tech), int(n_orders),
                float(epsilon), float(gamma)
            )
            result = json.loads(raw) if isinstance(raw, str) else raw
            st.session_state["ch01_result"] = result
            st.session_state["ch01_steps"]  = result.get("steps", [])

        if "ch01_steps" not in st.session_state:
            st.info("Configure settings and click **▶ Run Single Episode** or **📈 Run Learning Curve**.")
            return

        steps  = st.session_state["ch01_steps"]
        result = st.session_state["ch01_result"]
        curve  = st.session_state.get("ch01_curve", [])

        # ── Compute episode metrics ───────────────────────────────────────
        n_steps    = len(steps)
        total_gt   = result.get("total_gt", 0)
        sla_rate   = sum(1 for s in steps if s.get("sla_met"))   / max(n_steps, 1)
        skill_rate = sum(1 for s in steps if s.get("skill_match")) / max(n_steps, 1)
        exp_rate   = sum(1 for s in steps if s.get("explored"))  / max(n_steps, 1)
        avg_dist   = sum(s.get("distance", 0) for s in steps)    / max(n_steps, 1)
        avg_reward = sum(s.get("reward", 0) for s in steps)      / max(n_steps, 1)
        sla_saved  = sum(1 for s in steps if s.get("sla_met"))

        # ── KPI row ───────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(tx["metric_gt"],      f"{total_gt:.1f}")
        c2.metric(tx["metric_sla"],     f"{sla_rate*100:.1f}%")
        c3.metric(tx["metric_skill"],   f"{skill_rate*100:.1f}%")
        c4.metric(tx["metric_explore"], f"{exp_rate*100:.1f}%")
        c5.metric(tx["metric_dist"],    f"{avg_dist:.1f} km")

        # ── Episode selector (drives all charts below) ────────────────────
        if curve:
            n_eps = len(curve)
            st.info(
                "**Ch01 — No learning across episodes.** "
                "Q=0 throughout: the agent never updates its Q-table. "
                "Each episode is an independent random dispatch — expect **no upward trend** "
                "in the Learning Curve. The flat curve IS the result: it proves that without "
                f"learning, performance stays random. "
                f"Baseline G₀ = {sum(curve)/len(curve):.2f} — reference for Ch02–Ch18."
            )
            ep_sel = st.slider(
                "🎬 Select episode to inspect (updates all charts below)",
                0, n_eps - 1, 0, key="gb_ep_sel"
            )
            raw_ep = rlvr_py.run_ch01_episode(
                int(seed), int(n_tech), int(n_orders),
                float(epsilon), float(gamma)
            )
            ep_data   = json.loads(raw_ep) if isinstance(raw_ep, str) else raw_ep
            ep_steps  = ep_data.get("steps", [])
            ep_result = ep_data
            ep_gt     = curve[ep_sel]
            st.caption(
                f"Episode {ep_sel + 1}/{n_eps} — "
                f"Total Gₜ = **{ep_gt:.3f}** · "
                f"Mean across all episodes = {sum(curve)/len(curve):.3f}"
            )
        else:
            ep_steps  = steps
            ep_result = result
            ep_gt     = total_gt

        # ── Step slider ───────────────────────────────────────────────────
        ep_n_steps = len(ep_steps)
        sel = st.slider(tx["step_slider"], 0, max(ep_n_steps - 1, 0), 0)

        # ── Map ───────────────────────────────────────────────────────────
        st.subheader(tx["map_title"])
        _render_map(ep_steps, sel, tx)
        st.caption(tx["map_caption"])

        # ── Reward per Step ───────────────────────────────────────────────
        st.subheader("📊 Reward per Step")
        _render_reward_per_step(ep_steps)

        # ── Discounted Return Gt per Step ─────────────────────────────────
        st.subheader("📈 Discounted Return Gₜ per Step")
        _render_gt_per_step(ep_steps, tx)

        # ── Glass-Box ─────────────────────────────────────────────────────
        st.subheader(tx["glass_title"])
        _render_glass_box(ep_steps, sel, tx, gamma)

        # ── Episode Summary ───────────────────────────────────────────────
        st.subheader(tx["summary_title"])
        ep_sla_rate   = sum(1 for s in ep_steps if s.get("sla_met"))    / max(len(ep_steps), 1)
        ep_skill_rate = sum(1 for s in ep_steps if s.get("skill_match")) / max(len(ep_steps), 1)
        ep_exp_rate   = sum(1 for s in ep_steps if s.get("explored"))   / max(len(ep_steps), 1)
        ep_avg_dist   = sum(s.get("distance", 0) for s in ep_steps)     / max(len(ep_steps), 1)
        ep_avg_reward = sum(s.get("reward", 0) for s in ep_steps)       / max(len(ep_steps), 1)
        ep_sla_saved  = sum(1 for s in ep_steps if s.get("sla_met"))
        _render_summary(ep_steps, ep_result, ep_sla_rate, ep_skill_rate, ep_exp_rate,
                        ep_avg_dist, ep_avg_reward, ep_sla_saved, ep_gt, tx)

        # ── Learning Curve (always at the bottom) ─────────────────────────
        if curve:
            st.subheader(tx["curve_title"])
            st.caption(
                "📌 **Expected: no upward trend.** Ch01 uses Q=0 — the agent never learns. "
                "The curve should be flat (random noise around a constant mean). "
                "This flat baseline Gₜ is the reference for Ch02–Ch18."
            )
            _render_curve(curve, tx)
