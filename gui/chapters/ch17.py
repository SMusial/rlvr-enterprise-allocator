import streamlit as st
import json
import os
import math
import pandas as pd
import plotly.graph_objects as go
import altair as alt

AGENT_COLORS = ["#0082F0", "#FF8C0A", "#8B5CF6", "#0FC373", "#FF4B4B",
                "#00BCD4", "#E91E63", "#9C27B0", "#4CAF50", "#FF5722"]

# ---------------------------------------------------------------------------
# Variant descriptors
# ---------------------------------------------------------------------------
VARIANTS = {
    "V1": {
        "label":  "V1 — IQL · Individual Reward · No Communication",
        "arch":   "Independent Q-Learning (IQL)",
        "reward": "Individual",
        "comm":   "None",
        "color":  "#0082F0",
        "desc": (
            "**V1 — IQL / Individual Reward / No Communication** is the simplest MARL baseline. "
            "Each technician is a fully independent agent with its own Q-table. "
            "It receives only its own reward and has no knowledge of other agents. "
            "This is the MARL equivalent of single-agent Q-Learning applied independently to each technician."
        ),
        "fn": "run_ch17_episode",
        "extra_params": {},
    },
    "V2": {
        "label":  "V2 — IQL · Shared Reward · No Communication",
        "arch":   "Independent Q-Learning (IQL)",
        "reward": "Shared (team mean)",
        "comm":   "None",
        "color":  "#FF8C0A",
        "desc": (
            "**V2 — IQL / Shared Reward / No Communication** differs from V1 in one key way: "
            "all agents receive the **same shared reward** — the mean of all individual rewards. "
            "This encourages implicit cooperation even without communication. "
            "The key challenge is the **Credit Assignment Problem**: with shared reward, "
            "it is hard for each agent to know whether its own action was good or bad."
        ),
        "fn": "run_ch17_v2",
        "extra_params": {},
    },
    "V3": {
        "label":  "V3 — IQL · Individual Reward · Partial Observability",
        "arch":   "Independent Q-Learning (IQL)",
        "reward": "Individual",
        "comm":   "Partial (positions)",
        "color":  "#8B5CF6",
        "desc": (
            "**V3 — IQL / Individual Reward / Partial Observability** extends V1 by giving each agent "
            "**awareness of other technicians' positions**. "
            "When exploiting, each agent scores orders using: "
            "`score(t, o) = Q[t][o] + λ · min_colleague_distance(o)` — "
            "preferring orders far from colleagues. "
            "This is **distance-aware selection**: agents implicitly coordinate by avoiding "
            "areas already covered by colleagues, without explicit communication."
        ),
        "fn": "run_ch17_v3",
        "extra_params": {"lambda": 0.1},
    },
    "V4": {
        "label":  "V4 — CTDE (VDN) · Shared Reward · Full Observability",
        "arch":   "CTDE — Centralised Training, Decentralised Execution",
        "reward": "Shared (VDN)",
        "comm":   "Full (centralised training)",
        "color":  "#0FC373",
        "desc": (
            "**V4 — CTDE / VDN / Full Observability** is the first variant with a fundamentally "
            "different training architecture. "
            "During **training**, a centralised coordinator has access to all agents' Q-tables and positions. "
            "It selects the best technician using a **joint Q-score**: "
            "`Q_joint = Q[t][o] + distance_bonus` (VDN decomposition). "
            "All agents receive a **soft Q-table update** from the joint experience — "
            "not just the assigned agent. "
            "During **execution**, each agent uses its own Q-table independently (decentralised). "
            "This is the CTDE paradigm: train together, act alone."
        ),
        "fn": "run_ch17_v4",
        "extra_params": {},
    },
}

PLANNED_VARIANTS = [
    {"Variant": "V5", "Architecture": "CTDE",        "Reward": "Mixed",  "Communication": "Partial", "Status": "🔄 Planned"},
    {"Variant": "V6", "Architecture": "Cooperative", "Reward": "Shared", "Communication": "Full",    "Status": "🔄 Planned"},
]


# ---------------------------------------------------------------------------
# Variant comparison widget
# ---------------------------------------------------------------------------
def _render_variant_comparison(current_key: str):
    v = VARIANTS[current_key]
    st.markdown(v["desc"])
    st.markdown("#### All variants at a glance:")
    rows = []
    for k, vv in VARIANTS.items():
        rows.append({
            "Variant":       f"▶ {k}" if k == current_key else k,
            "Architecture":  vv["arch"],
            "Reward":        vv["reward"],
            "Communication": vv["comm"],
            "Status":        "▶ Current" if k == current_key else "✅ Done",
        })
    for p in PLANNED_VARIANTS:
        rows.append({
            "Variant":       p["Variant"],
            "Architecture":  p["Architecture"],
            "Reward":        p["Reward"],
            "Communication": p["Communication"],
            "Status":        p["Status"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
def _render_map(steps, sel, variant_color, is_v4=False):
    techs = {}
    for s in steps:
        if s["tech_idx"] not in techs:
            techs[s["tech_idx"]] = (s["tech_x"], s["tech_y"])
    for s in steps[:sel]:
        techs[s["tech_idx"]] = (s["order_x"], s["order_y"])

    orders = {}
    for s in steps:
        orders[s["order_idx"]] = (s["order_x"], s["order_y"])

    fig = go.Figure()

    for k, v in techs.items():
        color = AGENT_COLORS[k % len(AGENT_COLORS)]
        fig.add_trace(go.Scattermapbox(
            lat=[v[1]], lon=[v[0]], mode="markers+text",
            marker=dict(size=17, color=color),
            text=[f"T{k}"], textposition="top right",
            textfont=dict(size=12, color=color),
            name=f"T{k}",
        ))

    completed  = {}
    dispatched = {}
    for s in steps[:sel + 1]:
        completed[s["order_idx"]]  = s.get("sla_met", False)
        dispatched[s["order_idx"]] = s["tech_idx"]

    for k, v in orders.items():
        if k not in completed:
            fig.add_trace(go.Scattermapbox(
                lat=[v[1]], lon=[v[0]], mode="markers",
                marker=dict(size=14, color="#000000"),
                name=f"W{k}_b", showlegend=False,
            ))
            fig.add_trace(go.Scattermapbox(
                lat=[v[1]], lon=[v[0]], mode="markers+text",
                marker=dict(size=10, color="white"),
                text=[f"W{k}"], textposition="top right",
                textfont=dict(size=12, color="#000000"),
                name=f"W{k}", showlegend=False,
            ))
        elif completed[k]:
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

    if sel < len(steps):
        s = steps[sel]
        line_color = "#0FC373" if s.get("sla_met") else "#FF4B4B"
        label = "✅ SLA met" if s.get("sla_met") else "❌ SLA breach"
        if s.get("collision_avoided"):
            label += " 🔀"
        if is_v4 and not s.get("explored"):
            label += " 🧠 CTDE"
        fig.add_trace(go.Scattermapbox(
            lat=[s["tech_y"], s["order_y"]],
            lon=[s["tech_x"], s["order_x"]],
            mode="lines",
            line=dict(width=3, color=line_color),
            name=f"Step {sel}: T{s['tech_idx']}→W{s['order_idx']} ({label})",
        ))

    all_lats = [v[1] for v in techs.values()] + [v[1] for v in orders.values()]
    all_lons = [v[0] for v in techs.values()] + [v[0] for v in orders.values()]
    lat_c = (min(all_lats) + max(all_lats)) / 2
    lon_c = (min(all_lons) + max(all_lons)) / 2
    span  = max((max(all_lats) - min(all_lats)) * 1.2,
                (max(all_lons) - min(all_lons)) * 1.2, 0.05)
    zoom  = max(9, min(13, round(8.5 - math.log2(span * 111))))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=lat_c, lon=lon_c), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0), height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Reward per Step
# ---------------------------------------------------------------------------
def _render_reward_per_step(steps, sel, variant_color):
    rewards = [s["reward"] for s in steps]
    eff_sel = sel if sel < len(rewards) else None
    colors  = [variant_color if r >= 0 else "#FF4B4B" for r in rewards]
    if eff_sel is not None:
        colors[eff_sel] = "#FFD700"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(steps))), y=rewards,
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
        x=list(range(len(steps))), y=td_errors,
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
# V3: Nearest Colleague Distance
# ---------------------------------------------------------------------------
def _render_colleague_distance(steps, sel):
    dists   = [s.get("nearest_colleague_km", 0) for s in steps]
    avoided = [s.get("collision_avoided", False) for s in steps]
    eff_sel = sel if sel < len(dists) else None
    colors  = ["#FF8C0A" if a else "#8B5CF6" for a in avoided]
    if eff_sel is not None:
        colors[eff_sel] = "#FFD700"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(steps))), y=dists,
        marker_color=colors,
        hovertemplate="Step %{x}<br>Nearest colleague: %{y:.1f} km<extra></extra>",
    ))
    if eff_sel is not None:
        fig.add_vline(x=eff_sel, line_dash="dot", line_color="#FFD700", line_width=2)
    fig.update_layout(
        xaxis_title="Step", yaxis_title="Nearest Colleague (km)",
        height=220, margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    n_avoided = sum(1 for a in avoided if a)
    st.caption(f"Orange = overlap avoided · Purple = normal · {n_avoided}/{len(steps)} overlaps avoided")


# ---------------------------------------------------------------------------
# V4: Joint Q per Step
# ---------------------------------------------------------------------------
def _render_joint_q(steps, sel, mean_joint_q):
    jqs     = [s.get("joint_q", 0) for s in steps]
    eff_sel = sel if sel < len(jqs) else None
    colors  = ["#0FC373"] * len(jqs)
    if eff_sel is not None:
        colors[eff_sel] = "#FFD700"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(steps))), y=jqs,
        marker_color=colors,
        hovertemplate="Step %{x}<br>Joint Q: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    if eff_sel is not None:
        fig.add_vline(x=eff_sel, line_dash="dot", line_color="#FFD700", line_width=2)
    fig.update_layout(
        xaxis_title="Step", yaxis_title="Joint Q (VDN)",
        height=220, margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Joint Q = Σ Q^i(order) across all agents · "
        f"Mean Joint Q this episode = {mean_joint_q:.3f} · "
        f"VDN: Q_joint grows as agents learn better individual Q-values"
    )


# ---------------------------------------------------------------------------
# Glass-Box
# ---------------------------------------------------------------------------
def _render_glass_box(steps, sel, show_v3=False, show_v4=False):
    rows = []
    for i, s in enumerate(steps):
        row = {
            "Step":          i,
            "Agent":         f"T{s['tech_idx']}",
            "Work Order":    f"W{s['order_idx']}",
            "Action":        "explore" if s.get("explored") else "exploit",
            "ε":             f"{s.get('epsilon', 0):.3f}",
            "Reward":        round(s["reward"], 3),
            "Gₜ":            round(s["gt"], 3),
            "SLA":           "✅" if s.get("sla_met") else "❌",
            "Task vs Agent": f"{s.get('order_skill','?')} vs {s.get('tech_skill','?')}",
            "Skill Match":   "✅" if s.get("skill_match") else "❌",
            "Distance":      f"{s.get('distance_km', 0):.1f} km",
            "Q before":      round(s.get("q_before", 0), 4),
            "Q after":       round(s.get("q_after", 0), 4),
            "TD Error":      round(s.get("td_error", 0), 4),
        }
        if show_v3:
            row["Nearest Coll."] = f"{s.get('nearest_colleague_km', 0):.1f} km"
            row["Avoided"]       = "🔀" if s.get("collision_avoided") else ""
        if show_v4:
            row["Joint Q"]       = round(s.get("joint_q", 0), 4)
            row["CTDE"]          = "🧠" if not s.get("explored") else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    eff_sel = sel if sel < len(rows) else len(rows) - 1

    def highlight_row(row):
        return ["background-color: #252840"] * len(row) if row["Step"] == eff_sel else [""] * len(row)

    st.dataframe(df.style.apply(highlight_row, axis=1),
                 use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# Team Learning Curve (MA-5)
# ---------------------------------------------------------------------------
def _render_team_curve(curve, variant_color, variant_key):
    ma5 = []
    for i in range(len(curve)):
        w = curve[max(0, i - 4):i + 1]
        ma5.append(sum(w) / len(w))
    mean_gt = sum(curve) / len(curve)
    df = pd.DataFrame({"Episode": list(range(len(curve))), "Gₜ": curve, "MA-5": ma5})
    raw  = alt.Chart(df).mark_line(opacity=0.3, color=variant_color).encode(
        x="Episode:Q", y=alt.Y("Gₜ:Q", title="Team Gₜ"),
        tooltip=["Episode", alt.Tooltip("Gₜ:Q", format=".3f")])
    ma   = alt.Chart(df).mark_line(color=variant_color, strokeWidth=2).encode(
        x="Episode:Q", y="MA-5:Q",
        tooltip=["Episode", alt.Tooltip("MA-5:Q", format=".3f")])
    mean = alt.Chart(pd.DataFrame({"m": [mean_gt]})).mark_rule(
        color="#FF8C0A", strokeDash=[6, 3]).encode(y="m:Q")
    st.altair_chart((raw + ma + mean).properties(height=260), use_container_width=True)
    notes = {
        "V1": "📌 V1: individual rewards — each agent learns independently. Higher variance.",
        "V2": "📌 V2: shared reward — agents implicitly cooperate. Smoother curve.",
        "V3": "📌 V3: partial observability — agents avoid colleague overlap. Lower avg distance.",
        "V4": "📌 V4: CTDE/VDN — centralised training, all agents learn from joint experience. Expect faster convergence.",
    }
    st.caption(f"{notes.get(variant_key, '')} Mean G₀ = {mean_gt:.2f}")


# ---------------------------------------------------------------------------
# Per-Agent Learning Curves
# ---------------------------------------------------------------------------
def _render_agent_curves(agent_curves, n_tech):
    fig = go.Figure()
    for t in range(n_tech):
        if t < len(agent_curves) and agent_curves[t]:
            curve = agent_curves[t]
            ma5   = []
            for i in range(len(curve)):
                w = curve[max(0, i - 4):i + 1]
                ma5.append(sum(w) / len(w))
            fig.add_trace(go.Scatter(
                x=list(range(len(curve))), y=ma5,
                mode="lines", name=f"T{t} (MA-5)",
                line=dict(color=AGENT_COLORS[t % len(AGENT_COLORS)], width=2),
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
    st.caption("Each line = one agent. V4: agents should converge more uniformly due to centralised training.")


# ---------------------------------------------------------------------------
# Agent Stats
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Q-Table Heatmap
# ---------------------------------------------------------------------------
def _render_qtable(final_q_tables, n_tech, n_orders):
    fig = go.Figure(data=go.Heatmap(
        z=final_q_tables,
        x=[f"W{i}" for i in range(n_orders)],
        y=[f"T{i}" for i in range(n_tech)],
        colorscale="RdYlGn", zmid=0,
        hovertemplate="Agent T%{y}<br>Order W%{x}<br>Q = %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Work Order", yaxis_title="Agent (Technician)",
        height=300, margin=dict(l=60, r=20, t=20, b=40),
        paper_bgcolor="#0f1117", font=dict(color="#e8eaf6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = high Q-value · Red = low Q-value · V4: Q-tables should be more uniform (centralised training)")


# ---------------------------------------------------------------------------
# Episode Summary
# ---------------------------------------------------------------------------
def _render_summary(steps, total_gt, team_sla, ep_data=None):
    n = len(steps)
    skill_rate = sum(1 for s in steps if s.get("skill_match")) / max(n, 1)
    avg_dist   = sum(s.get("distance_km", 0) for s in steps) / max(n, 1)
    avg_reward = sum(s.get("reward", 0) for s in steps) / max(n, 1)
    exp_rate   = sum(1 for s in steps if s.get("explored")) / max(n, 1)
    rows = [
        {"Metric": "Team Total Gₜ",    "Value": f"{total_gt:.3f}"},
        {"Metric": "Team SLA Rate",    "Value": f"{team_sla*100:.1f}%"},
        {"Metric": "Skill Match Rate", "Value": f"{skill_rate*100:.1f}%"},
        {"Metric": "Exploration Rate", "Value": f"{exp_rate*100:.1f}%"},
        {"Metric": "Avg Distance",     "Value": f"{avg_dist:.1f} km"},
        {"Metric": "Avg Step Reward",  "Value": f"{avg_reward:.3f}"},
        {"Metric": "Steps",            "Value": str(n)},
    ]
    if ep_data:
        if ep_data.get("collisions_avoided", 0) > 0:
            rows.append({"Metric": "Overlaps Avoided (V3)", "Value": str(ep_data["collisions_avoided"])})
        if ep_data.get("mean_joint_q", 0) != 0:
            rows.append({"Metric": "Mean Joint Q (V4)", "Value": f"{ep_data['mean_joint_q']:.3f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar params
# ---------------------------------------------------------------------------
def _sidebar_params():
    st.sidebar.header("Ch17 Settings")
    n_tech    = st.sidebar.slider("Technicians (Agents)", 2, 10, 5)
    n_orders  = st.sidebar.slider("Work Orders",          4, 20, 10)
    n_ep      = st.sidebar.slider("Episodes",             5, 200, 50)
    alpha     = st.sidebar.slider("α (learning rate)",    0.01, 1.0, 0.1, 0.01)
    gamma     = st.sidebar.slider("γ (discount factor)",  0.5,  1.0, 0.95, 0.01)
    eps_start = st.sidebar.slider("ε start",              0.1,  1.0, 1.0, 0.05)
    eps_end   = st.sidebar.slider("ε end",                0.0,  0.5, 0.05, 0.01)
    seed      = st.sidebar.number_input("Random seed",    0, 9999, 42)
    return n_tech, n_orders, n_ep, alpha, gamma, eps_start, eps_end, seed


# ---------------------------------------------------------------------------
# Single variant lab
# ---------------------------------------------------------------------------
def _render_variant_lab(variant_key, rlvr_py, n_tech, n_orders, n_ep,
                         alpha, gamma, eps_start, eps_end, seed):
    v = VARIANTS[variant_key]
    variant_color = v["color"]
    ss_key = f"ch17_{variant_key}_data"
    is_v3  = variant_key == "V3"
    is_v4  = variant_key == "V4"

    _render_variant_comparison(variant_key)
    st.divider()

    lam = 0.1
    if is_v3:
        lam = st.slider(
            "λ (collision avoidance weight)", 0.0, 1.0, 0.1, 0.01,
            help="0 = same as V1, 1 = strong avoidance of colleague overlap",
            key="v3_lambda"
        )

    if st.button(f"▶ Run {variant_key} Training", type="primary", key=f"run_{variant_key}"):
        fn = getattr(rlvr_py, v["fn"])
        with st.spinner(f"Running {n_ep} episodes ({variant_key})..."):
            if is_v3:
                raw = fn(int(seed), int(n_tech), int(n_orders), int(n_ep),
                         float(alpha), float(gamma), float(eps_start), float(eps_end),
                         float(lam))
            else:
                raw = fn(int(seed), int(n_tech), int(n_orders), int(n_ep),
                         float(alpha), float(gamma), float(eps_start), float(eps_end))
        data = json.loads(raw) if isinstance(raw, str) else raw
        st.session_state[ss_key]                          = data
        st.session_state[f"ch17_{variant_key}_n_tech"]   = n_tech
        st.session_state[f"ch17_{variant_key}_n_orders"] = n_orders

    if ss_key not in st.session_state:
        st.info(f"Configure settings and click **▶ Run {variant_key} Training**.")
        return

    data     = st.session_state[ss_key]
    n_tech   = st.session_state[f"ch17_{variant_key}_n_tech"]
    n_orders = st.session_state[f"ch17_{variant_key}_n_orders"]

    episodes       = data["episodes"]
    curve          = data["curve"]
    agent_curves   = data["agent_curves"]
    final_q_tables = data["final_q_tables"]
    n_eps          = len(episodes)
    mean_g0        = sum(curve) / len(curve)

    st.info(
        f"**{v['label']}** · {n_tech} agents · {n_eps} episodes · "
        f"α={alpha:.2f} · γ={gamma:.2f} · ε: {eps_start:.2f}→{eps_end:.2f} · "
        f"Mean team G₀ = **{mean_g0:.2f}**"
    )

    ep_sel   = st.slider("🎬 Select episode", 0, n_eps - 1, n_eps - 1, key=f"ep_sel_{variant_key}")
    ep_data  = episodes[ep_sel]
    ep_steps = ep_data["steps"]
    ep_gt    = ep_data["total_gt"]
    ep_sla   = ep_data["team_sla_rate"]
    ep_eps   = ep_steps[0]["epsilon"] if ep_steps else 0.0
    ep_mjq   = ep_data.get("mean_joint_q", 0.0)

    caption = (f"Episode {ep_sel + 1}/{n_eps} — "
               f"Team Gₜ = **{ep_gt:.3f}** · SLA = {ep_sla*100:.1f}% · ε = {ep_eps:.3f}")
    if is_v3:
        caption += f" · Overlaps avoided = {ep_data.get('collisions_avoided', 0)}"
    if is_v4:
        caption += f" · Mean Joint Q = {ep_mjq:.3f}"
    st.caption(caption)

    n_steps = len(ep_steps)
    sel = st.slider("🔍 Highlight step on map", 0, n_steps, 0, key=f"step_sel_{variant_key}")
    if sel == n_steps:
        st.caption("📍 All work orders dispatched — no travel line shown.")

    st.subheader("🗺️ Warsaw Dispatch Map")
    _render_map(ep_steps, sel, variant_color, is_v4=is_v4)
    caption_map = "Blue = Technicians · White = Pending · Dark green = SLA met · Red = SLA breach"
    if is_v3: caption_map += " · 🔀 = overlap avoided"
    if is_v4: caption_map += " · 🧠 = CTDE exploit"
    st.caption(caption_map)

    st.subheader("📊 Reward per Step")
    _render_reward_per_step(ep_steps, sel, variant_color)

    if is_v3:
        st.subheader("🔀 Nearest Colleague Distance per Step")
        _render_colleague_distance(ep_steps, sel)

    if is_v4:
        st.subheader("🧠 Joint Q per Step (VDN)")
        _render_joint_q(ep_steps, sel, ep_mjq)

    st.subheader("📉 TD Error per Step")
    _render_td_error(ep_steps, sel)

    st.subheader("🔬 Glass-Box — IQL Step Trace")
    _render_glass_box(ep_steps, sel, show_v3=is_v3, show_v4=is_v4)

    st.subheader("📋 Episode Summary")
    _render_summary(ep_steps, ep_gt, ep_sla, ep_data)

    st.subheader("🤖 Agent Stats")
    _render_agent_stats(ep_data["agent_stats"])

    st.subheader("📊 Per-Agent Learning Curves (MA-5)")
    _render_agent_curves(agent_curves, n_tech)

    st.subheader("📈 Team Learning Curve — Gₜ over Episodes (MA-5)")
    _render_team_curve(curve, variant_color, variant_key)

    st.subheader("🧮 Q-Tables (Final)")
    _render_qtable(final_q_tables, n_tech, n_orders)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render():
    st.title("Chapter 17 — Multi-Agent RL: MARL Variants")
    st.caption("Warsaw ASP · V1: Individual · V2: Shared · V3: Partial Obs. · V4: CTDE/VDN")

    tab_v1, tab_v2, tab_v3, tab_v4, tab_handbook = st.tabs([
        "🤖 V1 — Individual",
        "🤝 V2 — Shared",
        "👁️ V3 — Partial Obs.",
        "🧠 V4 — CTDE/VDN",
        "📘 Hands-On Guide EN",
    ])

    with tab_handbook:
        _render_handbook()

    try:
        import rlvr_py
    except ImportError:
        st.error("❌ Rust engine not found. Run: cd rlvr-py && maturin develop --release")
        return

    n_tech, n_orders, n_ep, alpha, gamma, eps_start, eps_end, seed = _sidebar_params()

    with tab_v1:
        _render_variant_lab("V1", rlvr_py, n_tech, n_orders, n_ep,
                            alpha, gamma, eps_start, eps_end, seed)
    with tab_v2:
        _render_variant_lab("V2", rlvr_py, n_tech, n_orders, n_ep,
                            alpha, gamma, eps_start, eps_end, seed)
    with tab_v3:
        _render_variant_lab("V3", rlvr_py, n_tech, n_orders, n_ep,
                            alpha, gamma, eps_start, eps_end, seed)
    with tab_v4:
        _render_variant_lab("V4", rlvr_py, n_tech, n_orders, n_ep,
                            alpha, gamma, eps_start, eps_end, seed)
