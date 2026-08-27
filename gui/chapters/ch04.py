import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
T = {"EN": {
        "title": "Chapter 04 — Dynamic Programming: Policy & Value Iteration",
        "subtitle": "ASP Operational State Optimisation · PI vs VI vs Async VI · Warsaw Region",
        "engine_missing": "⚙️ Rust engine not found. Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ DP Settings",
        "gamma": "γ — Discount factor",
        "theta": "θ — Convergence threshold",
        "seed": "Random seed",
        "run_btn": "▶ Run All Three DP Algorithms",
        "kpi_pi_iters": "PI outer iterations",
        "kpi_vi_iters": "VI iterations",
        "kpi_async_iters": "Async VI iterations",
        "kpi_policy_match": "PI = VI policy",
        "conv_title": "📈 Convergence Comparison — PI vs VI vs Async VI",
        "conv_x": "Sweep",
        "conv_y": "Max Bellman residual ‖δV‖∞",
        "conv_caption": "All three algorithms converge to the same V*. Async VI prioritises high-residual states.",
        "policy_evo_title": "🔄 Policy Evolution — Policy Iteration Steps",
        "policy_evo_caption": "Each row = one PI outer iteration. Cells show optimal action per state.",
        "residual_title": "🗺️ Bellman Residual per State (after convergence)",
        "residual_caption": "Higher residual = state was harder to optimise. S5–S7 typically highest.",
        "value_title": "📊 Final Value Function V*(s) — All Three Algorithms",
        "value_caption": "All three algorithms should produce identical V*(s). Any difference = numerical precision only.",
        "policy_title": "🎯 Optimal Policy π*(s) — PI vs VI",
        "policy_caption": "PI and VI must find the same optimal policy. Differences indicate a bug.",
        "glass_title": "🔬 Glass-Box — Policy Iteration Trace",
        "glass_headers": ["PI Step", "State", "Old Action", "New Action", "Changed"],
        "summary_title": "📊 Episode Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "DP Algorithms — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "theory_policy_eval": r"""
**Policy Evaluation** computes V^π(s) for a fixed policy π using the Bellman expectation equation:

V^π(s) = Σ_s' P(s'|s,π(s)) · [R(s,π(s)) + γ · V^π(s')]

Starting from V = 0, iterate until ‖V^(k+1) - V^(k)‖∞ < θ.

This is a **fixed-point iteration** — the Bellman expectation operator is a contraction mapping,
guaranteeing convergence to the unique V^π.

Implemented in `policy_evaluation()` in `ch04_dp.rs`.
""",
        "theory_policy_improve": r"""
**Policy Improvement** derives a better policy π' from V^π by acting greedily:

π'(s) = argmax_a Σ_s' P(s'|s,a) · [R(s,a) + γ · V^π(s')]

**Policy Improvement Theorem**: if π'(s) ≠ π(s) for any state, then V^π'(s) ≥ V^π(s) for all s.
The new policy is always at least as good as the old one.

Implemented in `policy_improvement()` in `ch04_dp.rs`.
""",
        "theory_pi": r"""
**Policy Iteration** alternates between evaluation and improvement until the policy stabilises:

1. Initialise π arbitrarily
2. **Evaluate**: compute V^π using Bellman expectation equation
3. **Improve**: π' = greedy(V^π)
4. If π' = π → STOP (optimal). Else π ← π', go to 2.

**Convergence**: guaranteed in finite steps (finite state/action space).
PI typically converges in very few outer iterations (3–10) even for large state spaces.

Implemented in `policy_iteration()` in `ch04_dp.rs`.
""",
        "theory_async_dp": r"""
**Asynchronous DP** updates states selectively rather than all at once.

**Bellman Residual** for state s:
Residual(s) = |V^(k+1)(s) - V^(k)(s)|

States with high residuals are updated first — they have the most to gain from an update.

**Prioritized Sweeping**: maintain a priority queue ordered by residual.
Pop highest-residual state, update it, propagate to predecessors.

**Advantage**: focuses computation on states that matter most.
In ASP: S5 and S7 (critical states) get updated first → faster convergence in crisis scenarios.

Implemented in `async_value_iteration()` in `ch04_dp.rs`.
""",
        "pros_list": {
            "pi": [
                "Converges in very few outer iterations (3-10 typically)",
                "Policy improvement theorem guarantees monotone improvement",
                "Exact policy evaluation at each step",
                "Natural for problems where policy is the primary output",
            ],
            "vi": [
                "Simpler implementation — no inner/outer loop",
                "Each iteration is a single Bellman sweep",
                "Often faster total computation than PI",
                "Direct application of Bellman optimality equation",
            ],
            "async": [
                "Focuses computation on high-impact states",
                "Faster convergence in practice for large state spaces",
                "Natural for online/real-time settings",
                "Prioritized Sweeping is near-optimal update schedule",
            ],
        },
        "cons_list": {
            "pi": [
                "Each outer iteration requires full policy evaluation (expensive)",
                "Requires full model P(s'|s,a) — not model-free",
                "Synchronous updates — all states updated each sweep",
                "Overkill for small state spaces",
            ],
            "vi": [
                "Requires full model P(s'|s,a)",
                "Synchronous — all states updated each iteration",
                "More iterations than PI outer loops",
                "No intermediate policy available during convergence",
            ],
            "async": [
                "Requires full model P(s'|s,a)",
                "Residual computation adds overhead per iteration",
                "Update order affects convergence path (not final result)",
                "More complex implementation than synchronous VI",
            ],
        },
        "algo_labels": {
            "pi": "Policy Iteration",
            "vi": "Value Iteration",
            "async": "Async VI",
        },
    }}

COLORS = {"pi": "#0082F0", "vi": "#FF8C0A", "async": "#0FC373"}

# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def _tx(lang=None):
    import copy
    return copy.deepcopy(T.get("EN", {}))

def _render_handbook():
    import os as _os
    _col1, _col2 = st.columns([8, 1])
    with _col1:
        st.subheader("Hands-On Guide \u2014 Chapter 04 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch04_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("\U0001f4be Save", data=_html, file_name="handson_ch04_en.html", mime="text/html")
    st.iframe(_html, height=4000)

def render():
    lang = "EN"
    tx = _tx(lang)


    st.title(tx["title"])
    st.caption(tx["subtitle"])

    try:
        import rlvr_py
    except ImportError:
        st.error(tx["engine_missing"])
        return

    st.sidebar.header(tx["sidebar_title"])
    gamma = st.sidebar.slider(tx["gamma"], 0.50, 0.999, 0.95, 0.005)
    theta = st.sidebar.select_slider(
        tx["theta"],
        options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}",
    )
    seed = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    run = st.button(tx["run_btn"], type="primary")

    if run:
        with st.spinner("Running Rust DP engine..."):
            result = rlvr_py.run_ch04_dp(int(seed), float(gamma), float(theta))
        st.session_state["ch04_result"] = result

    if "ch04_result" not in st.session_state:
        st.info("Configure settings and click **▶ Run All Three DP Algorithms**.")
        return

    result      = st.session_state["ch04_result"]
    pi          = result["pi"]
    vi          = result["vi"]
    av          = result["async_vi"]
    state_names = result["state_names"]
    action_names = result["action_names"]
    residuals   = result["residuals"]

    policy_match = pi["policy"] == vi["policy"]

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tx["kpi_pi_iters"],    str(pi["pi_iterations"]))
    c2.metric(tx["kpi_vi_iters"],    str(vi["iterations"]))
    c3.metric(tx["kpi_async_iters"], str(av["iterations"]))
    c4.metric(tx["kpi_policy_match"], "✅ Yes" if policy_match else "❌ No")

    # Convergence comparison
    st.subheader(tx["conv_title"])
    _render_convergence(pi, vi, av, tx)
    st.caption(tx["conv_caption"])

    # Value function comparison
    st.subheader(tx["value_title"])
    _render_value_comparison(pi, vi, av, state_names, tx)
    st.caption(tx["value_caption"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(tx["policy_title"])
        _render_policy_comparison(pi, vi, state_names, action_names, tx)
        st.caption(tx["policy_caption"])
    with col2:
        st.subheader(tx["residual_title"])
        _render_residuals(residuals, state_names, tx)
        st.caption(tx["residual_caption"])

    # Policy evolution
    st.subheader(tx["policy_evo_title"])
    _render_policy_evolution(pi, state_names, action_names, tx)
    st.caption(tx["policy_evo_caption"])


    _tab_main, _tab_handbook = st.tabs(["\U0001f52c Interactive Lab", "\U0001f4d8 Hands-On Guide EN"])
    with _tab_handbook:
        _render_handbook()
    with _tab_main:


        # Glass-Box
        st.subheader(tx["glass_title"])
        _render_glass_box(pi, state_names, action_names, tx)

        # Summary
        st.subheader(tx["summary_title"])
        _render_summary(pi, vi, av, tx)

        # Theory


        # ---------------------------------------------------------------------------
        # Convergence comparison
        # ---------------------------------------------------------------------------
def _render_convergence(pi, vi, av, tx):
    fig = go.Figure()
    for key, label, data in [
        ("pi",    tx["algo_labels"]["pi"],    pi["convergence_curve"]),
        ("vi",    tx["algo_labels"]["vi"],    vi["curve"]),
        ("async", tx["algo_labels"]["async"], av["curve"]),
    ]:
        fig.add_trace(go.Scatter(
            x=list(range(len(data))), y=data,
            mode="lines", name=label,
            line=dict(color=COLORS[key], width=2),
        ))
    fig.update_layout(
        height=320, margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title=tx["conv_x"],
        yaxis_title=tx["conv_y"],
        yaxis_type="log",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Value function comparison
# ---------------------------------------------------------------------------
def _render_value_comparison(pi, vi, av, state_names, tx):
    short = [f"S{i}" for i in range(len(pi["values"]))]
    fig = go.Figure()
    for key, label, vals in [
        ("pi",    tx["algo_labels"]["pi"],    pi["values"]),
        ("vi",    tx["algo_labels"]["vi"],    vi["values"]),
        ("async", tx["algo_labels"]["async"], av["values"]),
    ]:
        fig.add_trace(go.Bar(
            x=short, y=vals, name=label,
            marker_color=COLORS[key], opacity=0.8,
        ))
    fig.update_layout(
        height=300, barmode="group",
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Policy comparison table
# ---------------------------------------------------------------------------
def _render_policy_comparison(pi, vi, state_names, action_names, tx):
    rows = []
    for s in range(len(pi["policy"])):
        pi_a = pi["policy"][s]
        vi_a = vi["policy"][s]
        rows.append({
            "State": f"S{s}",
            "Situation": state_names[s].split(":")[1].strip()[:25],
            f"PI: {tx['algo_labels']['pi']}": f"A{pi_a}",
            f"VI: {tx['algo_labels']['vi']}": f"A{vi_a}",
            "Match": "✅" if pi_a == vi_a else "❌",
        })
    st.dataframe(rows, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Bellman residual heatmap
# ---------------------------------------------------------------------------
def _render_residuals(residuals, state_names, tx):
    short = [f"S{i}" for i in range(len(residuals))]
    colors = ["#0FC373" if r < 0.001 else "#FF8C0A" if r < 0.01 else "#FF3232"
              for r in residuals]
    fig = go.Figure(go.Bar(
        x=short, y=residuals,
        marker_color=colors,
        text=[f"{r:.2e}" for r in residuals],
        textposition="outside",
    ))
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=20, b=40),
        yaxis_title="Residual",
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Policy evolution table
# ---------------------------------------------------------------------------
def _render_policy_evolution(pi, state_names, action_names, tx):
    history = pi["policy_history"]
    if not history:
        return
    rows = []
    for step_idx, pol in enumerate(history):
        row = {"PI Step": step_idx}
        for s, a in enumerate(pol):
            row[f"S{s}"] = f"A{a}"
        rows.append(row)
    st.dataframe(rows, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Glass-Box — PI step trace
# ---------------------------------------------------------------------------
def _render_glass_box(pi, state_names, action_names, tx):
    history = pi["policy_history"]
    if len(history) < 2:
        st.info("Policy converged in 1 step — no changes to show.")
        return

    rows = []
    for step_idx in range(1, len(history)):
        old_pol = history[step_idx - 1]
        new_pol = history[step_idx]
        for s in range(len(old_pol)):
            old_a = old_pol[s]
            new_a = new_pol[s]
            changed = old_a != new_a
            rows.append({
                tx["glass_headers"][0]: step_idx,
                tx["glass_headers"][1]: f"S{s}",
                tx["glass_headers"][2]: f"A{old_a}: {action_names[old_a].split(':')[1].strip()[:20]}",
                tx["glass_headers"][3]: f"A{new_a}: {action_names[new_a].split(':')[1].strip()[:20]}",
                tx["glass_headers"][4]: "🔄 Yes" if changed else "—",
            })
    st.dataframe(rows, width='stretch', height=300)

    # Bellman equations
    st.markdown("---")
    st.markdown("**Policy Evaluation (inner loop):**")
    st.latex(r"V^\pi(s) \leftarrow \sum_{s'} P(s'|s,\pi(s))\left[R(s,\pi(s)) + \gamma V^\pi(s')\right]")
    st.markdown("**Policy Improvement:**")
    st.latex(r"\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)\left[R(s,a) + \gamma V^\pi(s')\right]")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _render_summary(pi, vi, av, tx):
    st.markdown(f"#### {tx['summary_results']}")
    rows = [
        {
            "Algorithm":       tx["algo_labels"]["pi"],
            "Outer iters":     str(pi["pi_iterations"]),
            "Total sweeps":    str(len(pi["convergence_curve"])),
            "V*(S0)":          f"{pi['values'][0]:.3f}",
            "V*(S7)":          f"{pi['values'][7]:.3f}",
        },
        {
            "Algorithm":       tx["algo_labels"]["vi"],
            "Outer iters":     "N/A",
            "Total sweeps":    str(vi["iterations"]),
            "V*(S0)":          f"{vi['values'][0]:.3f}",
            "V*(S7)":          f"{vi['values'][7]:.3f}",
        },
        {
            "Algorithm":       tx["algo_labels"]["async"],
            "Outer iters":     "N/A",
            "Total sweeps":    str(av["iterations"]),
            "V*(S0)":          f"{av['values'][0]:.3f}",
            "V*(S7)":          f"{av['values'][7]:.3f}",
        },
    ]
    st.dataframe(rows, hide_index=True)

    st.markdown(f"#### {tx['summary_pros_cons']}")
    for key in ["pi", "vi", "async"]:
        label = tx["algo_labels"][key]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{label} — {tx['pros']}**")
            for p in tx["pros_list"][key]:
                st.markdown(f"- {p}")
        with col2:
            st.markdown(f"**{label} — {tx['cons']}**")
            for c in tx["cons_list"][key]:
                st.markdown(f"- {c}")
        st.markdown("---")

