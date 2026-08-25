import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
COLORS = {
    "epsilon_greedy": "#3498db",
    "ucb":            "#e67e22",
    "thompson":       "#2ecc71"
}

# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    
    

    st.title(tx["title"])
    st.caption(tx["subtitle"])

    try:
        import rlvr_py
    except ImportError:
        st.error(tx["engine_missing"])
        return

    st.sidebar.header(tx["sidebar_title"])
    n_steps       = st.sidebar.slider(tx["n_steps"],       50, 2000, 500, 50)
    epsilon       = st.sidebar.slider(tx["epsilon"],       0.0, 1.0, 0.3, 0.05)
    epsilon_decay = st.sidebar.slider(tx["epsilon_decay"], 0.0, 0.1, 0.01, 0.001,
                                      format="%.3f")
    ucb_c         = st.sidebar.slider(tx["ucb_c"],         0.1, 5.0, 2.0, 0.1)
    seed          = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    run = st.button(tx["run_btn"], type="primary")

    if run:
        with st.spinner("Running Rust bandit engine..."):
            raw = rlvr_py.run_ch03_bandits(
                int(seed), int(n_steps),
                float(epsilon), float(epsilon_decay), float(ucb_c)
            )
        st.session_state["ch03_raw"] = raw

    if "ch03_raw" not in st.session_state:
        st.info("Configure settings and click **▶ Run All Three Algorithms**.")
        return

    raw       = st.session_state["ch03_raw"]
    results   = raw["results"]
    arm_names = raw["arm_names"]
    true_rates = raw["true_rates"]

    # KPI row
    cols = st.columns(3)
    for i, res in enumerate(results):
        algo = res["algorithm"]
        label = tx["algo_labels"][algo]
        cols[i].metric(
            label,
            f"Regret: {res['total_regret']:.1f}",
            f"Reward: {res['total_reward']:.0f}",
        )

    # Cumulative regret
    st.subheader(tx["regret_title"])
    _render_regret(results, tx, arm_names)
    st.caption(tx["regret_caption"])

    # Cumulative reward
    st.subheader(tx["reward_title"])
    _render_reward(results, tx)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(tx["pulls_title"])
        _render_pulls(results, arm_names, true_rates, tx)
        st.caption(tx["pulls_caption"])
    with col2:
        st.subheader(tx["qval_title"])
        _render_qvalues(results, arm_names, true_rates, tx)
        st.caption(tx["qval_caption"])

    # Glass-Box
    st.subheader(tx["glass_title"])
    _render_glass_box(results, arm_names, tx)

    # Summary
    st.subheader(tx["summary_title"])
    _render_summary(results, arm_names, true_rates, tx)

    # Theory


# ---------------------------------------------------------------------------
# Cumulative regret chart
# ---------------------------------------------------------------------------
def _render_regret(results, tx, arm_names):
    fig = go.Figure()
    for res in results:
        algo  = res["algorithm"]
        label = tx["algo_labels"][algo]
        steps = [s["step"] for s in res["steps"]]
        regret = [s["cumulative_regret"] for s in res["steps"]]
        fig.add_trace(go.Scatter(
            x=steps, y=regret,
            mode="lines",
            name=label,
            line=dict(color=COLORS[algo], width=2),
        ))
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title=tx["regret_x"],
        yaxis_title=tx["regret_y"],
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Cumulative reward chart
# ---------------------------------------------------------------------------
def _render_reward(results, tx):
    fig = go.Figure()
    for res in results:
        algo  = res["algorithm"]
        label = tx["algo_labels"][algo]
        steps = [s["step"] for s in res["steps"]]
        cum_r = []
        total = 0.0
        for s in res["steps"]:
            total += s["reward"]
            cum_r.append(total)
        fig.add_trace(go.Scatter(
            x=steps, y=cum_r,
            mode="lines",
            name=label,
            line=dict(color=COLORS[algo], width=2),
        ))
    fig.update_layout(
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title=tx["reward_x"],
        yaxis_title=tx["reward_y"],
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Arm pull distribution
# ---------------------------------------------------------------------------
def _render_pulls(results, arm_names, true_rates, tx):
    fig = go.Figure()
    for res in results:
        algo  = res["algorithm"]
        label = tx["algo_labels"][algo]
        fig.add_trace(go.Bar(
            x=arm_names,
            y=res["final_n_pulls"],
            name=label,
            marker_color=COLORS[algo],
            opacity=0.8,
        ))
    fig.update_layout(
        height=280,
        barmode="group",
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Q-value convergence
# ---------------------------------------------------------------------------
def _render_qvalues(results, arm_names, true_rates, tx):
    fig = go.Figure()
    # True rates as dashed lines
    for i, (name, rate) in enumerate(zip(arm_names, true_rates)):
        fig.add_hline(
            y=rate,
            line_dash="dash",
            line_color="grey",
            opacity=0.5,
            annotation_text=f"{name} {rate:.0%}",
            annotation_position="right",
        )
    # Final Q-values as bars
    for res in results:
        algo  = res["algorithm"]
        label = tx["algo_labels"][algo]
        fig.add_trace(go.Bar(
            x=arm_names,
            y=res["final_q_values"],
            name=label,
            marker_color=COLORS[algo],
            opacity=0.8,
        ))
    fig.update_layout(
        height=280,
        barmode="group",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Glass-Box
# ---------------------------------------------------------------------------
def _render_glass_box(results, arm_names, tx):
    algo_options = {tx["algo_labels"][r["algorithm"]]: r for r in results}
    selected_label = st.selectbox(tx["glass_algo"], list(algo_options.keys()))
    res   = algo_options[selected_label]
    steps = res["steps"]
    n     = len(steps)

    sel = st.slider(tx["glass_step_slider"], 0, max(n - 1, 0), 0)

    rows = []
    for s in steps:
        rows.append({
            tx["glass_headers"][0]: s["step"],
            tx["glass_headers"][1]: s["arm"],
            tx["glass_headers"][2]: arm_names[s["arm"]],
            tx["glass_headers"][3]: f"{s['reward']:.0f}",
            tx["glass_headers"][4]: f"{s['regret']:.3f}",
            tx["glass_headers"][5]: f"{s['cumulative_regret']:.2f}",
            tx["glass_headers"][6]: f"{s['epsilon']:.3f}",
            tx["glass_headers"][7]: "🔍 Explore" if s["explored"] else "🎯 Exploit"
        })

    st.dataframe(rows, width='stretch', height=280)

    # Selected step detail
    s = steps[sel]
    algo = res["algorithm"]
    st.markdown(f"""
**Step {sel} detail ({tx['algo_labels'][algo]}):**
- Arm: **{arm_names[s['arm']]}** (arm {s['arm']})
- Reward: **{s['reward']:.0f}** · Regret: **{s['regret']:.3f}**
- Cumulative regret: **{s['cumulative_regret']:.2f}**
- ε = {s['epsilon']:.3f} · Mode: {"🔍 Explore" if s['explored'] else "🎯 Exploit"}
""")

    if algo == "ucb" and any(v > 0 for v in s["ucb_values"]):
        st.markdown("**UCB values at this step:**")
        for i, (name, ucb) in enumerate(zip(arm_names, s["ucb_values"])):
            marker = " ← selected" if i == s["arm"] else ""
            st.markdown(f"- {name}: `{ucb:.4f}`{marker}")
        st.latex(r"\text{UCB}(a) = Q(a) + c\sqrt{\frac{\ln t}{N(a)}}")

    if algo == "thompson" and any(v > 0 for v in s["thompson_samples"]):
        st.markdown("**Thompson samples at this step:**")
        for i, (name, ts) in enumerate(zip(arm_names, s["thompson_samples"])):
            marker = " ← selected" if i == s["arm"] else ""
            st.markdown(f"- {name}: `{ts:.4f}`{marker}")
        st.latex(r"\theta_a \sim \text{Beta}(\alpha_a, \beta_a)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _render_summary(results, arm_names, true_rates, tx):
    st.markdown(f"#### {tx['summary_results']}")

    # Comparison table
    rows = []
    for res in results:
        algo = res["algorithm"]
        rows.append({
            "Algorithm": tx["algo_labels"][algo],
            tx["total_reward_label"]: f"{res['total_reward']:.0f}",
            tx["total_regret_label"]: f"{res['total_regret']:.2f}",
            tx["best_arm_label"]: f"{arm_names[res['best_arm']]} (arm {res['best_arm']})"
        })
    st.dataframe(rows, width='stretch', hide_index=True)

    # True rates reveal
    st.markdown(f"**{tx['true_rates_label']}:**")
    for name, rate in zip(arm_names, true_rates):
        marker = " ⭐ optimal" if rate == max(true_rates) else ""
        st.markdown(f"- {name}: **{rate:.0%}**{marker}")

    # Pros & cons per algorithm
    st.markdown(f"#### {tx['summary_pros_cons']}")
    for res in results:
        algo  = res["algorithm"]
        label = tx["algo_labels"][algo]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{label} — {tx['pros']}**")
            for p in tx["pros_list"][algo]:
                st.markdown(f"- {p}")
        with col2:
            st.markdown(f"**{label} — {tx['cons']}**")
            for c in tx["cons_list"][algo]:
                st.markdown(f"- {c}")
        st.markdown("---")


# ---------------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------------