import streamlit as st
import plotly.graph_objects as go

T = {
    "EN": {
        "title":    "Chapter 10 - Model-Based RL: World Models",
        "subtitle": "WM Q-Learning - Prioritised Sweeping - MBPO - Uncertainty Bonus - Warsaw ASP",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title":  "Settings",
        "n_episodes":     "Episodes",
        "gamma":          "Gamma - Discount",
        "alpha":          "Alpha - Learning rate",
        "epsilon":        "Epsilon - Exploration",
        "epsilon_decay":  "Epsilon decay",
        "planning_steps": "k - Planning steps per real step",
        "priority_threshold": "Priority threshold (Prioritised Sweeping)",
        "uncertainty_beta":   "Beta - Uncertainty bonus weight",
        "seed":           "Seed",
        "run_btn":        "Run All Four Algorithms",
        "guide_title":    "How to use this chapter",
        "guide": (
            "Step 1 - World Model\n"
            "The agent learns T(s,a,s') and R(s,a) from real experience.\n"
            "Planning uses the learned model instead of the real environment.\n\n"
            "Step 2 - WM Q-Learning vs Dyna-Q (Ch07)\n"
            "Same idea as Dyna-Q but with an explicit tabular world model object.\n"
            "Model accuracy chart shows how well T(s,a,s') is learned.\n\n"
            "Step 3 - Prioritised Sweeping\n"
            "Plan from states with highest |delta| first.\n"
            "Propagates value updates to predecessors - faster convergence.\n\n"
            "Step 4 - MBPO (Model-Based Policy Gradient)\n"
            "Use learned model to generate synthetic rollouts for REINFORCE.\n"
            "Combines model-based efficiency with policy gradient flexibility.\n\n"
            "Step 5 - Uncertainty Bonus\n"
            "UCB-style: Q_bonus(s,a) = Q(s,a) + beta/sqrt(N(s,a)+1).\n"
            "Encourages exploration of rarely-visited state-action pairs."
        ),
        "returns_title":    "Episode Returns - All Four Algorithms",
        "returns_caption":  "MA-30. Prioritised Sweeping should converge fastest.",
        "accuracy_title":   "Model Accuracy",
        "accuracy_caption": "Fraction of (s,a) where learned T matches true T. Increases with experience.",
        "planning_title":   "Planning Steps Used per Episode",
        "planning_caption": "Actual planning steps executed. Prioritised Sweeping may use fewer.",
        "value_title":      "Value Function V(s)",
        "value_caption":    "S7 (SLA breach) should be lowest across all algorithms.",
        "qtable_title":     "Q-Table Heatmap",
        "qtable_caption":   "Q(s,a) values. Select algorithm.",
        "glass_title":      "Glass-Box - World Model Mechanics",
        "summary_title":    "Summary",
        "summary_results":  "Algorithm Comparison",
        "summary_pros_cons":"Algorithms - Pros and Cons",
        "pros": "Pros", "cons": "Cons",
        "theory_title":     "Theory - Chapter 10",
        "theory_sections": {
            "wm":   "10.1 World Models",
            "ps":   "10.2 Prioritised Sweeping",
            "mbpo": "10.3 Model-Based Policy Gradient",
            "ub":   "10.4 Uncertainty Bonus"
        },
        "theory_wm":   "T(s,a,s') = P(s'|s,a) learned from counts.\nR(s,a) = mean observed reward.\nPlanning: k Q-Learning steps on model samples.",
        "theory_ps":   "Priority(s,a) = |R(s,a) + gamma*max Q(s') - Q(s,a)|\nPlan from highest priority first.\nPropagate to predecessors after each update.",
        "theory_mbpo": "Real step: collect (s,a,r,s'), update model.\nSynthetic rollout: generate trajectory from model.\nREINFORCE update on synthetic trajectory.",
        "theory_ub":   "Q_bonus(s,a) = Q(s,a) + beta / sqrt(N(s,a) + 1)\nAction selection uses Q_bonus.\nQ update uses standard Q-Learning (no bonus).",
        "algo_labels": {
            "wm_qlearning": "WM Q-Learning",
            "pri_sweeping": "Prioritised Sweeping",
            "mbpo":         "MBPO (PG)",
            "uncertainty":  "Uncertainty Bonus"
        },
        "pros_list": {
            "wm_qlearning": ["Simple extension of Dyna-Q", "Explicit model reuse", "k planning steps tunable"],
            "pri_sweeping": ["Fastest convergence", "Efficient planning budget", "Propagates value to predecessors"],
            "mbpo":         ["Policy gradient flexibility", "No Q-table needed", "Combines model + PG"],
            "uncertainty":  ["Principled exploration", "No epsilon needed", "Adapts to visit counts"]
        },
        "cons_list": {
            "wm_qlearning": ["Random planning - inefficient", "Model errors compound", "Same as Dyna-Q"],
            "pri_sweeping": ["Priority queue overhead", "Predecessor search O(|S||A|)", "Sensitive to threshold"],
            "mbpo":         ["Model errors in rollouts", "Two learning rates", "High variance PG"],
            "uncertainty":  ["Beta must be tuned", "Bonus fades with visits", "May over-explore"]
        }
    }
}

COLORS = {
    "wm_qlearning": "#8B5CF6",
    "pri_sweeping": "#0082F0",
    "mbpo":         "#0FC373",
    "uncertainty":  "#FF8C0A"
}
ALGOS = ["wm_qlearning", "pri_sweeping", "mbpo", "uncertainty"]

def _ma(data, w=30):
    r = []
    for i in range(len(data)):
        s = max(0, i - w + 1)
        r.append(sum(data[s:i+1]) / (i - s + 1))
    return r


def _tx(lang):
    """Return translation dict for lang, filling missing keys from EN."""
    base = dict(T.get("EN", {}))
    over = T.get(lang, {})
    for k, v in over.items():
        base[k] = v
    return base

def render():
    lang = "EN"
    tx   = _tx(lang)
    st.title(tx["title"])
    st.caption(tx["subtitle"])
    try:
        import rlvr_py
    except ImportError:
        st.error(tx["engine_missing"]); return

    st.sidebar.header(tx["sidebar_title"])
    n_ep  = st.sidebar.slider(tx["n_episodes"],         50, 3000, 500, 50)
    gamma = st.sidebar.slider(tx["gamma"],              0.5, 0.999, 0.95, 0.005)
    alpha = st.sidebar.slider(tx["alpha"],              0.01, 1.0, 0.1, 0.01)
    eps   = st.sidebar.slider(tx["epsilon"],            0.0, 1.0, 0.3, 0.05)
    edec  = st.sidebar.slider(tx["epsilon_decay"],      0.0, 0.1, 0.01, 0.001, format="%.3f")
    kplan = st.sidebar.slider(tx["planning_steps"],     0, 50, 5, 1)
    pthr  = st.sidebar.slider(tx["priority_threshold"], 0.001, 0.1, 0.01, 0.001, format="%.3f")
    beta  = st.sidebar.slider(tx["uncertainty_beta"],   0.0, 5.0, 1.0, 0.1)
    seed  = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running Rust world-model engine..."):
            res = rlvr_py.run_ch10_world_model(
                int(seed), int(n_ep), float(gamma), float(alpha),
                float(eps), float(edec), int(kplan),
                float(pthr), float(beta),
            )
        st.session_state["ch10_result"] = res

    if "ch10_result" not in st.session_state:
        st.info("Configure settings and click Run."); _theory(tx); return

    res   = st.session_state["ch10_result"]
    short = [f"S{i}" for i in range(res["n_states"])]

    # KPI row
    cols = st.columns(4)
    for i, k in enumerate(ALGOS):
        avg = sum(res[k]["returns_curve"][-50:]) / min(50, len(res[k]["returns_curve"]))
        acc = sum(res[k]["model_accuracy"][-50:]) / max(1, min(50, len(res[k]["model_accuracy"])))
        cols[i].metric(tx["algo_labels"][k], f"Avg:{avg:.2f}", f"Acc:{acc:.2f}")

    # Returns
    st.subheader(tx["returns_title"])
    fig = go.Figure()
    for k in ALGOS:
        fig.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["returns_curve"]),
            mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
    fig.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
                      xaxis_title="Episode", yaxis_title="Return (MA-30)",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch')
    st.caption(tx["returns_caption"])

    # Model accuracy + Planning steps
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(tx["accuracy_title"])
        f2 = go.Figure()
        for k in ALGOS:
            f2.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["model_accuracy"]),
                mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
        f2.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40),
                         xaxis_title="Episode", yaxis_title="Accuracy",
                         legend=dict(orientation="h"))
        st.plotly_chart(f2, width='stretch')
        st.caption(tx["accuracy_caption"])
    with c2:
        st.subheader(tx["planning_title"])
        f3 = go.Figure()
        for k in ALGOS:
            f3.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["planning_steps_used"]),
                mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
        f3.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40),
                         xaxis_title="Episode", yaxis_title="Planning steps",
                         legend=dict(orientation="h"))
        st.plotly_chart(f3, width='stretch')
        st.caption(tx["planning_caption"])

    # Value function
    st.subheader(tx["value_title"])
    f4 = go.Figure()
    for k in ALGOS:
        f4.add_trace(go.Bar(x=short, y=res[k]["values"],
            name=tx["algo_labels"][k], marker_color=COLORS[k], opacity=0.8))
    f4.update_layout(height=260, barmode="group",
                     margin=dict(l=40,r=20,t=20,b=40), legend=dict(orientation="h"))
    st.plotly_chart(f4, width='stretch')
    st.caption(tx["value_caption"])

    # Q-table heatmap
    st.subheader(tx["qtable_title"])
    sel  = st.selectbox("Algorithm", [tx["algo_labels"][k] for k in ALGOS])
    ks   = {tx["algo_labels"][k]: k for k in ALGOS}.get(sel, "wm_qlearning")
    qt   = res[ks]["q_table"]
    ash  = [f"A{i}" for i in range(res["n_actions"])]
    f5   = go.Figure(go.Heatmap(z=qt, x=ash, y=short, colorscale="Purples",
        text=[[f"{qt[s][a]:.2f}" for a in range(res["n_actions"])] for s in range(res["n_states"])],
        texttemplate="%{text}"))
    f5.update_layout(height=280, margin=dict(l=60,r=20,t=20,b=40))
    st.plotly_chart(f5, width='stretch')

    st.subheader(tx["glass_title"]); _glass(res, tx)
    st.subheader(tx["summary_title"]); _summary(res, tx)
    _theory(tx)

def _glass(res, tx):
    opts = {tx["algo_labels"][k]: k for k in ALGOS}
    sel  = st.selectbox("Algorithm", list(opts.keys()), key="gb10")
    k    = opts[sel]; r = res[k]
    ep   = st.slider("Episode", 0, max(len(r["returns_curve"])-1, 0),
                     max(len(r["returns_curve"])-1, 0), key="gb10ep")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Return",         f"{r['returns_curve'][ep]:.3f}")
    c2.metric("TD error",       f"{r['td_error_curve'][ep]:.4f}")
    c3.metric("Model accuracy", f"{r['model_accuracy'][ep]:.3f}")
    c4.metric("Plan steps",     f"{r['planning_steps_used'][ep]:.0f}")
    if "pri_sweeping" in k:
        st.latex(r"\text{Priority}(s,a) = |R(s,a) + \gamma \max_{a'} Q(s',a') - Q(s,a)|")
        st.markdown("Plan from highest priority. Propagate to predecessors.")
    elif "mbpo" in k:
        st.latex(r"G_t^{\text{model}} = \sum_{k=t}^{H} \gamma^{k-t} \hat{R}(s_k, a_k)")
        st.latex(r"\theta \leftarrow \theta + \alpha \gamma^t G_t^{\text{model}} \nabla \log \pi(a_t|s_t)")
    elif "uncertainty" in k:
        st.latex(r"Q_{\text{bonus}}(s,a) = Q(s,a) + \frac{\beta}{\sqrt{N(s,a)+1}}")
        st.markdown("Action selection uses Q_bonus. Q update uses standard Q-Learning.")
    else:
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]")
        st.markdown("Planning: k Q-Learning steps on random model samples.")

def _summary(res, tx):
    rows = []
    for k in ALGOS:
        r   = res[k]
        avg = sum(r["returns_curve"][-100:]) / min(100, len(r["returns_curve"]))
        acc = sum(r["model_accuracy"]) / max(1, len(r["model_accuracy"]))
        rows.append({
            "Algorithm":             tx["algo_labels"][k],
            "Avg return (last 100)": f"{avg:.3f}",
            "Total steps":           str(r["total_steps"]),
            "Model size":            str(r["model_size"]),
            "Avg accuracy":          f"{acc:.3f}",
            "V*(S0)":                f"{r['values'][0]:.3f}",
            "V*(S7)":                f"{r['values'][7]:.3f}"
        })
    st.dataframe(rows, hide_index=True)
    for k in ALGOS:
        label = tx["algo_labels"][k]; c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{label} - {tx['pros']}**")
            for p in tx["pros_list"][k]: st.markdown(f"- {p}")
        with c2:
            st.markdown(f"**{label} - {tx['cons']}**")
            for c in tx["cons_list"][k]: st.markdown(f"- {c}")
        st.markdown("---")

def _theory(tx):
    st.markdown("---"); st.subheader(tx["theory_title"])
    for k in ["wm", "ps", "mbpo", "ub"]:
        with st.expander(tx["theory_sections"][k], expanded=False):
            st.markdown(tx[f"theory_{k}"])
