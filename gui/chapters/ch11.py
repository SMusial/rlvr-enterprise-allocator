import os
import streamlit as st
import plotly.graph_objects as go

T = {"EN": {
        "title":    "Chapter 11 - Multi-Agent RL",
        "subtitle": "IQL - JAL - Lenient Q - Mean Field Q - 2 Dispatchers - Warsaw ASP",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title":  "Settings",
        "n_episodes":     "Episodes",
        "gamma":          "Gamma - Discount",
        "alpha":          "Alpha - Learning rate",
        "epsilon":        "Epsilon - Exploration",
        "epsilon_decay":  "Epsilon decay",
        "leniency_mu":    "Mu - Leniency (0=IQL, 1=full lenient)",
        "mf_beta":        "Beta - Mean Field influence",
        "seed":           "Seed",
        "run_btn":        "Run All Four Algorithms",
        "guide": (
            "Scenario: 2 dispatchers share the Warsaw ASP 8-state MDP.\n"
            "Each dispatcher acts independently but their rewards interact.\n\n"
            "Step 1 - IQL (Independent Q-Learning)\n"
            "Each agent runs standard Q-Learning, ignoring the other.\n"
            "Baseline: simplest MARL approach.\n\n"
            "Step 2 - JAL (Joint Action Learning)\n"
            "Each agent models the other's action frequency.\n"
            "Q update uses expected value over estimated partner policy.\n\n"
            "Step 3 - Lenient Q-Learning\n"
            "Negative TD errors ignored with probability mu.\n"
            "Prevents penalising good actions due to partner's mistakes.\n\n"
            "Step 4 - Mean Field Q-Learning\n"
            "Approximate joint action by mean action of neighbours.\n"
            "Scales to many agents - foundation for large MARL.\n\n"
            "Watch the Cooperation chart: fraction of steps both agents\n"
            "chose the same action. Higher = more coordinated behaviour."
        ),
        "returns_title":     "Joint Episode Returns",
        "returns_caption":   "MA-30. Mean return across both agents.",
        "cooperation_title": "Cooperation Rate",
        "cooperation_caption": "Fraction of steps both agents chose same action. Higher = more coordinated.",
        "value_title":       "Joint Value Function V(s)",
        "value_caption":     "Mean V(s) across both agents. S7 (SLA breach) should be lowest.",
        "qtable_title":      "Q-Table Heatmap",
        "qtable_caption":    "Q(s,a) for selected algorithm and agent.",
        "glass_title":       "Glass-Box - MARL Mechanics",
        "summary_title":     "Summary",
        "summary_results":   "Algorithm Comparison",
        "summary_pros_cons": "Algorithms - Pros and Cons",
        "pros": "Pros", "cons": "Cons",
        "theory_sections": {
            "iql":  "11.1 Independent Q-Learning",
            "jal":  "11.2 Joint Action Learning",
            "lq":   "11.3 Lenient Q-Learning",
            "mf":   "11.4 Mean Field Q-Learning",
        },
        "theory_iql":  "Each agent i runs Q-Learning independently:\nQ_i(s,a) += alpha * [r + gamma * max Q_i(s') - Q_i(s,a)]\nOther agents treated as part of environment.",
        "theory_jal":  "Agent i models partner j's policy pi_j(a|s) from action frequencies.\nQ update uses expected value: E_{a_j ~ pi_j}[max Q_i(s')].",
        "theory_lq":   "delta < 0: apply with probability (1 - mu)\ndelta >= 0: always apply\nmu=0 -> IQL. mu=1 -> never penalise.",
        "theory_mf":   "mean_a_j(s) = running mean of partner's actions in state s\nQ_i(s,a) += alpha * [r + beta*mean_a_j/N_A + gamma*max Q_i(s') - Q_i(s,a)]",
        "algo_labels": {
            "iql":       "IQL",
            "jal":       "JAL",
            "lenient":   "Lenient Q",
            "meanfield": "Mean Field Q",
        },
        "pros_list": {
            "iql":       ["Simplest MARL", "No communication needed", "Scales to N agents"],
            "jal":       ["Models partner policy", "Better coordination", "Principled joint value"],
            "lenient":   ["Robust to partner mistakes", "Avoids miscoordination", "Tunable via mu"],
            "meanfield": ["Scales to large N", "Principled mean-field theory", "Low communication"],
        },
        "cons_list": {
            "iql":       ["Non-stationary environment", "No coordination", "May not converge"],
            "jal":       ["Requires observing partner actions", "O(|A|^N) joint space", "Slow model update"],
            "lenient":   ["mu must be tuned", "May ignore valid penalties", "Slower convergence"],
            "meanfield": ["Mean approximation loses info", "Beta must be tuned", "Assumes homogeneous agents"],
        },
    }}

COLORS = {
    "iql":       "#8B5CF6",
    "jal":       "#0082F0",
    "lenient":   "#0FC373",
    "meanfield": "#FF8C0A",
}
ALGOS = ["iql", "jal", "lenient", "meanfield"]

def _ma(data, w=30):
    r = []
    for i in range(len(data)):
        s = max(0, i - w + 1)
        r.append(sum(data[s:i+1]) / (i - s + 1))
    return r


def _tx(lang=None):
    import copy
    return copy.deepcopy(T.get("EN", {}))

def _render_handbook():
    import os as _os
    _col1, _col2 = st.columns([8, 1])
    with _col1:
        st.subheader("Hands-On Guide \u2014 Chapter 11 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch11_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("\U0001f4be Save", data=_html, file_name="handson_ch11_en.html", mime="text/html")
    st.iframe(_html, height=4000)

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
    n_ep  = st.sidebar.slider(tx["n_episodes"],    50, 3000, 500, 50)
    gamma = st.sidebar.slider(tx["gamma"],         0.5, 0.999, 0.95, 0.005)
    alpha = st.sidebar.slider(tx["alpha"],         0.01, 1.0, 0.1, 0.01)
    eps   = st.sidebar.slider(tx["epsilon"],       0.0, 1.0, 0.3, 0.05)
    edec  = st.sidebar.slider(tx["epsilon_decay"], 0.0, 0.1, 0.01, 0.001, format="%.3f")
    mu    = st.sidebar.slider(tx["leniency_mu"],   0.0, 1.0, 0.5, 0.05)
    beta  = st.sidebar.slider(tx["mf_beta"],       0.0, 2.0, 0.5, 0.1)
    seed  = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running Rust MARL engine..."):
            res = rlvr_py.run_ch11_multiagent(
                int(seed), int(n_ep), float(gamma), float(alpha),
                float(eps), float(edec), float(mu), float(beta),
            )
        st.session_state["ch11_result"] = res

    if "ch11_result" not in st.session_state:
        st.info("Configure settings and click Run."); return

    res   = st.session_state["ch11_result"]
    short = [f"S{i}" for i in range(res["n_states"])]

    # KPI row
    cols = st.columns(4)
    for i, k in enumerate(ALGOS):
        avg  = sum(res[k]["returns_curve"][-50:]) / min(50, len(res[k]["returns_curve"]))
        coop = sum(res[k]["cooperation_curve"][-50:]) / max(1, min(50, len(res[k]["cooperation_curve"])))
        cols[i].metric(tx["algo_labels"][k], f"Avg:{avg:.2f}", f"Coop:{coop:.2f}")

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

    # Cooperation + Value
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(tx["cooperation_title"])
        f2 = go.Figure()
        for k in ALGOS:
            f2.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["cooperation_curve"]),
                mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
        f2.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40),
                         xaxis_title="Episode", yaxis_title="Cooperation rate",
                         legend=dict(orientation="h"))
        st.plotly_chart(f2, width='stretch')
        st.caption(tx["cooperation_caption"])
    with c2:
        st.subheader(tx["value_title"])
        f3 = go.Figure()
        for k in ALGOS:
            f3.add_trace(go.Bar(x=short, y=res[k]["values"],
                name=tx["algo_labels"][k], marker_color=COLORS[k], opacity=0.8))
        f3.update_layout(height=260, barmode="group",
                         margin=dict(l=40,r=20,t=20,b=40), legend=dict(orientation="h"))
        st.plotly_chart(f3, width='stretch')
        st.caption(tx["value_caption"])

    # Q-table heatmap (per agent)
    st.subheader(tx["qtable_title"])
    col_a, col_b = st.columns(2)
    sel  = st.selectbox("Algorithm", [tx["algo_labels"][k] for k in ALGOS])
    ks   = {tx["algo_labels"][k]: k for k in ALGOS}.get(sel, "iql")
    ash  = [f"A{i}" for i in range(res["n_actions"])]
    for agent_idx, col in enumerate([col_a, col_b]):
        with col:
            st.markdown(f"**Agent {agent_idx}**")
            qt = res[ks]["q_tables"][agent_idx]
            f4 = go.Figure(go.Heatmap(z=qt, x=ash, y=short, colorscale="Purples",
                text=[[f"{qt[s][a]:.2f}" for a in range(res["n_actions"])]
                      for s in range(res["n_states"])],
                texttemplate="%{text}"))
            f4.update_layout(height=260, margin=dict(l=60,r=10,t=20,b=40))
            st.plotly_chart(f4, width='stretch')

    st.subheader(tx["glass_title"]); _glass(res, tx)
    st.subheader(tx["summary_title"]); _summary(res, tx)

def _glass(res, tx):
    opts = {tx["algo_labels"][k]: k for k in ALGOS}
    sel  = st.selectbox("Algorithm", list(opts.keys()), key="gb11")
    k    = opts[sel]; r = res[k]
    ep   = st.slider("Episode", 0, max(len(r["returns_curve"])-1, 0),
                     max(len(r["returns_curve"])-1, 0), key="gb11ep")
    c1, c2, c3 = st.columns(3)
    c1.metric("Joint return",   f"{r['returns_curve'][ep]:.3f}")
    c2.metric("TD error",       f"{r['td_error_curve'][ep]:.4f}")
    c3.metric("Cooperation",    f"{r['cooperation_curve'][ep]:.3f}")
    if k == "iql":
        st.latex(r"Q_i(s,a) \leftarrow Q_i(s,a) + \alpha [r + \gamma \max_{a'} Q_i(s',a') - Q_i(s,a)]")
        st.markdown("Each agent acts independently. Other agent = part of environment.")
    elif k == "jal":
        st.latex(r"\hat{\pi}_j(a|s) = \frac{N_j(s,a)}{\sum_{a'} N_j(s,a')}")
        st.latex(r"Q_i(s,a) \leftarrow Q_i(s,a) + \alpha [r + \gamma \mathbb{E}_{\hat{\pi}_j}[\max Q_i(s')] - Q_i(s,a)]")
    elif k == "lenient":
        st.latex(r"\delta = r + \gamma \max Q_i(s') - Q_i(s,a)")
        st.markdown("Apply update only if delta >= 0, or with prob (1-mu) if delta < 0.")
    else:
        st.latex(r"\bar{a}_j(s) = \frac{1}{N}\sum_{t} a_j^{(t)}(s)")
        st.latex(r"Q_i(s,a) \leftarrow Q_i(s,a) + \alpha [r + \beta \bar{a}_j/|A| + \gamma \max Q_i(s') - Q_i(s,a)]")

def _summary(res, tx):
    rows = []
    for k in ALGOS:
        r    = res[k]
        avg  = sum(r["returns_curve"][-100:]) / min(100, len(r["returns_curve"]))
        coop = sum(r["cooperation_curve"]) / max(1, len(r["cooperation_curve"]))
        rows.append({
            "Algorithm":             tx["algo_labels"][k],
            "Avg return (last 100)": f"{avg:.3f}",
            "Total steps":           str(r["total_steps"]),
            "Avg cooperation":       f"{coop:.3f}",
            "V*(S0)":                f"{r['values'][0]:.3f}",
            "V*(S7)":                f"{r['values'][7]:.3f}",
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
