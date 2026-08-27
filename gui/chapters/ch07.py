import os
import streamlit as st
import plotly.graph_objects as go

T = {"EN": {
        "title": "Chapter 07 — n-Step TD & Planning with Dyna-Q",
        "subtitle": "n-Step TD · n-Step SARSA · Dyna-Q · Dyna-Q+ · Warsaw ASP",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Settings",
        "n_episodes": "Number of episodes",
        "gamma": "γ — Discount factor",
        "alpha": "α — Learning rate",
        "epsilon": "ε — Initial exploration",
        "epsilon_decay": "ε decay rate",
        "n_step": "n — Step size (1=TD, large=MC)",
        "planning_steps": "k — Planning steps (Dyna-Q)",
        "kappa": "κ — Exploration bonus (Dyna-Q+)",
        "seed": "Random seed",
        "run_btn": "▶ Run All Four Algorithms",
        "guide": """
**Step 1 — Understand n-Step TD**
n=1 → TD(0). n=∞ → Monte Carlo. n=3-5 → sweet spot.
The n-step return: G^(n)_t = R_{t+1} + γR_{t+2} + ... + γ^(n-1)R_{t+n} + γ^n V(S_{t+n})

**Step 2 — Understand Dyna-Q**
Dyna-Q = Q-Learning + model learning + planning.
After each real step: do k simulated steps using the learned model.
More planning steps k → faster convergence, more computation per step.

**Step 3 — Understand Dyna-Q+**
Same as Dyna-Q but adds exploration bonus κ√τ(s,a) to rarely-tried transitions.
τ(s,a) = steps since (s,a) was last tried. Prevents stagnation in changing environments.

**Step 4 — Try n=1 vs n=5 vs n=10**
Watch how larger n improves early learning but increases variance.

**Step 5 — Try k=0 vs k=5 vs k=20 (planning steps)**
k=0 → pure Q-Learning. k=20 → Dyna-Q learns much faster per episode.

**Step 6 — Read the Model Size metric**
How many (s,a) pairs has Dyna-Q learned? Should grow toward N_STATES × N_ACTIONS = 32.
""",
        "returns_title": "📈 Episode Returns — All Four Algorithms",
        "returns_caption": "Moving average. Dyna-Q should converge fastest due to planning.",
        "td_error_title": "📉 TD Error Curve",
        "td_error_caption": "TD error decays as agent learns. Dyna-Q error drops fastest.",
        "value_title": "📊 Value Function V(s)",
        "value_caption": "All algorithms should converge to similar V*(s).",
        "nstep_title": "📊 n-Step Return Comparison (n=1 vs n=3 vs n=10)",
        "nstep_caption": "Larger n = more MC-like. Sweet spot typically n=3-5.",
        "model_title": "🗺️ Dyna-Q Model Coverage",
        "model_caption": "How many (s,a) pairs the model has learned. Max = 32.",
        "qtable_title": "📊 Q-Table Heatmap",
        "qtable_caption": "Q(s,a) values learned. Select algorithm.",
        "glass_title": "🔬 Glass-Box — Planning Steps Trace",
        "summary_title": "📊 Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "Algorithms — Pros & Cons",
        "pros": "✅ Pros", "cons": "❌ Cons",
        "theory_sections": {
            "nstep":    "§7.1 n-Step TD Returns",
            "nsarsa":   "§7.2 n-Step SARSA",
            "dynaq":    "§7.4 Dyna-Q",
            "dynaqplus":"§7.5 Dyna-Q+",
        },
        "theory_nstep": r"""
**n-Step TD** generalises TD(0) and MC:

G^(n)_t = R_{t+1} + γR_{t+2} + ... + γ^(n-1)R_{t+n} + γ^n V(S_{t+n})

V(S_t) <- V(S_t) + α [G^(n)_t - V(S_t)]

- n=1: TD(0) — maximum bootstrapping, minimum variance
- n=∞: Monte Carlo — no bootstrapping, maximum variance
- n=3-5: sweet spot — low bias, manageable variance

Implemented in `nstep_td_prediction()` in `ch07_nstep.rs`.
""",
        "theory_nsarsa": r"""
**n-Step SARSA** extends SARSA to n steps:

G^(n)_t = R_{t+1} + ... + γ^(n-1)R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})

Q(S_t,A_t) <- Q(S_t,A_t) + α [G^(n)_t - Q(S_t,A_t)]

On-policy: A_{t+n} is chosen by the same epsilon-greedy policy.
Implemented in `nstep_sarsa()` in `ch07_nstep.rs`.
""",
        "theory_dynaq": r"""
**Dyna-Q** integrates learning and planning:

For each real step:
1. Q-Learning update: Q(s,a) += α[R + γ max_a' Q(s',a') - Q(s,a)]
2. Model update: Model(s,a) = (R, s')
3. Planning (k times): sample random (s,a) from model, do Q-Learning update

With k=5 planning steps, Dyna-Q is ~5x more sample-efficient than Q-Learning.
Implemented in `dyna_q()` in `ch07_nstep.rs`.
""",
        "theory_dynaqplus": r"""
**Dyna-Q+** adds an exploration bonus to rarely-tried transitions:

r' = r + κ√τ(s,a)

where τ(s,a) = steps since (s,a) was last tried.

This prevents the agent from ignoring transitions it hasn't tried recently.
Critical in non-stationary environments where the model may become stale.
κ controls the exploration bonus strength (default: 0.001).
Implemented in `dyna_q_plus()` in `ch07_nstep.rs`.
""",
        "algo_labels": {
            "nstep_td":    "n-Step TD",
            "nstep_sarsa": "n-Step SARSA",
            "dyna_q":      "Dyna-Q",
            "dyna_q_plus": "Dyna-Q+",
        },
        "pros_list": {
            "nstep_td":    ["Bridges TD and MC", "Tunable bias-variance via n", "No model needed"],
            "nstep_sarsa": ["On-policy, safe", "n-step reduces variance vs TD(0)", "No model needed"],
            "dyna_q":      ["Sample efficient — k planning steps", "Learns model of environment", "Converges faster than Q-Learning"],
            "dyna_q_plus": ["Handles non-stationary environments", "Exploration bonus prevents stagnation", "Best of Dyna-Q + exploration"],
        },
        "cons_list": {
            "nstep_td":    ["Must wait n steps before update", "Higher memory (stores n transitions)", "n must be tuned"],
            "nstep_sarsa": ["On-policy — needs epsilon > 0", "n must be tuned", "Slower than Dyna-Q"],
            "dyna_q":      ["Model may be wrong (stale)", "k planning steps add computation", "Assumes stationary environment"],
            "dyna_q_plus": ["κ must be tuned", "More complex than Dyna-Q", "Bonus can cause over-exploration"],
        },
    }}

COLORS = {
    "nstep_td":    "#0082F0",
    "nstep_sarsa": "#FF8C0A",
    "dyna_q":      "#0FC373",
    "dyna_q_plus": "#FF3232",
}

def _moving_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def _tx(lang=None):
    import copy
    return copy.deepcopy(T.get("EN", {}))

def _render_handbook():
    import os as _os
    _col1, _col2 = st.columns([8, 1])
    with _col1:
        st.subheader("Hands-On Guide \u2014 Chapter 07 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch07_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("\U0001f4be Save", data=_html, file_name="handson_ch07_en.html", mime="text/html")
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
    n_episodes     = st.sidebar.slider(tx["n_episodes"],     50, 3000, 500, 50)
    gamma          = st.sidebar.slider(tx["gamma"],          0.5, 0.999, 0.95, 0.005)
    alpha          = st.sidebar.slider(tx["alpha"],          0.01, 1.0, 0.1, 0.01)
    epsilon        = st.sidebar.slider(tx["epsilon"],        0.0, 1.0, 0.3, 0.05)
    epsilon_decay  = st.sidebar.slider(tx["epsilon_decay"],  0.0, 0.1, 0.01, 0.001, format="%.3f")
    n_step         = st.sidebar.slider(tx["n_step"],         1, 20, 3, 1)
    planning_steps = st.sidebar.slider(tx["planning_steps"], 0, 50, 5, 1)
    kappa          = st.sidebar.slider(tx["kappa"],          0.0, 0.01, 0.001, 0.0001, format="%.4f")
    seed           = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running Rust n-step/Dyna engine..."):
            result = rlvr_py.run_ch07_nstep(
                int(seed), int(n_episodes), float(gamma), float(alpha),
                float(epsilon), float(epsilon_decay),
                int(n_step), int(planning_steps), float(kappa)
            )
        st.session_state["ch07_result"] = result

    if "ch07_result" not in st.session_state:
        st.info("Configure settings and click **▶ Run All Four Algorithms**.")
        return

    result       = st.session_state["ch07_result"]
    state_names  = result["state_names"]
    action_names = result["action_names"]
    algos        = ["nstep_td", "nstep_sarsa", "dyna_q", "dyna_q_plus"]

    # KPI
    cols = st.columns(4)
    for i, key in enumerate(algos):
        r = result[key]
        avg = sum(r["returns_curve"][-50:]) / min(50, len(r["returns_curve"]))
        extra = f"Model: {r['model_size']}" if r["model_size"] > 0 else f"Steps: {r['total_steps']:,}"
        cols[i].metric(tx["algo_labels"][key], f"Avg: {avg:.2f}", extra)

    # Returns
    st.subheader(tx["returns_title"])
    fig = go.Figure()
    for key in algos:
        ma = _moving_avg(result[key]["returns_curve"], 30)
        fig.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
            mode="lines", name=tx["algo_labels"][key],
            line=dict(color=COLORS[key], width=2)))
    fig.update_layout(height=300, margin=dict(l=40,r=20,t=20,b=40),
                      xaxis_title="Episode", yaxis_title="Return (MA-30)",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch')
    st.caption(tx["returns_caption"])

    # TD Error
    st.subheader(tx["td_error_title"])
    fig2 = go.Figure()
    for key in ["nstep_sarsa", "dyna_q", "dyna_q_plus"]:
        ma = _moving_avg(result[key]["td_error_curve"], 30)
        fig2.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
            mode="lines", name=tx["algo_labels"][key],
            line=dict(color=COLORS[key], width=2)))
    fig2.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40),
                       xaxis_title="Episode", yaxis_title="Avg TD Error",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig2, width='stretch')
    st.caption(tx["td_error_caption"])

    # Value function
    st.subheader(tx["value_title"])
    short = [f"S{i}" for i in range(result["n_states"])]
    fig3 = go.Figure()
    for key in algos:
        fig3.add_trace(go.Bar(x=short, y=result[key]["values"],
            name=tx["algo_labels"][key], marker_color=COLORS[key], opacity=0.8))
    fig3.update_layout(height=280, barmode="group",
                       margin=dict(l=40,r=20,t=20,b=40),
                       legend=dict(orientation="h"))
    st.plotly_chart(fig3, width='stretch')
    st.caption(tx["value_caption"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(tx["model_title"])
        model_sizes = {tx["algo_labels"][k]: result[k]["model_size"] for k in ["dyna_q","dyna_q_plus"]}
        max_model = result["n_states"] * result["n_actions"]
        fig4 = go.Figure()
        for label, size in model_sizes.items():
            fig4.add_trace(go.Bar(x=[label], y=[size],
                marker_color="#0082F0" if "Plus" not in label and "+" not in label else "#0FC373"))
        fig4.add_hline(y=max_model, line_dash="dash", line_color="grey",
                       annotation_text=f"Max={max_model}")
        fig4.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40),
                           yaxis_title="(s,a) pairs learned")
        st.plotly_chart(fig4, width='stretch')
        st.caption(tx["model_caption"])

    with col2:
        st.subheader(tx["qtable_title"])
        algo_sel = st.selectbox("Algorithm",
            [tx["algo_labels"][k] for k in ["nstep_sarsa","dyna_q","dyna_q_plus"]])
        key_map = {tx["algo_labels"][k]: k for k in ["nstep_sarsa","dyna_q","dyna_q_plus"]}
        key_sel = key_map.get(algo_sel, "dyna_q")
        qt = result[key_sel]["q_table"]
        action_short = [f"A{i}" for i in range(result["n_actions"])]
        fig5 = go.Figure(go.Heatmap(
            z=qt, x=action_short, y=short, colorscale="Blues",
            text=[[f"{qt[s][a]:.2f}" for a in range(result["n_actions"])]
                  for s in range(result["n_states"])],
            texttemplate="%{text}",
        ))
        fig5.update_layout(height=300, margin=dict(l=60,r=20,t=20,b=40))
        st.plotly_chart(fig5, width='stretch')
        st.caption(tx["qtable_caption"])

    # Glass-Box
    st.subheader(tx["glass_title"])
    _render_glass_box(result, tx)

    # Summary
    st.subheader(tx["summary_title"])
    _render_summary(result, tx, algos)



def _render_glass_box(result, tx):
    algo_options = {tx["algo_labels"][k]: k for k in
                    ["nstep_td","nstep_sarsa","dyna_q","dyna_q_plus"]}
    selected = st.selectbox("Algorithm", list(algo_options.keys()), key="gb7_algo")
    key = algo_options[selected]
    r = result[key]
    ep_idx = st.slider("Episode", 0, max(len(r["returns_curve"])-1, 0),
                       max(len(r["returns_curve"])-1, 0), key="gb7_ep")
    col1, col2, col3 = st.columns(3)
    col1.metric("Episode return",    f"{r['returns_curve'][ep_idx]:.3f}")
    col2.metric("Avg TD error",      f"{r['td_error_curve'][ep_idx]:.4f}")
    col3.metric("Model size",        str(r["model_size"]))

    if "nstep" in key:
        st.latex(r"G^{(n)}_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V(S_{t+n})")
    elif key == "dyna_q":
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\left[R + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]")
        st.markdown("**Planning:** sample random (s,a) from model → Q-Learning update × k")
    else:
        st.latex(r"r' = r + \kappa\sqrt{\tau(s,a)}")
        st.markdown("**Exploration bonus:** κ√τ(s,a) added to rarely-tried transitions")


def _render_summary(result, tx, algos):
    st.markdown(f"#### {tx['summary_results']}")
    rows = []
    for key in algos:
        r = result[key]
        avg = sum(r["returns_curve"][-100:]) / min(100, len(r["returns_curve"]))
        rows.append({
            "Algorithm":           tx["algo_labels"][key],
            "Avg return (last 100)": f"{avg:.3f}",
            "Total steps":         str(r["total_steps"]),
            "Model size":          str(r["model_size"]),
            "V*(S0)":              f"{r['values'][0]:.3f}",
            "V*(S7)":              f"{r['values'][7]:.3f}",
        })
    st.dataframe(rows, hide_index=True)
    st.markdown(f"#### {tx['summary_pros_cons']}")
    for key in algos:
        label = tx["algo_labels"][key]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{label} — {tx['pros']}**")
            for p in tx["pros_list"][key]: st.markdown(f"- {p}")
        with c2:
            st.markdown(f"**{label} — {tx['cons']}**")
            for c in tx["cons_list"][key]: st.markdown(f"- {c}")
        st.markdown("---")

