import os
import streamlit as st
import plotly.graph_objects as go

T = {"EN": {
        "title": "Chapter 05 — Monte Carlo Methods",
        "subtitle": "ASP Dispatch Learning from Episodes · No Model Required · Warsaw Region",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ MC Settings",
        "n_episodes": "Number of episodes",
        "gamma": "γ — Discount factor",
        "epsilon": "ε — Initial exploration",
        "epsilon_decay": "ε decay rate α",
        "seed": "Random seed",
        "run_btn": "▶ Run All Four MC Algorithms",
        "guide": """
**Step 1 — Understand the key difference from Ch04**
MC methods learn WITHOUT a model — no P(s\'|s,a) needed.
The agent generates episodes by interacting with the environment and learns from the returns.

**Step 2 — Set number of episodes**
More episodes = better estimates. Try 200 first, then 2000 to see convergence.

**Step 3 — Click ▶ Run All Four MC Algorithms**
- First-Visit MC: updates V(s) only on first visit to s per episode
- Every-Visit MC: updates V(s) on every visit to s per episode
- On-Policy Control: learns Q*(s,a) and π* simultaneously (epsilon-soft)
- Off-Policy Control: learns from behaviour policy using Importance Sampling

**Step 4 — Read the Returns Curve**
Watch average episode return improve over time — this is the learning signal.

**Step 5 — Read the Value Function comparison**
MC estimates of V*(s) should converge toward the DP solution from Ch04.

**Step 6 — Read the Visit Count heatmap**
Which states were visited most? Rarely visited states have high variance estimates.

**Step 7 — Read the Glass-Box**
See exact returns G_t for each state in a selected episode.
""",
        "returns_title": "📈 Episode Returns — All Four Algorithms",
        "returns_caption": "Moving average of episode returns. On-policy control should improve over time.",
        "value_title": "📊 Value Function V(s) — MC vs DP Reference",
        "value_caption": "MC estimates converge toward DP solution (Ch04) with more episodes.",
        "visits_title": "🗺️ State Visit Counts — First-Visit MC",
        "visits_caption": "States visited rarely have high-variance V(s) estimates.",
        "conv_title": "📈 Convergence — Max |V^(k) - V^(k-1)|",
        "conv_caption": "MC convergence is noisier than DP — stochastic environment.",
        "qtable_title": "📊 Q-Table Heatmap — On-Policy Control",
        "qtable_caption": "Q(s,a) values learned by on-policy MC. Brighter = higher value.",
        "glass_title": "🔬 Glass-Box — Episode Trace",
        "glass_headers": ["Step", "State", "Action", "Reward", "Return G_t"],
        "summary_title": "📊 Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "MC Algorithms — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "theory_sections": {
            "intro":       "§5.1 Monte Carlo Methods — Introduction",
            "first_visit": "§5.2 First-Visit MC Prediction",
            "every_visit": "§5.2 Every-Visit MC Prediction",
            "on_policy":   "§5.3 On-Policy MC Control",
            "off_policy":  "§5.4 Off-Policy MC with Importance Sampling",
        },
        "theory_intro": r"""
**Monte Carlo Methods** learn directly from episodes of experience — no model of P(s\'|s,a) needed.

Key properties:
- **Model-free**: learns from raw experience, not transition probabilities
- **Episode-based**: must wait until end of episode to update (unlike TD)
- **Unbiased**: estimates are unbiased (unlike TD which bootstraps)
- **High variance**: estimates can be noisy, especially for rarely visited states

The return from step t:
G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ... = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}

Implemented in `ch05_mc.rs` — `generate_episode()`, `mc_first_visit_prediction()`.
""",
        "theory_first_visit": r"""
**First-Visit MC Prediction** estimates V^π(s) by averaging returns from the FIRST visit to s per episode:

V(s) ← average of G_t for all first visits to s across all episodes

- Unbiased estimator of V^π(s)
- Each episode contributes at most one return per state
- Converges to V^π(s) as number of episodes → ∞

Implemented in `mc_first_visit_prediction()` in `ch05_mc.rs`.
""",
        "theory_every_visit": r"""
**Every-Visit MC Prediction** averages returns from ALL visits to s (not just first):

V(s) ← average of G_t for ALL visits to s across all episodes

- Biased but consistent estimator
- More data per episode → lower variance
- Converges to V^π(s) as number of episodes → ∞

Implemented in `mc_every_visit_prediction()` in `ch05_mc.rs`.
""",
        "theory_on_policy": r"""
**On-Policy MC Control** (epsilon-soft) learns Q*(s,a) and π* simultaneously:

1. Generate episode using epsilon-soft policy
2. For each (s,a) pair (first-visit): G ← discounted return
3. Q(s,a) ← average(G) — incremental update
4. π(s) ← argmax_a Q(s,a) — greedy improvement

The epsilon-soft policy ensures all (s,a) pairs are visited infinitely often.
As epsilon → 0, the policy converges to the optimal greedy policy.

Implemented in `mc_on_policy_control()` in `ch05_mc.rs`.
""",
        "theory_off_policy": r"""
**Off-Policy MC Control** with Weighted Importance Sampling:

- **Behaviour policy b**: generates episodes (uniform random)
- **Target policy π**: what we want to optimise (greedy)
- **Importance Sampling Ratio**: ρ = π(a|s) / b(a|s)

Weighted IS update:
Q(s,a) ← Q(s,a) + (W / C(s,a)) * [G - Q(s,a)]
C(s,a) ← C(s,a) + W

Off-policy MC can learn the optimal policy while following a different (exploratory) policy.
This is the foundation for modern off-policy algorithms like Q-Learning (Ch06).

Implemented in `mc_off_policy_control()` in `ch05_mc.rs`.
""",
        "algo_labels": {
            "first_visit": "First-Visit MC",
            "every_visit": "Every-Visit MC",
            "on_policy":   "On-Policy Control",
            "off_policy":  "Off-Policy (IS)",
        },
        "pros_list": {
            "first_visit": ["Unbiased estimator", "Simple implementation", "No model needed"],
            "every_visit": ["More data per episode", "Lower variance than first-visit", "No model needed"],
            "on_policy":   ["Learns Q* and π* simultaneously", "No model needed", "Guaranteed to visit all (s,a)"],
            "off_policy":  ["Learns from any behaviour policy", "Foundation for Q-Learning", "Can reuse historical data"],
        },
        "cons_list": {
            "first_visit": ["High variance", "Must wait for episode end", "Slow for long episodes"],
            "every_visit": ["Biased estimator", "Must wait for episode end", "Slow for long episodes"],
            "on_policy":   ["Epsilon must stay > 0", "Slower convergence than DP", "High variance"],
            "off_policy":  ["IS ratio can explode", "High variance", "Complex implementation"],
        },
    }}

COLORS = {
    "first_visit": "#0082F0",
    "every_visit": "#FF8C0A",
    "on_policy":   "#0FC373",
    "off_policy":  "#FF3232",
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
        st.subheader("Hands-On Guide \u2014 Chapter 05 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch05_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("\U0001f4be Save", data=_html, file_name="handson_ch05_en.html", mime="text/html")
    st.iframe(_html, height=4000)

def render():
    lang = "EN"
    tx = _tx(lang)

    _tab_main, _tab_handbook = st.tabs(["\U0001f4ca Chapter", "\U0001f4d8 Hands-On Guide EN"])
    with _tab_handbook:
        _render_handbook()
    with _tab_main:

        st.title(tx["title"])
        st.caption(tx["subtitle"])
        try:
            import rlvr_py
        except ImportError:
            st.error(tx["engine_missing"])
            return

        st.sidebar.header(tx["sidebar_title"])
        n_episodes    = st.sidebar.slider(tx["n_episodes"],    50, 5000, 500, 50)
        gamma         = st.sidebar.slider(tx["gamma"],         0.5, 0.999, 0.95, 0.005)
        epsilon       = st.sidebar.slider(tx["epsilon"],       0.0, 1.0, 0.3, 0.05)
        epsilon_decay = st.sidebar.slider(tx["epsilon_decay"], 0.0, 0.1, 0.01, 0.001, format="%.3f")
        seed          = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

        if st.button(tx["run_btn"], type="primary"):
            with st.spinner("Running Rust MC engine..."):
                result = rlvr_py.run_ch05_mc(
                    int(seed), int(n_episodes), float(gamma),
                    float(epsilon), float(epsilon_decay)
                )
            st.session_state["ch05_result"] = result

        if "ch05_result" not in st.session_state:
            st.info("Configure settings and click **▶ Run All Four MC Algorithms**.")
            return

        result      = st.session_state["ch05_result"]
        state_names = result["state_names"]
        action_names= result["action_names"]
        algos       = ["first_visit", "every_visit", "on_policy", "off_policy"]

        # KPI
        cols = st.columns(4)
        for i, key in enumerate(algos):
            r = result[key]
            avg_ret = sum(r["returns_curve"][-50:]) / min(50, len(r["returns_curve"]))
            cols[i].metric(tx["algo_labels"][key], f"Avg return: {avg_ret:.2f}")

        # Returns curve
        st.subheader(tx["returns_title"])
        fig = go.Figure()
        for key in algos:
            r = result[key]
            ma = _moving_avg(r["returns_curve"], 30)
            fig.add_trace(go.Scatter(
                x=list(range(len(ma))), y=ma,
                mode="lines", name=tx["algo_labels"][key],
                line=dict(color=COLORS[key], width=2),
            ))
        fig.update_layout(height=300, margin=dict(l=40,r=20,t=20,b=40),
                          xaxis_title="Episode", yaxis_title="Return (MA-30)",
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, width='stretch')
        st.caption(tx["returns_caption"])

        # Value function comparison
        st.subheader(tx["value_title"])
        short = [f"S{i}" for i in range(result["n_states"])]
        fig2 = go.Figure()
        for key in algos:
            fig2.add_trace(go.Bar(
                x=short, y=result[key]["values"],
                name=tx["algo_labels"][key],
                marker_color=COLORS[key], opacity=0.8,
            ))
        fig2.update_layout(height=300, barmode="group",
                           margin=dict(l=40,r=20,t=20,b=40),
                           legend=dict(orientation="h"))
        st.plotly_chart(fig2, width='stretch')
        st.caption(tx["value_caption"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(tx["visits_title"])
            vc = result["first_visit"]["visit_counts"]
            colors_vc = ["#0FC373" if v > 50 else "#FF8C0A" if v > 10 else "#FF3232" for v in vc]
            fig3 = go.Figure(go.Bar(x=short, y=vc, marker_color=colors_vc,
                                    text=[str(v) for v in vc], textposition="outside"))
            fig3.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40))
            st.plotly_chart(fig3, width='stretch')
            st.caption(tx["visits_caption"])

        with col2:
            st.subheader(tx["conv_title"])
            fig4 = go.Figure()
            for key in ["first_visit", "on_policy"]:
                fig4.add_trace(go.Scatter(
                    x=list(range(len(result[key]["convergence_curve"]))),
                    y=result[key]["convergence_curve"],
                    mode="lines", name=tx["algo_labels"][key],
                    line=dict(color=COLORS[key], width=1.5),
                ))
            fig4.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
                               yaxis_type="log", legend=dict(orientation="h"))
            st.plotly_chart(fig4, width='stretch')
            st.caption(tx["conv_caption"])

        # Q-table heatmap
        st.subheader(tx["qtable_title"])
        qt = result["on_policy"]["q_table"]
        action_short = [f"A{i}" for i in range(result["n_actions"])]
        fig5 = go.Figure(go.Heatmap(
            z=qt, x=action_short, y=short,
            colorscale="Blues",
            text=[[f"{qt[s][a]:.2f}" for a in range(result["n_actions"])]
                  for s in range(result["n_states"])],
            texttemplate="%{text}",
        ))
        fig5.update_layout(height=320, margin=dict(l=60,r=20,t=20,b=40))
        st.plotly_chart(fig5, width='stretch')
        st.caption(tx["qtable_caption"])

        # Glass-Box
        st.subheader(tx["glass_title"])
        _render_glass_box(result, tx, state_names, action_names)

        # Summary
        st.subheader(tx["summary_title"])
        _render_summary(result, tx, algos)



def _render_glass_box(result, tx, state_names, action_names):
    algo_options = {tx["algo_labels"][k]: k for k in ["first_visit", "every_visit", "on_policy", "off_policy"]}
    selected = st.selectbox("Algorithm", list(algo_options.keys()))
    key = algo_options[selected]
    r = result[key]
    curve = r["returns_curve"]
    ep_idx = st.slider("Episode", 0, max(len(curve)-1, 0), max(len(curve)-1, 0))
    st.metric("Episode return", f"{curve[ep_idx]:.3f}")
    st.latex(r"G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}")


def _render_summary(result, tx, algos):
    st.markdown(f"#### {tx['summary_results']}")
    rows = []
    for key in algos:
        r = result[key]
        avg_last = sum(r["returns_curve"][-100:]) / min(100, len(r["returns_curve"]))
        rows.append({
            "Algorithm":        tx["algo_labels"][key],
            "Avg return (last 100)": f"{avg_last:.3f}",
            "Best V*(S0)":      f"{r['values'][0]:.3f}",
            "Worst V*(S7)":     f"{r['values'][7]:.3f}",
            "Best action S7":   f"A{r['policy'][7]}",
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

