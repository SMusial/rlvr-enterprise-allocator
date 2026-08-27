import os
import streamlit as st
import plotly.graph_objects as go

T = {"EN": {
        "title": "Chapter 06 — Temporal Difference Learning",
        "subtitle": "TD(0) · SARSA · Q-Learning · ASP Dispatch · Warsaw Region",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ TD Settings",
        "n_episodes": "Number of episodes",
        "gamma": "γ — Discount factor",
        "alpha": "α — Learning rate",
        "epsilon": "ε — Initial exploration",
        "epsilon_decay": "ε decay rate",
        "seed": "Random seed",
        "run_btn": "▶ Run TD(0), SARSA and Q-Learning",
        "guide": """
**Step 1 — Understand the key difference from Ch05 (MC)**
TD methods update after EVERY step, not after episode end.
They bootstrap: use current estimates V(s') to update V(s).
This makes them faster and applicable to continuing tasks.

**Step 2 — Understand SARSA vs Q-Learning**
Both are TD control methods. The difference is one line:
- SARSA: uses Q(s', a') where a' is chosen by epsilon-greedy (on-policy)
- Q-Learning: uses max_a' Q(s', a') regardless of what action is taken (off-policy)

**Step 3 — Set α (learning rate)**
α=0.1 is a good start. Higher α = faster learning but less stable.
Lower α = slower but more stable convergence.

**Step 4 — Click ▶ Run TD(0), SARSA and Q-Learning**
All three run simultaneously. Results appear side by side.

**Step 5 — Read the TD Error curve**
TD error = R + γV(s') - V(s). Watch it decay toward zero.
This is the learning signal — the agent is "surprised" less over time.

**Step 6 — Compare Q-Learning vs SARSA policies**
Q-Learning finds the optimal policy (off-policy).
SARSA finds the safest policy given epsilon-soft exploration (on-policy).

**Step 7 — Compare with DP reference (Ch04)**
TD methods should converge to the same policy as DP — without needing P(s'|s,a).
""",
        "returns_title": "📈 Episode Returns — TD(0), SARSA, Q-Learning",
        "returns_caption": "Moving average of episode returns. Q-Learning should converge fastest.",
        "td_error_title": "📉 TD Error — |R + γV(s') - V(s)|",
        "td_error_caption": "TD error decays as the agent learns. Smaller = better estimates.",
        "value_title": "📊 Value Function V(s) — TD vs DP Reference",
        "value_caption": "TD estimates should converge toward DP solution (Ch04) with more episodes.",
        "policy_title": "🎯 Optimal Policy — SARSA vs Q-Learning",
        "policy_caption": "Q-Learning finds optimal policy. SARSA finds safest policy under epsilon-soft.",
        "qtable_title": "📊 Q-Table Heatmap",
        "qtable_caption": "Q(s,a) values. Select algorithm to display.",
        "glass_title": "🔬 Glass-Box — TD Update Trace",
        "glass_headers": ["Episode", "Step", "State", "Action", "Reward", "Next State", "TD Error"],
        "summary_title": "📊 Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "TD Algorithms — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "theory_td0": r"""
**TD(0) Prediction** updates V(s) after every step:

V(S_t) <- V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]

- α = learning rate (step size)
- R_{t+1} + γ V(S_{t+1}) = TD target
- R_{t+1} + γ V(S_{t+1}) - V(S_t) = TD error δ_t

Converges to V^π for any fixed policy π.
Implemented in `td0_prediction()` in `ch06_td.rs`.
""",
        "theory_sarsa": r"""
**SARSA** (State-Action-Reward-State-Action) is on-policy TD control:

Q(S_t, A_t) <- Q(S_t, A_t) + α [R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

Key: A_{t+1} is chosen by the SAME epsilon-greedy policy used for behaviour.
This makes SARSA on-policy — it learns the value of the epsilon-soft policy.

SARSA is conservative: it accounts for the exploration cost in its Q estimates.
Implemented in `sarsa()` in `ch06_td.rs`.
""",
        "theory_qlearning": r"""
**Q-Learning** is off-policy TD control:

Q(S_t, A_t) <- Q(S_t, A_t) + α [R_{t+1} + γ max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]

Key: uses max_{a'} Q(S_{t+1}, a') — the GREEDY action, regardless of what was actually taken.
This makes Q-Learning off-policy — it directly learns Q* (optimal action-value function).

Q-Learning converges to Q* regardless of the behaviour policy (as long as all (s,a) are visited).
Implemented in `q_learning()` in `ch06_td.rs`.
""",
        "theory_comparison": r"""
**SARSA vs Q-Learning — the one-line difference:**

SARSA:      Q(s,a) += α [R + γ Q(s', a') - Q(s,a)]   where a' ~ ε-greedy
Q-Learning: Q(s,a) += α [R + γ max_a' Q(s', a') - Q(s,a)]

**When does it matter?**
In risky environments (like S7 — SLA breach imminent):
- SARSA avoids risky states because it accounts for epsilon-exploration
- Q-Learning ignores exploration cost and finds the theoretically optimal policy

**Cliff Walking analogy:**
- SARSA walks safely away from the cliff (accounts for accidental falls)
- Q-Learning walks along the cliff edge (optimal but risky during learning)

In ASP: SARSA is safer during training, Q-Learning finds the better final policy.
""",
        "algo_labels": {"td0": "TD(0)", "sarsa": "SARSA", "qlearning": "Q-Learning"},
        "pros_list": {
            "td0":       ["Online learning — updates every step", "No model needed", "Lower variance than MC"],
            "sarsa":     ["Safe during learning", "On-policy — consistent with behaviour", "Converges to optimal epsilon-soft policy"],
            "qlearning": ["Directly learns Q*", "Off-policy — can learn from any data", "Converges to optimal greedy policy"],
        },
        "cons_list": {
            "td0":       ["Only predicts V^pi — needs separate policy", "Biased (bootstrapping)", "Sensitive to alpha"],
            "sarsa":     ["Suboptimal if epsilon stays high", "On-policy — needs epsilon > 0", "Slower than Q-Learning in safe environments"],
            "qlearning": ["Can be risky during learning", "Overestimates Q values (maximisation bias)", "Sensitive to alpha"],
        },
    }}

COLORS = {"td0": "#0082F0", "sarsa": "#FF8C0A", "qlearning": "#0FC373"}

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
        st.subheader("Hands-On Guide \u2014 Chapter 06 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch06_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("\U0001f4be Save", data=_html, file_name="handson_ch06_en.html", mime="text/html")
    st.iframe(_html, height=4000)

def render():
    lang = "EN"
    tx = _tx(lang)
    st.title(tx["title"])
    st.caption(tx["subtitle"])

    _tab_main, _tab_handbook = st.tabs(["\U0001f52c Interactive Lab", "\U0001f4d8 Hands-On Guide EN"])
    with _tab_handbook:
        _render_handbook()
    with _tab_main:

        try:
            import rlvr_py
        except ImportError:
            st.error(tx["engine_missing"])
            return

        st.sidebar.header(tx["sidebar_title"])
        n_episodes    = st.sidebar.slider(tx["n_episodes"],    50, 5000, 1000, 50)
        gamma         = st.sidebar.slider(tx["gamma"],         0.5, 0.999, 0.95, 0.005)
        alpha         = st.sidebar.slider(tx["alpha"],         0.01, 1.0, 0.1, 0.01)
        epsilon       = st.sidebar.slider(tx["epsilon"],       0.0, 1.0, 0.3, 0.05)
        epsilon_decay = st.sidebar.slider(tx["epsilon_decay"], 0.0, 0.1, 0.01, 0.001, format="%.3f")
        seed          = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

        if st.button(tx["run_btn"], type="primary"):
            with st.spinner("Running Rust TD engine..."):
                result = rlvr_py.run_ch06_td(
                    int(seed), int(n_episodes), float(gamma),
                    float(alpha), float(epsilon), float(epsilon_decay)
                )
            st.session_state["ch06_result"] = result

        if "ch06_result" not in st.session_state:
            st.info("Configure settings and click **▶ Run TD(0), SARSA and Q-Learning**.")
            return

        result       = st.session_state["ch06_result"]
        state_names  = result["state_names"]
        action_names = result["action_names"]
        algos        = ["td0", "sarsa", "qlearning"]

        # KPI
        cols = st.columns(3)
        for i, key in enumerate(algos):
            r = result[key]
            avg = sum(r["returns_curve"][-50:]) / min(50, len(r["returns_curve"]))
            cols[i].metric(tx["algo_labels"][key],
                           f"Avg return: {avg:.2f}",
                           f"Steps: {r['total_steps']:,}")

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
        for key in ["sarsa", "qlearning"]:
            ma = _moving_avg(result[key]["td_error_curve"], 30)
            fig2.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
                mode="lines", name=tx["algo_labels"][key],
                line=dict(color=COLORS[key], width=2)))
        fig2.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
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
        fig3.update_layout(height=300, barmode="group",
                           margin=dict(l=40,r=20,t=20,b=40),
                           legend=dict(orientation="h"))
        st.plotly_chart(fig3, width='stretch')
        st.caption(tx["value_caption"])

        # Policy comparison
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(tx["policy_title"])
            rows = []
            for s in range(result["n_states"]):
                sa = result["sarsa"]["policy"][s]
                ql = result["qlearning"]["policy"][s]
                rows.append({
                    "State": f"S{s}",
                    "SARSA": f"A{sa}",
                    "Q-Learning": f"A{ql}",
                    "Match": "✅" if sa == ql else "🔄",
                })
            st.dataframe(rows, hide_index=True)
            st.caption(tx["policy_caption"])

        with col2:
            st.subheader(tx["qtable_title"])
            algo_sel = st.selectbox("Algorithm", [tx["algo_labels"][k] for k in ["sarsa","qlearning"]])
            key_sel = "sarsa" if "SARSA" in algo_sel or "sarsa" in algo_sel.lower() else "qlearning"
            qt = result[key_sel]["q_table"]
            action_short = [f"A{i}" for i in range(result["n_actions"])]
            fig4 = go.Figure(go.Heatmap(
                z=qt, x=action_short, y=short,
                colorscale="Blues",
                text=[[f"{qt[s][a]:.2f}" for a in range(result["n_actions"])]
                      for s in range(result["n_states"])],
                texttemplate="%{text}",
            ))
            fig4.update_layout(height=320, margin=dict(l=60,r=20,t=20,b=40))
            st.plotly_chart(fig4, width='stretch')
            st.caption(tx["qtable_caption"])

        # Glass-Box
        st.subheader(tx["glass_title"])
        _render_glass_box(result, tx)

        # Summary
        st.subheader(tx["summary_title"])
        _render_summary(result, tx, algos)



def _render_glass_box(result, tx):
        algo_options = {tx["algo_labels"][k]: k for k in ["td0","sarsa","qlearning"]}
        selected = st.selectbox("Algorithm", list(algo_options.keys()), key="gb_algo")
        key = algo_options[selected]
        r = result[key]
        ep_idx = st.slider("Episode", 0, max(len(r["returns_curve"])-1, 0),
                           max(len(r["returns_curve"])-1, 0), key="gb_ep")
        st.metric("Episode return", f"{r['returns_curve'][ep_idx]:.3f}")
        st.metric("Avg TD error this episode", f"{r['td_error_curve'][ep_idx]:.4f}")
        if key == "sarsa":
            st.latex(r"Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]")
        elif key == "qlearning":
            st.latex(r"Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)]")
        else:
            st.latex(r"V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]")


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
                "V*(S0)":              f"{r['values'][0]:.3f}",
                "V*(S7)":              f"{r['values'][7]:.3f}",
                "Policy S7":           f"A{r['policy'][7]}",
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

