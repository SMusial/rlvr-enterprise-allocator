import streamlit as st
import plotly.graph_objects as go

COLORS = {"td0": "#0082F0", "sarsa": "#FF8C0A", "qlearning": "#0FC373"}

def _moving_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def render():
    
    
    st.title(tx["title"])
    st.caption(tx["subtitle"])
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
                       f"Steps: {r['total_steps']:}")

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
                "Match": "✅" if sa == ql else "🔄"
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
            "Policy S7":           f"A{r['policy'][7]}"
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

