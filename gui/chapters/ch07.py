import streamlit as st
import plotly.graph_objects as go

COLORS = {
    "nstep_td":    "#0082F0",
    "nstep_sarsa": "#FF8C0A",
    "dyna_q":      "#0FC373",
    "dyna_q_plus": "#FF3232"
}

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
        extra = f"Model: {r['model_size']}" if r["model_size"] > 0 else f"Steps: {r['total_steps']:}"
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
            "V*(S7)":              f"{r['values'][7]:.3f}"
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

