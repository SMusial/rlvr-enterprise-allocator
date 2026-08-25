import streamlit as st
import plotly.graph_objects as go

COLORS = {
    "first_visit": "#0082F0",
    "every_visit": "#FF8C0A",
    "on_policy":   "#0FC373",
    "off_policy":  "#FF3232"
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
            "Best action S7":   f"A{r['policy'][7]}"
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

