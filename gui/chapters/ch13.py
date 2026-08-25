
import streamlit as st
import plotly.graph_objects as go

COLORS = {"iql":"#8B5CF6","vdn":"#0082F0","qmix":"#0FC373","qmix_cg":"#FF8C0A"}
ALGOS  = ["iql","vdn","qmix","qmix_cg"]

TX = {"EN": {
        "title":    "Chapter 13 - Cooperative MARL: VDN and QMIX",
        "subtitle": "IQL Baseline - VDN - QMIX - QMIX+CG - 2 Agents - Warsaw ASP",
        "run":      "Run All Four Algorithms",
        "ret":      "Joint Episode Returns",
        "mix":      "Mixing Weights (QMIX)",
        "jq":       "Joint Q_tot",
        "val":      "Joint Value Function V(s)",
        "glass":    "Glass-Box",
        "summary":  "Summary",
        "episodes": "Episodes", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon":  "Epsilon",  "edecay": "Epsilon decay",
        "mhidden":  "Mixing network hidden units", "seed": "Seed",
        "labels":   {"iql":"IQL Baseline","vdn":"VDN","qmix":"QMIX","qmix_cg":"QMIX+CG"},
        "guide": (
            "Cooperative setting: both agents share the SAME joint reward.\n"
            "Goal: maximise joint return, not individual returns.\n\n"
            "IQL Baseline: each agent ignores the other (Ch11 style).\n"
            "VDN: Q_tot = Q_0 + Q_1. Simple additive decomposition.\n"
            "QMIX: Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s). Monotone mixing.\n"
            "QMIX+CG: QMIX + counterfactual baseline for credit assignment.\n\n"
            "Key property - IGM (Individual-Global-Max):\n"
            "  argmax Q_tot = (argmax Q_0, argmax Q_1)\n"
            "This enables decentralised execution with centralised training."
        ),
        "theory_igm":   "IGM: argmax_a Q_tot(s,a) = (argmax Q_0(s_0,.), argmax Q_1(s_1,.))",
        "theory_vdn":   "Q_tot = Q_0(s_0,a_0) + Q_1(s_1,a_1)\ndQ_tot/dQ_i = 1 (equal gradient)",
        "theory_qmix":  "Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s)\nw_i(s) >= 0 (monotonicity)\ndQ_tot/dQ_i = w_i(s) (state-dependent gradient)",
        "theory_cg":    "Counterfactual advantage:\nA_i = Q_tot(s,a) - Q_tot(s, a_{-i}, argmax Q_i)\nIsolates agent i contribution to joint value."
    }}

def _ma(data, w=30):
    r = []
    for i in range(len(data)):
        s = max(0, i-w+1); r.append(sum(data[s:i+1])/(i-s+1))
    return r


def _tx(lang=None):
    import copy
    return copy.deepcopy(TX.get("EN", {}))

def render():
    lang = "EN"
    tx   = _tx()
    lb   = tx["labels"]
    st.title(tx["title"]); st.caption(tx["subtitle"])
    try: import rlvr_py
    except ImportError: st.error("Run: cd rlvr-py && maturin develop"); return

    st.sidebar.header("Settings")
    n_ep  = st.sidebar.slider(tx["episodes"],  50, 3000, 500, 50)
    gamma = st.sidebar.slider(tx["gamma"],     0.5, 0.999, 0.95, 0.005)
    alpha = st.sidebar.slider(tx["alpha"],     0.01, 1.0, 0.1, 0.01)
    eps   = st.sidebar.slider(tx["epsilon"],   0.0, 1.0, 0.3, 0.05)
    edec  = st.sidebar.slider(tx["edecay"],    0.0, 0.1, 0.01, 0.001, format="%.3f")
    mh    = st.sidebar.slider(tx["mhidden"],   4, 32, 8, 4)
    seed  = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    with st.expander("Guide", expanded=False): st.markdown(tx["guide"])

    if st.button(tx["run"], type="primary"):
        with st.spinner("Running Rust cooperative MARL engine..."):
            res = rlvr_py.run_ch13_coop_marl(int(seed),int(n_ep),float(gamma),float(alpha),float(eps),float(edec),int(mh))
        st.session_state["ch13_result"] = res

    if "ch13_result" not in st.session_state:
        st.info("Click Run."); _theory(tx); return

    res   = st.session_state["ch13_result"]
    short = [f"S{i}" for i in range(res["n_states"])]

    # KPI
    cols = st.columns(4)
    for i,k in enumerate(ALGOS):
        avg = sum(res[k]["returns_curve"][-50:])/min(50,len(res[k]["returns_curve"]))
        mw  = sum(res[k]["mixing_weights"][-50:])/max(1,min(50,len(res[k]["mixing_weights"])))
        cols[i].metric(lb[k], f"Avg:{avg:.2f}", f"W:{mw:.2f}")

    # Returns
    st.subheader(tx["ret"])
    fig = go.Figure()
    for k in ALGOS:
        fig.add_trace(go.Scatter(x=list(range(n_ep)),y=_ma(res[k]["returns_curve"]),
            mode="lines",name=lb[k],line=dict(color=COLORS[k],width=2)))
    fig.update_layout(height=280,margin=dict(l=40,r=20,t=20,b=40),
                      xaxis_title="Episode",yaxis_title="Return (MA-30)",legend=dict(orientation="h"))
    st.plotly_chart(fig,width='stretch')

    # Mixing weights + Joint Q
    c1,c2 = st.columns(2)
    with c1:
        st.subheader(tx["mix"])
        f2 = go.Figure()
        for k in ["qmix","qmix_cg"]:
            f2.add_trace(go.Scatter(x=list(range(n_ep)),y=_ma(res[k]["mixing_weights"]),
                mode="lines",name=lb[k],line=dict(color=COLORS[k],width=2)))
        f2.update_layout(height=260,margin=dict(l=40,r=20,t=20,b=40),
                         xaxis_title="Episode",yaxis_title="Mean |w|",legend=dict(orientation="h"))
        st.plotly_chart(f2,width='stretch')
    with c2:
        st.subheader(tx["jq"])
        f3 = go.Figure()
        for k in ALGOS:
            f3.add_trace(go.Scatter(x=list(range(n_ep)),y=_ma(res[k]["joint_q_curve"]),
                mode="lines",name=lb[k],line=dict(color=COLORS[k],width=2)))
        f3.update_layout(height=260,margin=dict(l=40,r=20,t=20,b=40),
                         xaxis_title="Episode",yaxis_title="Q_tot",legend=dict(orientation="h"))
        st.plotly_chart(f3,width='stretch')

    # Value function
    st.subheader(tx["val"])
    f4 = go.Figure()
    for k in ALGOS:
        f4.add_trace(go.Bar(x=short,y=res[k]["values"],name=lb[k],marker_color=COLORS[k],opacity=0.8))
    f4.update_layout(height=260,barmode="group",margin=dict(l=40,r=20,t=20,b=40),legend=dict(orientation="h"))
    st.plotly_chart(f4,width='stretch')

    # Q-table heatmaps per agent
    st.subheader("Q-Table Heatmap")
    sel = st.selectbox("Algorithm",[lb[k] for k in ALGOS])
    ks  = {lb[k]:k for k in ALGOS}.get(sel,"vdn")
    ash = [f"A{i}" for i in range(res["n_actions"])]
    c1,c2 = st.columns(2)
    for agent_idx,col in enumerate([c1,c2]):
        with col:
            st.markdown(f"**Agent {agent_idx}**")
            qt = res[ks]["q_tables"][agent_idx]
            f5 = go.Figure(go.Heatmap(z=qt,x=ash,y=short,colorscale="Purples",
                text=[[f"{qt[s][a]:.2f}" for a in range(res["n_actions"])] for s in range(res["n_states"])],
                texttemplate="%{text}"))
            f5.update_layout(height=260,margin=dict(l=60,r=10,t=20,b=40))
            st.plotly_chart(f5,width='stretch')

    st.subheader(tx["glass"]); _glass(res,lb)
    st.subheader(tx["summary"]); _summary(res,lb)
    _theory(tx)

def _glass(res,lb):
    opts = {lb[k]:k for k in ALGOS}
    sel  = st.selectbox("Algorithm",list(opts.keys()),key="gb13")
    k    = opts[sel]; r = res[k]
    ep   = st.slider("Episode",0,max(len(r["returns_curve"])-1,0),max(len(r["returns_curve"])-1,0),key="gb13ep")
    c1,c2,c3 = st.columns(3)
    c1.metric("Joint return",  f"{r['returns_curve'][ep]:.3f}")
    c2.metric("TD error",      f"{r['td_error_curve'][ep]:.4f}")
    c3.metric("Mixing weight", f"{r['mixing_weights'][ep]:.3f}")
    if k=="iql":
        st.latex(r"Q_i(s,a) \leftarrow Q_i + \alpha[r_{joint} + \gamma \max Q_i(s') - Q_i(s,a)]")
        st.markdown("No coordination. Joint reward used but agents act independently.")
    elif k=="vdn":
        st.latex(r"Q_{tot} = Q_0(s_0,a_0) + Q_1(s_1,a_1)")
        st.latex(r"\frac{\partial Q_{tot}}{\partial Q_i} = 1")
    elif k=="qmix":
        st.latex(r"Q_{tot} = w_0(s)Q_0 + w_1(s)Q_1 + b(s),\quad w_i \geq 0")
        st.latex(r"\frac{\partial Q_{tot}}{\partial Q_i} = w_i(s)")
    else:
        st.latex(r"A_i = Q_{tot}(s,\mathbf{a}) - Q_{tot}(s, \mathbf{a}_{-i}, \arg\max Q_i)")
        st.markdown("Counterfactual baseline isolates each agent's contribution.")

def _summary(res,lb):
    rows = []
    for k in ALGOS:
        r   = res[k]
        avg = sum(r["returns_curve"][-100:])/min(100,len(r["returns_curve"]))
        mw  = sum(r["mixing_weights"])/max(1,len(r["mixing_weights"]))
        rows.append({"Algorithm":lb[k],"Avg return (last 100)":f"{avg:.3f}",
                     "Steps":str(r["total_steps"]),"Avg mixing w":f"{mw:.3f}",
                     "V*(S0)":f"{r['values'][0]:.3f}","V*(S7)":f"{r['values'][7]:.3f}"})
    st.dataframe(rows,hide_index=True)

def _theory(tx):
    st.markdown("---"); st.subheader("Theory")
    for k,label in [("igm","13.1 IGM Property"),("vdn","13.2 VDN"),
                    ("qmix","13.3 QMIX"),("cg","13.4 Counterfactual Baseline")]:
        with st.expander(label,expanded=False):
            st.markdown(tx.get(f"theory_{k}",""))
