import streamlit as st
import plotly.graph_objects as go

COLORS = {"nash_q":"#8B5CF6","correlated_q":"#0082F0","minimax_q":"#0FC373","fictitious":"#FF8C0A"}
ALGOS  = ["nash_q","correlated_q","minimax_q","fictitious"]
LABELS = {"nash_q":"Nash Q","correlated_q":"Correlated Q","minimax_q":"Minimax Q","fictitious":"Fictitious Play"}

def _ma(data,w=30):
    r=[]
    for i in range(len(data)):
        s=max(0,i-w+1); r.append(sum(data[s:i+1])/(i-s+1))
    return r

def render():
    lang = st.session_state.get("lang","EN")
    st.title("Chapter 12 - Game Theory and Nash Equilibrium")
    st.caption("Nash Q - Correlated Q - Minimax Q - Fictitious Play - Warsaw ASP")
    try: import rlvr_py
    except ImportError: st.error("Run: cd rlvr-py && maturin develop"); return

    st.sidebar.header("Settings")
    n_ep     = st.sidebar.slider("Episodes",  50,3000,500,50)
    gamma    = st.sidebar.slider("Gamma",     0.5,0.999,0.95,0.005)
    alpha    = st.sidebar.slider("Alpha",     0.01,1.0,0.1,0.01)
    eps      = st.sidebar.slider("Epsilon",   0.0,1.0,0.3,0.05)
    edec     = st.sidebar.slider("Epsilon decay",0.0,0.1,0.01,0.001,format="%.3f")
    zero_sum = st.sidebar.checkbox("Zero-sum game (Minimax mode)",value=False)
    seed     = st.sidebar.number_input("Seed",0,9999,42)

    with st.expander("Guide",expanded=False):
        st.markdown(
            "Nash Q: converges to Nash equilibrium - neither player can improve unilaterally.\n\n"
            "Correlated Q: broader than Nash - agents coordinate via joint distribution.\n\n"
            "Minimax Q: zero-sum game - player 0 maximises worst-case payoff.\n\n"
            "Fictitious Play: best response to opponent empirical average strategy.\n\n"
            "Watch Nash Gap chart: lower = closer to equilibrium."
        )

    if st.button("Run All Four Algorithms",type="primary"):
        with st.spinner("Running..."):
            res = rlvr_py.run_ch12_game_theory(
                int(seed),int(n_ep),float(gamma),float(alpha),
                float(eps),float(edec),bool(zero_sum))
        st.session_state["ch12_result"] = res

    if "ch12_result" not in st.session_state:
        st.info("Configure settings and click Run.")
        _theory()
        return

    res   = st.session_state["ch12_result"]
    short = [f"S{i}" for i in range(res["n_states"])]

    cols = st.columns(4)
    for i,k in enumerate(ALGOS):
        avg = sum(res[k]["returns_curve"][-50:])/min(50,len(res[k]["returns_curve"]))
        ng  = sum(res[k]["nash_gap_curve"][-50:])/max(1,min(50,len(res[k]["nash_gap_curve"])))
        cols[i].metric(LABELS[k],f"Avg:{avg:.2f}",f"Gap:{ng:.3f}")

    st.subheader("Joint Episode Returns")
    fig=go.Figure()
    for k in ALGOS:
        fig.add_trace(go.Scatter(x=list(range(n_ep)),y=_ma(res[k]["returns_curve"]),
            mode="lines",name=LABELS[k],line=dict(color=COLORS[k],width=2)))
    fig.update_layout(height=280,margin=dict(l=40,r=20,t=20,b=40),
        xaxis_title="Episode",yaxis_title="Return (MA-30)",legend=dict(orientation="h"))
    st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        st.subheader("Nash Gap (Exploitability)")
        f2=go.Figure()
        for k in ALGOS:
            f2.add_trace(go.Scatter(x=list(range(n_ep)),y=_ma(res[k]["nash_gap_curve"]),
                mode="lines",name=LABELS[k],line=dict(color=COLORS[k],width=2)))
        f2.update_layout(height=260,margin=dict(l=40,r=20,t=20,b=40),
            xaxis_title="Episode",yaxis_title="Nash Gap",legend=dict(orientation="h"))
        st.plotly_chart(f2,use_container_width=True)
    with c2:
        st.subheader("Joint Value Function V(s)")
        f3=go.Figure()
        for k in ALGOS:
            f3.add_trace(go.Bar(x=short,y=res[k]["values"],name=LABELS[k],marker_color=COLORS[k],opacity=0.8))
        f3.update_layout(height=260,barmode="group",margin=dict(l=40,r=20,t=20,b=40),legend=dict(orientation="h"))
        st.plotly_chart(f3,use_container_width=True)

    st.subheader("Mixed Strategy Profile")
    col_s,state_s=st.columns(2)
    with col_s: sel=st.selectbox("Algorithm",[LABELS[k] for k in ALGOS])
    with state_s: sidx=st.slider("State",0,res["n_states"]-1,0)
    ks={LABELS[k]:k for k in ALGOS}.get(sel,"nash_q")
    ash=[f"A{i}" for i in range(res["n_actions"])]
    sd=[[res[ks]["strategies"][p][sidx][a] for a in range(res["n_actions"])] for p in range(res["n_players"])]
    f4=go.Figure(go.Heatmap(z=sd,x=ash,y=["P0","P1"],colorscale="Purples",
        text=[[f"{sd[p][a]:.3f}" for a in range(res["n_actions"])] for p in range(res["n_players"])],
        texttemplate="%{text}",zmin=0,zmax=1))
    f4.update_layout(height=180,margin=dict(l=60,r=20,t=20,b=40))
    st.plotly_chart(f4,use_container_width=True)

    st.subheader("Glass-Box")
    _glass(res)
    st.subheader("Summary")
    _summary(res)
    _theory()

def _glass(res):
    opts={LABELS[k]:k for k in ALGOS}
    sel=st.selectbox("Algorithm",list(opts.keys()),key="gb12")
    k=opts[sel]; r=res[k]
    ep=st.slider("Episode",0,max(len(r["returns_curve"])-1,0),max(len(r["returns_curve"])-1,0),key="gb12ep")
    c1,c2,c3=st.columns(3)
    c1.metric("Joint return",f"{r['returns_curve'][ep]:.3f}")
    c2.metric("Nash gap",f"{r['nash_gap_curve'][ep]:.4f}")
    c3.metric("Exploitability",f"{r['exploitability'][ep]:.4f}")
    if k=="nash_q":
        st.latex(r"Q_i(s,a_0,a_1) \leftarrow Q_i + \alpha[r_i + \gamma V_i^{Nash}(s') - Q_i(s,a_0,a_1)]")
    elif k=="correlated_q":
        st.latex(r"\sigma(a_0,a_1|s) \text{ updated via regret matching}")
    elif k=="minimax_q":
        st.latex(r"V_0^{mm}(s) = \max_{\pi_0}\min_{\pi_1}\sum_{a_0,a_1}\pi_0(a_0)\pi_1(a_1)Q_0(s,a_0,a_1)")
    else:
        st.latex(r"\hat{\pi}_j(a|s) = N_j(s,a)/\sum_{a'}N_j(s,a')")

def _summary(res):
    rows=[]
    for k in ALGOS:
        r=res[k]
        avg=sum(r["returns_curve"][-100:])/min(100,len(r["returns_curve"]))
        ng=sum(r["nash_gap_curve"])/max(1,len(r["nash_gap_curve"]))
        rows.append({"Algorithm":LABELS[k],"Avg return (last 100)":f"{avg:.3f}","Steps":str(r["total_steps"]),"Avg Nash gap":f"{ng:.4f}","V*(S0)":f"{r['values'][0]:.3f}","V*(S7)":f"{r['values'][7]:.3f}"})
    st.dataframe(rows,hide_index=True)

def _theory():
    st.markdown("---")
    st.subheader("Theory - Chapter 12")
    with st.expander("12.1 Nash Equilibrium",expanded=False):
        st.markdown("V_i(s, pi_i*, pi_j*) >= V_i(s, pi_i, pi_j*) for all pi_i")
    with st.expander("12.2 Nash Q-Learning",expanded=False):
        st.markdown("Q_i(s,a0,a1) += alpha * [r_i + gamma * V_i^Nash(s') - Q_i(s,a0,a1)]")
    with st.expander("12.3 Correlated Q-Learning",expanded=False):
        st.markdown("Joint distribution sigma(a0,a1|s) updated via regret matching.")
    with st.expander("12.4 Minimax Q-Learning",expanded=False):
        st.markdown("Zero-sum: max_{pi_0} min_{pi_1} V_0(s, pi_0, pi_1)")
    with st.expander("12.5 Fictitious Play",expanded=False):
        st.markdown("Best response to empirical average: pi_j(a|s) = N_j(s,a) / sum N_j(s,a)")
