import streamlit as st
import plotly.graph_objects as go

TX = {"EN": {
        "title": "Chapter 12 - Game Theory and Nash Equilibrium",
        "subtitle": "Nash Q - Correlated Q - Minimax Q - Fictitious Play - Warsaw ASP",
        "engine_missing": "Run: cd rlvr-py && maturin develop",
        "guide": (
            "Nash Q: converges to Nash equilibrium.\n\n"
            "Correlated Q: broader than Nash - agents coordinate via joint distribution.\n\n"
            "Minimax Q: zero-sum game - player 0 maximises worst-case payoff.\n\n"
            "Fictitious Play: best response to opponent empirical average strategy.\n\n"
            "Watch Nash Gap chart: lower = closer to equilibrium."
        ),
        "labels": {"nash_q":"Nash Q","correlated_q":"Correlated Q","minimax_q":"Minimax Q","fictitious":"Fictitious Play"},
        "settings": "Settings", "episodes": "Episodes", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon": "Epsilon", "edecay": "Epsilon decay", "zerosum": "Zero-sum game (Minimax mode)", "seed": "Seed",
        "run": "Run All Four Algorithms", "guide_title": "Guide",
        "ret": "Joint Episode Returns", "gap": "Nash Gap (Exploitability)",
        "strat": "Mixed Strategy Profile", "val": "Joint Value Function V(s)",
        "glass": "Glass-Box", "summary": "Summary"
    }}


COLORS = {"nash_q":"#8B5CF6","correlated_q":"#0082F0","minimax_q":"#0FC373","fictitious":"#FF8C0A"}
ALGOS  = ["nash_q","correlated_q","minimax_q","fictitious"]
LABELS = {"nash_q":"Nash Q","correlated_q":"Correlated Q","minimax_q":"Minimax Q","fictitious":"Fictitious Play"}

def _ma(data,w=30):
    r=[]
    for i in range(len(data)):
        s=max(0,i-w+1); r.append(sum(data[s:i+1])/(i-s+1))
    return r



def _tx(lang=None):
    import copy
    return copy.deepcopy(TX.get("EN", {}))

def render():
    
    
    st.title(tx["title"])
    st.caption(tx["subtitle"])
    try: import rlvr_py
    except ImportError: st.error("Run: cd rlvr-py && maturin develop"); return

    st.sidebar.header(tx["settings"])
    n_ep     = st.sidebar.slider(tx["episodes"],  50,3000,500,50)
    gamma    = st.sidebar.slider(tx["gamma"],     0.5,0.999,0.95,0.005)
    alpha    = st.sidebar.slider(tx["alpha"],     0.01,1.0,0.1,0.01)
    eps      = st.sidebar.slider(tx["epsilon"],   0.0,1.0,0.3,0.05)
    edec     = st.sidebar.slider(tx["edecay"],0.0,0.1,0.01,0.001,format="%.3f")
    zero_sum = st.sidebar.checkbox(tx["zerosum"],value=False)
    seed     = st.sidebar.number_input(tx["seed"],0,9999,42)
    if False: st.markdown(
            "Nash Q: converges to Nash equilibrium - neither player can improve unilaterally.\n\n"
            "Correlated Q: broader than Nash - agents coordinate via joint distribution.\n\n"
            "Minimax Q: zero-sum game - player 0 maximises worst-case payoff.\n\n"
            "Fictitious Play: best response to opponent empirical average strategy.\n\n"
            "Watch Nash Gap chart: lower = closer to equilibrium."
        )

    if st.button(tx["run"],type="primary"):
        with st.spinner("Running..."):
            res = rlvr_py.run_ch12_game_theory(
                int(seed),int(n_ep),float(gamma),float(alpha),
                float(eps),float(edec),bool(zero_sum))
        st.session_state["ch12_result"] = res

    if "ch12_result" not in st.session_state:
        st.info("Configure settings and click Run.")
        return

    res   = st.session_state["ch12_result"]
    short = [f"S{i}" for i in range(res["n_states"])]

    cols = st.columns(4)
    for i,k in enumerate(ALGOS):
        lbl = tx["labels"] if tx else LABELS
        avg = sum(res[k]["returns_curve"][-50:])/min(50,len(res[k]["returns_curve"]))
        ng  = sum(res[k]["nash_gap_curve"][-50:])/max(1,min(50,len(res[k]["nash_gap_curve"])))
        cols[i].metric(lbl[k],f"Avg:{avg:.2f}",f"Gap:{ng:.3f}")

    st.subheader(tx["ret"])
    fig=go.Figure()
    for k in ALGOS:
        fig.add_trace(go.Scatter(x=list(range(n_ep)),y=_ma(res[k]["returns_curve"]),
            mode="lines",name=LABELS[k],line=dict(color=COLORS[k],width=2)))
    fig.update_layout(height=280,margin=dict(l=40,r=20,t=20,b=40),
        xaxis_title="Episode",yaxis_title="Return (MA-30)",legend=dict(orientation="h"))
    st.plotly_chart(fig,width='stretch')

    c1,c2=st.columns(2)
    with c1:
        st.subheader(tx["gap"])
        f2=go.Figure()
        for k in ALGOS:
            f2.add_trace(go.Scatter(x=list(range(n_ep)),y=_ma(res[k]["nash_gap_curve"]),
                mode="lines",name=LABELS[k],line=dict(color=COLORS[k],width=2)))
        f2.update_layout(height=260,margin=dict(l=40,r=20,t=20,b=40),
            xaxis_title="Episode",yaxis_title="Nash Gap",legend=dict(orientation="h"))
        st.plotly_chart(f2,width='stretch')
    with c2:
        st.subheader(tx["val"])
        f3=go.Figure()
        for k in ALGOS:
            f3.add_trace(go.Bar(x=short,y=res[k]["values"],name=LABELS[k],marker_color=COLORS[k],opacity=0.8))
        f3.update_layout(height=260,barmode="group",margin=dict(l=40,r=20,t=20,b=40),legend=dict(orientation="h"))
        st.plotly_chart(f3,width='stretch')

    st.subheader(tx["strat"])
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
    st.plotly_chart(f4,width='stretch')

    st.subheader(tx["glass"])
    _glass(res, tx)
    st.subheader(tx["summary"])
    _summary(res, tx)

def _glass(res, tx=None):
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

def _summary(res, tx=None):
    rows=[]
    for k in ALGOS:
        r=res[k]
        avg=sum(r["returns_curve"][-100:])/min(100,len(r["returns_curve"]))
        ng=sum(r["nash_gap_curve"])/max(1,len(r["nash_gap_curve"]))
        rows.append({"Algorithm":LABELS[k],"Avg return (last 100)":f"{avg:.3f}","Steps":str(r["total_steps"]),"Avg Nash gap":f"{ng:.4f}","V*(S0)":f"{r['values'][0]:.3f}","V*(S7)":f"{r['values'][7]:.3f}"})
    st.dataframe(rows,hide_index=True)
