import os
import streamlit as st
import plotly.graph_objects as go

T = {"EN": {
        "title": "Chapter 09 — Policy Gradient: REINFORCE & Softmax",
        "subtitle": "REINFORCE · REINFORCE+Baseline · Actor-Critic TD(0) · Temperature · Warsaw ASP",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Settings",
        "n_episodes": "Episodes", "gamma": "γ Discount", "alpha": "α Policy LR",
        "alpha_baseline": "α_v Baseline LR", "temperature": "τ Temperature", "seed": "Seed",
        "run_btn": "▶ Run All Four Algorithms",
        "returns_title": "📈 Episode Returns", "returns_caption": "MA-30. REINFORCE+Baseline most stable.",
        "pg_loss_title": "📉 PG Magnitude", "pg_loss_caption": "Mean |Δθ| per episode.",
        "entropy_title": "🌡️ Policy Entropy", "entropy_caption": "H(π(·|s)). Decreases as policy sharpens.",
        "value_title": "🏛️ V(s)", "value_caption": "S7 (SLA breach) should be lowest.",
        "theta_title": "🗺️ θ[s][a] Heatmap", "theta_caption": "Softmax logits. Higher = more likely.",
        "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Summary", "summary_results": "Comparison",
        "summary_pros_cons": "Pros & Cons", "pros": "✅ Pros", "cons": "❌ Cons",
        "theory_reinf": "G_t = Σ γ^{k-t} R_{k+1}\nθ ← θ + α γ^t G_t ∇log π(A_t|S_t,θ)",
        "theory_baseline": "θ ← θ + α γ^t (G_t−b(S_t)) ∇log π\nb(s) does not introduce bias.",
        "algo_labels": {"reinforce": "REINFORCE", "reinforce_baseline": "REINFORCE+Baseline", "softmax_td0": "Actor-Critic (TD0)", "reinforce_temp": "REINFORCE τ=0.5"},
        "pros_list": {"reinforce": ["Unbiased","Simple","No critic"], "reinforce_baseline": ["Lower variance","Still unbiased","Recommended"], "softmax_td0": ["Online updates","Lower variance than MC","Foundation A2C/PPO"], "reinforce_temp": ["Sharper policy","Less exploration needed","Good deterministic envs"]},
        "cons_list": {"reinforce": ["High variance","Slow","Full episode needed"], "reinforce_baseline": ["Needs α_v","Slightly complex","Still MC"], "softmax_td0": ["Biased (TD)","Two LRs to tune","Can be unstable"], "reinforce_temp": ["Less exploration","Sensitive to τ","May miss global opt"]},
    }}
COLORS = {"reinforce": "#8B5CF6", "reinforce_baseline": "#0082F0", "softmax_td0": "#0FC373", "reinforce_temp": "#FF8C0A"}
ALGOS  = ["reinforce", "reinforce_baseline", "softmax_td0", "reinforce_temp"]

def _ma(data, w=30):
    r = []
    for i in range(len(data)):
        s = max(0, i-w+1); r.append(sum(data[s:i+1])/(i-s+1))
    return r


def _tx(lang=None):
    import copy
    return copy.deepcopy(T.get("EN", {}))

def _render_handbook():
    import os as _os
    _col1, _col2 = st.columns([8, 1])
    with _col1:
        st.subheader("Hands-On Guide \u2014 Chapter 09 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch09_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("\U0001f4be Save", data=_html, file_name="handson_ch09_en.html", mime="text/html")
    st.iframe(_html, height=4000)

def render():
    lang = "EN"; tx = _tx(lang)
    st.title(tx["title"]); st.caption(tx["subtitle"])
    try: import rlvr_py
    except ImportError: st.error(tx["engine_missing"]); return
    st.sidebar.header(tx["sidebar_title"])
    n_ep  = st.sidebar.slider(tx["n_episodes"],    50, 3000, 500, 50)
    gamma = st.sidebar.slider(tx["gamma"],         0.5, 0.999, 0.95, 0.005)
    alpha = st.sidebar.slider(tx["alpha"],         0.001, 0.1, 0.01, 0.001, format="%.3f")
    ab    = st.sidebar.slider(tx["alpha_baseline"],0.01, 0.5, 0.1, 0.01)
    tau   = st.sidebar.slider(tx["temperature"],   0.1, 3.0, 1.0, 0.1)
    seed  = st.sidebar.number_input(tx["seed"], 0, 9999, 42)
    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running..."):
            res = rlvr_py.run_ch09_policy_gradient(int(seed), int(n_ep), float(gamma), float(alpha), float(ab), float(tau))
        st.session_state["ch09_result"] = res
    if "ch09_result" not in st.session_state:
        st.info("Click ▶ to run."); return
    res = st.session_state["ch09_result"]
    short = [f"S{i}" for i in range(res["n_states"])]
    cols = st.columns(4)
    for i, k in enumerate(ALGOS):
        avg = sum(res[k]["returns_curve"][-50:]) / min(50, len(res[k]["returns_curve"]))
        aen = sum(res[k]["entropy_curve"][-50:]) / max(1, min(50, len(res[k]["entropy_curve"])))
        cols[i].metric(tx["algo_labels"][k], f"Avg:{avg:.2f}", f"H:{aen:.2f}")
    st.subheader(tx["returns_title"])
    fig = go.Figure()
    for k in ALGOS: fig.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["returns_curve"]), mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
    fig.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40), xaxis_title="Episode", yaxis_title="Return (MA-30)", legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch'); st.caption(tx["returns_caption"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(tx["pg_loss_title"])
        f2 = go.Figure()
        for k in ALGOS: f2.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["pg_loss_curve"]), mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
        f2.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40), xaxis_title="Episode", yaxis_title="Mean |Δθ|", legend=dict(orientation="h"))
        st.plotly_chart(f2, width='stretch'); st.caption(tx["pg_loss_caption"])
    with c2:
        st.subheader(tx["entropy_title"])
        f3 = go.Figure()
        for k in ALGOS: f3.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["entropy_curve"]), mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
        f3.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40), xaxis_title="Episode", yaxis_title="H(π(·|s))", legend=dict(orientation="h"))
        st.plotly_chart(f3, width='stretch'); st.caption(tx["entropy_caption"])
    st.subheader(tx["value_title"])
    f4 = go.Figure()
    for k in ALGOS: f4.add_trace(go.Bar(x=short, y=res[k]["values"], name=tx["algo_labels"][k], marker_color=COLORS[k], opacity=0.8))
    f4.update_layout(height=260, barmode="group", margin=dict(l=40,r=20,t=20,b=40), legend=dict(orientation="h"))
    st.plotly_chart(f4, width='stretch'); st.caption(tx["value_caption"])
    st.subheader(tx["theta_title"])
    sel  = st.selectbox("Algorithm", [tx["algo_labels"][k] for k in ALGOS])
    ks   = {tx["algo_labels"][k]: k for k in ALGOS}.get(sel, "reinforce")
    th   = res[ks]["theta"]; ash = [f"A{i}" for i in range(res["n_actions"])]
    f5   = go.Figure(go.Heatmap(z=th, x=ash, y=short, colorscale="Purples", text=[[f"{th[s][a]:.3f}" for a in range(res["n_actions"])] for s in range(res["n_states"])], texttemplate="%{text}"))
    f5.update_layout(height=280, margin=dict(l=60,r=20,t=20,b=40)); st.plotly_chart(f5, width='stretch')
    st.subheader(tx["glass_title"]); _glass(res, tx)
    st.subheader(tx["summary_title"]); _summary(res, tx)

def _glass(res, tx):
    opts = {tx["algo_labels"][k]: k for k in ALGOS}
    sel  = st.selectbox("Algorithm", list(opts.keys()), key="gb9")
    k    = opts[sel]; r = res[k]
    ep   = st.slider("Episode", 0, max(len(r["returns_curve"])-1,0), max(len(r["returns_curve"])-1,0), key="gb9ep")
    c1,c2,c3 = st.columns(3)
    c1.metric("Return", f"{r['returns_curve'][ep]:.3f}")
    c2.metric("|Δθ|",   f"{r['pg_loss_curve'][ep]:.4f}")
    c3.metric("H(π)",   f"{r['entropy_curve'][ep]:.4f}")
    if "actor_critic" in k:
        st.latex(r"\delta_t=R_{t+1}+\gamma V(S')-V(S)")
        st.latex(r"V(S)\leftarrow V(S)+\alpha_v\delta_t")
        st.latex(r"\theta\leftarrow\theta+\alpha\gamma^t\delta_t\nabla\log\pi(A_t|S_t,\theta)")
    else:
        st.latex(r"G_t=\sum_{k=t}^T\gamma^{k-t}R_{k+1}")
        st.latex(r"\theta\leftarrow\theta+\alpha\gamma^t G_t\nabla\log\pi(A_t|S_t,\theta)")
        st.latex(r"\nabla\log\pi(a|s)=\mathbf{1}[a=A_t]-\pi(a|s,\theta)")

def _summary(res, tx):
    rows = []
    for k in ALGOS:
        r = res[k]; avg = sum(r["returns_curve"][-100:])/min(100,len(r["returns_curve"]))
        aen = sum(r["entropy_curve"])/max(1,len(r["entropy_curve"]))
        rows.append({"Algorithm": tx["algo_labels"][k], "Avg return (last 100)": f"{avg:.3f}", "Steps": str(r["total_steps"]), "Avg H": f"{aen:.3f}", "V*(S0)": f"{r['values'][0]:.3f}", "V*(S7)": f"{r['values'][7]:.3f}"})
    st.dataframe(rows, hide_index=True)
    for k in ALGOS:
        label = tx["algo_labels"][k]; c1,c2 = st.columns(2)
        with c1: st.markdown(f"**{label} — {tx['pros']}**"); [st.markdown(f"- {p}") for p in tx["pros_list"][k]]
        with c2: st.markdown(f"**{label} — {tx['cons']}**"); [st.markdown(f"- {c}") for c in tx["cons_list"][k]]
        st.markdown("---")
