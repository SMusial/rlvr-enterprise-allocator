import streamlit as st
import plotly.graph_objects as go

COLORS = {"iql":"#8B5CF6","vdn":"#0082F0","qmix":"#0FC373","qmix_cg":"#FF8C0A"}
ALGOS  = ["iql","vdn","qmix","qmix_cg"]

TX = {
    "EN": {
        "title":    "Chapter 13 - Cooperative MARL: VDN and QMIX",
        "subtitle": "IQL Baseline - VDN - QMIX - QMIX+CG - 2 Agents - Warsaw ASP",
        "run":      "\u25b6 Run All Four Algorithms",
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
        "guide": """
**Cooperative setting: both agents share the SAME joint reward.**

**Step 1 \u2014 IQL Baseline**
Each agent ignores the other. Baseline: what happens without coordination?

**Step 2 \u2014 VDN (Value Decomposition Networks)**
Q_tot = Q_0 + Q_1. Simple additive decomposition.
IGM: argmax Q_tot = (argmax Q_0, argmax Q_1) \u2014 enables decentralised execution.

**Step 3 \u2014 QMIX**
Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s). Monotone mixing.
w_i(s) >= 0 enforces IGM. State-dependent weights = more expressive than VDN.

**Step 4 \u2014 QMIX+CG (Counterfactual Baseline)**
A_i = Q_tot(s,a) - Q_tot(s, a_{-i}, argmax Q_i). Isolates each agent's contribution.

**Step 5 \u2014 Watch the Mixing Weights chart**
How does QMIX learn state-dependent coordination weights?
""",
        "theory_igm":   "IGM: argmax_a Q_tot(s,a) = (argmax Q_0(s_0,.), argmax Q_1(s_1,.))",
        "theory_vdn":   "Q_tot = Q_0(s_0,a_0) + Q_1(s_1,a_1)",
        "theory_qmix":  "Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s), w_i(s) >= 0",
        "theory_cg":    "A_i = Q_tot(s,a) - Q_tot(s, a_{-i}, argmax Q_i)",
    },
    "DE": {
        "title":    "Kapitel 13 \u2014 Kooperatives MARL: VDN und QMIX",
        "subtitle": "IQL Baseline \u2014 VDN \u2014 QMIX \u2014 QMIX+CG \u2014 2 Agenten \u2014 ASP Warschau",
        "run":      "\u25b6 Alle vier Algorithmen starten",
        "ret":      "Gemeinsame Episodenr\u00fcckgaben",
        "mix":      "Mischgewichte (QMIX)",
        "jq":       "Gemeinsames Q_tot",
        "val":      "Gemeinsame Wertfunktion V(s)",
        "glass":    "Glass-Box", "summary": "Zusammenfassung",
        "episodes": "Episoden", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon":  "Epsilon", "edecay": "Epsilon-Abklingrate",
        "mhidden":  "Versteckte Einheiten des Mischnetzwerks", "seed": "Zufallsseed",
        "labels":   {"iql":"IQL Baseline","vdn":"VDN","qmix":"QMIX","qmix_cg":"QMIX+CG"},
        "guide": """
**Kooperatives Szenario: beide Agenten teilen DIESELBE gemeinsame Belohnung.**

**Schritt 1 \u2014 IQL Basislinie**
Jeder Agent ignoriert den anderen. Was passiert ohne Koordination?

**Schritt 2 \u2014 VDN (Value Decomposition Networks)**
Q_tot = Q_0 + Q_1. Einfache additive Zerlegung.
IGM: argmax Q_tot = (argmax Q_0, argmax Q_1) \u2014 erm\u00f6glicht dezentrale Ausf\u00fchrung.

**Schritt 3 \u2014 QMIX**
Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s). Monotones Mischen.
w_i(s) >= 0 erzwingt IGM. Zustandsabh\u00e4ngige Gewichte = ausdrucksst\u00e4rkere Koordination.

**Schritt 4 \u2014 QMIX+CG (Kontrafaktische Basislinie)**
A_i = Q_tot(s,a) - Q_tot(s, a_{{-i}}, argmax Q_i). Isoliert den Beitrag jedes Agenten.

**Schritt 5 \u2014 Mischgewichte-Diagramm beobachten**
Wie lernt QMIX zustandsabh\u00e4ngige Koordinationsgewichte?
""",
        "theory_igm":  "IGM: argmax Q_tot = (argmax Q_0, argmax Q_1)",
        "theory_vdn":  "Q_tot = Q_0 + Q_1",
        "theory_qmix": "Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s), w_i >= 0",
        "theory_cg":   "A_i = Q_tot(s,a) - Q_tot(s, a_{{-i}}, argmax Q_i)",
    },
    "FR": {
        "title":    "Chapitre 13 \u2014 MARL Coop\u00e9ratif: VDN et QMIX",
        "subtitle": "IQL \u2014 VDN \u2014 QMIX \u2014 QMIX+CG \u2014 ASP Varsovie",
        "run":      "\u25b6 Lancer", "ret": "Retours joints",
        "mix":      "Poids de m\u00e9lange", "jq": "Q_tot joint",
        "val":      "V(s)", "glass": "Glass-Box", "summary": "R\u00e9sum\u00e9",
        "episodes": "\u00c9pisodes", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon":  "Epsilon", "edecay": "D\u00e9croissance",
        "mhidden":  "Unit\u00e9s cach\u00e9es", "seed": "Graine",
        "labels":   {"iql":"IQL","vdn":"VDN","qmix":"QMIX","qmix_cg":"QMIX+CG"},
        "guide": """
**Sc\u00e9nario coop\u00e9ratif : les deux agents partagent LA M\u00caMe r\u00e9compense commune.**

**\u00c9tape 1 \u2014 IQL Baseline**
Chaque agent ignore l'autre. Que se passe-t-il sans coordination ?

**\u00c9tape 2 \u2014 VDN**
Q_tot = Q_0 + Q_1. D\u00e9composition additive. IGM permet l'ex\u00e9cution d\u00e9centralis\u00e9e.

**\u00c9tape 3 \u2014 QMIX**
Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s). M\u00e9lange monotone. w_i(s) >= 0 applique IGM.

**\u00c9tape 4 \u2014 QMIX+CG**
Baseline contrefactuelle isole la contribution de chaque agent.

**\u00c9tape 5 \u2014 Observer les poids de m\u00e9lange**
Comment QMIX apprend-il des poids d\u00e9pendants de l'\u00e9tat ?
""",
        "theory_igm":  "IGM: argmax Q_tot = (argmax Q_0, argmax Q_1)",
        "theory_vdn":  "Q_tot = Q_0 + Q_1",
        "theory_qmix": "Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s), w_i >= 0",
        "theory_cg":   "A_i = Q_tot(s,a) - Q_tot(s, a_{{-i}}, argmax Q_i)",
    },
    "ES": {
        "title":    "Cap\u00edtulo 13 \u2014 MARL Cooperativo: VDN y QMIX",
        "subtitle": "IQL \u2014 VDN \u2014 QMIX \u2014 QMIX+CG \u2014 ASP Varsovia",
        "run":      "\u25b6 Ejecutar", "ret": "Retornos conjuntos",
        "mix":      "Pesos de mezcla", "jq": "Q_tot conjunto",
        "val":      "V(s)", "glass": "Glass-Box", "summary": "Resumen",
        "episodes": "Episodios", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon":  "Epsilon", "edecay": "Decaimiento",
        "mhidden":  "Unidades ocultas", "seed": "Semilla",
        "labels":   {"iql":"IQL","vdn":"VDN","qmix":"QMIX","qmix_cg":"QMIX+CG"},
        "guide": """
**Escenario cooperativo: ambos agentes comparten LA MISMA recompensa conjunta.**

**Paso 1 \u2014 IQL Baseline**
Cada agente ignora al otro. \u00bfQu\u00e9 pasa sin coordinaci\u00f3n?

**Paso 2 \u2014 VDN**
Q_tot = Q_0 + Q_1. Descomposici\u00f3n aditiva. IGM permite ejecuci\u00f3n descentralizada.

**Paso 3 \u2014 QMIX**
Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s). Mezcla mon\u00f3tona. w_i(s) >= 0 aplica IGM.

**Paso 4 \u2014 QMIX+CG**
L\u00ednea base contrafactual a\u00edsl\u00e1 la contribuci\u00f3n de cada agente.

**Paso 5 \u2014 Observar los pesos de mezcla**
\u00bfC\u00f3mo aprende QMIX pesos dependientes del estado?
""",
        "theory_igm":  "IGM: argmax Q_tot = (argmax Q_0, argmax Q_1)",
        "theory_vdn":  "Q_tot = Q_0 + Q_1",
        "theory_qmix": "Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s), w_i >= 0",
        "theory_cg":   "A_i = Q_tot(s,a) - Q_tot(s, a_{{-i}}, argmax Q_i)",
    },
    "PL": {
        "title":    "Rozdzia\u0142 13 \u2014 Kooperacyjny MARL: VDN i QMIX",
        "subtitle": "IQL Baseline \u2014 VDN \u2014 QMIX \u2014 QMIX+CG \u2014 2 Agenci \u2014 ASP Warszawa",
        "run":      "\u25b6 Uruchom wszystkie cztery algorytmy",
        "ret":      "Wsp\u00f3lne zwroty epizod\u00f3w",
        "mix":      "Wagi mieszania (QMIX)", "jq": "Wsp\u00f3lne Q_tot",
        "val":      "Wsp\u00f3lna funkcja warto\u015bci V(s)",
        "glass":    "Glass-Box", "summary": "Podsumowanie",
        "episodes": "Epizody", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon":  "Epsilon", "edecay": "Zanik epsilon",
        "mhidden":  "Ukryte jednostki sieci mieszania", "seed": "Ziarno",
        "labels":   {"iql":"IQL Baseline","vdn":"VDN","qmix":"QMIX","qmix_cg":"QMIX+CG"},
        "guide": """
**Scenariusz kooperacyjny: obaj agenci dziel\u0105 T\u0118 SAM\u0104 wsp\u00f3ln\u0105 nagrod\u0119.**

**Krok 1 \u2014 IQL Baseline**
Ka\u017cdy agent ignoruje drugiego. Co si\u0119 dzieje bez koordynacji?

**Krok 2 \u2014 VDN (Value Decomposition Networks)**
Q_tot = Q_0 + Q_1. Prosta dekompozycja addytywna.
IGM: argmax Q_tot = (argmax Q_0, argmax Q_1) \u2014 umo\u017cliwia zdecentralizowane wykonanie.

**Krok 3 \u2014 QMIX**
Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s). Monotoniczna mieszanka.
w_i(s) >= 0 wymusza IGM. Zale\u017cne od stanu wagi = bardziej ekspresywna koordynacja.

**Krok 4 \u2014 QMIX+CG (Kontrfaktyczna linia bazowa)**
A_i = Q_tot(s,a) - Q_tot(s, a_{{-i}}, argmax Q_i). Izoluje wk\u0142ad ka\u017cdego agenta.

**Krok 5 \u2014 Obserwuj wykres wag mieszania**
Jak QMIX uczy si\u0119 zale\u017cnych od stanu wag koordynacji?
""",
        "theory_igm":  "IGM: argmax Q_tot = (argmax Q_0, argmax Q_1)",
        "theory_vdn":  "Q_tot = Q_0 + Q_1",
        "theory_qmix": "Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s), w_i >= 0",
        "theory_cg":   "A_i = Q_tot(s,a) - Q_tot(s, a_{{-i}}, argmax Q_i)",
    },
}

def _ma(data, w=30):
    r = []
    for i in range(len(data)):
        s = max(0, i-w+1); r.append(sum(data[s:i+1])/(i-s+1))
    return r

def _tx(lang):
    import copy
    base = copy.deepcopy(TX.get("EN", {}))
    over = TX.get(lang, {})
    for k, v in over.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base

def render():
    lang = st.session_state.get("lang","EN")
    tx   = _tx(lang)
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
        with st.spinner("Running..."):
            res = rlvr_py.run_ch13_coop_marl(int(seed),int(n_ep),float(gamma),float(alpha),float(eps),float(edec),int(mh))
        st.session_state["ch13_result"] = res
    if "ch13_result" not in st.session_state:
        st.info("Click Run."); _theory(tx); return
    res   = st.session_state["ch13_result"]
    short = [f"S{i}" for i in range(res["n_states"])]
    cols  = st.columns(4)
    for i,k in enumerate(ALGOS):
        avg = sum(res[k]["returns_curve"][-50:])/min(50,len(res[k]["returns_curve"]))
        mw  = sum(res[k]["mixing_weights"][-50:])/max(1,min(50,len(res[k]["mixing_weights"])))
        cols[i].metric(lb[k], f"Avg:{avg:.2f}", f"W:{mw:.2f}")
    st.subheader(tx["ret"])
    fig = go.Figure()
    for k in ALGOS:
        fig.add_trace(go.Scatter(x=list(range(n_ep)),y=_ma(res[k]["returns_curve"]),
            mode="lines",name=lb[k],line=dict(color=COLORS[k],width=2)))
    fig.update_layout(height=280,margin=dict(l=40,r=20,t=20,b=40),
        xaxis_title="Episode",yaxis_title="Return (MA-30)",legend=dict(orientation="h"))
    st.plotly_chart(fig,width='stretch')
    c1,c2 = st.columns(2)
    with c1:
        st.subheader(tx["mix"])
        f2 = go.Figure()
        for k in ["qmix","qmix_cg"]:
            f2.add_trace(go.Scatter(x=list(range(n_ep)),y=_ma(res[k]["mixing_weights"]),
                mode="lines",name=lb[k],line=dict(color=COLORS[k],width=2)))
        f2.update_layout(height=260,margin=dict(l=40,r=20,t=20,b=40),legend=dict(orientation="h"))
        st.plotly_chart(f2,width='stretch')
    with c2:
        st.subheader(tx["jq"])
        f3 = go.Figure()
        for k in ALGOS:
            f3.add_trace(go.Scatter(x=list(range(n_ep)),y=_ma(res[k]["joint_q_curve"]),
                mode="lines",name=lb[k],line=dict(color=COLORS[k],width=2)))
        f3.update_layout(height=260,margin=dict(l=40,r=20,t=20,b=40),legend=dict(orientation="h"))
        st.plotly_chart(f3,width='stretch')
    st.subheader(tx["val"])
    f4 = go.Figure()
    for k in ALGOS:
        f4.add_trace(go.Bar(x=short,y=res[k]["values"],name=lb[k],marker_color=COLORS[k],opacity=0.8))
    f4.update_layout(height=260,barmode="group",margin=dict(l=40,r=20,t=20,b=40),legend=dict(orientation="h"))
    st.plotly_chart(f4,width='stretch')
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
