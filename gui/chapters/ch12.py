import streamlit as st
import plotly.graph_objects as go

TX = {
    "EN": {
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
        "glass": "Glass-Box", "summary": "Summary",
    },
    "DE": {
        "title": "Kapitel 12 — Spieltheorie & Nash-Gleichgewicht",
        "subtitle": "Nash Q — Correlated Q — Minimax Q — Fictitious Play — ASP Warschau",
        "engine_missing": "Ausführen: cd rlvr-py && maturin develop",
        "guide": """
**Nash Q: konvergiert zum Nash-Gleichgewicht.**

**Schritt 1 — Nash-Gleichgewicht verstehen**
Kein Agent kann sich durch einseitiges Ändern seiner Strategie verbessern.

**Schritt 2 — Correlated Q**
Breiter als Nash — Agenten koordinieren über gemeinsame Verteilung σ(a0,a1|s).

**Schritt 3 — Minimax Q**
Nullsummenspiel — Spieler 0 maximiert den Worst-Case-Auszahlungswert.
Aktivieren Sie die Nullsummen-Checkbox um diesen Modus zu sehen.

**Schritt 4 — Fictitious Play**
Beste Antwort auf die empirische Durchschnittsstrategie des Gegners.

**Schritt 5 — Nash-Lücke-Diagramm beobachten**
Niedrigerer Wert = näher am Gleichgewicht. Null = perfektes Nash-Gleichgewicht erreicht.
""",
        "labels": {"nash_q":"Nash Q","correlated_q":"Correlated Q","minimax_q":"Minimax Q","fictitious":"Fictitious Play"},
        "settings": "Einstellungen", "episodes": "Episoden", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon": "Epsilon", "edecay": "Epsilon-Abklingrate",
        "zerosum": "Nullsummenspiel (Minimax-Modus)", "seed": "Zufallsseed",
        "run": "▶ Alle vier Algorithmen starten", "guide_title": "ℹ️ Anleitung",
        "ret": "Gemeinsame Episodenrückgaben", "gap": "Nash-Lücke (Exploitierbarkeit)",
        "strat": "Gemischtes Strategieprofil", "val": "Gemeinsame Wertfunktion V(s)",
        "glass": "Glass-Box", "summary": "Zusammenfassung",
    },
    "FR": {
        "title": "Chapitre 12 — Théorie des jeux & Équilibre de Nash",
        "subtitle": "Nash Q — Correlated Q — Minimax Q — Fictitious Play — ASP Varsovie",
        "engine_missing": "Exécutez: cd rlvr-py && maturin develop",
        "guide": """
**Nash Q : converge vers l'équilibre de Nash.**

**Étape 1 — Comprendre l'équilibre de Nash**
Aucun agent ne peut améliorer sa valeur en changeant unilatéralement sa stratégie.

**Étape 2 — Correlated Q**
Plus large que Nash — les agents coordonnent via une distribution conjointe σ(a0,a1|s).

**Étape 3 — Minimax Q**
Jeu à somme nulle — le joueur 0 maximise le pire cas de gain.
Activer la case à cocher jeu à somme nulle pour voir ce mode.

**Étape 4 — Fictitious Play**
Meilleure réponse à la stratégie moyenne empirique de l'adversaire.

**Étape 5 — Observer le graphique de lacune Nash**
Plus bas = plus proche de l'équilibre. Zéro = équilibre de Nash parfait atteint.
""",
        "labels": {"nash_q":"Nash Q","correlated_q":"Correlated Q","minimax_q":"Minimax Q","fictitious":"Fictitious Play"},
        "settings": "Paramètres", "episodes": "Épisodes", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon": "Epsilon", "edecay": "Décroissance", "zerosum": "Jeu à somme nulle", "seed": "Graine",
        "run": "Lancer", "guide_title": "Guide",
        "ret": "Retours joints", "gap": "Lacune Nash", "strat": "Profil stratégie", "val": "V(s)",
        "glass": "Glass-Box", "summary": "Résumé",
    },
    "ES": {
        "title": "Capítulo 12 — Teoría de juegos & Equilibrio de Nash",
        "subtitle": "Nash Q — Correlated Q — Minimax Q — Fictitious Play — ASP Varsovia",
        "engine_missing": "Ejecute: cd rlvr-py && maturin develop",
        "guide": """
**Nash Q: converge al equilibrio de Nash.**

**Paso 1 — Entender el equilibrio de Nash**
Ningún agente puede mejorar su valor cambiando unilateralmente su estrategia.

**Paso 2 — Correlated Q**
Más amplio que Nash — los agentes coordinan a través de distribución conjunta σ(a0,a1|s).

**Paso 3 — Minimax Q**
Juego de suma cero — el jugador 0 maximiza el pago en el peor caso.
Activar la casilla de verificación de juego de suma cero para ver este modo.

**Paso 4 — Fictitious Play**
Mejor respuesta a la estrategia promedio empírica del oponente.

**Paso 5 — Observar el gráfico de brecha Nash**
Más bajo = más cerca del equilibrio. Cero = equilibrio de Nash perfecto alcanzado.
""",
        "labels": {"nash_q":"Nash Q","correlated_q":"Correlated Q","minimax_q":"Minimax Q","fictitious":"Fictitious Play"},
        "settings": "Configuración", "episodes": "Episodios", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon": "Epsilon", "edecay": "Decaimiento", "zerosum": "Juego suma cero", "seed": "Semilla",
        "run": "Ejecutar", "guide_title": "Guía",
        "ret": "Retornos conjuntos", "gap": "Brecha Nash", "strat": "Perfil estrategia", "val": "V(s)",
        "glass": "Glass-Box", "summary": "Resumen",
    },
    "PL": {
        "title": "Rozdział 12 — Teoria gier & Równowaga Nasha",
        "subtitle": "Nash Q — Correlated Q — Minimax Q — Fictitious Play — ASP Warszawa",
        "engine_missing": "Uruchom: cd rlvr-py && maturin develop",
        "guide": """
**Nash Q: zbiega do równowagi Nasha.**

**Krok 1 — Zrozum równowagę Nasha**
Żaden agent nie może poprawić swojej wartości przez jednostronna zmianę strategii.

**Krok 2 — Correlated Q**
Szersze niż Nash — agenci koordynują przez wspólną dystrybucję σ(a0,a1|s).

**Krok 3 — Minimax Q**
Gra zerowa — gracz 0 maksymalizuje wartość najgorszego przypadku wypłaty.
Włącz pole wyboru gry zerowej aby zobaczyć ten tryb.

**Krok 4 — Fictitious Play**
Najlepsza odpowiedź na empiryczną średną strategię przeciwnika.

**Krok 5 — Obserwuj wykres luki Nasha**
Niżej = bliżej równowagi. Zero = osiągnięto doskonałą równowagę Nasha.
""",
        "labels": {"nash_q":"Nash Q","correlated_q":"Correlated Q","minimax_q":"Minimax Q","fictitious":"Fictitious Play"},
        "settings": "Ustawienia", "episodes": "Epizody", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon": "Epsilon", "edecay": "Zanik epsilon", "zerosum": "Gra zerowa (tryb Minimax)", "seed": "Ziarno",
        "run": "▶ Uruchom wszystkie cztery algorytmy", "guide_title": "Przewodnik",
        "ret": "Wspólne zwroty epizodów", "gap": "Luka Nasha", "strat": "Profil strategii mieszanej", "val": "V(s)",
        "glass": "Glass-Box", "summary": "Podsumowanie",
    },
}


COLORS = {"nash_q":"#8B5CF6","correlated_q":"#0082F0","minimax_q":"#0FC373","fictitious":"#FF8C0A"}
ALGOS  = ["nash_q","correlated_q","minimax_q","fictitious"]
LABELS = {"nash_q":"Nash Q","correlated_q":"Correlated Q","minimax_q":"Minimax Q","fictitious":"Fictitious Play"}

def _ma(data,w=30):
    r=[]
    for i in range(len(data)):
        s=max(0,i-w+1); r.append(sum(data[s:i+1])/(i-s+1))
    return r


def _tx(lang):
    """Deep merge: DE overrides EN, but missing keys/subkeys fall back to EN."""
    import copy
    base = copy.deepcopy(TX.get("EN", {}))
    over = TX.get(lang, {})
    for k, v in over.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            # Deep merge nested dicts (e.g. theory_sections, algo_labels)
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base

def render():
    lang = st.session_state.get("lang","EN")
    tx = _tx(lang)
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

    with st.expander(tx["guide_title"],expanded=False):
        st.markdown(tx["guide"])
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
        _theory()
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
    _theory()

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
