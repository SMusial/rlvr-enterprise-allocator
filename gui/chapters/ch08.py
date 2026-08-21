import streamlit as st
import plotly.graph_objects as go

T = {
    "EN": {
        "title": "Chapter 08 — Eligibility Traces & TD(λ)",
        "subtitle": "SARSA(λ) · Q(λ) Watkins · λ=0 (TD0) · λ=0.99 (≈MC) · Warsaw ASP",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Settings",
        "n_episodes": "Episodes", "gamma": "γ Discount", "alpha": "α Learning rate",
        "epsilon": "ε Exploration", "epsilon_decay": "ε decay", "lambda_": "λ Trace decay (0=TD,1≈MC)",
        "replacing": "Replacing traces", "seed": "Seed",
        "run_btn": "▶ Run All Four Algorithms",
        "guide_title": "📖 Guide",
        "guide": "λ=0→TD(0). λ=1→MC. λ=0.7 sweet spot. SARSA(λ) on-policy. Q(λ) off-policy with Watkins cut.",
        "returns_title": "📈 Episode Returns", "returns_caption": "MA-30. SARSA(λ=0.7) converges faster than TD0.",
        "td_error_title": "📉 TD Error", "td_error_caption": "",
        "value_title": "🏛️ V(s)", "value_caption": "S7 (SLA breach) should be lowest.",
        "trace_title": "🔍 Active Traces", "trace_caption": "Higher λ → more active traces.",
        "qtable_title": "🗺️ Q-Table Heatmap", "qtable_caption": "",
        "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Summary", "summary_results": "Comparison",
        "summary_pros_cons": "Pros & Cons", "pros": "✅ Pros", "cons": "❌ Cons",
        "theory_title": "📚 Theory",
        "theory_sections": {"et": "8.1 Eligibility Traces", "sarsal": "8.2 SARSA(λ)", "ql": "8.3 Q(λ) Watkins", "lambda": "8.4 Choosing λ"},
        "theory_et": "e_t(s,a) = γλ·e_{t-1}(s,a) + 𝟙[s=S_t,a=A_t]  (replacing: set to 1)\nDecay γλ=0.665 for γ=0.95,λ=0.7. Pruned at 1e-8.",
        "theory_sarsal": "δ_t = R+γQ(S',A')−Q(S,A)\nQ(s,a) ← Q(s,a)+α δ_t e_t(s,a)  ∀(s,a)",
        "theory_ql": "TD target: max_a Q(S',a). Traces CUT on non-greedy action (Watkins 1989).",
        "theory_lambda": "λ=0→TD(0) | λ=0.7→sweet spot | λ=0.99→≈MC\nReduce α if unstable with high λ.",
        "algo_labels": {"sarsa_lambda": "SARSA(λ)", "q_lambda": "Q(λ) Watkins", "sarsa_td0": "SARSA λ=0 (TD0)", "sarsa_mc": "SARSA λ=0.99 (≈MC)"},
        "pros_list": {"sarsa_lambda": ["On-policy, safe","Fast backward credit","λ tunes bias-variance"], "q_lambda": ["Off-policy","Watkins cut prevents divergence","Aggressive exploration"], "sarsa_td0": ["Simplest","Low variance","Baseline ref"], "sarsa_mc": ["Near-zero bias","Full propagation","Upper bound λ"]},
        "cons_list": {"sarsa_lambda": ["Needs ε>0","O(|S||A|)/step","λ to tune"], "q_lambda": ["Traces cut on explore","Less propagation","Instability risk"], "sarsa_td0": ["Slow credit","Many episodes","No backward prop"], "sarsa_mc": ["High variance","Needs episode end","Unstable small α"]},
    },
    "PL": {
        "title": "Rozdział 08 — Ślady Kwalifikowalności & TD(λ)",
        "subtitle": "SARSA(λ) · Q(λ) Watkins · λ=0 (TD0) · λ=0.99 (≈MC) · ASP Warszawa",
        "engine_missing": "Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Ustawienia",
        "n_episodes": "Epizody", "gamma": "γ Dyskonto", "alpha": "α Uczenie",
        "epsilon": "ε Eksploracja", "epsilon_decay": "Zanik ε", "lambda_": "λ Zanikanie śladów",
        "replacing": "Replacing traces", "seed": "Ziarno",
        "run_btn": "▶ Uruchom wszystkie cztery algorytmy",
        "guide_title": "📖 Przewodnik",
        "guide": """
**Krok 1 — Zrozum Eligibility Traces**
Ślady rozwiązują problem przypisania kredytu.
e_t(s,a) = γλ·e_{t-1}(s,a) + 1[s=S_t, a=A_t]
λ=0 → TD(0). λ=1 → MC. λ=0.7 → optymalny zakres.

**Krok 2 — Zrozum SARSA(λ) vs Q(λ)**
SARSA(λ): on-policy, cel = R + γQ(s',a').
Q(λ) Watkins: off-policy, ślady przecinane przy niezachłannych akcjach.

**Krok 3 — Ustaw λ**
Zacznij od λ=0.7. Porównaj λ=0.0 vs λ=0.99.

**Krok 4 — Kliknij ▶ Uruchom wszystkie cztery algorytmy**
SARSA(λ), Q(λ), SARSA λ=0 (bazowe TD0), SARSA λ=0.99 (podobne do MC).

**Krok 5 — Odczytaj wykres aktywnych śladów**
Wyższe λ → więcej aktywnych śladów → szybsza propagacja kredytu.

**Krok 6 — Odczytaj krzywą zwrotów**
SARSA(λ=0.7) powinno zbiec szybciej niż SARSA λ=0.

**Krok 7 — Porównaj z Ch06 TD(0)**
Eligibility traces powinny osiągnąć to samo V* co TD(0) w mniejszej liczbie epizodów.
""",
        "returns_title": "📈 Zwroty epizodów", "returns_caption": "MA-30. SARSA(λ=0.7) zbiega szybciej niż TD0.",
        "td_error_title": "📉 Błąd TD", "td_error_caption": "",
        "value_title": "🏛️ V(s)", "value_caption": "S7 powinno być najniższe.",
        "trace_title": "🔍 Aktywne ślady", "trace_caption": "Wyższe λ → więcej aktywnych śladów.",
        "qtable_title": "🗺️ Heatmapa Q", "qtable_caption": "",
        "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Podsumowanie", "summary_results": "Porównanie",
        "summary_pros_cons": "Zalety i Wady", "pros": "✅ Zalety", "cons": "❌ Wady",
        "theory_title": "📚 Teoria",
        "theory_sections": {"et": "8.1 Ślady kwalifikowalności", "sarsal": "8.2 SARSA(λ)", "ql": "8.3 Q(λ) Watkins", "lambda": "8.4 Dobór λ"},
        "theory_et": "e_t(s,a) = γλ·e_{t-1}(s,a) + 𝟙[s=S_t,a=A_t]\nZanik γλ=0.665 dla γ=0.95,λ=0.7.",
        "theory_sarsal": "Q(s,a) ← Q(s,a)+α δ_t e_t(s,a)  dla WSZYSTKICH (s,a)",
        "theory_ql": "Cel TD: max_a Q(S',a). Ślady CIĘTE przy akcji niegreedy.",
        "theory_lambda": "λ=0→TD(0) | λ=0.7→optimum | λ=0.99→≈MC",
        "algo_labels": {"sarsa_lambda": "SARSA(λ)", "q_lambda": "Q(λ) Watkins", "sarsa_td0": "SARSA λ=0 (TD0)", "sarsa_mc": "SARSA λ=0.99 (≈MC)"},
        "pros_list": {"sarsa_lambda": ["On-policy","Szybkie przypisanie zasługi","λ reguluje bias-variance"], "q_lambda": ["Off-policy","Cięcie Watkinsa","Agresywna eksploracja"], "sarsa_td0": ["Najprostszy","Niski variance","Punkt odniesienia"], "sarsa_mc": ["Prawie zerowy bias","Pełna propagacja","Górna granica λ"]},
        "cons_list": {"sarsa_lambda": ["Wymaga ε>0","O(|S||A|)/krok","λ do strojenia"], "q_lambda": ["Ślady cięte","Mniej propagacji","Ryzyko niestabilności"], "sarsa_td0": ["Wolne przypisanie","Wiele epizodów","Brak propagacji wstecz"], "sarsa_mc": ["Wysoki variance","Wymaga zakończenia","Niestabilny przy małym α"]},
    },
        "DE": {
        "title": "Kapitel 08 — Eligibility Traces",
        "subtitle": "TD(λ) — SARSA(λ) — Q(λ) Watkins — ASP Warschau",
        "engine_missing": "Ausführen: `cd rlvr-py && maturin develop`",
        "sidebar_title": "Einstellungen",
        "n_episodes": "Episoden", "gamma": "Gamma", "alpha": "Alpha",
        "epsilon": "Epsilon", "epsilon_decay": "Epsilon-Abklingrate",
        "lambda_val": "λ — Trace-Abklingrate", "seed": "Zufallsseed",
        "run_btn": "▶ Alle Algorithmen starten",
        "guide_title": "Anleitung",
        "guide": """
**Krok 1 — Zrozum Eligibility Traces**
Ślady rozwiązują problem przypisania kredytu.
e_t(s,a) = γλ·e_{t-1}(s,a) + 1[s=S_t, a=A_t]
λ=0 → TD(0). λ=1 → MC. λ=0.7 → optymalny zakres.

**Krok 2 — Zrozum SARSA(λ) vs Q(λ)**
SARSA(λ): on-policy, cel = R + γQ(s',a').
Q(λ) Watkins: off-policy, ślady przecinane przy niezachłannych akcjach.

**Krok 3 — Ustaw λ (współczynnik zaniku śladów)**
Zacznij od λ=0.7. Porównaj λ=0.0 vs λ=0.99 aby zobaczyć różnicę.

**Krok 4 — Kliknij ▶ Uruchom wszystkie cztery algorytmy**
SARSA(λ), Q(λ), SARSA λ=0 (bazowe TD0), SARSA λ=0.99 (podobne do MC).

**Krok 5 — Odczytaj wykres aktywnych śladów**
Wyższe λ → więcej aktywnych śladów → szybsza propagacja kredytu.

**Krok 6 — Odczytaj krzywą zwrotów**
SARSA(λ=0.7) powinno zbiec szybciej niż SARSA λ=0.

**Krok 7 — Porównaj z Ch06 TD(0)**
Eligibility traces powinny osiągnąć to samo V* co TD(0) ale w mniejszej liczbie epizodów.
""",
        "returns_title": "Episodenrückgaben",
        "returns_caption": "Gleitender Durchschnitt.",
        "value_title": "Wertfunktion V(s)",
        "value_caption": "",
        "glass_title": "Glass-Box",
        "summary_title": "Zusammenfassung", "summary_results": "Vergleich",
        "summary_pros_cons": "Vor- & Nachteile",
        "pros": "Vorteile", "cons": "Nachteile",
        "theory_title": "Theorie — Kapitel 08",
        "theory_sections": {"traces": "8.1 Eligibility Traces", "tdlambda": "8.2 TD(λ)", "sarsa": "8.3 SARSA(λ)"},
        "algo_labels": {"td_lambda": "TD(λ)", "sarsa_lambda": "SARSA(λ)", "q_lambda": "Q(λ) Watkins"},
        "pros_list": {
            "td_lambda": ["Brückt TD und MC", "Schnellere Kreditvergabe"],
            "sarsa_lambda": ["On-Policy mit Traces", "Bessere Konvergenz"],
            "q_lambda": ["Off-Policy mit Traces", "Watkins-Schnitt bei Exploration"],
        },
        "cons_list": {
            "td_lambda": ["λ muss eingestellt werden"],
            "sarsa_lambda": ["Speicher für Traces"],
            "q_lambda": ["Watkins-Schnitt reduziert Effizienz"],
        },
        "theory_traces": r"$e_t(s) = \gamma\lambda e_{t-1}(s) + \mathbf{1}[S_t = s]$",
        "theory_tdlambda": r"$V(s) \leftarrow V(s) + lpha\delta_t e_t(s)$",
        "theory_sarsa": r"$Q(s,a) \leftarrow Q(s,a) + lpha\delta_t e_t(s,a)$",
    },
    "FR": {
        "title": "Chapitre 08 — Traces d'éligibilité & TD(λ)", "subtitle": "SARSA(λ) · Q(λ) · ASP Varsovie",
        "engine_missing": "Exécutez: `cd rlvr-py && maturin develop`", "sidebar_title": "⚙️ Paramètres",
        "n_episodes": "Épisodes", "gamma": "γ", "alpha": "α", "epsilon": "ε", "epsilon_decay": "Décroissance ε",
        "lambda_": "λ traces", "replacing": "Replacing", "seed": "Graine", "run_btn": "▶ Lancer",
        "guide_title": "📖 Guide", "guide": """
**Étape 1 — Comprendre les Eligibility Traces**
Les traces résolvent le problème d'attribution du crédit.
e_t(s,a) = γλ·e_{t-1}(s,a) + 1[s=S_t, a=A_t]
λ=0 → TD(0). λ=1 → MC. λ=0.7 → zone optimale.

**Étape 2 — Comprendre SARSA(λ) vs Q(λ)**
SARSA(λ) : on-policy, cible = R + γQ(s',a').
Q(λ) Watkins : off-policy, traces coupées sur actions non-gloutonnes.

**Étape 3 — Régler λ**
Commencer avec λ=0.7. Essayer λ=0.0 vs λ=0.99.

**Étape 4 — Cliquer ▶ Lancer les quatre algorithmes**
SARSA(λ), Q(λ), SARSA λ=0 (baseline TD0), SARSA λ=0.99 (type MC).

**Étape 5 — Lire le graphique des traces actives**
λ plus élevé → plus d'états actifs → propagation du crédit plus rapide.

**Étape 6 — Lire la courbe de retours**
SARSA(λ=0.7) devrait converger plus vite que SARSA λ=0.

**Étape 7 — Comparer avec Ch06 TD(0)**
Les traces devraient atteindre le même V* que TD(0) en moins d'épisodes.
""",
        "returns_title": "📈 Retours", "returns_caption": "", "td_error_title": "📉 Erreur TD", "td_error_caption": "",
        "value_title": "🏛️ V(s)", "value_caption": "", "trace_title": "🔍 Traces", "trace_caption": "",
        "qtable_title": "🗺️ Table Q", "qtable_caption": "", "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Résumé", "summary_results": "Comparaison", "summary_pros_cons": "Avantages & Inconvénients",
        "pros": "✅ Pros", "cons": "❌ Cons", "theory_title": "📚 Théorie",
        "theory_sections": {"et": "8.1 Traces", "sarsal": "8.2 SARSA(λ)", "ql": "8.3 Q(λ)", "lambda": "8.4 Choix λ"},
        "theory_et": "e_t(s,a)=γλ·e_{t-1}(s,a)+𝟙[...]", "theory_sarsal": "Q←Q+αδe ∀(s,a)",
        "theory_ql": "Coupure Watkins si action non-greedy.", "theory_lambda": "λ=0→TD | λ=0.7→optimum | λ=0.99→MC",
        "algo_labels": {"sarsa_lambda": "SARSA(λ)", "q_lambda": "Q(λ)", "sarsa_td0": "λ=0", "sarsa_mc": "λ=0.99"},
        "pros_list": {"sarsa_lambda": ["On-policy"], "q_lambda": ["Off-policy"], "sarsa_td0": ["Simple"], "sarsa_mc": ["Faible biais"]},
        "cons_list": {"sarsa_lambda": ["λ à régler"], "q_lambda": ["Coupure"], "sarsa_td0": ["Lent"], "sarsa_mc": ["Variance"]},
    },
    "ES": {
        "title": "Capítulo 08 — Trazas de Elegibilidad & TD(λ)", "subtitle": "SARSA(λ) · Q(λ) · ASP Varsovia",
        "engine_missing": "Ejecute: `cd rlvr-py && maturin develop`", "sidebar_title": "⚙️ Configuración",
        "n_episodes": "Episodios", "gamma": "γ", "alpha": "α", "epsilon": "ε", "epsilon_decay": "Decaimiento ε",
        "lambda_": "λ trazas", "replacing": "Replacing", "seed": "Semilla", "run_btn": "▶ Ejecutar",
        "guide_title": "📖 Guía", "guide": """
**Paso 1 — Entender los Eligibility Traces**
Los traces resuelven el problema de asignación de crédito.
e_t(s,a) = γλ·e_{t-1}(s,a) + 1[s=S_t, a=A_t]
λ=0 → TD(0). λ=1 → MC. λ=0.7 → zona óptima.

**Paso 2 — Entender SARSA(λ) vs Q(λ)**
SARSA(λ): on-policy, objetivo = R + γQ(s',a').
Q(λ) Watkins: off-policy, traces cortados en acciones no codiciosas.

**Paso 3 — Ajustar λ**
Comenzar con λ=0.7. Probar λ=0.0 vs λ=0.99.

**Paso 4 — Hacer clic en ▶ Ejecutar los cuatro algoritmos**
SARSA(λ), Q(λ), SARSA λ=0 (línea base TD0), SARSA λ=0.99 (tipo MC).

**Paso 5 — Leer el gráfico de traces activos**
λ más alto → más estados activos → propagación de crédito más rápida.

**Paso 6 — Leer la curva de retornos**
SARSA(λ=0.7) debería converger más rápido que SARSA λ=0.

**Paso 7 — Comparar con Ch06 TD(0)**
Los traces deberían alcanzar el mismo V* que TD(0) en menos episodios.
""",
        "returns_title": "📈 Retornos", "returns_caption": "", "td_error_title": "📉 Error TD", "td_error_caption": "",
        "value_title": "🏛️ V(s)", "value_caption": "", "trace_title": "🔍 Trazas", "trace_caption": "",
        "qtable_title": "🗺️ Tabla Q", "qtable_caption": "", "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Resumen", "summary_results": "Comparación", "summary_pros_cons": "Pros y Contras",
        "pros": "✅ Pros", "cons": "❌ Cons", "theory_title": "📚 Teoría",
        "theory_sections": {"et": "8.1 Trazas", "sarsal": "8.2 SARSA(λ)", "ql": "8.3 Q(λ)", "lambda": "8.4 Elección λ"},
        "theory_et": "e_t(s,a)=γλ·e_{t-1}(s,a)+𝟙[...]", "theory_sarsal": "Q←Q+αδe ∀(s,a)",
        "theory_ql": "Corte Watkins si acción no-greedy.", "theory_lambda": "λ=0→TD | λ=0.7→óptimo | λ=0.99→MC",
        "algo_labels": {"sarsa_lambda": "SARSA(λ)", "q_lambda": "Q(λ)", "sarsa_td0": "λ=0", "sarsa_mc": "λ=0.99"},
        "pros_list": {"sarsa_lambda": ["On-policy"], "q_lambda": ["Off-policy"], "sarsa_td0": ["Simple"], "sarsa_mc": ["Bajo sesgo"]},
        "cons_list": {"sarsa_lambda": ["λ a ajustar"], "q_lambda": ["Corte"], "sarsa_td0": ["Lento"], "sarsa_mc": ["Varianza"]},
    },
}
COLORS = {"sarsa_lambda": "#8B5CF6", "q_lambda": "#0082F0", "sarsa_td0": "#FF8C0A", "sarsa_mc": "#0FC373"}
ALGOS  = ["sarsa_lambda", "q_lambda", "sarsa_td0", "sarsa_mc"]

def _ma(data, w=30):
    r = []
    for i in range(len(data)):
        s = max(0, i-w+1); r.append(sum(data[s:i+1])/(i-s+1))
    return r


def _tx(lang):
    """Deep merge: DE overrides EN, but missing keys/subkeys fall back to EN."""
    import copy
    base = copy.deepcopy(T.get("EN", {}))
    over = T.get(lang, {})
    for k, v in over.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            # Deep merge nested dicts (e.g. theory_sections, algo_labels)
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base

def render():
    lang = st.session_state.get("lang", "EN"); tx = _tx(lang)
    st.title(tx["title"]); st.caption(tx["subtitle"])
    try: import rlvr_py
    except ImportError: st.error(tx["engine_missing"]); return
    st.sidebar.header(tx["sidebar_title"])
    n_ep  = st.sidebar.slider(tx["n_episodes"],    50, 3000, 500, 50)
    gamma = st.sidebar.slider(tx["gamma"],         0.5, 0.999, 0.95, 0.005)
    alpha = st.sidebar.slider(tx["alpha"],         0.01, 1.0, 0.1, 0.01)
    eps   = st.sidebar.slider(tx["epsilon"],       0.0, 1.0, 0.3, 0.05)
    edec  = st.sidebar.slider(tx["epsilon_decay"], 0.0, 0.1, 0.01, 0.001, format="%.3f")
    lam   = st.sidebar.slider(tx["lambda_"],       0.0, 1.0, 0.7, 0.05)
    repl  = st.sidebar.checkbox(tx["replacing"], value=True)
    seed  = st.sidebar.number_input(tx["seed"], 0, 9999, 42)
    with st.expander(tx["guide_title"], expanded=False): st.markdown(tx["guide"])
    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running..."):
            res = rlvr_py.run_ch08_eligibility(int(seed), int(n_ep), float(gamma), float(alpha), float(eps), float(edec), float(lam), bool(repl))
        st.session_state["ch08_result"] = res
    if "ch08_result" not in st.session_state:
        st.info("Click ▶ to run."); _theory(tx); return
    res = st.session_state["ch08_result"]
    short = [f"S{i}" for i in range(res["n_states"])]
    cols = st.columns(4)
    for i, k in enumerate(ALGOS):
        avg = sum(res[k]["returns_curve"][-50:]) / min(50, len(res[k]["returns_curve"]))
        atr = sum(res[k]["trace_stats"][-50:]) / max(1, min(50, len(res[k]["trace_stats"])))
        cols[i].metric(tx["algo_labels"][k], f"Avg:{avg:.2f}", f"Tr:{atr:.1f}")
    st.subheader(tx["returns_title"])
    fig = go.Figure()
    for k in ALGOS: fig.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["returns_curve"]), mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
    fig.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40), xaxis_title="Episode", yaxis_title="Return (MA-30)", legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch'); st.caption(tx["returns_caption"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(tx["value_title"])
        f2 = go.Figure()
        for k in ALGOS: f2.add_trace(go.Bar(x=short, y=res[k]["values"], name=tx["algo_labels"][k], marker_color=COLORS[k], opacity=0.8))
        f2.update_layout(height=260, barmode="group", margin=dict(l=40,r=20,t=20,b=40), legend=dict(orientation="h"))
        st.plotly_chart(f2, width='stretch'); st.caption(tx["value_caption"])
    with c2:
        st.subheader(tx["trace_title"])
        f3 = go.Figure()
        for k in ["sarsa_lambda","q_lambda"]: f3.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["trace_stats"]), mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
        f3.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40), xaxis_title="Episode", yaxis_title="Max traces", legend=dict(orientation="h"))
        st.plotly_chart(f3, width='stretch'); st.caption(tx["trace_caption"])
    st.subheader(tx["qtable_title"])
    sel = st.selectbox("Algorithm", [tx["algo_labels"][k] for k in ALGOS])
    ks  = {tx["algo_labels"][k]: k for k in ALGOS}.get(sel, "sarsa_lambda")
    qt  = res[ks]["q_table"]; ash = [f"A{i}" for i in range(res["n_actions"])]
    f4  = go.Figure(go.Heatmap(z=qt, x=ash, y=short, colorscale="Purples", text=[[f"{qt[s][a]:.2f}" for a in range(res["n_actions"])] for s in range(res["n_states"])], texttemplate="%{text}"))
    f4.update_layout(height=280, margin=dict(l=60,r=20,t=20,b=40)); st.plotly_chart(f4, width='stretch')
    st.subheader(tx["glass_title"]); _glass(res, tx)
    st.subheader(tx["summary_title"]); _summary(res, tx)
    _theory(tx)

def _glass(res, tx):
    opts = {tx["algo_labels"][k]: k for k in ALGOS}
    sel  = st.selectbox("Algorithm", list(opts.keys()), key="gb8")
    k    = opts[sel]; r = res[k]
    ep   = st.slider("Episode", 0, max(len(r["returns_curve"])-1,0), max(len(r["returns_curve"])-1,0), key="gb8ep")
    c1,c2,c3 = st.columns(3)
    c1.metric("Return", f"{r['returns_curve'][ep]:.3f}")
    c2.metric("TD error", f"{r['td_error_curve'][ep]:.4f}")
    c3.metric("Max traces", f"{r['trace_stats'][ep]:.0f}")
    if "sarsa" in k:
        st.latex(r"\delta_t=R_{t+1}+\gamma Q(S',A')-Q(S,A)")
        st.latex(r"e_t(s,a)=\gamma\lambda e_{t-1}(s,a)+\mathbf{1}[s=S_t,a=A_t]")
        st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\delta_t e_t(s,a)\;\forall(s,a)")
    else:
        st.latex(r"\delta_t=R_{t+1}+\gamma\max_{a'}Q(S',a')-Q(S,A)")
        st.markdown("**Watkins cut:** traces=0 on non-greedy action.")
        st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\delta_t e_t(s,a)\;\forall(s,a)")

def _summary(res, tx):
    rows = []
    for k in ALGOS:
        r = res[k]; avg = sum(r["returns_curve"][-100:])/min(100,len(r["returns_curve"]))
        atr = sum(r["trace_stats"])/max(1,len(r["trace_stats"]))
        rows.append({"Algorithm": tx["algo_labels"][k], "Avg return (last 100)": f"{avg:.3f}", "Steps": str(r["total_steps"]), "Avg traces": f"{atr:.1f}", "V*(S0)": f"{r['values'][0]:.3f}", "V*(S7)": f"{r['values'][7]:.3f}"})
    st.dataframe(rows, hide_index=True)
    for k in ALGOS:
        label = tx["algo_labels"][k]; c1,c2 = st.columns(2)
        with c1: st.markdown(f"**{label} — {tx['pros']}**"); [st.markdown(f"- {p}") for p in tx["pros_list"][k]]
        with c2: st.markdown(f"**{label} — {tx['cons']}**"); [st.markdown(f"- {c}") for c in tx["cons_list"][k]]
        st.markdown("---")

def _theory(tx):
    st.markdown("---"); st.subheader(tx["theory_title"])
    for k in ["et","sarsal","ql","lambda"]:
        with st.expander(tx["theory_sections"][k], expanded=False): st.markdown(tx[f"theory_{k}"])
