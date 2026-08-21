import streamlit as st
import plotly.graph_objects as go

T = {
    "EN": {
        "title": "Chapter 05 — Monte Carlo Methods",
        "subtitle": "ASP Dispatch Learning from Episodes · No Model Required · Warsaw Region",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ MC Settings",
        "n_episodes": "Number of episodes",
        "gamma": "γ — Discount factor",
        "epsilon": "ε — Initial exploration",
        "epsilon_decay": "ε decay rate α",
        "seed": "Random seed",
        "run_btn": "▶ Run All Four MC Algorithms",
        "guide_title": "🎓 How to use this chapter",
        "guide": """
**Step 1 — Understand the key difference from Ch04**
MC methods learn WITHOUT a model — no P(s\'|s,a) needed.
The agent generates episodes by interacting with the environment and learns from the returns.

**Step 2 — Set number of episodes**
More episodes = better estimates. Try 200 first, then 2000 to see convergence.

**Step 3 — Click ▶ Run All Four MC Algorithms**
- First-Visit MC: updates V(s) only on first visit to s per episode
- Every-Visit MC: updates V(s) on every visit to s per episode
- On-Policy Control: learns Q*(s,a) and π* simultaneously (epsilon-soft)
- Off-Policy Control: learns from behaviour policy using Importance Sampling

**Step 4 — Read the Returns Curve**
Watch average episode return improve over time — this is the learning signal.

**Step 5 — Read the Value Function comparison**
MC estimates of V*(s) should converge toward the DP solution from Ch04.

**Step 6 — Read the Visit Count heatmap**
Which states were visited most? Rarely visited states have high variance estimates.

**Step 7 — Read the Glass-Box**
See exact returns G_t for each state in a selected episode.
""",
        "returns_title": "📈 Episode Returns — All Four Algorithms",
        "returns_caption": "Moving average of episode returns. On-policy control should improve over time.",
        "value_title": "📊 Value Function V(s) — MC vs DP Reference",
        "value_caption": "MC estimates converge toward DP solution (Ch04) with more episodes.",
        "visits_title": "🗺️ State Visit Counts — First-Visit MC",
        "visits_caption": "States visited rarely have high-variance V(s) estimates.",
        "conv_title": "📈 Convergence — Max |V^(k) - V^(k-1)|",
        "conv_caption": "MC convergence is noisier than DP — stochastic environment.",
        "qtable_title": "📊 Q-Table Heatmap — On-Policy Control",
        "qtable_caption": "Q(s,a) values learned by on-policy MC. Brighter = higher value.",
        "glass_title": "🔬 Glass-Box — Episode Trace",
        "glass_headers": ["Step", "State", "Action", "Reward", "Return G_t"],
        "summary_title": "📊 Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "MC Algorithms — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "theory_title": "📖 Theory — Chapter 05",
        "theory_sections": {
            "intro":       "§5.1 Monte Carlo Methods — Introduction",
            "first_visit": "§5.2 First-Visit MC Prediction",
            "every_visit": "§5.2 Every-Visit MC Prediction",
            "on_policy":   "§5.3 On-Policy MC Control",
            "off_policy":  "§5.4 Off-Policy MC with Importance Sampling",
        },
        "theory_intro": r"""
**Monte Carlo Methods** learn directly from episodes of experience — no model of P(s\'|s,a) needed.

Key properties:
- **Model-free**: learns from raw experience, not transition probabilities
- **Episode-based**: must wait until end of episode to update (unlike TD)
- **Unbiased**: estimates are unbiased (unlike TD which bootstraps)
- **High variance**: estimates can be noisy, especially for rarely visited states

The return from step t:
G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ... = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}

Implemented in `ch05_mc.rs` — `generate_episode()`, `mc_first_visit_prediction()`.
""",
        "theory_first_visit": r"""
**First-Visit MC Prediction** estimates V^π(s) by averaging returns from the FIRST visit to s per episode:

V(s) ← average of G_t for all first visits to s across all episodes

- Unbiased estimator of V^π(s)
- Each episode contributes at most one return per state
- Converges to V^π(s) as number of episodes → ∞

Implemented in `mc_first_visit_prediction()` in `ch05_mc.rs`.
""",
        "theory_every_visit": r"""
**Every-Visit MC Prediction** averages returns from ALL visits to s (not just first):

V(s) ← average of G_t for ALL visits to s across all episodes

- Biased but consistent estimator
- More data per episode → lower variance
- Converges to V^π(s) as number of episodes → ∞

Implemented in `mc_every_visit_prediction()` in `ch05_mc.rs`.
""",
        "theory_on_policy": r"""
**On-Policy MC Control** (epsilon-soft) learns Q*(s,a) and π* simultaneously:

1. Generate episode using epsilon-soft policy
2. For each (s,a) pair (first-visit): G ← discounted return
3. Q(s,a) ← average(G) — incremental update
4. π(s) ← argmax_a Q(s,a) — greedy improvement

The epsilon-soft policy ensures all (s,a) pairs are visited infinitely often.
As epsilon → 0, the policy converges to the optimal greedy policy.

Implemented in `mc_on_policy_control()` in `ch05_mc.rs`.
""",
        "theory_off_policy": r"""
**Off-Policy MC Control** with Weighted Importance Sampling:

- **Behaviour policy b**: generates episodes (uniform random)
- **Target policy π**: what we want to optimise (greedy)
- **Importance Sampling Ratio**: ρ = π(a|s) / b(a|s)

Weighted IS update:
Q(s,a) ← Q(s,a) + (W / C(s,a)) * [G - Q(s,a)]
C(s,a) ← C(s,a) + W

Off-policy MC can learn the optimal policy while following a different (exploratory) policy.
This is the foundation for modern off-policy algorithms like Q-Learning (Ch06).

Implemented in `mc_off_policy_control()` in `ch05_mc.rs`.
""",
        "algo_labels": {
            "first_visit": "First-Visit MC",
            "every_visit": "Every-Visit MC",
            "on_policy":   "On-Policy Control",
            "off_policy":  "Off-Policy (IS)",
        },
        "pros_list": {
            "first_visit": ["Unbiased estimator", "Simple implementation", "No model needed"],
            "every_visit": ["More data per episode", "Lower variance than first-visit", "No model needed"],
            "on_policy":   ["Learns Q* and π* simultaneously", "No model needed", "Guaranteed to visit all (s,a)"],
            "off_policy":  ["Learns from any behaviour policy", "Foundation for Q-Learning", "Can reuse historical data"],
        },
        "cons_list": {
            "first_visit": ["High variance", "Must wait for episode end", "Slow for long episodes"],
            "every_visit": ["Biased estimator", "Must wait for episode end", "Slow for long episodes"],
            "on_policy":   ["Epsilon must stay > 0", "Slower convergence than DP", "High variance"],
            "off_policy":  ["IS ratio can explode", "High variance", "Complex implementation"],
        },
    },
        "DE": {
        "title": "Kapitel 05 — Monte-Carlo-Methoden",
        "subtitle": "MC-Vorhersage — MC-Kontrolle — ASP Warschau",
        "engine_missing": "Ausführen: `cd rlvr-py && maturin develop`",
        "sidebar_title": "Einstellungen",
        "n_episodes": "Episoden", "gamma": "Gamma", "epsilon": "Epsilon", "seed": "Zufallsseed",
        "run_btn": "▶ Monte-Carlo starten",
        "guide_title": "Anleitung",
        "guide": "MC lernt aus vollständigen Episoden. G_t wird rückwärts berechnet und zum Aktualisieren von V(s) verwendet.",
        "theory_title": "Theorie — Kapitel 05",
        "theory_sections": {"mc": "5.1 Monte-Carlo-Vorhersage", "control": "5.2 MC-Kontrolle"},
        "summary_title": "Zusammenfassung", "summary_results": "Ergebnisse",
        "summary_pros_cons": "Monte-Carlo — Vor- & Nachteile",
        "pros": "Vorteile", "cons": "Nachteile",
    },


    "FR": {
        "title": "Chapitre 05 — Méthodes de Monte Carlo",
        "subtitle": "Apprentissage ASP par épisodes · Sans modèle · Région de Varsovie",
        "engine_missing": "Exécutez : `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Paramètres MC",
        "n_episodes": "Nombre d\'épisodes",
        "gamma": "γ — Facteur d\'actualisation",
        "epsilon": "ε — Exploration initiale",
        "epsilon_decay": "Taux de décroissance ε α",
        "seed": "Graine aléatoire",
        "run_btn": "▶ Lancer les quatre algorithmes MC",
        "guide_title": "🎓 Comment utiliser ce chapitre",
        "guide": "Quatre algorithmes MC sans modèle. Augmentez le nombre d\'épisodes pour voir la convergence.",
        "returns_title": "📈 Retours par épisode — Quatre algorithmes",
        "returns_caption": "Moyenne mobile des retours. Le contrôle on-policy devrait s\'améliorer.",
        "value_title": "📊 Fonction de valeur V(s) — MC vs référence DP",
        "value_caption": "Les estimations MC convergent vers la solution DP (Ch04).",
        "visits_title": "🗺️ Comptage des visites — First-Visit MC",
        "visits_caption": "Les états rarement visités ont des estimations V(s) à haute variance.",
        "conv_title": "📈 Convergence — Max |V^(k) - V^(k-1)|",
        "conv_caption": "La convergence MC est plus bruyante que DP.",
        "qtable_title": "📊 Table Q — Contrôle On-Policy",
        "qtable_caption": "Valeurs Q(s,a) apprises par MC on-policy.",
        "glass_title": "🔬 Glass-Box — Trace d\'épisode",
        "glass_headers": ["Étape", "État", "Action", "Récompense", "Retour G_t"],
        "summary_title": "📊 Résumé",
        "summary_results": "Comparaison des algorithmes",
        "summary_pros_cons": "Algorithmes MC — Avantages & Inconvénients",
        "pros": "✅ Avantages", "cons": "❌ Inconvénients",
        "theory_title": "📖 Théorie — Chapitre 05",
        "theory_sections": {
            "intro": "§5.1 Introduction aux méthodes MC",
            "first_visit": "§5.2 Prédiction MC First-Visit",
            "every_visit": "§5.2 Prédiction MC Every-Visit",
            "on_policy": "§5.3 Contrôle MC On-Policy",
            "off_policy": "§5.4 MC Off-Policy avec échantillonnage d\'importance",
        },
        "theory_intro": "G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ...",
        "theory_first_visit": "V(s) ← moyenne des G_t pour les premières visites à s.",
        "theory_every_visit": "V(s) ← moyenne des G_t pour toutes les visites à s.",
        "theory_on_policy": "Générer épisode → mettre à jour Q(s,a) → améliorer π greedily.",
        "theory_off_policy": "Q(s,a) ← Q(s,a) + (W/C) * [G - Q(s,a)]",
        "algo_labels": {"first_visit": "First-Visit MC", "every_visit": "Every-Visit MC", "on_policy": "Contrôle On-Policy", "off_policy": "Off-Policy (IS)"},
        "pros_list": {"first_visit": ["Estimateur non biaisé", "Sans modèle"], "every_visit": ["Plus de données", "Sans modèle"], "on_policy": ["Apprend Q* et π*", "Sans modèle"], "off_policy": ["Apprend de n\'importe quelle politique", "Base du Q-Learning"]},
        "cons_list": {"first_visit": ["Haute variance", "Attend la fin de l\'épisode"], "every_visit": ["Estimateur biaisé", "Attend la fin"], "on_policy": ["ε doit rester > 0", "Convergence lente"], "off_policy": ["Ratio IS peut exploser", "Haute variance"]},
    },
    "ES": {
        "title": "Capítulo 05 — Métodos de Monte Carlo",
        "subtitle": "Aprendizaje ASP por episodios · Sin modelo · Región de Varsovia",
        "engine_missing": "Ejecute: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Configuración MC",
        "n_episodes": "Número de episodios",
        "gamma": "γ — Factor de descuento",
        "epsilon": "ε — Exploración inicial",
        "epsilon_decay": "Tasa de decaimiento ε α",
        "seed": "Semilla aleatoria",
        "run_btn": "▶ Ejecutar los cuatro algoritmos MC",
        "guide_title": "🎓 Cómo usar este capítulo",
        "guide": "Cuatro algoritmos MC sin modelo. Aumente el número de episodios para ver la convergencia.",
        "returns_title": "📈 Retornos por episodio — Cuatro algoritmos",
        "returns_caption": "Media móvil de retornos. El control on-policy debería mejorar.",
        "value_title": "📊 Función de valor V(s) — MC vs referencia DP",
        "value_caption": "Las estimaciones MC convergen hacia la solución DP (Ch04).",
        "visits_title": "🗺️ Conteo de visitas — First-Visit MC",
        "visits_caption": "Los estados raramente visitados tienen estimaciones V(s) de alta varianza.",
        "conv_title": "📈 Convergencia — Max |V^(k) - V^(k-1)|",
        "conv_caption": "La convergencia MC es más ruidosa que DP.",
        "qtable_title": "📊 Tabla Q — Control On-Policy",
        "qtable_caption": "Valores Q(s,a) aprendidos por MC on-policy.",
        "glass_title": "🔬 Glass-Box — Traza de episodio",
        "glass_headers": ["Paso", "Estado", "Acción", "Recompensa", "Retorno G_t"],
        "summary_title": "📊 Resumen",
        "summary_results": "Comparación de algoritmos",
        "summary_pros_cons": "Algoritmos MC — Pros y Contras",
        "pros": "✅ Pros", "cons": "❌ Contras",
        "theory_title": "📖 Teoría — Capítulo 05",
        "theory_sections": {
            "intro": "§5.1 Introducción a los métodos MC",
            "first_visit": "§5.2 Predicción MC First-Visit",
            "every_visit": "§5.2 Predicción MC Every-Visit",
            "on_policy": "§5.3 Control MC On-Policy",
            "off_policy": "§5.4 MC Off-Policy con muestreo de importancia",
        },
        "theory_intro": "G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ...",
        "theory_first_visit": "V(s) ← promedio de G_t para primeras visitas a s.",
        "theory_every_visit": "V(s) ← promedio de G_t para todas las visitas a s.",
        "theory_on_policy": "Generar episodio → actualizar Q(s,a) → mejorar π greedy.",
        "theory_off_policy": "Q(s,a) ← Q(s,a) + (W/C) * [G - Q(s,a)]",
        "algo_labels": {"first_visit": "First-Visit MC", "every_visit": "Every-Visit MC", "on_policy": "Control On-Policy", "off_policy": "Off-Policy (IS)"},
        "pros_list": {"first_visit": ["Estimador insesgado", "Sin modelo"], "every_visit": ["Más datos", "Sin modelo"], "on_policy": ["Aprende Q* y π*", "Sin modelo"], "off_policy": ["Aprende de cualquier política", "Base del Q-Learning"]},
        "cons_list": {"first_visit": ["Alta varianza", "Espera fin de episodio"], "every_visit": ["Estimador sesgado", "Espera fin"], "on_policy": ["ε debe ser > 0", "Convergencia lenta"], "off_policy": ["Ratio IS puede explotar", "Alta varianza"]},
    },
    "PL": {
        "title": "Rozdział 05 — Metody Monte Carlo",
        "subtitle": "Uczenie ASP z epizodów · Bez modelu · Region Warszawy",
        "engine_missing": "Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Ustawienia MC",
        "n_episodes": "Liczba epizodów",
        "gamma": "γ — Współczynnik dyskontowania",
        "epsilon": "ε — Eksploracja początkowa",
        "epsilon_decay": "Współczynnik zaniku ε α",
        "seed": "Ziarno losowości",
        "run_btn": "▶ Uruchom wszystkie cztery algorytmy MC",
        "guide_title": "🎓 Jak korzystać z tego rozdziału",
        "guide": """
**Krok 1**
MC uczy się BEZ modelu P(s\'|s,a) — tylko z epizodów.

**Krok 2**
Ustaw liczbę epizodów.
Zacznij od 200, potem 2000.

**Krok 3**
Kliknij ▶ aby uruchomić wszystkie cztery algorytmy.

**Krok 4**
Odczytaj krzywą zwrotów — powinna rosnąć dla on-policy.

**Krok 5**
Porównaj V(s) MC z rozwiązaniem DP z Ch04.

**Krok 6**
Odczytaj mapę ciepła wizyt — rzadko odwiedzane stany mają wysoką wariancję.

**Krok 7**
Odczytaj Glass-Box — dokładne zwroty G_t dla wybranego epizodu.
""",
        "returns_title": "📈 Zwroty epizodów — Cztery algorytmy",
        "returns_caption": "Średnia krocząca zwrotów. On-policy control powinien się poprawiać.",
        "value_title": "📊 Funkcja wartości V(s) — MC vs referencja DP",
        "value_caption": "Estymaty MC zbiegają do rozwiązania DP (Ch04) przy większej liczbie epizodów.",
        "visits_title": "🗺️ Liczba wizyt — First-Visit MC",
        "visits_caption": "Rzadko odwiedzane stany mają estymaty V(s) z wysoką wariancją.",
        "conv_title": "📈 Zbieżność — Max |V^(k) - V^(k-1)|",
        "conv_caption": "Zbieżność MC jest bardziej zaszumiona niż DP — środowisko stochastyczne.",
        "qtable_title": "📊 Tabela Q — On-Policy Control",
        "qtable_caption": "Wartości Q(s,a) wyuczone przez on-policy MC.",
        "glass_title": "🔬 Glass-Box — Ślad epizodu",
        "glass_headers": ["Krok", "Stan", "Akcja", "Nagroda", "Zwrot G_t"],
        "summary_title": "📊 Podsumowanie",
        "summary_results": "Porównanie algorytmów",
        "summary_pros_cons": "Algorytmy MC — Zalety i Wady",
        "pros": "✅ Zalety", "cons": "❌ Wady",
        "theory_title": "📖 Teoria — Rozdział 05",
        "theory_sections": {
            "intro": "§5.1 Wprowadzenie do metod Monte Carlo",
            "first_visit": "§5.2 Predykcja MC First-Visit",
            "every_visit": "§5.2 Predykcja MC Every-Visit",
            "on_policy": "§5.3 Sterowanie MC On-Policy",
            "off_policy": "§5.4 MC Off-Policy z próbkowaniem ważności",
        },
        "theory_intro": r"""
**Metody Monte Carlo** uczą się bezpośrednio z epizodów — bez modelu P(s\'|s,a).
G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ...
Implementacja: `ch05_mc.rs` — `generate_episode()`, `mc_first_visit_prediction()`.
""",
        "theory_first_visit": "V(s) ← średnia G_t dla pierwszych wizyt w s.",
        "theory_every_visit": "V(s) ← średnia G_t dla wszystkich wizyt w s.",
        "theory_on_policy": "Generuj epizod → aktualizuj Q(s,a) → popraw π zachłannie.",
        "theory_off_policy": "Q(s,a) ← Q(s,a) + (W/C) * [G - Q(s,a)]",
        "algo_labels": {"first_visit": "First-Visit MC", "every_visit": "Every-Visit MC", "on_policy": "On-Policy Control", "off_policy": "Off-Policy (IS)"},
        "pros_list": {"first_visit": ["Nieobciążony estymator", "Bez modelu", "Prosta implementacja"], "every_visit": ["Więcej danych na epizod", "Bez modelu"], "on_policy": ["Uczy Q* i π* jednocześnie", "Bez modelu"], "off_policy": ["Uczy z dowolnej polityki", "Podstawa Q-Learning"]},
        "cons_list": {"first_visit": ["Wysoka wariancja", "Czeka na koniec epizodu"], "every_visit": ["Obciążony estymator", "Czeka na koniec"], "on_policy": ["ε musi być > 0", "Wolna zbieżność"], "off_policy": ["Współczynnik IS może eksplodować", "Wysoka wariancja"]},
    },
}

COLORS = {
    "first_visit": "#0082F0",
    "every_visit": "#FF8C0A",
    "on_policy":   "#0FC373",
    "off_policy":  "#FF3232",
}

def _moving_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


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
    lang = st.session_state.get("lang", "EN")
    tx = _tx(lang)
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

    with st.expander(tx["guide_title"], expanded=False):
        st.markdown(tx["guide"])

    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running Rust MC engine..."):
            result = rlvr_py.run_ch05_mc(
                int(seed), int(n_episodes), float(gamma),
                float(epsilon), float(epsilon_decay)
            )
        st.session_state["ch05_result"] = result

    if "ch05_result" not in st.session_state:
        st.info("Configure settings and click **▶ Run All Four MC Algorithms**.")
        _render_theory(tx)
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

    _render_theory(tx)


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
            "Best action S7":   f"A{r['policy'][7]}",
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


def _render_theory(tx):
    st.markdown("---")
    st.subheader(tx["theory_title"])
    for key in ["intro", "first_visit", "every_visit", "on_policy", "off_policy"]:
        with st.expander(tx["theory_sections"][key], expanded=False):
            st.markdown(tx[f"theory_{key}"])
