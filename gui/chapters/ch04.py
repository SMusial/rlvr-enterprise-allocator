import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
T = {
    "EN": {
        "title": "Chapter 04 — Dynamic Programming: Policy & Value Iteration",
        "subtitle": "ASP Operational State Optimisation · PI vs VI vs Async VI · Warsaw Region",
        "engine_missing": "⚙️ Rust engine not found. Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ DP Settings",
        "gamma": "γ — Discount factor",
        "theta": "θ — Convergence threshold",
        "seed": "Random seed",
        "run_btn": "▶ Run All Three DP Algorithms",
        "kpi_pi_iters": "PI outer iterations",
        "kpi_vi_iters": "VI iterations",
        "kpi_async_iters": "Async VI iterations",
        "kpi_policy_match": "PI = VI policy",
        "conv_title": "📈 Convergence Comparison — PI vs VI vs Async VI",
        "conv_x": "Sweep",
        "conv_y": "Max Bellman residual ‖δV‖∞",
        "conv_caption": "All three algorithms converge to the same V*. Async VI prioritises high-residual states.",
        "policy_evo_title": "🔄 Policy Evolution — Policy Iteration Steps",
        "policy_evo_caption": "Each row = one PI outer iteration. Cells show optimal action per state.",
        "residual_title": "🗺️ Bellman Residual per State (after convergence)",
        "residual_caption": "Higher residual = state was harder to optimise. S5–S7 typically highest.",
        "value_title": "📊 Final Value Function V*(s) — All Three Algorithms",
        "value_caption": "All three algorithms should produce identical V*(s). Any difference = numerical precision only.",
        "policy_title": "🎯 Optimal Policy π*(s) — PI vs VI",
        "policy_caption": "PI and VI must find the same optimal policy. Differences indicate a bug.",
        "glass_title": "🔬 Glass-Box — Policy Iteration Trace",
        "glass_headers": ["PI Step", "State", "Old Action", "New Action", "Changed"],
        "summary_title": "📊 Episode Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "DP Algorithms — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "theory_policy_eval": r"""
**Policy Evaluation** computes V^π(s) for a fixed policy π using the Bellman expectation equation:

V^π(s) = Σ_s' P(s'|s,π(s)) · [R(s,π(s)) + γ · V^π(s')]

Starting from V = 0, iterate until ‖V^(k+1) - V^(k)‖∞ < θ.

This is a **fixed-point iteration** — the Bellman expectation operator is a contraction mapping,
guaranteeing convergence to the unique V^π.

Implemented in `policy_evaluation()` in `ch04_dp.rs`.
""",
        "theory_policy_improve": r"""
**Policy Improvement** derives a better policy π' from V^π by acting greedily:

π'(s) = argmax_a Σ_s' P(s'|s,a) · [R(s,a) + γ · V^π(s')]

**Policy Improvement Theorem**: if π'(s) ≠ π(s) for any state, then V^π'(s) ≥ V^π(s) for all s.
The new policy is always at least as good as the old one.

Implemented in `policy_improvement()` in `ch04_dp.rs`.
""",
        "theory_pi": r"""
**Policy Iteration** alternates between evaluation and improvement until the policy stabilises:

1. Initialise π arbitrarily
2. **Evaluate**: compute V^π using Bellman expectation equation
3. **Improve**: π' = greedy(V^π)
4. If π' = π → STOP (optimal). Else π ← π', go to 2.

**Convergence**: guaranteed in finite steps (finite state/action space).
PI typically converges in very few outer iterations (3–10) even for large state spaces.

Implemented in `policy_iteration()` in `ch04_dp.rs`.
""",
        "theory_async_dp": r"""
**Asynchronous DP** updates states selectively rather than all at once.

**Bellman Residual** for state s:
Residual(s) = |V^(k+1)(s) - V^(k)(s)|

States with high residuals are updated first — they have the most to gain from an update.

**Prioritized Sweeping**: maintain a priority queue ordered by residual.
Pop highest-residual state, update it, propagate to predecessors.

**Advantage**: focuses computation on states that matter most.
In ASP: S5 and S7 (critical states) get updated first → faster convergence in crisis scenarios.

Implemented in `async_value_iteration()` in `ch04_dp.rs`.
""",
        "pros_list": {
            "pi": [
                "Converges in very few outer iterations (3-10 typically)",
                "Policy improvement theorem guarantees monotone improvement",
                "Exact policy evaluation at each step",
                "Natural for problems where policy is the primary output",
            ],
            "vi": [
                "Simpler implementation — no inner/outer loop",
                "Each iteration is a single Bellman sweep",
                "Often faster total computation than PI",
                "Direct application of Bellman optimality equation",
            ],
            "async": [
                "Focuses computation on high-impact states",
                "Faster convergence in practice for large state spaces",
                "Natural for online/real-time settings",
                "Prioritized Sweeping is near-optimal update schedule",
            ],
        },
        "cons_list": {
            "pi": [
                "Each outer iteration requires full policy evaluation (expensive)",
                "Requires full model P(s'|s,a) — not model-free",
                "Synchronous updates — all states updated each sweep",
                "Overkill for small state spaces",
            ],
            "vi": [
                "Requires full model P(s'|s,a)",
                "Synchronous — all states updated each iteration",
                "More iterations than PI outer loops",
                "No intermediate policy available during convergence",
            ],
            "async": [
                "Requires full model P(s'|s,a)",
                "Residual computation adds overhead per iteration",
                "Update order affects convergence path (not final result)",
                "More complex implementation than synchronous VI",
            ],
        },
        "algo_labels": {
            "pi": "Policy Iteration",
            "vi": "Value Iteration",
            "async": "Async VI",
        },
    },
    "DE": {
        "title": "Kapitel 04 — Dynamische Programmierung: Policy & Value Iteration",
        "subtitle": "ASP-Zustandsoptimierung — PI vs VI vs Async VI — Region Warschau",
        "engine_missing": "⚠ Rust-Engine nicht gefunden. Ausführen: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ DP-Einstellungen",
        "gamma": "γ — Diskontierungsfaktor",
        "theta": "θ — Konvergenzschwelle",
        "seed": "Zufallsseed",
        "run_btn": "▶ Alle drei DP-Algorithmen starten",
        "kpi_pi_iters": "PI-Außeniterationen",
        "kpi_vi_iters": "VI-Iterationen",
        "kpi_async_iters": "Async-VI-Iterationen",
        "kpi_policy_match": "PI = VI Strategie",
        "conv_title": "📉 Konvergenzvergleich — PI vs VI vs Async VI",
        "conv_x": "Durchlauf",
        "conv_y": "Max. Bellman-Residual ΔV",
        "conv_caption": "Alle drei Algorithmen konvergieren zum selben V*. Async VI priorisiert Zustände mit hohem Residual.",
        "policy_evo_title": "🔄 Strategieentwicklung — PI-Schritte",
        "policy_evo_caption": "Jede Zeile = eine PI-Außeniterierung. Zellen zeigen optimale Aktion pro Zustand.",
        "residual_title": "🧮 Bellman-Residual pro Zustand (nach Konvergenz)",
        "residual_caption": "Hohes Residual = Zustand schwieriger zu optimieren. S5–S7 typischerweise am höchsten.",
        "value_title": "📊 Finale Wertfunktion V*(s) — Alle drei Algorithmen",
        "value_caption": "Alle drei Algorithmen sollten identisches V*(s) liefern.",
        "policy_title": "🎯 Optimale Strategie π*(s) — PI vs VI",
        "policy_caption": "PI und VI müssen dieselbe optimale Strategie finden.",
        "glass_title": "🔍 Glass-Box — Policy-Iteration-Protokoll",
        "glass_headers": ["PI-Schritt", "Zustand", "Alte Aktion", "Neue Aktion", "Geändert"],
        "summary_title": "📋 Zusammenfassung",
        "summary_results": "Algorithmenvergleich",
        "summary_pros_cons": "DP-Algorithmen — Vor- & Nachteile",
        "pros": "✅ Vorteile",
        "cons": "❌ Nachteile",
        "theory_policy_eval": r"""**Strategiebewertung** berechnet V^π(s) für eine feste Strategie π:
V^π(s) = Σ_s' P(s'|s,π(s)) [R(s,π(s)) + γ V^π(s')]
Iterieren bis ‖V^(k+1) - V^(k)‖ < θ.
""",
        "theory_policy_improve": r"""**Strategieverbesserung** leitet eine bessere Strategie π' aus V^π ab:
π'(s) = argmax_a Σ_s' P(s'|s,a) [R(s,a) + γ V^π(s')]
Theorem: V^π'(s) ≥ V^π(s) für alle s.
""",
        "theory_pi": r"""**Policy Iteration** wechselt zwischen Bewertung und Verbesserung bis zur Stabilisierung:
1. π beliebig initialisieren
2. **Bewerten**: V^π berechnen
3. **Verbessern**: π' = greedy(V^π)
4. Falls π' = π → STOP. Sonst π ← π', weiter zu 2.
Konvergenz in endlich vielen Schritten garantiert.
""",
        "theory_async_dp": r"""**Asynchrones DP** aktualisiert Zustände selektiv nach Bellman-Residual:
Residual(s) = |V^(k+1)(s) - V^(k)(s)|
Zustände mit hohem Residual werden zuerst aktualisiert.
""",
        "pros_list": {
            "pi": [
                "Konvergiert in wenigen Außeniterationen (3–10 typisch)",
                "Strategieverbesserungstheorem garantiert monotone Verbesserung",
                "Exakte Strategiebewertung in jedem Schritt",
            ],
            "vi": [
                "Einfachere Implementierung — keine innere/äußere Schleife",
                "Jede Iteration ist ein einzelner Bellman-Durchlauf",
                "Oft schnellere Gesamtberechnung als PI",
            ],
            "async": [
                "Fokussiert Berechnung auf wichtige Zustände",
                "Schnellere Konvergenz in der Praxis",
                "Natürlich für Online-/Echtzeit-Einstellungen",
            ],
        },
        "cons_list": {
            "pi": [
                "Jede Außeniterierung erfordert vollständige Strategiebewertung",
                "Benötigt vollständiges Modell P(s'|s,a)",
                "Synchrone Aktualisierungen",
            ],
            "vi": [
                "Benötigt vollständiges Modell P(s'|s,a)",
                "Mehr Iterationen als PI-Außenschleifen",
                "Keine Zwischenstrategie während der Konvergenz",
            ],
            "async": [
                "Benötigt vollständiges Modell P(s'|s,a)",
                "Residualberechnung erhöht Overhead",
                "Komplexere Implementierung",
            ],
        },
        "algo_labels": {
            "pi": "Policy Iteration",
            "vi": "Value Iteration",
            "async": "Async VI",
        },
    },
    "FR": {
        "title": "Chapitre 04 — Programmation Dynamique : Itération de Politique et de Valeur",
        "subtitle": "Optimisation des états ASP · PI vs VI vs VI Async · Région de Varsovie",
        "engine_missing": "⚙️ Moteur Rust introuvable. Exécutez : `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Paramètres DP",
        "gamma": "γ — Facteur d'actualisation",
        "theta": "θ — Seuil de convergence",
        "seed": "Graine aléatoire",
        "run_btn": "▶ Lancer les trois algorithmes DP",
        "kpi_pi_iters": "Itérations PI externes",
        "kpi_vi_iters": "Itérations VI",
        "kpi_async_iters": "Itérations VI Async",
        "kpi_policy_match": "PI = VI politique",
        "conv_title": "📈 Comparaison de convergence — PI vs VI vs VI Async",
        "conv_x": "Balayage",
        "conv_y": "Résidu de Bellman max ‖δV‖∞",
        "conv_caption": "Les trois algorithmes convergent vers le même V*. VI Async priorise les états à résidu élevé.",
        "policy_evo_title": "🔄 Évolution de la politique — Étapes PI",
        "policy_evo_caption": "Chaque ligne = une itération PI externe. Les cellules montrent l'action optimale par état.",
        "residual_title": "🗺️ Résidu de Bellman par état (après convergence)",
        "residual_caption": "Résidu élevé = état plus difficile à optimiser. S5–S7 généralement les plus élevés.",
        "value_title": "📊 Fonction de valeur finale V*(s) — Trois algorithmes",
        "value_caption": "Les trois algorithmes produisent le même V*(s).",
        "policy_title": "🎯 Politique optimale π*(s) — PI vs VI",
        "policy_caption": "PI et VI doivent trouver la même politique optimale.",
        "glass_title": "🔬 Glass-Box — Trace d'itération de politique",
        "glass_headers": ["Étape PI", "État", "Ancienne action", "Nouvelle action", "Changé"],
        "summary_title": "📊 Résumé",
        "summary_results": "Comparaison des algorithmes",
        "summary_pros_cons": "Algorithmes DP — Avantages & Inconvénients",
        "pros": "✅ Avantages",
        "cons": "❌ Inconvénients",
        "theory_policy_eval": r"""
**Évaluation de politique** : V^π(s) = Σ_s' P(s'|s,π(s)) · [R(s,π(s)) + γ · V^π(s')]
Itérer jusqu'à ‖V^(k+1) - V^(k)‖∞ < θ.
""",
        "theory_policy_improve": r"""
**Amélioration de politique** : π'(s) = argmax_a Σ_s' P(s'|s,a) · [R(s,a) + γ · V^π(s')]
Théorème : V^π'(s) ≥ V^π(s) pour tout s.
""",
        "theory_pi": r"""
**Itération de politique** : alterner évaluation et amélioration jusqu'à stabilisation.
Convergence garantie en nombre fini d'étapes.
""",
        "theory_async_dp": r"""
**DP asynchrone** : mettre à jour les états par ordre de résidu de Bellman décroissant.
Résidu(s) = |V^(k+1)(s) - V^(k)(s)|
""",
        "pros_list": {
            "pi": ["Peu d'itérations externes", "Amélioration monotone garantie", "Évaluation exacte"],
            "vi": ["Implémentation simple", "Un seul balayage par itération", "Souvent plus rapide que PI"],
            "async": ["Focalise sur les états importants", "Convergence plus rapide en pratique"],
        },
        "cons_list": {
            "pi": ["Évaluation complète coûteuse", "Nécessite le modèle P(s'|s,a)"],
            "vi": ["Nécessite le modèle P(s'|s,a)", "Plus d'itérations que PI"],
            "async": ["Calcul de résidu supplémentaire", "Implémentation plus complexe"],
        },
        "algo_labels": {"pi": "Itération de politique", "vi": "Itération de valeur", "async": "VI Async"},
    },
    "ES": {
        "title": "Capítulo 04 — Programación Dinámica: Iteración de Política y Valor",
        "subtitle": "Optimización de estados ASP · PI vs VI vs VI Async · Región de Varsovia",
        "engine_missing": "⚙️ Motor Rust no encontrado. Ejecute: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Configuración DP",
        "gamma": "γ — Factor de descuento",
        "theta": "θ — Umbral de convergencia",
        "seed": "Semilla aleatoria",
        "run_btn": "▶ Ejecutar los tres algoritmos DP",
        "kpi_pi_iters": "Iteraciones PI externas",
        "kpi_vi_iters": "Iteraciones VI",
        "kpi_async_iters": "Iteraciones VI Async",
        "kpi_policy_match": "PI = VI política",
        "conv_title": "📈 Comparación de convergencia — PI vs VI vs VI Async",
        "conv_x": "Barrido",
        "conv_y": "Residuo de Bellman máx ‖δV‖∞",
        "conv_caption": "Los tres algoritmos convergen al mismo V*. VI Async prioriza estados de alto residuo.",
        "policy_evo_title": "🔄 Evolución de política — Pasos PI",
        "policy_evo_caption": "Cada fila = una iteración PI externa. Las celdas muestran la acción óptima por estado.",
        "residual_title": "🗺️ Residuo de Bellman por estado (tras convergencia)",
        "residual_caption": "Residuo alto = estado más difícil de optimizar. S5–S7 generalmente los más altos.",
        "value_title": "📊 Función de valor final V*(s) — Tres algoritmos",
        "value_caption": "Los tres algoritmos producen el mismo V*(s).",
        "policy_title": "🎯 Política óptima π*(s) — PI vs VI",
        "policy_caption": "PI y VI deben encontrar la misma política óptima.",
        "glass_title": "🔬 Glass-Box — Traza de iteración de política",
        "glass_headers": ["Paso PI", "Estado", "Acción antigua", "Acción nueva", "Cambió"],
        "summary_title": "📊 Resumen",
        "summary_results": "Comparación de algoritmos",
        "summary_pros_cons": "Algoritmos DP — Pros y Contras",
        "pros": "✅ Pros",
        "cons": "❌ Contras",
        "theory_policy_eval": r"""
**Evaluación de política**: V^π(s) = Σ_s' P(s'|s,π(s)) · [R(s,π(s)) + γ · V^π(s')]
""",
        "theory_policy_improve": r"""
**Mejora de política**: π'(s) = argmax_a Σ_s' P(s'|s,a) · [R(s,a) + γ · V^π(s')]
""",
        "theory_pi": r"""
**Iteración de política**: alternar evaluación y mejora hasta estabilización.
""",
        "theory_async_dp": r"""
**DP asíncrono**: actualizar estados por orden de residuo de Bellman decreciente.
""",
        "pros_list": {
            "pi": ["Pocas iteraciones externas", "Mejora monotónica garantizada"],
            "vi": ["Implementación simple", "A menudo más rápido que PI"],
            "async": ["Focaliza en estados importantes", "Convergencia más rápida en práctica"],
        },
        "cons_list": {
            "pi": ["Evaluación completa costosa", "Requiere modelo P(s'|s,a)"],
            "vi": ["Requiere modelo P(s'|s,a)", "Más iteraciones que PI"],
            "async": ["Cálculo de residuo adicional", "Implementación más compleja"],
        },
        "algo_labels": {"pi": "Iteración de política", "vi": "Iteración de valor", "async": "VI Async"},
    },
    "PL": {
        "title": "Rozdział 04 — Programowanie Dynamiczne: Iteracja Polityki i Wartości",
        "subtitle": "Optymalizacja stanów ASP · PI vs VI vs Async VI · Region Warszawy",
        "engine_missing": "⚙️ Silnik Rust nie znaleziony. Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Ustawienia DP",
        "gamma": "γ — Współczynnik dyskontowania",
        "theta": "θ — Próg zbieżności",
        "seed": "Ziarno losowości",
        "run_btn": "▶ Uruchom wszystkie trzy algorytmy DP",
        "kpi_pi_iters": "Iteracje PI zewnętrzne",
        "kpi_vi_iters": "Iteracje VI",
        "kpi_async_iters": "Iteracje Async VI",
        "kpi_policy_match": "PI = VI polityka",
        "conv_title": "📈 Porównanie zbieżności — PI vs VI vs Async VI",
        "conv_x": "Przebieg",
        "conv_y": "Maks. residual Bellmana ‖δV‖∞",
        "conv_caption": "Wszystkie trzy algorytmy zbiegają do tego samego V*. Async VI priorytetyzuje stany z wysokim residualem.",
        "policy_evo_title": "🔄 Ewolucja polityki — Kroki PI",
        "policy_evo_caption": "Każdy wiersz = jedna zewnętrzna iteracja PI. Komórki pokazują optymalną akcję dla każdego stanu.",
        "residual_title": "🗺️ Residual Bellmana per stan (po zbieżności)",
        "residual_caption": "Wysoki residual = stan trudniejszy do optymalizacji. S5–S7 zazwyczaj najwyższe.",
        "value_title": "📊 Końcowa funkcja wartości V*(s) — Trzy algorytmy",
        "value_caption": "Wszystkie trzy algorytmy powinny dać identyczne V*(s).",
        "policy_title": "🎯 Optymalna polityka π*(s) — PI vs VI",
        "policy_caption": "PI i VI muszą znaleźć tę samą optymalną politykę.",
        "glass_title": "🔬 Glass-Box — Ślad iteracji polityki",
        "glass_headers": ["Krok PI", "Stan", "Stara akcja", "Nowa akcja", "Zmiana"],
        "summary_title": "📊 Podsumowanie",
        "summary_results": "Porównanie algorytmów",
        "summary_pros_cons": "Algorytmy DP — Zalety i Wady",
        "pros": "✅ Zalety",
        "cons": "❌ Wady",
        "theory_policy_eval": r"""
**Ewaluacja polityki**: V^π(s) = Σ_s' P(s'|s,π(s)) · [R(s,π(s)) + γ · V^π(s')]
Iteruj aż ‖V^(k+1) - V^(k)‖∞ < θ.
""",
        "theory_policy_improve": r"""
**Poprawa polityki**: π'(s) = argmax_a Σ_s' P(s'|s,a) · [R(s,a) + γ · V^π(s')]
Twierdzenie: V^π'(s) ≥ V^π(s) dla każdego s.
""",
        "theory_pi": r"""
**Iteracja polityki**: naprzemiennie ewaluacja i poprawa aż do stabilizacji.
Zbieżność gwarantowana w skończonej liczbie kroków.
""",
        "theory_async_dp": r"""
**Asynchroniczne DP**: aktualizuj stany w kolejności malejącego residualu Bellmana.
Residual(s) = |V^(k+1)(s) - V^(k)(s)|
""",
        "pros_list": {
            "pi": ["Mało zewnętrznych iteracji", "Monotoniczne polepszanie polityki", "Dokładna ewaluacja"],
            "vi": ["Prosta implementacja", "Jeden przebieg Bellmana na iterację", "Często szybszy niż PI"],
            "async": ["Skupia się na ważnych stanach", "Szybsza zbieżność w praktyce"],
        },
        "cons_list": {
            "pi": ["Kosztowna pełna ewaluacja", "Wymaga modelu P(s'|s,a)"],
            "vi": ["Wymaga modelu P(s'|s,a)", "Więcej iteracji niż PI"],
            "async": ["Dodatkowe obliczenie residuali", "Bardziej złożona implementacja"],
        },
        "algo_labels": {"pi": "Iteracja polityki", "vi": "Iteracja wartości", "async": "Async VI"},
    },
}

COLORS = {"pi": "#0082F0", "vi": "#FF8C0A", "async": "#0FC373"}

# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def _tx(lang):
    """Return translation dict for lang, filling missing keys from EN."""
    base = dict(T.get("EN", {}))
    over = T.get(lang, {})
    for k, v in over.items():
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
    gamma = st.sidebar.slider(tx["gamma"], 0.50, 0.999, 0.95, 0.005)
    theta = st.sidebar.select_slider(
        tx["theta"],
        options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}",
    )
    seed = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    run = st.button(tx["run_btn"], type="primary")

    if run:
        with st.spinner("Running Rust DP engine..."):
            result = rlvr_py.run_ch04_dp(int(seed), float(gamma), float(theta))
        st.session_state["ch04_result"] = result

    if "ch04_result" not in st.session_state:
        st.info("Configure settings and click **▶ Run All Three DP Algorithms**.")
        return

    result      = st.session_state["ch04_result"]
    pi          = result["pi"]
    vi          = result["vi"]
    av          = result["async_vi"]
    state_names = result["state_names"]
    action_names = result["action_names"]
    residuals   = result["residuals"]

    policy_match = pi["policy"] == vi["policy"]

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tx["kpi_pi_iters"],    str(pi["pi_iterations"]))
    c2.metric(tx["kpi_vi_iters"],    str(vi["iterations"]))
    c3.metric(tx["kpi_async_iters"], str(av["iterations"]))
    c4.metric(tx["kpi_policy_match"], "✅ Yes" if policy_match else "❌ No")

    # Convergence comparison
    st.subheader(tx["conv_title"])
    _render_convergence(pi, vi, av, tx)
    st.caption(tx["conv_caption"])

    # Value function comparison
    st.subheader(tx["value_title"])
    _render_value_comparison(pi, vi, av, state_names, tx)
    st.caption(tx["value_caption"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(tx["policy_title"])
        _render_policy_comparison(pi, vi, state_names, action_names, tx)
        st.caption(tx["policy_caption"])
    with col2:
        st.subheader(tx["residual_title"])
        _render_residuals(residuals, state_names, tx)
        st.caption(tx["residual_caption"])

    # Policy evolution
    st.subheader(tx["policy_evo_title"])
    _render_policy_evolution(pi, state_names, action_names, tx)
    st.caption(tx["policy_evo_caption"])

    # Glass-Box
    st.subheader(tx["glass_title"])
    _render_glass_box(pi, state_names, action_names, tx)

    # Summary
    st.subheader(tx["summary_title"])
    _render_summary(pi, vi, av, tx)

    # Theory


# ---------------------------------------------------------------------------
# Convergence comparison
# ---------------------------------------------------------------------------
def _render_convergence(pi, vi, av, tx):
    fig = go.Figure()
    for key, label, data in [
        ("pi",    tx["algo_labels"]["pi"],    pi["convergence_curve"]),
        ("vi",    tx["algo_labels"]["vi"],    vi["curve"]),
        ("async", tx["algo_labels"]["async"], av["curve"]),
    ]:
        fig.add_trace(go.Scatter(
            x=list(range(len(data))), y=data,
            mode="lines", name=label,
            line=dict(color=COLORS[key], width=2),
        ))
    fig.update_layout(
        height=320, margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title=tx["conv_x"],
        yaxis_title=tx["conv_y"],
        yaxis_type="log",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Value function comparison
# ---------------------------------------------------------------------------
def _render_value_comparison(pi, vi, av, state_names, tx):
    short = [f"S{i}" for i in range(len(pi["values"]))]
    fig = go.Figure()
    for key, label, vals in [
        ("pi",    tx["algo_labels"]["pi"],    pi["values"]),
        ("vi",    tx["algo_labels"]["vi"],    vi["values"]),
        ("async", tx["algo_labels"]["async"], av["values"]),
    ]:
        fig.add_trace(go.Bar(
            x=short, y=vals, name=label,
            marker_color=COLORS[key], opacity=0.8,
        ))
    fig.update_layout(
        height=300, barmode="group",
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Policy comparison table
# ---------------------------------------------------------------------------
def _render_policy_comparison(pi, vi, state_names, action_names, tx):
    rows = []
    for s in range(len(pi["policy"])):
        pi_a = pi["policy"][s]
        vi_a = vi["policy"][s]
        rows.append({
            "State": f"S{s}",
            "Situation": state_names[s].split(":")[1].strip()[:25],
            f"PI: {tx['algo_labels']['pi']}": f"A{pi_a}",
            f"VI: {tx['algo_labels']['vi']}": f"A{vi_a}",
            "Match": "✅" if pi_a == vi_a else "❌",
        })
    st.dataframe(rows, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Bellman residual heatmap
# ---------------------------------------------------------------------------
def _render_residuals(residuals, state_names, tx):
    short = [f"S{i}" for i in range(len(residuals))]
    colors = ["#0FC373" if r < 0.001 else "#FF8C0A" if r < 0.01 else "#FF3232"
              for r in residuals]
    fig = go.Figure(go.Bar(
        x=short, y=residuals,
        marker_color=colors,
        text=[f"{r:.2e}" for r in residuals],
        textposition="outside",
    ))
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=20, b=40),
        yaxis_title="Residual",
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Policy evolution table
# ---------------------------------------------------------------------------
def _render_policy_evolution(pi, state_names, action_names, tx):
    history = pi["policy_history"]
    if not history:
        return
    rows = []
    for step_idx, pol in enumerate(history):
        row = {"PI Step": step_idx}
        for s, a in enumerate(pol):
            row[f"S{s}"] = f"A{a}"
        rows.append(row)
    st.dataframe(rows, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Glass-Box — PI step trace
# ---------------------------------------------------------------------------
def _render_glass_box(pi, state_names, action_names, tx):
    history = pi["policy_history"]
    if len(history) < 2:
        st.info("Policy converged in 1 step — no changes to show.")
        return

    rows = []
    for step_idx in range(1, len(history)):
        old_pol = history[step_idx - 1]
        new_pol = history[step_idx]
        for s in range(len(old_pol)):
            old_a = old_pol[s]
            new_a = new_pol[s]
            changed = old_a != new_a
            rows.append({
                tx["glass_headers"][0]: step_idx,
                tx["glass_headers"][1]: f"S{s}",
                tx["glass_headers"][2]: f"A{old_a}: {action_names[old_a].split(':')[1].strip()[:20]}",
                tx["glass_headers"][3]: f"A{new_a}: {action_names[new_a].split(':')[1].strip()[:20]}",
                tx["glass_headers"][4]: "🔄 Yes" if changed else "—",
            })
    st.dataframe(rows, width='stretch', height=300)

    # Bellman equations
    st.markdown("---")
    st.markdown("**Policy Evaluation (inner loop):**")
    st.latex(r"V^\pi(s) \leftarrow \sum_{s'} P(s'|s,\pi(s))\left[R(s,\pi(s)) + \gamma V^\pi(s')\right]")
    st.markdown("**Policy Improvement:**")
    st.latex(r"\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)\left[R(s,a) + \gamma V^\pi(s')\right]")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _render_summary(pi, vi, av, tx):
    st.markdown(f"#### {tx['summary_results']}")
    rows = [
        {
            "Algorithm":       tx["algo_labels"]["pi"],
            "Outer iters":     str(pi["pi_iterations"]),
            "Total sweeps":    str(len(pi["convergence_curve"])),
            "V*(S0)":          f"{pi['values'][0]:.3f}",
            "V*(S7)":          f"{pi['values'][7]:.3f}",
        },
        {
            "Algorithm":       tx["algo_labels"]["vi"],
            "Outer iters":     "N/A",
            "Total sweeps":    str(vi["iterations"]),
            "V*(S0)":          f"{vi['values'][0]:.3f}",
            "V*(S7)":          f"{vi['values'][7]:.3f}",
        },
        {
            "Algorithm":       tx["algo_labels"]["async"],
            "Outer iters":     "N/A",
            "Total sweeps":    str(av["iterations"]),
            "V*(S0)":          f"{av['values'][0]:.3f}",
            "V*(S7)":          f"{av['values'][7]:.3f}",
        },
    ]
    st.dataframe(rows, hide_index=True)

    st.markdown(f"#### {tx['summary_pros_cons']}")
    for key in ["pi", "vi", "async"]:
        label = tx["algo_labels"][key]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{label} — {tx['pros']}**")
            for p in tx["pros_list"][key]:
                st.markdown(f"- {p}")
        with col2:
            st.markdown(f"**{label} — {tx['cons']}**")
            for c in tx["cons_list"][key]:
                st.markdown(f"- {c}")
        st.markdown("---")

