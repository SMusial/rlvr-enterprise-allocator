import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Translations — EN / FR / ES / PL  (mirrors ch07.py structure exactly)
# ---------------------------------------------------------------------------
T = {
    "EN": {
        "title":    "Chapter 08 — Eligibility Traces & TD(λ)",
        "subtitle": "SARSA(λ) · Q(λ) Watkins · λ=0 (TD0) · λ=0.99 (≈MC) · Warsaw ASP",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title":  "⚙️ Settings",
        "n_episodes":     "Number of episodes",
        "gamma":          "γ  Discount factor",
        "alpha":          "α  Learning rate",
        "epsilon":        "ε  Initial exploration",
        "epsilon_decay":  "ε decay rate",
        "lambda_":        "λ  Trace decay (0=TD, 1≈MC)",
        "replacing":      "Replacing traces (vs Accumulating)",
        "seed":           "Random seed",
        "run_btn":        "▶ Run All Four Algorithms",
        "guide_title":    "📖 How to use this chapter",
        "guide": """**Step 1 — Understand λ**
λ=0 → TD(0): only current (s,a) updated. No backward propagation.
λ=1 → Monte Carlo: full episode propagation. High variance.
λ=0.7 → sweet spot: fast propagation, stable learning.

**Step 2 — SARSA(λ) vs Q(λ)**
SARSA(λ): on-policy. Traces persist through exploratory actions.
Q(λ) Watkins: off-policy. Traces CUT when a non-greedy action is taken.

**Step 3 — Replacing vs Accumulating**
Replacing: e(s,a) = 1 on visit. Stable, recommended for tabular.
Accumulating: e(s,a) += 1 on visit. Can grow > 1 in loops.

**Step 4 — Watch the Active Traces chart**
Higher λ → more active traces → faster backward credit assignment.
λ=0 → max 1 active trace per step (only current (s,a)).

**Step 5 — Compare convergence speed**
SARSA(λ=0.7) should converge faster than SARSA(λ=0) baseline.""",
        "returns_title":   "📈 Episode Returns — All Four Algorithms",
        "returns_caption": "Moving average (window=30). SARSA(λ=0.7) should converge faster than TD0 baseline.",
        "td_error_title":  "📉 TD Error Curve",
        "td_error_caption":"TD error decays as agent learns. λ>0 propagates errors faster → quicker decay.",
        "value_title":     "🏛️ Value Function V(s)",
        "value_caption":   "All algorithms should converge to similar V*(s). S7 (SLA breach) lowest.",
        "trace_title":     "🔍 Active Traces per Episode",
        "trace_caption":   "Higher λ → more active traces → faster backward credit assignment.",
        "qtable_title":    "🗺️ Q-Table Heatmap",
        "qtable_caption":  "Q(s,a) values learned by selected algorithm.",
        "glass_title":     "🔬 Glass-Box — Eligibility Trace Mechanics",
        "summary_title":   "📊 Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "Algorithms — Pros & Cons",
        "pros": "✅ Pros", "cons": "❌ Cons",
        "theory_title":    "📚 Theory — Chapter 08",
        "theory_sections": {
            "et":     "8.1 Eligibility Traces",
            "sarsal": "8.2 SARSA(λ) — on-policy",
            "ql":     "8.3 Q(λ) — Watkins off-policy",
            "lambda": "8.4 Choosing λ",
        },
        "theory_et": r"""**Eligibility trace** e(s,a) tracks how recently (s,a) was visited:

Accumulating:  e_t(s,a) = γλ · e_{t-1}(s,a) + 𝟙[s=S_t, a=A_t]
Replacing:     e_t(s,a) = γλ · e_{t-1}(s,a) + 1  (set to 1 on visit)

Decay factor: γλ. With γ=0.95, λ=0.7 → γλ=0.665 per step.
After k steps without visit: e ≈ (0.665)^k → pruned at 1e-8 (~50 steps).
Implemented in `sarsa_lambda()` in `ch08_eligibility.rs`.""",
        "theory_sarsal": r"""**SARSA(λ)** — on-policy backward view:

δ_t = R_{t+1} + γ Q(S_{t+1},A_{t+1}) − Q(S_t,A_t)
e_t(s,a) = γλ · e_{t-1}(s,a) + 𝟙[s=S_t, a=A_t]
Q(s,a) ← Q(s,a) + α δ_t e_t(s,a)   for ALL (s,a)

λ=0 → only Q(S_t,A_t) updated (TD(0))
λ=1 → equivalent to Monte Carlo (γ=1)
Implemented in `sarsa_lambda()` in `ch08_eligibility.rs`""",
        "theory_ql": r"""**Q(λ) — Watkins' traces** — off-policy:

Same as SARSA(λ) but:
- TD target uses greedy action: max_a Q(S',a)
- Traces CUT to 0 when non-greedy action taken (Watkins, 1989)

Watkins' cut prevents off-policy contamination.
Allows more aggressive exploration vs SARSA(λ).
Implemented in `q_lambda()` in `ch08_eligibility.rs`""",
        "theory_lambda": r"""**Choosing λ**:

λ=0.0 → TD(0): fast per step, slow credit propagation
λ=0.5 → moderate: good for stochastic environments
λ=0.7 → sweet spot: fast convergence, stable (recommended)
λ=0.9 → near-MC: fast but high variance (reduce α)
λ=1.0 → Monte Carlo: requires episode completion

Rule of thumb: start at λ=0.7, reduce α if unstable.""",
        "algo_labels": {
            "sarsa_lambda": "SARSA(λ)",
            "q_lambda":     "Q(λ) Watkins",
            "sarsa_td0":    "SARSA λ=0 (TD0)",
            "sarsa_mc":     "SARSA λ=0.99 (≈MC)",
        },
        "pros_list": {
            "sarsa_lambda": ["On-policy, safe", "Fast backward credit assignment", "λ tunes bias-variance tradeoff"],
            "q_lambda":     ["Off-policy, learns optimal policy", "Watkins' cut prevents divergence", "More aggressive exploration possible"],
            "sarsa_td0":    ["Simplest, no traces overhead", "Low variance", "Baseline reference (Ch06)"],
            "sarsa_mc":     ["Near-zero bias", "Full episode propagation", "Upper bound on λ performance"],
        },
        "cons_list": {
            "sarsa_lambda": ["On-policy: needs ε>0", "O(|S||A|) per step", "λ must be tuned"],
            "q_lambda":     ["Traces cut on exploration", "Less propagation than SARSA(λ)", "Off-policy instability risk"],
            "sarsa_td0":    ["Slow credit assignment", "Many episodes to converge", "No backward propagation"],
            "sarsa_mc":     ["High variance", "Requires episode completion", "Unstable with small α"],
        },
    },
    "PL": {
        "title":    "Rozdział 08 — Ślady Kwalifikowalności & TD(λ)",
        "subtitle": "SARSA(λ) · Q(λ) Watkins · λ=0 (TD0) · λ=0.99 (≈MC) · ASP Warszawa",
        "engine_missing": "Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title":  "⚙️ Ustawienia",
        "n_episodes":     "Liczba epizodów",
        "gamma":          "γ  Współczynnik dyskontowania",
        "alpha":          "α  Współczynnik uczenia",
        "epsilon":        "ε  Eksploracja początkowa",
        "epsilon_decay":  "Współczynnik zaniku ε",
        "lambda_":        "λ  Zanikanie śladów (0=TD, 1≈MC)",
        "replacing":      "Replacing traces (vs Accumulating)",
        "seed":           "Ziarno losowości",
        "run_btn":        "▶ Uruchom wszystkie cztery algorytmy",
        "guide_title":    "📖 Jak korzystać z tego rozdziału",
        "guide": """**Krok 1 — Zrozum λ**
λ=0 → TD(0): tylko bieżąca para (s,a) aktualizowana.
λ=1 → Monte Carlo: propagacja przez cały epizod.
λ=0.7 → optimum: szybka propagacja, stabilne uczenie.

**Krok 2 — SARSA(λ) vs Q(λ)**
SARSA(λ): on-policy. Ślady trwają przez akcje eksploracyjne.
Q(λ) Watkins: off-policy. Ślady CIĘTE przy akcji niegreedy.

**Krok 3 — Replacing vs Accumulating**
Replacing: e(s,a) = 1 przy wizycie. Stabilne, zalecane.
Accumulating: e(s,a) += 1. Może rosnąć > 1 w pętlach.

**Krok 4 — Obserwuj wykres Active Traces**
Wyższe λ → więcej aktywnych śladów → szybsze przypisanie zasługi.

**Krok 5 — Porównaj szybkość zbieżności**
SARSA(λ=0.7) powinien zbiegać szybciej niż SARSA(λ=0).""",
        "returns_title":   "📈 Zwroty epizodów — Cztery algorytmy",
        "returns_caption": "Średnia krocząca (okno=30). SARSA(λ=0.7) powinien zbiegać szybciej niż TD0.",
        "td_error_title":  "📉 Krzywa błędu TD",
        "td_error_caption":"Błąd TD maleje w miarę uczenia. λ>0 propaguje błędy szybciej.",
        "value_title":     "🏛️ Funkcja wartości V(s)",
        "value_caption":   "Wszystkie algorytmy powinny zbiegać do podobnych V*(s). S7 najniższe.",
        "trace_title":     "🔍 Aktywne ślady na epizod",
        "trace_caption":   "Wyższe λ → więcej aktywnych śladów → szybsze przypisanie zasługi wstecz.",
        "qtable_title":    "🗺️ Heatmapa tabeli Q",
        "qtable_caption":  "Wartości Q(s,a) wyuczone przez wybrany algorytm.",
        "glass_title":     "🔬 Glass-Box — Mechanika śladów kwalifikowalności",
        "summary_title":   "📊 Podsumowanie",
        "summary_results": "Porównanie algorytmów",
        "summary_pros_cons": "Algorytmy — Zalety i Wady",
        "pros": "✅ Zalety", "cons": "❌ Wady",
        "theory_title":    "📚 Teoria — Rozdział 08",
        "theory_sections": {
            "et":     "8.1 Ślady kwalifikowalności",
            "sarsal": "8.2 SARSA(λ) — on-policy",
            "ql":     "8.3 Q(λ) — Watkins off-policy",
            "lambda": "8.4 Dobór λ",
        },
        "theory_et": r"""**Ślad kwalifikowalności** e(s,a) śledzi jak niedawno odwiedzono (s,a):

Accumulating: e_t(s,a) = γλ · e_{t-1}(s,a) + 𝟙[s=S_t, a=A_t]
Replacing:    e_t(s,a) = 1 przy wizycie

Czynnik zaniku: γλ. Dla γ=0.95, λ=0.7 → γλ=0.665 na krok.
Implementacja: `sarsa_lambda()` w `ch08_eligibility.rs`.""",
        "theory_sarsal": r"""**SARSA(λ)** — on-policy, widok do tyłu:

δ_t = R_{t+1} + γ Q(S_{t+1},A_{t+1}) − Q(S_t,A_t)
Q(s,a) ← Q(s,a) + α δ_t e_t(s,a)   dla WSZYSTKICH (s,a)

Implementacja: `sarsa_lambda()` w `ch08_eligibility.rs`""",
        "theory_ql": r"""**Q(λ) — ślady Watkinsa** — off-policy:

Cel TD: max_a Q(S',a) (greedy)
Ślady CIĘTE do 0 gdy akcja niegreedy.
Implementacja: `q_lambda()` w `ch08_eligibility.rs`""",
        "theory_lambda": r"""**Dobór λ**:
λ=0.0 → TD(0) | λ=0.7 → optimum | λ=0.99 → ≈MC
Zasada: zacznij od λ=0.7, zmniejsz α jeśli niestabilne.""",
        "algo_labels": {
            "sarsa_lambda": "SARSA(λ)",
            "q_lambda":     "Q(λ) Watkins",
            "sarsa_td0":    "SARSA λ=0 (TD0)",
            "sarsa_mc":     "SARSA λ=0.99 (≈MC)",
        },
        "pros_list": {
            "sarsa_lambda": ["On-policy, bezpieczny", "Szybkie przypisanie zasługi", "λ reguluje bias-variance"],
            "q_lambda":     ["Off-policy, uczy optymalnej polityki", "Cięcie Watkinsa zapobiega rozbieżności", "Agresywna eksploracja"],
            "sarsa_td0":    ["Najprostszy, bez śladów", "Niski variance", "Punkt odniesienia (Ch06)"],
            "sarsa_mc":     ["Prawie zerowy bias", "Pełna propagacja epizodyczna", "Górna granica λ"],
        },
        "cons_list": {
            "sarsa_lambda": ["On-policy: wymaga ε>0", "O(|S||A|) na krok", "λ wymaga strojenia"],
            "q_lambda":     ["Ślady cięte przy eksploracji", "Mniej propagacji niż SARSA(λ)", "Ryzyko niestabilności"],
            "sarsa_td0":    ["Wolne przypisanie zasługi", "Wiele epizodów do zbieżności", "Brak propagacji wstecz"],
            "sarsa_mc":     ["Wysoki variance", "Wymaga zakończenia epizodu", "Niestabilny przy małym α"],
        },
    },
    "FR": {
        "title":    "Chapitre 08 — Traces d'éligibilité & TD(λ)",
        "subtitle": "SARSA(λ) · Q(λ) Watkins · λ=0 (TD0) · λ=0.99 (≈MC) · ASP Varsovie",
        "engine_missing": "Exécutez : `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Paramètres",
        "n_episodes": "Épisodes", "gamma": "γ", "alpha": "α",
        "epsilon": "ε", "epsilon_decay": "Décroissance ε",
        "lambda_": "λ  Décroissance traces (0=TD, 1≈MC)",
        "replacing": "Replacing traces", "seed": "Graine",
        "run_btn": "▶ Lancer les quatre algorithmes",
        "guide_title": "📖 Guide",
        "guide": "λ=0→TD(0). λ=1→MC. λ=0.7 optimum. SARSA(λ) on-policy. Q(λ) off-policy avec coupure Watkins.",
        "returns_title": "📈 Retours", "returns_caption": "SARSA(λ=0.7) converge plus vite que TD0.",
        "td_error_title": "📉 Erreur TD", "td_error_caption": "",
        "value_title": "🏛️ V(s)", "value_caption": "",
        "trace_title": "🔍 Traces actives", "trace_caption": "λ élevé → plus de traces actives.",
        "qtable_title": "🗺️ Table Q", "qtable_caption": "",
        "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Résumé", "summary_results": "Comparaison",
        "summary_pros_cons": "Avantages & Inconvénients",
        "pros": "✅ Pros", "cons": "❌ Cons",
        "theory_title": "📚 Théorie",
        "theory_sections": {"et": "8.1 Traces", "sarsal": "8.2 SARSA(λ)", "ql": "8.3 Q(λ)", "lambda": "8.4 Choix λ"},
        "theory_et": "e_t(s,a) = γλ·e_{t-1}(s,a) + 𝟙[s=S_t,a=A_t]",
        "theory_sarsal": "Q(s,a) ← Q(s,a) + α δ_t e_t(s,a) pour tous (s,a)",
        "theory_ql": "Coupure Watkins: traces=0 si action non-greedy.",
        "theory_lambda": "λ=0→TD(0) | λ=0.7→optimum | λ=0.99→≈MC",
        "algo_labels": {"sarsa_lambda": "SARSA(λ)", "q_lambda": "Q(λ) Watkins",
                        "sarsa_td0": "SARSA λ=0", "sarsa_mc": "SARSA λ=0.99"},
        "pros_list": {"sarsa_lambda": ["On-policy"], "q_lambda": ["Off-policy"],
                      "sarsa_td0": ["Simple"], "sarsa_mc": ["Faible biais"]},
        "cons_list": {"sarsa_lambda": ["λ à régler"], "q_lambda": ["Coupure traces"],
                      "sarsa_td0": ["Lent"], "sarsa_mc": ["Variance élevée"]},
    },
    "ES": {
        "title":    "Capítulo 08 — Trazas de Elegibilidad & TD(λ)",
        "subtitle": "SARSA(λ) · Q(λ) Watkins · λ=0 (TD0) · λ=0.99 (≈MC) · ASP Varsovia",
        "engine_missing": "Ejecute: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Configuración",
        "n_episodes": "Episodios", "gamma": "γ", "alpha": "α",
        "epsilon": "ε", "epsilon_decay": "Decaimiento ε",
        "lambda_": "λ  Decaimiento trazas (0=TD, 1≈MC)",
        "replacing": "Replacing traces", "seed": "Semilla",
        "run_btn": "▶ Ejecutar los cuatro algoritmos",
        "guide_title": "📖 Guía",
        "guide": "λ=0→TD(0). λ=1→MC. λ=0.7 óptimo. SARSA(λ) on-policy. Q(λ) off-policy con corte Watkins.",
        "returns_title": "📈 Retornos", "returns_caption": "SARSA(λ=0.7) converge más rápido que TD0.",
        "td_error_title": "📉 Error TD", "td_error_caption": "",
        "value_title": "🏛️ V(s)", "value_caption": "",
        "trace_title": "🔍 Trazas activas", "trace_caption": "λ alto → más trazas activas.",
        "qtable_title": "🗺️ Tabla Q", "qtable_caption": "",
        "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Resumen", "summary_results": "Comparación",
        "summary_pros_cons": "Pros y Contras",
        "pros": "✅ Pros", "cons": "❌ Cons",
        "theory_title": "📚 Teoría",
        "theory_sections": {"et": "8.1 Trazas", "sarsal": "8.2 SARSA(λ)", "ql": "8.3 Q(λ)", "lambda": "8.4 Elección λ"},
        "theory_et": "e_t(s,a) = γλ·e_{t-1}(s,a) + 𝟙[s=S_t,a=A_t]",
        "theory_sarsal": "Q(s,a) ← Q(s,a) + α δ_t e_t(s,a) para todos (s,a)",
        "theory_ql": "Corte Watkins: trazas=0 si acción no-greedy.",
        "theory_lambda": "λ=0→TD(0) | λ=0.7→óptimo | λ=0.99→≈MC",
        "algo_labels": {"sarsa_lambda": "SARSA(λ)", "q_lambda": "Q(λ) Watkins",
                        "sarsa_td0": "SARSA λ=0", "sarsa_mc": "SARSA λ=0.99"},
        "pros_list": {"sarsa_lambda": ["On-policy"], "q_lambda": ["Off-policy"],
                      "sarsa_td0": ["Simple"], "sarsa_mc": ["Bajo sesgo"]},
        "cons_list": {"sarsa_lambda": ["λ a ajustar"], "q_lambda": ["Corte trazas"],
                      "sarsa_td0": ["Lento"], "sarsa_mc": ["Alta varianza"]},
    },
}

COLORS = {
    "sarsa_lambda": "#8B5CF6",
    "q_lambda":     "#0082F0",
    "sarsa_td0":    "#FF8C0A",
    "sarsa_mc":     "#0FC373",
}
ALGOS = ["sarsa_lambda", "q_lambda", "sarsa_td0", "sarsa_mc"]


def _moving_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def render():
    lang = st.session_state.get("lang", "EN")
    tx   = T[lang]

    st.title(tx["title"])
    st.caption(tx["subtitle"])

    try:
        import rlvr_py
    except ImportError:
        st.error(tx["engine_missing"])
        return

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.header(tx["sidebar_title"])
    n_episodes    = st.sidebar.slider(tx["n_episodes"],    50, 3000, 500, 50)
    gamma         = st.sidebar.slider(tx["gamma"],         0.5, 0.999, 0.95, 0.005)
    alpha         = st.sidebar.slider(tx["alpha"],         0.01, 1.0, 0.1, 0.01)
    epsilon       = st.sidebar.slider(tx["epsilon"],       0.0, 1.0, 0.3, 0.05)
    epsilon_decay = st.sidebar.slider(tx["epsilon_decay"], 0.0, 0.1, 0.01, 0.001, format="%.3f")
    lambda_       = st.sidebar.slider(tx["lambda_"],       0.0, 1.0, 0.7, 0.05)
    replacing     = st.sidebar.checkbox(tx["replacing"], value=True)
    seed          = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    with st.expander(tx["guide_title"], expanded=False):
        st.markdown(tx["guide"])

    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running Rust eligibility-trace engine..."):
            result = rlvr_py.run_ch08_eligibility(
                int(seed), int(n_episodes), float(gamma), float(alpha),
                float(epsilon), float(epsilon_decay),
                float(lambda_), bool(replacing),
            )
        st.session_state["ch08_result"] = result

    if "ch08_result" not in st.session_state:
        st.info("Configure settings and click **▶ Run All Four Algorithms**.")
        _render_theory(tx)
        return

    result       = st.session_state["ch08_result"]
    state_names  = result["state_names"]
    action_names = result["action_names"]

    # ── KPI row ──────────────────────────────────────────────────────────
    cols = st.columns(4)
    for i, key in enumerate(ALGOS):
        r   = result[key]
        avg = sum(r["returns_curve"][-50:]) / min(50, len(r["returns_curve"]))
        avg_traces = sum(r["trace_stats"][-50:]) / max(1, min(50, len(r["trace_stats"])))
        cols[i].metric(tx["algo_labels"][key], f"Avg: {avg:.2f}", f"Traces: {avg_traces:.1f}")

    # ── Returns ──────────────────────────────────────────────────────────
    st.subheader(tx["returns_title"])
    fig = go.Figure()
    for key in ALGOS:
        ma = _moving_avg(result[key]["returns_curve"], 30)
        fig.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
            mode="lines", name=tx["algo_labels"][key],
            line=dict(color=COLORS[key], width=2)))
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=20, b=40),
                      xaxis_title="Episode", yaxis_title="Return (MA-30)",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(tx["returns_caption"])

    # ── TD Error ─────────────────────────────────────────────────────────
    st.subheader(tx["td_error_title"])
    fig2 = go.Figure()
    for key in ALGOS:
        ma = _moving_avg(result[key]["td_error_curve"], 30)
        fig2.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
            mode="lines", name=tx["algo_labels"][key],
            line=dict(color=COLORS[key], width=2)))
    fig2.update_layout(height=260, margin=dict(l=40, r=20, t=20, b=40),
                       xaxis_title="Episode", yaxis_title="Avg TD Error",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(tx["td_error_caption"])

    # ── Value function + Active traces ───────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(tx["value_title"])
        short = [f"S{i}" for i in range(result["n_states"])]
        fig3  = go.Figure()
        for key in ALGOS:
            fig3.add_trace(go.Bar(x=short, y=result[key]["values"],
                name=tx["algo_labels"][key], marker_color=COLORS[key], opacity=0.8))
        fig3.update_layout(height=280, barmode="group",
                           margin=dict(l=40, r=20, t=20, b=40),
                           legend=dict(orientation="h"))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(tx["value_caption"])

    with col2:
        st.subheader(tx["trace_title"])
        fig4 = go.Figure()
        for key in ["sarsa_lambda", "q_lambda"]:
            ma = _moving_avg(result[key]["trace_stats"], 30)
            fig4.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
                mode="lines", name=tx["algo_labels"][key],
                line=dict(color=COLORS[key], width=2)))
        fig4.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=40),
                           xaxis_title="Episode", yaxis_title="Max active traces",
                           legend=dict(orientation="h"))
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(tx["trace_caption"])

    # ── Q-Table heatmap ──────────────────────────────────────────────────
    st.subheader(tx["qtable_title"])
    algo_sel = st.selectbox("Algorithm", [tx["algo_labels"][k] for k in ALGOS])
    key_map  = {tx["algo_labels"][k]: k for k in ALGOS}
    key_sel  = key_map.get(algo_sel, "sarsa_lambda")
    qt       = result[key_sel]["q_table"]
    action_short = [f"A{i}" for i in range(result["n_actions"])]
    fig5 = go.Figure(go.Heatmap(
        z=qt, x=action_short, y=short, colorscale="Purples",
        text=[[f"{qt[s][a]:.2f}" for a in range(result["n_actions"])]
              for s in range(result["n_states"])],
        texttemplate="%{text}",
    ))
    fig5.update_layout(height=300, margin=dict(l=60, r=20, t=20, b=40))
    st.plotly_chart(fig5, use_container_width=True)
    st.caption(tx["qtable_caption"])

    # ── Glass-Box ────────────────────────────────────────────────────────
    st.subheader(tx["glass_title"])
    _render_glass_box(result, tx)

    # ── Summary ──────────────────────────────────────────────────────────
    st.subheader(tx["summary_title"])
    _render_summary(result, tx)

    _render_theory(tx)


def _render_glass_box(result, tx):
    algo_options = {tx["algo_labels"][k]: k for k in ALGOS}
    selected     = st.selectbox("Algorithm", list(algo_options.keys()), key="gb8_algo")
    key          = algo_options[selected]
    r            = result[key]
    ep_idx = st.slider("Episode", 0, max(len(r["returns_curve"]) - 1, 0),
                       max(len(r["returns_curve"]) - 1, 0), key="gb8_ep")
    col1, col2, col3 = st.columns(3)
    col1.metric("Episode return",    f"{r['returns_curve'][ep_idx]:.3f}")
    col2.metric("Avg TD error",      f"{r['td_error_curve'][ep_idx]:.4f}")
    col3.metric("Max active traces", f"{r['trace_stats'][ep_idx]:.0f}")

    if "sarsa" in key:
        st.latex(r"\delta_t = R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)")
        st.latex(r"e_t(s,a) = \gamma\lambda \cdot e_{t-1}(s,a) + \mathbf{1}[s=S_t,a=A_t]")
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t e_t(s,a) \quad \forall (s,a)")
    else:
        st.latex(r"\delta_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)")
        st.markdown("**Watkins' cut:** traces reset to 0 when non-greedy action taken.")
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t e_t(s,a) \quad \forall (s,a)")


def _render_summary(result, tx):
    st.markdown(f"#### {tx['summary_results']}")
    rows = []
    for key in ALGOS:
        r   = result[key]
        avg = sum(r["returns_curve"][-100:]) / min(100, len(r["returns_curve"]))
        avg_traces = sum(r["trace_stats"]) / max(1, len(r["trace_stats"]))
        rows.append({
            "Algorithm":             tx["algo_labels"][key],
            "Avg return (last 100)": f"{avg:.3f}",
            "Total steps":           str(r["total_steps"]),
            "Avg traces/ep":         f"{avg_traces:.1f}",
            "V*(S0)":                f"{r['values'][0]:.3f}",
            "V*(S7)":                f"{r['values'][7]:.3f}",
        })
    st.dataframe(rows, hide_index=True)

    st.markdown(f"#### {tx['summary_pros_cons']}")
    for key in ALGOS:
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
    for key in ["et", "sarsal", "ql", "lambda"]:
        with st.expander(tx["theory_sections"][key], expanded=False):
            st.markdown(tx[f"theory_{key}"])
