import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
T = {
    "EN": {
        "title": "Chapter 03 — Multi-Armed Bandit & Exploration Strategies",
        "subtitle": "ASP Skill-Slot Optimisation · Warsaw Region",
        "engine_missing": "⚙️ Rust engine not found. Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Bandit Settings",
        "n_steps": "Steps (pulls)",
        "epsilon": "ε — Initial exploration rate",
        "epsilon_decay": "ε decay rate α",
        "ucb_c": "c — UCB exploration constant",
        "seed": "Random seed",
        "run_btn": "▶ Run All Three Algorithms",
        "regret_title": "📈 Cumulative Regret — All Three Algorithms",
        "regret_x": "Step",
        "regret_y": "Cumulative Regret",
        "regret_caption": "Lower is better. Regret = gap between optimal arm and chosen arm, accumulated over time.",
        "pulls_title": "🎰 Arm Pull Distribution",
        "pulls_caption": "How many times each skill slot was selected. True best arm = Plumbing (88% SLA).",
        "qval_title": "📊 Q-value Convergence vs True SLA Rates",
        "qval_caption": "Dashed lines = true SLA rates (unknown to agent). Solid bars = learned Q(a) estimates.",
        "reward_title": "💰 Cumulative Reward — All Three Algorithms",
        "reward_x": "Step",
        "reward_y": "Cumulative Reward",
        "glass_title": "🔬 Glass-Box Inspector — Step Trace",
        "glass_algo": "Algorithm",
        "glass_step_slider": "🔍 Highlight step",
        "glass_headers": ["Step", "Arm", "Skill", "Reward", "Regret", "Cum. Regret", "ε", "Mode"],
        "summary_title": "📊 Episode Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "Bandit Algorithms — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "algo_labels": {
            "epsilon_greedy": "ε-Greedy",
            "ucb": "UCB1",
            "thompson": "Thompson Sampling",
        },
        "true_rates_label": "True SLA rates (hidden from agent)",
        "best_arm_label": "Best arm identified",
        "total_regret_label": "Total regret",
        "total_reward_label": "Total reward",
        "theory_egreedy": r"""
**Epsilon-Greedy with Annealing** — the baseline bandit algorithm.

$$a_t = \begin{cases} \text{random arm} & \text{with probability } \varepsilon_t \\ \arg\max_a Q(a) & \text{with probability } 1-\varepsilon_t \end{cases}$$

**Annealing schedule:**
$$\varepsilon_t = \max\left(\varepsilon_{\min},\ \frac{\varepsilon_0}{1 + \alpha t}\right)$$

- Starts with high exploration, gradually shifts to exploitation
- Simple but wastes pulls on clearly inferior arms
- Q-value update (incremental mean):

$$Q(a) \leftarrow Q(a) + \frac{1}{N(a)} \left[ R - Q(a) \right]$$

Implemented in `epsilon_greedy_select()` in `ch03_bandit.rs`.
""",
        "theory_ucb": r"""
**UCB1 (Upper Confidence Bound)** — optimism in the face of uncertainty.

$$\text{UCB}(a) = Q(a) + c \sqrt{\frac{\ln t}{N(a)}}$$

- $Q(a)$ = estimated value (exploitation term)
- $c\sqrt{\ln t / N(a)}$ = uncertainty bonus (exploration term)
- Arms pulled rarely have high uncertainty → get explored more
- As $N(a)$ grows, uncertainty shrinks → exploitation dominates

**Regret bound:** $O(\sqrt{KT \ln T})$ — sublinear, guaranteed to converge.

Implemented in `ucb_select()` in `ch03_bandit.rs`.
""",
        "theory_thompson": r"""
**Thompson Sampling** — Bayesian exploration via posterior sampling.

For Bernoulli rewards, maintain a Beta posterior for each arm:
$$\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$$

- Prior: $\text{Beta}(1, 1)$ = uniform (no prior knowledge)
- After each pull: if reward=1 → $\alpha_a += 1$, if reward=0 → $\beta_a += 1$
- At each step: sample $\theta_a$ from each posterior, pick $\arg\max_a \theta_a$

**Why it works:** arms with uncertain posteriors have high variance → get sampled more.
As evidence accumulates, the best arm's posterior concentrates near its true rate.

Implemented in `thompson_select()` in `ch03_bandit.rs`.
""",
        "theory_regret": r"""
**Regret Analysis** — how fast does each algorithm learn?

| Algorithm | Regret bound | Type |
|---|---|---|
| Random | $O(T)$ | Linear — never learns |
| Epsilon-greedy (fixed) | $O(T^{2/3})$ | Sublinear |
| Epsilon-greedy (annealing) | $O(\sqrt{T \ln T})$ | Near-optimal |
| UCB1 | $O(\sqrt{KT \ln T})$ | Near-optimal, deterministic |
| Thompson Sampling | $O(\sqrt{KT \ln T})$ | Near-optimal, Bayesian |

**Key insight:** UCB and Thompson Sampling achieve near-optimal regret bounds.
In practice, Thompson Sampling often outperforms UCB on Bernoulli bandits.

The Glass-Box shows per-step regret so you can see exactly when each algorithm
makes a suboptimal choice and how quickly it recovers.
""",
        "pros_list": {
            "epsilon_greedy": [
                "Simple — one parameter ε controls everything",
                "Works with any reward distribution",
                "Annealing schedule improves long-term performance",
                "Good baseline for comparison",
            ],
            "ucb": [
                "Deterministic — same inputs always give same outputs",
                "Provable regret bound O(sqrt(KT ln T))",
                "No prior knowledge needed",
                "Naturally balances exploration and exploitation",
            ],
            "thompson": [
                "Best empirical performance on Bernoulli bandits",
                "Naturally incorporates uncertainty via Beta posterior",
                "Scales well to many arms",
                "Bayesian — prior knowledge can be incorporated",
            ],
        },
        "cons_list": {
            "epsilon_greedy": [
                "Explores uniformly — wastes pulls on bad arms",
                "Decay rate α must be tuned manually",
                "No uncertainty quantification",
                "Suboptimal regret bound vs UCB/Thompson",
            ],
            "ucb": [
                "Exploration constant c must be tuned",
                "Can over-explore in early steps",
                "Assumes stationary reward distributions",
                "Less natural for Bayesian settings",
            ],
            "thompson": [
                "Requires conjugate prior (Beta for Bernoulli)",
                "Stochastic — results vary between runs",
                "More complex to implement than ε-greedy",
                "Prior choice affects early performance",
            ],
        },
    },
        "DE": {
        "title": "Kapitel 03 — Mehrarmiger Bandit & Explorationsstrategien",
        "subtitle": "ASP Qualifikationsslot-Optimierung — Region Warschau",
        "engine_missing": "⚠ Rust-Engine nicht gefunden. Ausführen: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Bandit-Einstellungen",
        "n_steps": "Schritte (Ziehungen)",
        "epsilon": "ε — Anfängliche Explorationsrate",
        "epsilon_decay": "ε-Abklingrate",
        "ucb_c": "c — UCB-Explorationskonstante",
        "seed": "Zufallsseed",
        "run_btn": "▶ Alle drei Algorithmen starten",
        "regret_title": "📉 Kumulatives Bedauern — Alle drei Algorithmen",
        "regret_x": "Schritt",
        "regret_y": "Kumulatives Bedauern",
        "regret_caption": "Niedriger = besser. Bedauern = Lücke zwischen optimalem und gewähltem Arm.",
        "pulls_title": "📊 Armziehungsverteilung",
        "pulls_caption": "Wie oft jeder Qualifikationsslot ausgewählt wurde. Bester Arm = Sanitär (88% SLA).",
        "qval_title": "📊 Q-Wert-Konvergenz vs. echte SLA-Raten",
        "qval_caption": "Gestrichelte Linien = echte SLA-Raten. Balken = gelernte Q(a)-Schätzungen.",
        "reward_title": "📈 Kumulative Belohnung — Alle drei Algorithmen",
        "reward_x": "Schritt",
        "reward_y": "Kumulative Belohnung",
        "glass_title": "🔍 Glass-Box-Inspektor — Schrittprotokoll",
        "glass_algo": "Algorithmus",
        "glass_step_slider": "🔍 Schritt hervorheben",
        "glass_headers": ["Schritt", "Arm", "Qualifikation", "Belohnung", "Bedauern", "Kum. Bedauern", "ε", "Modus"],
        "summary_title": "📋 Zusammenfassung",
        "summary_results": "Algorithmenvergleich",
        "summary_pros_cons": "Bandit-Algorithmen — Vor- & Nachteile",
        "pros": "✅ Vorteile",
        "cons": "❌ Nachteile",
        "algo_labels": {
            "epsilon_greedy": "ε-Greedy",
            "ucb": "UCB1",
            "thompson": "Thompson-Sampling",
        },
        "true_rates_label": "Echte SLA-Raten (dem Agenten unbekannt)",
        "best_arm_label": "Identifizierter bester Arm",
        "total_regret_label": "Gesamtbedauern",
        "total_reward_label": "Gesamtbelohnung",
        "theory_egreedy": r"""**Epsilon-Greedy mit Abkühlung**:
$$\varepsilon_t = \max\left(\varepsilon_{\min},\ \frac{\varepsilon_0}{1 + \alpha t}\right)$$
$$Q(a) \leftarrow Q(a) + \frac{1}{N(a)} \left[ R - Q(a) \right]$$""",
        "theory_ucb": r"""**UCB1** — Optimismus angesichts von Unsicherheit:
$$\text{UCB}(a) = Q(a) + c \sqrt{\frac{\ln t}{N(a)}}$$
Bedauernschranke: $O(\sqrt{KT \ln T})$""",
        "theory_thompson": r"""**Thompson-Sampling**: $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$, wähle $\arg\max_a \theta_a$.""",
        "theory_regret": r"""**Bedauernanalyse**:
| Algorithmus | Bedauernschranke |
|---|---|
| Zufällig | $O(T)$ |
| ε-Greedy | $O(T^{2/3})$ |
| UCB1 | $O(\sqrt{KT \ln T})$ |
| Thompson | $O(\sqrt{KT \ln T})$ |""",
        "pros_list": {
            "epsilon_greedy": ["Einfach — ein Parameter ε", "Funktioniert mit jeder Verteilung", "Gute Basislinie"],
            "ucb": ["Deterministisch", "Beweisbare Bedauernschranke", "Kein Vorwissen nötig"],
            "thompson": ["Beste empirische Leistung", "Berücksichtigt Unsicherheit natürlich"],
        },
        "cons_list": {
            "epsilon_greedy": ["Erkundet gleichmäßig", "Abklingrate muss manuell eingestellt werden"],
            "ucb": ["Konstante c muss eingestellt werden", "Kann am Anfang über-erkunden"],
            "thompson": ["Benötigt konjugierten Prior", "Stochastisch"],
        },
    },
    "FR": {
        "title": "Chapitre 03 — Bandit Multi-Bras & Stratégies d'Exploration",
        "subtitle": "Optimisation des créneaux de compétences ASP · Région de Varsovie",
        "engine_missing": "⚙️ Moteur Rust introuvable. Exécutez : `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Paramètres Bandit",
        "n_steps": "Étapes (tirages)",
        "epsilon": "ε — Taux d'exploration initial",
        "epsilon_decay": "Taux de décroissance ε α",
        "ucb_c": "c — Constante d'exploration UCB",
        "seed": "Graine aléatoire",
        "run_btn": "▶ Lancer les trois algorithmes",
        "regret_title": "📈 Regret cumulatif — Trois algorithmes",
        "regret_x": "Étape",
        "regret_y": "Regret cumulatif",
        "regret_caption": "Plus bas = meilleur. Regret = écart entre le bras optimal et le bras choisi.",
        "pulls_title": "🎰 Distribution des tirages",
        "pulls_caption": "Nombre de fois que chaque créneau a été sélectionné. Meilleur bras = Plomberie (88% SLA).",
        "qval_title": "📊 Convergence des valeurs Q vs vrais taux SLA",
        "qval_caption": "Lignes pointillées = vrais taux SLA. Barres = estimations Q(a) apprises.",
        "reward_title": "💰 Récompense cumulée — Trois algorithmes",
        "reward_x": "Étape",
        "reward_y": "Récompense cumulée",
        "glass_title": "🔬 Inspecteur Glass-Box — Trace des étapes",
        "glass_algo": "Algorithme",
        "glass_step_slider": "🔍 Mettre en évidence l'étape",
        "glass_headers": ["Étape", "Bras", "Compétence", "Récompense", "Regret", "Regret cum.", "ε", "Mode"],
        "summary_title": "📊 Résumé",
        "summary_results": "Comparaison des algorithmes",
        "summary_pros_cons": "Algorithmes Bandit — Avantages & Inconvénients",
        "pros": "✅ Avantages",
        "cons": "❌ Inconvénients",
        "algo_labels": {
            "epsilon_greedy": "ε-Greedy",
            "ucb": "UCB1",
            "thompson": "Thompson Sampling",
        },
        "true_rates_label": "Vrais taux SLA (cachés de l'agent)",
        "best_arm_label": "Meilleur bras identifié",
        "total_regret_label": "Regret total",
        "total_reward_label": "Récompense totale",
        "theory_egreedy": r"""
**Epsilon-Greedy avec recuit** :
$$\varepsilon_t = \max\left(\varepsilon_{\min},\ \frac{\varepsilon_0}{1 + \alpha t}\right)$$
$$Q(a) \leftarrow Q(a) + \frac{1}{N(a)} \left[ R - Q(a) \right]$$
""",
        "theory_ucb": r"""
**UCB1** :
$$\text{UCB}(a) = Q(a) + c \sqrt{\frac{\ln t}{N(a)}}$$
Borne de regret : $O(\sqrt{KT \ln T})$
""",
        "theory_thompson": r"""
**Thompson Sampling** : $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$, sélectionner $\arg\max_a \theta_a$.
""",
        "theory_regret": r"""
**Analyse du regret** :

| Algorithme | Borne de regret |
|---|---|
| Aléatoire | $O(T)$ |
| ε-Greedy (fixe) | $O(T^{2/3})$ |
| UCB1 | $O(\sqrt{KT \ln T})$ |
| Thompson | $O(\sqrt{KT \ln T})$ |
""",
        "pros_list": {
            "epsilon_greedy": ["Simple", "Fonctionne avec toute distribution", "Bon point de référence"],
            "ucb": ["Déterministe", "Borne de regret prouvable", "Pas de connaissance préalable"],
            "thompson": ["Meilleures performances empiriques", "Incorpore l'incertitude naturellement"],
        },
        "cons_list": {
            "epsilon_greedy": ["Explore uniformément", "Taux de décroissance à régler manuellement"],
            "ucb": ["Constante c à régler", "Peut sur-explorer au début"],
            "thompson": ["Nécessite un prior conjugué", "Stochastique"],
        },
    },
    "ES": {
        "title": "Capítulo 03 — Bandido Multi-Brazo & Estrategias de Exploración",
        "subtitle": "Optimización de ranuras de habilidades ASP · Región de Varsovia",
        "engine_missing": "⚙️ Motor Rust no encontrado. Ejecute: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Configuración Bandido",
        "n_steps": "Pasos (tiradas)",
        "epsilon": "ε — Tasa de exploración inicial",
        "epsilon_decay": "Tasa de decaimiento ε α",
        "ucb_c": "c — Constante de exploración UCB",
        "seed": "Semilla aleatoria",
        "run_btn": "▶ Ejecutar los tres algoritmos",
        "regret_title": "📈 Arrepentimiento acumulado — Tres algoritmos",
        "regret_x": "Paso",
        "regret_y": "Arrepentimiento acumulado",
        "regret_caption": "Más bajo = mejor. Arrepentimiento = brecha entre el brazo óptimo y el elegido.",
        "pulls_title": "🎰 Distribución de tiradas",
        "pulls_caption": "Cuántas veces se seleccionó cada ranura. Mejor brazo = Fontanería (88% SLA).",
        "qval_title": "📊 Convergencia de valores Q vs tasas SLA reales",
        "qval_caption": "Líneas punteadas = tasas SLA reales. Barras = estimaciones Q(a) aprendidas.",
        "reward_title": "💰 Recompensa acumulada — Tres algoritmos",
        "reward_x": "Paso",
        "reward_y": "Recompensa acumulada",
        "glass_title": "🔬 Inspector Glass-Box — Traza de pasos",
        "glass_algo": "Algoritmo",
        "glass_step_slider": "🔍 Resaltar paso",
        "glass_headers": ["Paso", "Brazo", "Habilidad", "Recompensa", "Arrepent.", "Arrepent. acum.", "ε", "Modo"],
        "summary_title": "📊 Resumen",
        "summary_results": "Comparación de algoritmos",
        "summary_pros_cons": "Algoritmos Bandido — Pros y Contras",
        "pros": "✅ Pros",
        "cons": "❌ Contras",
        "algo_labels": {
            "epsilon_greedy": "ε-Greedy",
            "ucb": "UCB1",
            "thompson": "Thompson Sampling",
        },
        "true_rates_label": "Tasas SLA reales (ocultas al agente)",
        "best_arm_label": "Mejor brazo identificado",
        "total_regret_label": "Arrepentimiento total",
        "total_reward_label": "Recompensa total",
        "theory_egreedy": r"""
**Epsilon-Greedy con recocido** :
$$\varepsilon_t = \max\left(\varepsilon_{\min},\ \frac{\varepsilon_0}{1 + \alpha t}\right)$$
""",
        "theory_ucb": r"""
**UCB1** :
$$\text{UCB}(a) = Q(a) + c \sqrt{\frac{\ln t}{N(a)}}$$
""",
        "theory_thompson": r"""
**Muestreo de Thompson** : $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$, seleccionar $\arg\max_a \theta_a$.
""",
        "theory_regret": r"""
**Análisis del arrepentimiento** :

| Algoritmo | Cota de arrepentimiento |
|---|---|
| Aleatorio | $O(T)$ |
| ε-Greedy | $O(T^{2/3})$ |
| UCB1 | $O(\sqrt{KT \ln T})$ |
| Thompson | $O(\sqrt{KT \ln T})$ |
""",
        "pros_list": {
            "epsilon_greedy": ["Simple", "Funciona con cualquier distribución", "Buena línea base"],
            "ucb": ["Determinista", "Cota de arrepentimiento demostrable"],
            "thompson": ["Mejor rendimiento empírico", "Incorpora incertidumbre naturalmente"],
        },
        "cons_list": {
            "epsilon_greedy": ["Explora uniformemente", "Tasa de decaimiento a ajustar manualmente"],
            "ucb": ["Constante c a ajustar", "Puede sobre-explorar al inicio"],
            "thompson": ["Requiere prior conjugado", "Estocástico"],
        },
    },
    "PL": {
        "title": "Rozdział 03 — Wieloręki Bandyta & Strategie Eksploracji",
        "subtitle": "Optymalizacja slotów umiejętności ASP · Region Warszawy",
        "engine_missing": "⚙️ Silnik Rust nie znaleziony. Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Ustawienia Bandyty",
        "n_steps": "Kroki (losowania)",
        "epsilon": "ε — Początkowy współczynnik eksploracji",
        "epsilon_decay": "Współczynnik zaniku ε α",
        "ucb_c": "c — Stała eksploracji UCB",
        "seed": "Ziarno losowości",
        "run_btn": "▶ Uruchom wszystkie trzy algorytmy",
        "regret_title": "📈 Skumulowany żal — Trzy algorytmy",
        "regret_x": "Krok",
        "regret_y": "Skumulowany żal",
        "regret_caption": "Niżej = lepiej. Żal = różnica między optymalnym ramieniem a wybranym.",
        "pulls_title": "🎰 Rozkład losowań",
        "pulls_caption": "Ile razy wybrano każdy slot umiejętności. Najlepsze ramię = Hydraulika (88% SLA).",
        "qval_title": "📊 Zbieżność wartości Q vs prawdziwe wskaźniki SLA",
        "qval_caption": "Linie przerywane = prawdziwe wskaźniki SLA. Słupki = wyuczone estymaty Q(a).",
        "reward_title": "💰 Skumulowana nagroda — Trzy algorytmy",
        "reward_x": "Krok",
        "reward_y": "Skumulowana nagroda",
        "glass_title": "🔬 Inspektor Glass-Box — Ślad kroków",
        "glass_algo": "Algorytm",
        "glass_step_slider": "🔍 Podświetl krok",
        "glass_headers": ["Krok", "Ramię", "Umiejętność", "Nagroda", "Żal", "Żal skum.", "ε", "Tryb"],
        "summary_title": "📊 Podsumowanie",
        "summary_results": "Porównanie algorytmów",
        "summary_pros_cons": "Algorytmy Bandyty — Zalety i Wady",
        "pros": "✅ Zalety",
        "cons": "❌ Wady",
        "algo_labels": {
            "epsilon_greedy": "ε-Zachłanny",
            "ucb": "UCB1",
            "thompson": "Próbkowanie Thompsona",
        },
        "true_rates_label": "Prawdziwe wskaźniki SLA (ukryte przed agentem)",
        "best_arm_label": "Zidentyfikowane najlepsze ramię",
        "total_regret_label": "Całkowity żal",
        "total_reward_label": "Całkowita nagroda",
        "theory_egreedy": r"""
**Epsilon-zachłanny z wygaszaniem** :
$$\varepsilon_t = \max\left(\varepsilon_{\min},\ \frac{\varepsilon_0}{1 + \alpha t}\right)$$
$$Q(a) \leftarrow Q(a) + \frac{1}{N(a)} \left[ R - Q(a) \right]$$
""",
        "theory_ucb": r"""
**UCB1** :
$$\text{UCB}(a) = Q(a) + c \sqrt{\frac{\ln t}{N(a)}}$$
Granica żalu: $O(\sqrt{KT \ln T})$
""",
        "theory_thompson": r"""
**Próbkowanie Thompsona** : $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$, wybierz $\arg\max_a \theta_a$.
""",
        "theory_regret": r"""
**Analiza żalu** :

| Algorytm | Granica żalu |
|---|---|
| Losowy | $O(T)$ |
| ε-Zachłanny | $O(T^{2/3})$ |
| UCB1 | $O(\sqrt{KT \ln T})$ |
| Thompson | $O(\sqrt{KT \ln T})$ |
""",
        "pros_list": {
            "epsilon_greedy": ["Prosty", "Działa z dowolnym rozkładem", "Dobry punkt odniesienia"],
            "ucb": ["Deterministyczny", "Udowodniona granica żalu"],
            "thompson": ["Najlepsza wydajność empiryczna", "Naturalnie uwzględnia niepewność"],
        },
        "cons_list": {
            "epsilon_greedy": ["Eksploruje równomiernie", "Współczynnik zaniku wymaga ręcznego strojenia"],
            "ucb": ["Stała c wymaga strojenia", "Może nadmiernie eksplorować na początku"],
            "thompson": ["Wymaga sprzężonego prioru", "Stochastyczny"],
        },
    },
}

COLORS = {
    "epsilon_greedy": "#3498db",
    "ucb":            "#e67e22",
    "thompson":       "#2ecc71",
}

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
    n_steps       = st.sidebar.slider(tx["n_steps"],       50, 2000, 500, 50)
    epsilon       = st.sidebar.slider(tx["epsilon"],       0.0, 1.0, 0.3, 0.05)
    epsilon_decay = st.sidebar.slider(tx["epsilon_decay"], 0.0, 0.1, 0.01, 0.001,
                                      format="%.3f")
    ucb_c         = st.sidebar.slider(tx["ucb_c"],         0.1, 5.0, 2.0, 0.1)
    seed          = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    run = st.button(tx["run_btn"], type="primary")

    if run:
        with st.spinner("Running Rust bandit engine..."):
            raw = rlvr_py.run_ch03_bandits(
                int(seed), int(n_steps),
                float(epsilon), float(epsilon_decay), float(ucb_c)
            )
        st.session_state["ch03_raw"] = raw

    if "ch03_raw" not in st.session_state:
        st.info("Configure settings and click **▶ Run All Three Algorithms**.")
        return

    raw       = st.session_state["ch03_raw"]
    results   = raw["results"]
    arm_names = raw["arm_names"]
    true_rates = raw["true_rates"]

    # KPI row
    cols = st.columns(3)
    for i, res in enumerate(results):
        algo = res["algorithm"]
        label = tx["algo_labels"][algo]
        cols[i].metric(
            label,
            f"Regret: {res['total_regret']:.1f}",
            f"Reward: {res['total_reward']:.0f}",
        )

    # Cumulative regret
    st.subheader(tx["regret_title"])
    _render_regret(results, tx, arm_names)
    st.caption(tx["regret_caption"])

    # Cumulative reward
    st.subheader(tx["reward_title"])
    _render_reward(results, tx)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(tx["pulls_title"])
        _render_pulls(results, arm_names, true_rates, tx)
        st.caption(tx["pulls_caption"])
    with col2:
        st.subheader(tx["qval_title"])
        _render_qvalues(results, arm_names, true_rates, tx)
        st.caption(tx["qval_caption"])

    # Glass-Box
    st.subheader(tx["glass_title"])
    _render_glass_box(results, arm_names, tx)

    # Summary
    st.subheader(tx["summary_title"])
    _render_summary(results, arm_names, true_rates, tx)

    # Theory


# ---------------------------------------------------------------------------
# Cumulative regret chart
# ---------------------------------------------------------------------------
def _render_regret(results, tx, arm_names):
    fig = go.Figure()
    for res in results:
        algo  = res["algorithm"]
        label = tx["algo_labels"][algo]
        steps = [s["step"] for s in res["steps"]]
        regret = [s["cumulative_regret"] for s in res["steps"]]
        fig.add_trace(go.Scatter(
            x=steps, y=regret,
            mode="lines",
            name=label,
            line=dict(color=COLORS[algo], width=2),
        ))
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title=tx["regret_x"],
        yaxis_title=tx["regret_y"],
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Cumulative reward chart
# ---------------------------------------------------------------------------
def _render_reward(results, tx):
    fig = go.Figure()
    for res in results:
        algo  = res["algorithm"]
        label = tx["algo_labels"][algo]
        steps = [s["step"] for s in res["steps"]]
        cum_r = []
        total = 0.0
        for s in res["steps"]:
            total += s["reward"]
            cum_r.append(total)
        fig.add_trace(go.Scatter(
            x=steps, y=cum_r,
            mode="lines",
            name=label,
            line=dict(color=COLORS[algo], width=2),
        ))
    fig.update_layout(
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title=tx["reward_x"],
        yaxis_title=tx["reward_y"],
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Arm pull distribution
# ---------------------------------------------------------------------------
def _render_pulls(results, arm_names, true_rates, tx):
    fig = go.Figure()
    for res in results:
        algo  = res["algorithm"]
        label = tx["algo_labels"][algo]
        fig.add_trace(go.Bar(
            x=arm_names,
            y=res["final_n_pulls"],
            name=label,
            marker_color=COLORS[algo],
            opacity=0.8,
        ))
    fig.update_layout(
        height=280,
        barmode="group",
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Q-value convergence
# ---------------------------------------------------------------------------
def _render_qvalues(results, arm_names, true_rates, tx):
    fig = go.Figure()
    # True rates as dashed lines
    for i, (name, rate) in enumerate(zip(arm_names, true_rates)):
        fig.add_hline(
            y=rate,
            line_dash="dash",
            line_color="grey",
            opacity=0.5,
            annotation_text=f"{name} {rate:.0%}",
            annotation_position="right",
        )
    # Final Q-values as bars
    for res in results:
        algo  = res["algorithm"]
        label = tx["algo_labels"][algo]
        fig.add_trace(go.Bar(
            x=arm_names,
            y=res["final_q_values"],
            name=label,
            marker_color=COLORS[algo],
            opacity=0.8,
        ))
    fig.update_layout(
        height=280,
        barmode="group",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Glass-Box
# ---------------------------------------------------------------------------
def _render_glass_box(results, arm_names, tx):
    algo_options = {tx["algo_labels"][r["algorithm"]]: r for r in results}
    selected_label = st.selectbox(tx["glass_algo"], list(algo_options.keys()))
    res   = algo_options[selected_label]
    steps = res["steps"]
    n     = len(steps)

    sel = st.slider(tx["glass_step_slider"], 0, max(n - 1, 0), 0)

    rows = []
    for s in steps:
        rows.append({
            tx["glass_headers"][0]: s["step"],
            tx["glass_headers"][1]: s["arm"],
            tx["glass_headers"][2]: arm_names[s["arm"]],
            tx["glass_headers"][3]: f"{s['reward']:.0f}",
            tx["glass_headers"][4]: f"{s['regret']:.3f}",
            tx["glass_headers"][5]: f"{s['cumulative_regret']:.2f}",
            tx["glass_headers"][6]: f"{s['epsilon']:.3f}",
            tx["glass_headers"][7]: "🔍 Explore" if s["explored"] else "🎯 Exploit",
        })

    st.dataframe(rows, width='stretch', height=280)

    # Selected step detail
    s = steps[sel]
    algo = res["algorithm"]
    st.markdown(f"""
**Step {sel} detail ({tx['algo_labels'][algo]}):**
- Arm: **{arm_names[s['arm']]}** (arm {s['arm']})
- Reward: **{s['reward']:.0f}** · Regret: **{s['regret']:.3f}**
- Cumulative regret: **{s['cumulative_regret']:.2f}**
- ε = {s['epsilon']:.3f} · Mode: {"🔍 Explore" if s['explored'] else "🎯 Exploit"}
""")

    if algo == "ucb" and any(v > 0 for v in s["ucb_values"]):
        st.markdown("**UCB values at this step:**")
        for i, (name, ucb) in enumerate(zip(arm_names, s["ucb_values"])):
            marker = " ← selected" if i == s["arm"] else ""
            st.markdown(f"- {name}: `{ucb:.4f}`{marker}")
        st.latex(r"\text{UCB}(a) = Q(a) + c\sqrt{\frac{\ln t}{N(a)}}")

    if algo == "thompson" and any(v > 0 for v in s["thompson_samples"]):
        st.markdown("**Thompson samples at this step:**")
        for i, (name, ts) in enumerate(zip(arm_names, s["thompson_samples"])):
            marker = " ← selected" if i == s["arm"] else ""
            st.markdown(f"- {name}: `{ts:.4f}`{marker}")
        st.latex(r"\theta_a \sim \text{Beta}(\alpha_a, \beta_a)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _render_summary(results, arm_names, true_rates, tx):
    st.markdown(f"#### {tx['summary_results']}")

    # Comparison table
    rows = []
    for res in results:
        algo = res["algorithm"]
        rows.append({
            "Algorithm": tx["algo_labels"][algo],
            tx["total_reward_label"]: f"{res['total_reward']:.0f}",
            tx["total_regret_label"]: f"{res['total_regret']:.2f}",
            tx["best_arm_label"]: f"{arm_names[res['best_arm']]} (arm {res['best_arm']})",
        })
    st.dataframe(rows, width='stretch', hide_index=True)

    # True rates reveal
    st.markdown(f"**{tx['true_rates_label']}:**")
    for name, rate in zip(arm_names, true_rates):
        marker = " ⭐ optimal" if rate == max(true_rates) else ""
        st.markdown(f"- {name}: **{rate:.0%}**{marker}")

    # Pros & cons per algorithm
    st.markdown(f"#### {tx['summary_pros_cons']}")
    for res in results:
        algo  = res["algorithm"]
        label = tx["algo_labels"][algo]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{label} — {tx['pros']}**")
            for p in tx["pros_list"][algo]:
                st.markdown(f"- {p}")
        with col2:
            st.markdown(f"**{label} — {tx['cons']}**")
            for c in tx["cons_list"][algo]:
                st.markdown(f"- {c}")
        st.markdown("---")


# ---------------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------------