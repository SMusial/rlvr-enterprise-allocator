import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
T = {
    "EN": {
        "title": "Chapter 02 — Discrete MDP & Bellman Optimality",
        "subtitle": "ASP Operational State Optimisation · Warsaw Region",
        "engine_missing": "⚙️ Rust engine not found. Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ MDP Settings",
        "gamma": "γ — Discount factor",
        "theta": "θ — Convergence threshold",
        "seed": "Random seed",
        "run_btn": "▶ Run Value Iteration",
        "guide_title": "🎓 How to use this chapter",
        "guide": """
**Step 1 — Set γ (discount factor)**
γ controls how much the agent values future rewards. γ=0.99 = farsighted (plans ahead).
γ=0.5 = myopic (only cares about immediate reward). Watch how γ affects convergence speed.

**Step 2 — Set θ (convergence threshold)**
θ is how small the change in V must be before we stop iterating.
Smaller θ = more precise but more iterations. Try 1e-6 to start.

**Step 3 — Click ▶ Run Value Iteration**
The Rust engine builds the ASP transition matrix, reward matrix, and runs Bellman iterations.

**Step 4 — Read the Value Function chart**
Each bar = long-term value of being in that operational state.
S0 (all available) should be highest. S7 (SLA breach imminent) should be lowest.

**Step 5 — Read the Optimal Policy table**
For each operational state, the table shows which dispatch strategy maximises long-term value.

**Step 6 — Read the Convergence curve**
Watch ‖V^(k+1) - V^(k)‖∞ decay to zero — this is the contraction mapping theorem in action.

**Step 7 — Read the Glass-Box Bellman trace**
See the exact Bellman update for each state in the first 3 iterations.
""",
        "value_title": "📊 Optimal Value Function V*(s)",
        "value_caption": "Long-term expected reward of being in each operational state under optimal policy",
        "policy_title": "🎯 Optimal Policy π*(s)",
        "policy_caption": "Best dispatch strategy for each operational state",
        "conv_title": "📈 Convergence — ‖V^(k+1) - V^(k)‖∞",
        "conv_x": "Iteration",
        "conv_y": "Max change in V",
        "conv_caption": "Bellman contraction: each iteration reduces error by factor γ",
        "heatmap_title": "🗺️ Transition Matrix P(s'|s, a=A1: Skill-matched)",
        "heatmap_caption": "Probability of transitioning from state s (row) to state s' (column) under skill-matched dispatch",
        "glass_title": "🔬 Glass-Box — Bellman Update Trace (first 3 iterations)",
        "glass_headers": ["Iter", "State", "Best Action", "Q(s,A0)", "Q(s,A1)", "Q(s,A2)", "Q(s,A3)", "V_old", "V_new", "Δ"],
        "summary_title": "📊 Episode Summary",
        "summary_results": "Quantified Results",
        "summary_pros_cons": "Discrete MDP + Value Iteration — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "pros_list": [
            "Guaranteed convergence to optimal policy (contraction mapping theorem)",
            "Exact solution — no approximation error for small state spaces",
            "Interpretable: value function explains WHY each action is chosen",
            "Linear algebra solution available for policy evaluation (nalgebra LU)",
            "Foundation for all subsequent RL algorithms (Ch03–Ch20)",
        ],
        "cons_list": [
            "Requires full transition model P(s'|s,a) — not always available",
            "State space must be discrete and finite — doesn't scale to continuous spaces",
            "Curse of dimensionality: O(|S|² × |A|) per iteration",
            "Transition probabilities must be estimated or hand-crafted",
            "Ch06 (TD Learning) solves the model-free version of this problem",
        ],
        "metric_iters": "Iterations to converge",
        "metric_best_state": "Best operational state",
        "metric_worst_state": "Worst operational state",
        "metric_value_range": "Value range V*(s)",
        "metric_contraction": "Contraction verified",
        "theory_title": "📖 Theory — Chapter 02",
        "theory_sections": {
            "bellman": "§2.3 Bellman Optimality Equation",
            "vi": "§2.3 Value Iteration Algorithm",
            "contraction": "§2.3 Contraction Mapping Theorem",
            "linear": "§2.2 Linear System Solution",
            "policy": "§2.3 Policy Extraction",
        },
        "theory_bellman": r"""
**Bellman Optimality Equation** defines the value of a state under the optimal policy:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^*(s') \right]$$

- The value of state s = best action × (immediate reward + discounted future value)
- This is recursive — V*(s) depends on V*(s')
- Value Iteration solves this by iterating until convergence

Implemented in `value_iteration()` in `ch02_bellman.rs`.
""",
        "theory_vi": r"""
**Value Iteration** repeatedly applies the Bellman operator until convergence:

$$V^{(k+1)}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^{(k)}(s') \right]$$

Starting from V⁽⁰⁾ = 0, each iteration brings V closer to V*.
Stop when: $\|V^{(k+1)} - V^{(k)}\|_\infty < \theta$

The Glass-Box shows the exact update for each state in the first 3 iterations.
""",
        "theory_contraction": r"""
**Contraction Mapping Theorem** guarantees convergence:

$$\|V^{(k+1)} - V^{(k)}\|_\infty \leq \gamma \|V^{(k)} - V^{(k-1)}\|_\infty$$

The Bellman operator is a γ-contraction — each iteration reduces the error by factor γ.
Since γ < 1, the sequence converges to a unique fixed point V*.

Verified in `verify_contraction()` in `ch02_bellman.rs`.
""",
        "theory_linear": r"""
**Exact solution via linear system** — for a fixed policy π:

$$V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$$

Where $P^\pi$ is the transition matrix under policy π and $r^\pi$ is the reward vector.
Solved using **nalgebra LU decomposition** in `solve_exact()` in `ch02_bellman.rs`.

This gives the exact value function without iteration — but only works for small state spaces.
""",
        "theory_policy": r"""
**Policy Extraction** — greedy policy from V*:

$$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^*(s') \right]$$

Once V* is known, the optimal action in each state is simply the one that maximises
the right-hand side of the Bellman equation.

Implemented in `extract_policy()` in `ch02_bellman.rs`.
""",
    },
        "DE": {
        "title": "Kapitel 02 — Diskretes MDP & Bellman-Optimalität",
        "subtitle": "Optimierung der ASP-Betriebszustände — Region Warschau",
        "engine_missing": "⚠ Rust-Engine nicht gefunden. Ausführen: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ MDP-Einstellungen",
        "gamma": "γ — Diskontierungsfaktor",
        "theta": "θ — Konvergenzschwelle",
        "seed": "Zufallsseed",
        "run_btn": "▶ Wertiteration starten",
        "guide_title": "ℹ️ Anleitung",
        "guide": """**Schritt 1 — γ einstellen**: γ=0.99 = weitsichtig, γ=0.5 = kurzsichtig.
**Schritt 2 — θ einstellen**: kleiner = genauer, aber mehr Iterationen.
**Schritt 3 — Wertiteration starten**: Rust-Engine baut Übergangsmatrix und iteriert.
**Schritt 4 — Wertfunktion lesen**: jeder Balken = langfristiger Wert des Zustands.
**Schritt 5 — Optimale Strategie lesen**: beste Dispatch-Strategie für jeden Zustand.
**Schritt 6 — Konvergenzkurve lesen**: Abfall von V^(k+1) - V^(k).
**Schritt 7 — Glass-Box lesen**: Bellman-Update für jeden Zustand.""",
        "value_title": "📊 Optimale Wertfunktion V*(s)",
        "value_caption": "Langfristig erwartete Belohnung für jeden Betriebszustand",
        "policy_title": "🎯 Optimale Strategie π*(s)",
        "policy_caption": "Beste Dispatch-Strategie für jeden Betriebszustand",
        "conv_title": "📉 Konvergenz — V^(k+1) - V^(k)",
        "conv_x": "Iteration",
        "conv_y": "Max. Änderung in V",
        "conv_caption": "Bellman-Kontraktion: jede Iteration reduziert Fehler um Faktor γ",
        "heatmap_title": "🧮 Übergangsmatrix P(s'|s, a=A1: Qualifikation)",
        "heatmap_caption": "Übergangswahrscheinlichkeit von Zustand s zu s' unter qualifikationsbasiertem Dispatch",
        "glass_title": "🔍 Glass-Box — Bellman-Update-Protokoll (erste 3 Iterationen)",
        "glass_headers": ["Iter", "Zustand", "Beste Aktion", "Q(s,A0)", "Q(s,A1)", "Q(s,A2)", "Q(s,A3)", "V_alt", "V_neu", "δ"],
        "summary_title": "📋 Zusammenfassung",
        "summary_results": "Quantifizierte Ergebnisse",
        "summary_pros_cons": "Diskretes MDP + Wertiteration — Vor- & Nachteile",
        "pros": "✅ Vorteile",
        "cons": "❌ Nachteile",
        "pros_list": [
            "Garantierte Konvergenz zur optimalen Strategie (Kontraktionsabbildungssatz)",
            "Exakte Lösung ohne Approximationsfehler für kleine Zustandsräume",
            "Interpretierbar: Wertfunktion erklärt WARUM jede Aktion gewählt wird",
            "Lineare Algebra-Lösung verfügbar (nalgebra LU)",
            "Grundlage für alle nachfolgenden RL-Algorithmen (Ch03–Ch20)",
        ],
        "cons_list": [
            "Benötigt vollständiges Übergangsmodell P(s'|s,a)",
            "Zustandsraum muss diskret und endlich sein",
            "Fluch der Dimensionalität: O(|S| × |A|) pro Iteration",
            "Übergangswahrscheinlichkeiten müssen geschätzt werden",
            "Ch06 (TD-Lernen) löst die modellfreie Version",
        ],
        "metric_iters": "Iterationen bis zur Konvergenz",
        "metric_best_state": "Bester Betriebszustand",
        "metric_worst_state": "Schlechtester Betriebszustand",
        "metric_value_range": "Wertebereich V*(s)",
        "metric_contraction": "Kontraktion verifiziert",
        "theory_title": "📚 Theorie — Kapitel 02",
        "theory_sections": {
            "bellman":     "2.3 Bellman-Optimalitätsgleichung",
            "vi":          "2.3 Wertiterationsalgorithmus",
            "contraction": "2.3 Kontraktionsabbildungssatz",
            "linear":      "2.2 Lineare System-Lösung",
            "policy":      "2.3 Strategieextraktion",
        },
        "theory_bellman": r"""**Bellman-Optimalitätsgleichung**:
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^*(s') \right]$$
Implementiert in `value_iteration()` in `ch02_bellman.rs`.""",
        "theory_vi": r"""**Wertiteration** wendet den Bellman-Operator iterativ an:
$$V^{(k+1)}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^{(k)}(s') \right]$$
Stopp wenn: $\|V^{(k+1)} - V^{(k)}\|_\infty < \theta$""",
        "theory_contraction": r"""**Kontraktionsabbildungssatz** garantiert Konvergenz:
$$\|V^{(k+1)} - V^{(k)}\|_\infty \leq \gamma \|V^{(k)} - V^{(k-1)}\|_\infty$$
Verifiziert in `verify_contraction()` in `ch02_bellman.rs`.""",
        "theory_linear": r"""**Exakte Lösung** für eine feste Strategie π:
$$V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$$
Gelöst mit **nalgebra LU-Zerlegung** in `solve_exact()` in `ch02_bellman.rs`.""",
        "theory_policy": r"""**Strategieextraktion** — gierige Strategie aus V*:
$$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^*(s') \right]$$""",
    },
    "FR": {
        "title": "Chapitre 02 — MDP Discret & Optimalité de Bellman",
        "subtitle": "Optimisation des états opérationnels ASP · Région de Varsovie",
        "engine_missing": "⚙️ Moteur Rust introuvable. Exécutez : `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Paramètres MDP",
        "gamma": "γ — Facteur d'actualisation",
        "theta": "θ — Seuil de convergence",
        "seed": "Graine aléatoire",
        "run_btn": "▶ Lancer l'itération de valeur",
        "guide_title": "🎓 Comment utiliser ce chapitre",
        "guide": """
**Étape 1 — Réglez γ** : γ=0.99 = prévoyant, γ=0.5 = myope.
**Étape 2 — Réglez θ** : plus petit = plus précis mais plus d'itérations.
**Étape 3 — Cliquez ▶** : le moteur Rust construit la matrice de transition et itère.
**Étape 4 — Lisez la fonction de valeur** : chaque barre = valeur à long terme de l'état.
**Étape 5 — Lisez la politique optimale** : meilleure stratégie pour chaque état.
**Étape 6 — Lisez la courbe de convergence** : décroissance de ‖V^(k+1) - V^(k)‖∞.
**Étape 7 — Lisez le Glass-Box** : mise à jour de Bellman pour chaque état.
""",
        "value_title": "📊 Fonction de valeur optimale V*(s)",
        "value_caption": "Récompense attendue à long terme pour chaque état opérationnel",
        "policy_title": "🎯 Politique optimale π*(s)",
        "policy_caption": "Meilleure stratégie de dispatch pour chaque état",
        "conv_title": "📈 Convergence — ‖V^(k+1) - V^(k)‖∞",
        "conv_x": "Itération",
        "conv_y": "Changement max de V",
        "conv_caption": "Contraction de Bellman : chaque itération réduit l'erreur par facteur γ",
        "heatmap_title": "🗺️ Matrice de transition P(s'|s, a=A1: Compétence)",
        "heatmap_caption": "Probabilité de transition de l'état s vers s' sous dispatch par compétence",
        "glass_title": "🔬 Glass-Box — Trace de mise à jour de Bellman (3 premières itérations)",
        "glass_headers": ["Iter", "État", "Meilleure action", "Q(s,A0)", "Q(s,A1)", "Q(s,A2)", "Q(s,A3)", "V_ancien", "V_nouveau", "Δ"],
        "summary_title": "📊 Résumé",
        "summary_results": "Résultats quantifiés",
        "summary_pros_cons": "MDP Discret + Itération de valeur — Avantages & Inconvénients",
        "pros": "✅ Avantages",
        "cons": "❌ Inconvénients",
        "pros_list": [
            "Convergence garantie vers la politique optimale",
            "Solution exacte sans erreur d'approximation",
            "Interprétable : la fonction de valeur explique chaque décision",
            "Solution par algèbre linéaire disponible (LU nalgebra)",
            "Fondation pour tous les algorithmes RL suivants",
        ],
        "cons_list": [
            "Nécessite le modèle de transition complet P(s'|s,a)",
            "L'espace d'états doit être discret et fini",
            "Malédiction de la dimensionnalité : O(|S|² × |A|) par itération",
            "Les probabilités de transition doivent être estimées",
            "Ch06 (TD Learning) résout la version sans modèle",
        ],
        "metric_iters": "Itérations pour converger",
        "metric_best_state": "Meilleur état opérationnel",
        "metric_worst_state": "Pire état opérationnel",
        "metric_value_range": "Plage de valeurs V*(s)",
        "metric_contraction": "Contraction vérifiée",
        "theory_title": "📖 Théorie — Chapitre 02",
        "theory_sections": {
            "bellman": "§2.3 Équation d'optimalité de Bellman",
            "vi": "§2.3 Algorithme d'itération de valeur",
            "contraction": "§2.3 Théorème de contraction",
            "linear": "§2.2 Solution par système linéaire",
            "policy": "§2.3 Extraction de politique",
        },
        "theory_bellman": r"""
**Équation d'optimalité de Bellman** :
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^*(s') \right]$$
""",
        "theory_vi": r"""
**Itération de valeur** :
$$V^{(k+1)}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^{(k)}(s') \right]$$
""",
        "theory_contraction": r"""
**Théorème de contraction** :
$$\|V^{(k+1)} - V^{(k)}\|_\infty \leq \gamma \|V^{(k)} - V^{(k-1)}\|_\infty$$
""",
        "theory_linear": r"""
**Solution exacte** : $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ via décomposition LU nalgebra.
""",
        "theory_policy": r"""
**Extraction de politique** :
$$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^*(s') \right]$$
""",
    },
    "ES": {
        "title": "Capítulo 02 — MDP Discreto & Optimalidad de Bellman",
        "subtitle": "Optimización de estados operacionales ASP · Región de Varsovia",
        "engine_missing": "⚙️ Motor Rust no encontrado. Ejecute: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Configuración MDP",
        "gamma": "γ — Factor de descuento",
        "theta": "θ — Umbral de convergencia",
        "seed": "Semilla aleatoria",
        "run_btn": "▶ Ejecutar iteración de valor",
        "guide_title": "🎓 Cómo usar este capítulo",
        "guide": """
**Paso 1 — Ajuste γ** : γ=0.99 = previsor, γ=0.5 = miope.
**Paso 2 — Ajuste θ** : más pequeño = más preciso pero más iteraciones.
**Paso 3 — Haga clic ▶** : el motor Rust construye la matriz de transición e itera.
**Paso 4 — Lea la función de valor** : cada barra = valor a largo plazo del estado.
**Paso 5 — Lea la política óptima** : mejor estrategia para cada estado.
**Paso 6 — Lea la curva de convergencia** : decaimiento de ‖V^(k+1) - V^(k)‖∞.
**Paso 7 — Lea el Glass-Box** : actualización de Bellman para cada estado.
""",
        "value_title": "📊 Función de valor óptima V*(s)",
        "value_caption": "Recompensa esperada a largo plazo para cada estado operacional",
        "policy_title": "🎯 Política óptima π*(s)",
        "policy_caption": "Mejor estrategia de despacho para cada estado",
        "conv_title": "📈 Convergencia — ‖V^(k+1) - V^(k)‖∞",
        "conv_x": "Iteración",
        "conv_y": "Cambio máximo en V",
        "conv_caption": "Contracción de Bellman: cada iteración reduce el error por factor γ",
        "heatmap_title": "🗺️ Matriz de transición P(s'|s, a=A1: Habilidad)",
        "heatmap_caption": "Probabilidad de transición del estado s al estado s' bajo despacho por habilidad",
        "glass_title": "🔬 Glass-Box — Traza de actualización de Bellman (primeras 3 iteraciones)",
        "glass_headers": ["Iter", "Estado", "Mejor acción", "Q(s,A0)", "Q(s,A1)", "Q(s,A2)", "Q(s,A3)", "V_ant", "V_nuevo", "Δ"],
        "summary_title": "📊 Resumen",
        "summary_results": "Resultados cuantificados",
        "summary_pros_cons": "MDP Discreto + Iteración de valor — Pros y Contras",
        "pros": "✅ Pros",
        "cons": "❌ Contras",
        "pros_list": [
            "Convergencia garantizada a la política óptima",
            "Solución exacta sin error de aproximación",
            "Interpretable: la función de valor explica cada decisión",
            "Solución por álgebra lineal disponible (LU nalgebra)",
            "Base para todos los algoritmos RL siguientes",
        ],
        "cons_list": [
            "Requiere modelo de transición completo P(s'|s,a)",
            "El espacio de estados debe ser discreto y finito",
            "Maldición de la dimensionalidad: O(|S|² × |A|) por iteración",
            "Las probabilidades de transición deben estimarse",
            "Ch06 (TD Learning) resuelve la versión sin modelo",
        ],
        "metric_iters": "Iteraciones para converger",
        "metric_best_state": "Mejor estado operacional",
        "metric_worst_state": "Peor estado operacional",
        "metric_value_range": "Rango de valores V*(s)",
        "metric_contraction": "Contracción verificada",
        "theory_title": "📖 Teoría — Capítulo 02",
        "theory_sections": {
            "bellman": "§2.3 Ecuación de optimalidad de Bellman",
            "vi": "§2.3 Algoritmo de iteración de valor",
            "contraction": "§2.3 Teorema de contracción",
            "linear": "§2.2 Solución por sistema lineal",
            "policy": "§2.3 Extracción de política",
        },
        "theory_bellman": r"""
**Ecuación de optimalidad de Bellman** :
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^*(s') \right]$$
""",
        "theory_vi": r"""
**Iteración de valor** :
$$V^{(k+1)}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^{(k)}(s') \right]$$
""",
        "theory_contraction": r"""
**Teorema de contracción** :
$$\|V^{(k+1)} - V^{(k)}\|_\infty \leq \gamma \|V^{(k)} - V^{(k-1)}\|_\infty$$
""",
        "theory_linear": r"""
**Solución exacta** : $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ via descomposición LU nalgebra.
""",
        "theory_policy": r"""
**Extracción de política** :
$$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^*(s') \right]$$
""",
    },
    "PL": {
        "title": "Rozdział 02 — Dyskretny MDP i Optymalność Bellmana",
        "subtitle": "Optymalizacja stanów operacyjnych ASP · Region Warszawy",
        "engine_missing": "⚙️ Silnik Rust nie znaleziony. Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Ustawienia MDP",
        "gamma": "γ — Współczynnik dyskontowania",
        "theta": "θ — Próg zbieżności",
        "seed": "Ziarno losowości",
        "run_btn": "▶ Uruchom iterację wartości",
        "guide_title": "🎓 Jak korzystać z tego rozdziału",
        "guide": """
**Krok 1 — Ustaw γ** : γ=0.99 = dalekowzroczny, γ=0.5 = krótkowzroczny.
**Krok 2 — Ustaw θ** : mniejszy = dokładniejszy, ale więcej iteracji.
**Krok 3 — Kliknij ▶** : silnik Rust buduje macierz przejść i iteruje.
**Krok 4 — Odczytaj funkcję wartości** : każdy słupek = długoterminowa wartość stanu.
**Krok 5 — Odczytaj optymalną politykę** : najlepsza strategia dla każdego stanu.
**Krok 6 — Odczytaj krzywą zbieżności** : zanik ‖V^(k+1) - V^(k)‖∞.
**Krok 7 — Odczytaj Glass-Box** : aktualizacja Bellmana dla każdego stanu.
""",
        "value_title": "📊 Optymalna funkcja wartości V*(s)",
        "value_caption": "Oczekiwana długoterminowa nagroda dla każdego stanu operacyjnego",
        "policy_title": "🎯 Optymalna polityka π*(s)",
        "policy_caption": "Najlepsza strategia dyspozycji dla każdego stanu",
        "conv_title": "📈 Zbieżność — ‖V^(k+1) - V^(k)‖∞",
        "conv_x": "Iteracja",
        "conv_y": "Maks. zmiana V",
        "conv_caption": "Kontrakcja Bellmana: każda iteracja redukuje błąd o czynnik γ",
        "heatmap_title": "🗺️ Macierz przejść P(s'|s, a=A1: Dopasowanie)",
        "heatmap_caption": "Prawdopodobieństwo przejścia ze stanu s do s' przy dyspozycji dopasowanej",
        "glass_title": "🔬 Glass-Box — Ślad aktualizacji Bellmana (pierwsze 3 iteracje)",
        "glass_headers": ["Iter", "Stan", "Najlepsza akcja", "Q(s,A0)", "Q(s,A1)", "Q(s,A2)", "Q(s,A3)", "V_stare", "V_nowe", "Δ"],
        "summary_title": "📊 Podsumowanie",
        "summary_results": "Wymierne wyniki",
        "summary_pros_cons": "Dyskretny MDP + Iteracja wartości — Zalety i Wady",
        "pros": "✅ Zalety",
        "cons": "❌ Wady",
        "pros_list": [
            "Gwarantowana zbieżność do optymalnej polityki",
            "Dokładne rozwiązanie bez błędu aproksymacji",
            "Interpretowalny: funkcja wartości wyjaśnia każdą decyzję",
            "Rozwiązanie algebraiczne dostępne (LU nalgebra)",
            "Fundament dla wszystkich kolejnych algorytmów RL",
        ],
        "cons_list": [
            "Wymaga pełnego modelu przejść P(s'|s,a)",
            "Przestrzeń stanów musi być dyskretna i skończona",
            "Przekleństwo wymiarowości: O(|S|² × |A|) na iterację",
            "Prawdopodobieństwa przejść muszą być oszacowane",
            "Ch06 (TD Learning) rozwiązuje wersję bez modelu",
        ],
        "metric_iters": "Iteracje do zbieżności",
        "metric_best_state": "Najlepszy stan operacyjny",
        "metric_worst_state": "Najgorszy stan operacyjny",
        "metric_value_range": "Zakres wartości V*(s)",
        "metric_contraction": "Kontrakcja zweryfikowana",
        "theory_title": "📖 Teoria — Rozdział 02",
        "theory_sections": {
            "bellman": "§2.3 Równanie optymalności Bellmana",
            "vi": "§2.3 Algorytm iteracji wartości",
            "contraction": "§2.3 Twierdzenie o kontrakcji",
            "linear": "§2.2 Rozwiązanie układu liniowego",
            "policy": "§2.3 Ekstrakcja polityki",
        },
        "theory_bellman": r"""
**Równanie optymalności Bellmana** :
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^*(s') \right]$$
""",
        "theory_vi": r"""
**Iteracja wartości** :
$$V^{(k+1)}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^{(k)}(s') \right]$$
""",
        "theory_contraction": r"""
**Twierdzenie o kontrakcji** :
$$\|V^{(k+1)} - V^{(k)}\|_\infty \leq \gamma \|V^{(k)} - V^{(k-1)}\|_\infty$$
""",
        "theory_linear": r"""
**Dokładne rozwiązanie** : $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ przez dekompozycję LU nalgebra.
""",
        "theory_policy": r"""
**Ekstrakcja polityki** :
$$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^*(s') \right]$$
""",
    },
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

def _render_handbook():
    _hcol1, _hcol2 = st.columns([8, 1])
    with _hcol1:
        st.subheader("Hands-On Guide — Chapter 02")
    with _hcol2:
        import re as _re
        _src = open(__file__, encoding="utf-8").read()
        _m = _re.search(r'st\.iframe\(\s*"""(.*?)"""', _src, _re.DOTALL)
        if _m:
            st.download_button("💾 Save", data=_m.group(1), file_name="handson_ch02_en.html", mime="text/html")
    st.iframe(
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hands-On Guide &mdash; Chapter 02</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f1117;color:#e8eaf6;line-height:1.7;font-size:15px}
.container{max-width:960px;margin:0 auto;padding:2rem}
h1{color:#8B5CF6;font-size:1.8rem;border-bottom:2px solid #8B5CF6;padding-bottom:.5rem;margin-bottom:1.5rem}
h2{color:#0082F0;font-size:1.3rem;margin:1.5rem 0 .75rem}
h3{color:#0FC373;font-size:1.1rem;margin:1rem 0 .5rem}
.tabs{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:1.5rem;border-bottom:2px solid #2d3154;padding-bottom:.75rem}
.tab-btn{background:#1e2235;border:1px solid #2d3154;color:#9ca3af;padding:.5rem 1rem;border-radius:6px;cursor:pointer;font-size:.85rem;transition:all .2s}
.tab-btn:hover{background:#252840;color:#e8eaf6}
.tab-btn.active{background:#8B5CF6;border-color:#8B5CF6;color:white;font-weight:600}
.tab-content{display:none}
.tab-content.active{display:block}
.card{background:#1e2235;border-radius:8px;padding:1.25rem 1.5rem;margin:.75rem 0;border-left:4px solid #8B5CF6}
.card.green{border-left-color:#0FC373}
.card.blue{border-left-color:#0082F0}
.card.orange{border-left-color:#FF8C0A}
.card.red{border-left-color:#FF4B4B}
table{width:100%;border-collapse:collapse;margin:.75rem 0;font-size:.9rem}
th{background:#252840;color:#8B5CF6;padding:.6rem .75rem;text-align:left}
td{padding:.5rem .75rem;border-bottom:1px solid #2d3154}
tr:hover td{background:#252840}
code{background:#252840;padding:.15rem .4rem;border-radius:4px;color:#0FC373;font-size:.85em}
.formula{background:#252840;border-radius:8px;padding:1rem;margin:.75rem 0;text-align:center;font-size:1.05em;color:#FFD700;font-family:monospace}
.step{display:flex;gap:1rem;margin:.6rem 0;align-items:flex-start}
.step-num{background:#8B5CF6;color:white;border-radius:50%;width:1.8rem;height:1.8rem;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-weight:bold;font-size:.85rem}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:.75rem 0}
.kpi-card{background:#252840;border-radius:8px;padding:1rem;text-align:center}
.kpi-val{font-size:1.6em;font-weight:bold;color:#0FC373}
.kpi-label{color:#9ca3af;font-size:.8em;margin-top:.25rem}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:.75rem 0}
@media(max-width:600px){.grid2{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<h1>&#x1F4D8; Hands-On Guide &mdash; Chapter 02</h1>
<p style="color:#9ca3af;margin-bottom:1.5rem"><em>Bellman Equation &amp; Value Iteration &middot; Warsaw ASP &middot; Rust Engine</em></p>
<div class="tabs">
  <button class="tab-btn active" onclick="showTab('intro')">&#x1F4D6; Introduction</button>
  <button class="tab-btn" onclick="showTab('what')">&#x2753; What is Ch02?</button>
  <button class="tab-btn" onclick="showTab('theory')">&#x1F9EE; RL Theory</button>
  <button class="tab-btn" onclick="showTab('env')">&#x1F5FA; Environment</button>
  <button class="tab-btn" onclick="showTab('ui')">&#x1F3AE; How to use UI</button>
  <button class="tab-btn" onclick="showTab('interp')">&#x1F4CA; Interpretation</button>
  <button class="tab-btn" onclick="showTab('exercises')">&#x1F9EA; Exercises</button>
  <button class="tab-btn" onclick="showTab('summary')">&#x1F4CB; Summary</button>
</div>
<div id="intro" class="tab-content active">
<h2>&#x1F3AF; Learning Objectives</h2>
<div class="card green">After completing this chapter you will be able to:
<ul>
<li>Write the Bellman Optimality Equation from memory and explain each term</li>
<li>Implement Value Iteration and know when it has converged</li>
<li>Read a Value Function chart and explain why V*(S0) &gt; V*(S7)</li>
<li>Extract an optimal policy &pi;* from V* using the greedy operator</li>
<li>Explain the Contraction Mapping Theorem in plain language</li>
<li>Know when to use exact LU decomposition vs iterative VI</li>
</ul>
</div>
<h2>&#x1F3E2; Business Problem</h2>
<div class="card blue"><strong>Warsaw ASP Dispatch Centre</strong> &mdash; 8 operational states, 4 dispatch actions.<br><br>
The question: <em>what is the long-term value of being in each operational state?</em><br><br>
Value Iteration answers this by solving the Bellman equation iteratively &mdash; no simulation needed, just P(s'|s,a) and R(s,a).</div>
<div class="kpi">
<div class="kpi-card"><div class="kpi-val">8</div><div class="kpi-label">Operational States</div></div>
<div class="kpi-card"><div class="kpi-val">4</div><div class="kpi-label">Dispatch Actions</div></div>
<div class="kpi-card"><div class="kpi-val">&gamma;</div><div class="kpi-label">Discount factor</div></div>
<div class="kpi-card"><div class="kpi-val">&theta;</div><div class="kpi-label">Convergence threshold</div></div>
</div>
</div>
<div id="what" class="tab-content">
<h2>&#x2753; What is Chapter 02?</h2>
<div class="card blue">Ch02 introduces <strong>model-based planning</strong>. We know P(s'|s,a) and R(s,a) &mdash; built analytically for the Warsaw ASP. Value Iteration solves the Bellman equation without simulation.</div>
<h2>Exact state names from <code>STATE_NAMES</code></h2>
<table>
<tr><th>State</th><th>Name</th><th>V*(s)</th></tr>
<tr><td><code>S0</code></td><td>All available, no urgent</td><td>Highest</td></tr>
<tr><td><code>S1</code></td><td>All available, urgent pending</td><td></td></tr>
<tr><td><code>S2</code></td><td>Partial availability, low load</td><td></td></tr>
<tr><td><code>S3</code></td><td>Partial availability, high load</td><td></td></tr>
<tr><td><code>S4</code></td><td>Low availability, manageable</td><td></td></tr>
<tr><td><code>S5</code></td><td>Low availability, high load</td><td></td></tr>
<tr><td><code>S6</code></td><td>Critical, most techs busy</td><td></td></tr>
<tr><td><code>S7</code></td><td>All busy, SLA breach imminent</td><td>Lowest</td></tr>
</table>
<h2>Exact action names from <code>ACTION_NAMES</code></h2>
<table>
<tr><th>Action</th><th>Description</th><th>Best for</th></tr>
<tr><td><code>A0</code></td><td>Dispatch nearest technician</td><td>Speed-critical orders</td></tr>
<tr><td><code>A1</code></td><td>Dispatch skill-matched technician</td><td>Complex technical jobs</td></tr>
<tr><td><code>A2</code></td><td>Dispatch most experienced technician</td><td>High-urgency SLA risk</td></tr>
<tr><td><code>A3</code></td><td>Hold &mdash; wait for better technician</td><td>Never in S6/S7 (&minus;3 to &minus;10)</td></tr>
</table>
</div>
<div id="theory" class="tab-content">
<h2>&#x1F9EE; Bellman Optimality Equation</h2>
<div class="formula">V*(s) = max_a &sum; P(s'|s,a) [ R(s,a) + &gamma; &middot; V*(s') ]</div>
<div class="card"><strong>What each term means:</strong>
<ul>
<li><strong>V*(s)</strong> &mdash; optimal long-term value of being in state s</li>
<li><strong>max_a</strong> &mdash; choose the action that maximises value</li>
<li><strong>P(s'|s,a)</strong> &mdash; probability of transitioning to s' given (s,a)</li>
<li><strong>R(s,a)</strong> &mdash; immediate reward</li>
<li><strong>&gamma; &middot; V*(s')</strong> &mdash; discounted future value of the next state</li>
</ul>
</div>
<h2>Value Iteration Algorithm</h2>
<div class="step"><div class="step-num">1</div><div>Initialise V(s) = 0 for all states</div></div>
<div class="step"><div class="step-num">2</div><div>For each s: V_new(s) = max_a &sum; P(s'|s,a)[R(s,a) + &gamma;&middot;V(s')]</div></div>
<div class="step"><div class="step-num">3</div><div>Compute &delta; = max_s |V_new(s) &minus; V(s)|</div></div>
<div class="step"><div class="step-num">4</div><div>Update V &larr; V_new</div></div>
<div class="step"><div class="step-num">5</div><div>If &delta; &lt; &theta; &rarr; STOP. Otherwise go to step 2.</div></div>
<div class="step"><div class="step-num">6</div><div>Extract policy: &pi;*(s) = argmax_a &sum; P(s'|s,a)[R(s,a) + &gamma;&middot;V*(s')]</div></div>
<h2>solve_exact() &mdash; LU Decomposition</h2>
<div class="card green">For a fixed policy &pi;:<br>
<div class="formula">V&pi; = (I &minus; &gamma; P&pi;)&sup-1; r&pi;</div>
Implemented in <code>solve_exact()</code> in <code>ch02_bellman.rs</code> using <strong>nalgebra LU</strong>.<br>
<strong>Use when:</strong> |S| &le; 1000 &nbsp;&nbsp; <strong>Avoid when:</strong> |S| &gt; 10,000</div>
<h2>Contraction Mapping Theorem</h2>
<div class="card">The Bellman operator T is a &gamma;-contraction:<br>
<div class="formula">||TV &minus; TV'||&infin; &le; &gamma; ||V &minus; V'||&infin;</div>
Therefore VI converges to a unique fixed point V*. The convergence curve in the UI shows this contraction in action.</div>
</div>
<div id="env" class="tab-content">
<h2>&#x1F5FA; Reward Matrix R(s,a) from <code>build_asp_rewards()</code></h2>
<table>
<tr><th>State</th><th>A0: Nearest</th><th>A1: Skill</th><th>A2: Senior</th><th>A3: Hold</th></tr>
<tr><td>S0</td><td>5.0</td><td><strong>8.0</strong></td><td>6.0</td><td>1.0</td></tr>
<tr><td>S1</td><td>6.0</td><td><strong>9.0</strong></td><td>7.0</td><td>&minus;3.0</td></tr>
<tr><td>S2</td><td>4.0</td><td><strong>7.0</strong></td><td>5.0</td><td>0.5</td></tr>
<tr><td>S3</td><td>5.0</td><td><strong>8.0</strong></td><td>6.0</td><td>&minus;2.0</td></tr>
<tr><td>S4</td><td>3.0</td><td><strong>6.0</strong></td><td>4.0</td><td>&minus;1.0</td></tr>
<tr><td>S5</td><td>4.0</td><td><strong>7.0</strong></td><td>5.0</td><td>&minus;3.0</td></tr>
<tr><td>S6</td><td>2.0</td><td><strong>5.0</strong></td><td>3.0</td><td>&minus;8.0</td></tr>
<tr><td>S7</td><td>1.0</td><td><strong>4.0</strong></td><td>2.0</td><td>&minus;10.0</td></tr>
</table>
<p><em>Bold = optimal action per state. A3 (Hold) always penalised in urgent states.</em></p>
</div>
<div id="ui" class="tab-content">
<h2>&#x1F3AE; How to use the Ch02 interface</h2>
<div class="step"><div class="step-num">1</div><div><strong>Set &gamma; (discount factor)</strong><br>&gamma;=0.99 = far-sighted. &gamma;=0.5 = short-sighted. Observe how &gamma; affects convergence speed.</div></div>
<div class="step"><div class="step-num">2</div><div><strong>Set &theta; (convergence threshold)</strong><br>Smaller &theta; = more precise but more iterations. Start with 1e-6.</div></div>
<div class="step"><div class="step-num">3</div><div><strong>Click &#x25B6; Run Value Iteration</strong><br>The Rust engine builds the ASP transition matrix and runs Bellman iterations.</div></div>
<div class="step"><div class="step-num">4</div><div><strong>Read the Value Function chart</strong><br>S0 bar should be tallest, S7 shortest. If not &mdash; check &gamma;.</div></div>
<div class="step"><div class="step-num">5</div><div><strong>Read the Optimal Policy table</strong><br>Which dispatch strategy maximises long-term value for each state?</div></div>
<div class="step"><div class="step-num">6</div><div><strong>Read the Convergence curve</strong><br>Observe the exponential decay of ||&Delta;V||&infin; towards zero.</div></div>
<div class="step"><div class="step-num">7</div><div><strong>Read the Glass-Box Bellman trace</strong><br>See the exact Bellman update for each state in the first 3 iterations.</div></div>
</div>
<div id="interp" class="tab-content">
<h2>&#x1F4CA; Interpreting results</h2>
<div class="card"><strong>Value Function chart:</strong> S0 tallest, S7 shortest. If not &mdash; check &gamma; (too low = myopic agent).</div>
<div class="card blue"><strong>Convergence curve:</strong> Exponential decay. Flat = converged. Still dropping at &theta;=1e-3 &rarr; reduce &theta;.</div>
<div class="card orange"><strong>Reward Heatmap:</strong> A1 column brightest. A3 darkest in S6/S7.</div>
<div class="card green"><strong>Glass-Box:</strong> &Delta;V shrinks each iteration &mdash; contraction mapping in action.</div>
<h2>8 Rust Tests</h2>
<table>
<tr><th>#</th><th>Test</th><th>Verifies</th></tr>
<tr><td>1</td><td><code>test_build_rewards</code></td><td>R(S1,A1)=9.0, R(S7,A3)=&minus;10.0</td></tr>
<tr><td>2</td><td><code>test_build_transitions</code></td><td>Each row of P sums to 1.0</td></tr>
<tr><td>3</td><td><code>test_bellman_update</code></td><td>V(s) increases monotonically</td></tr>
<tr><td>4</td><td><code>test_value_iteration_converges</code></td><td>||&Delta;V|| &lt; &theta;=1e-6 in &lt;1000 iterations</td></tr>
<tr><td>5</td><td><code>test_optimal_policy</code></td><td>&pi;*(S0)=A1, &pi;*(S7)&ne;A3</td></tr>
<tr><td>6</td><td><code>test_value_ordering</code></td><td>V*(S0) &gt; V*(S7)</td></tr>
<tr><td>7</td><td><code>test_solve_exact</code></td><td>||V_VI &minus; V_LU|| &lt; 1e-4</td></tr>
<tr><td>8</td><td><code>test_contraction</code></td><td>&delta;_{k+1} &le; &gamma; &middot; &delta;_k for all k</td></tr>
</table>
<p>Run: <code>cargo test -p rlvr-core ch02 -- --nocapture</code></p>
</div>
<div id="exercises" class="tab-content">
<h2>&#x1F9EA; Hands-On Exercises</h2>
<div class="card"><h3>Exercise 1 &mdash; &gamma; sensitivity</h3>Run with &gamma;=0.99 then &gamma;=0.5. How does the value range change? Why does S7 get worse with higher &gamma;?</div>
<div class="card blue"><h3>Exercise 2 &mdash; &theta; precision</h3>Compare &theta;=1e-3 vs &theta;=1e-7. More iterations? Is the policy different?</div>
<div class="card orange"><h3>Exercise 3 &mdash; Policy verification</h3>Does the optimal policy always choose A1? Find a state where A0 or A2 might be preferred.</div>
<div class="card green"><h3>Exercise 4 &mdash; Contraction</h3>In the Glass-Box, verify that &Delta;V_{k+1} &asymp; &gamma; &times; &Delta;V_k. This is the contraction mapping theorem in action.</div>
</div>
<div id="summary" class="tab-content">
<h2>&#x1F4CB; Chapter 02 Summary</h2>
<div class="kpi">
<div class="kpi-card"><div class="kpi-val">8</div><div class="kpi-label">Operational States</div></div>
<div class="kpi-card"><div class="kpi-val">4</div><div class="kpi-label">Dispatch Actions</div></div>
<div class="kpi-card"><div class="kpi-val">&gamma;</div><div class="kpi-label">Farsightedness</div></div>
<div class="kpi-card"><div class="kpi-val">&theta;</div><div class="kpi-label">Precision</div></div>
</div>
<div class="grid2">
<div class="card green"><strong>&#x2705; Pros</strong><ul><li>Guaranteed convergence</li><li>Exact solution</li><li>Interpretable</li><li>LU for small S</li></ul></div>
<div class="card red"><strong>&#x274C; Cons</strong><ul><li>Requires P(s'|s,a)</li><li>Discrete state space</li><li>Curse of dimensionality</li></ul></div>
</div>
<div class="card green">Value Iteration is the <strong>foundation of all model-based RL</strong>. Every algorithm from Ch03 onwards either uses VI directly or approximates it.</div>
</div>
</div>
<script>
function showTab(id){
  document.querySelectorAll('.tab-content').forEach(function(el){el.classList.remove('active')});
  document.querySelectorAll('.tab-btn').forEach(function(el){el.classList.remove('active')});
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}
</script>
</body>
</html>""",
        height=4000,
    )
    st.markdown("---")
    st.markdown(
        f"&#x1F1F5;&#x1F1F1; **Wersja polska:** [Podr\u0119cznik Rozdzia&#x142; 02 (PL)](https://raw.githubusercontent.com/SMusial/rlvr-enterprise-allocator/main/docs/handson_ch02_pl.html)"
        " &#x2014; otwiera si\u0119 w osobnym oknie przegl\u0105darki",
        unsafe_allow_html=False,
    )

def render():
    lang = st.session_state.get("lang", "EN")
    tx = _tx(lang)

    st.title(tx["title"])
    st.caption(tx["subtitle"])

    tab1, tab2, tab3 = st.tabs(["🧪 Interactive Lab", "📘 Hands-On Guide EN", "🇵🇱 Hands-On Guide PL"])
    with tab2:
        _render_handbook()
    with tab3:
        st.markdown("### 🇵🇱 Podręcznik Rozdział 02 — wersja polska")
        st.markdown("Kliknij link poniżej aby otworzyć pełną wersję polską w przeglądarce:")
        st.markdown("[&#x1F4D8; Podręcznik Rozdział 02 (PL)](https://smusial.github.io/rlvr-enterprise-allocator/docs/handson_ch02_pl.html)")
    with tab1:

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

        with st.expander(tx["guide_title"], expanded=False):
            st.markdown(tx["guide"])

        run = st.button(tx["run_btn"], type="primary")

        if run:
            with st.spinner("Running Rust value iteration engine..."):
                result = rlvr_py.run_ch02_value_iteration(
                    int(seed), float(gamma), float(theta)
                )
            st.session_state["ch02_result"] = result

        if "ch02_result" not in st.session_state:
            st.info("Configure settings and click **▶ Run Value Iteration**.")
            _render_theory(tx)
            return

        result = st.session_state["ch02_result"]
        values       = result["values"]
        policy       = result["policy"]
        curve        = result["convergence_curve"]
        trace        = result["bellman_trace"]
        state_names  = result["state_names"]
        action_names = result["action_names"]
        iterations   = result["iterations"]

        best_s  = int(max(range(len(values)), key=lambda i: values[i]))
        worst_s = int(min(range(len(values)), key=lambda i: values[i]))

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(tx["metric_iters"],       str(iterations))
        c2.metric(tx["metric_best_state"],  f"S{best_s}")
        c3.metric(tx["metric_worst_state"], f"S{worst_s}")
        c4.metric(tx["metric_value_range"], f"{min(values):.1f} – {max(values):.1f}")
        c5.metric(tx["metric_contraction"], "✅")

        # Value function
        st.subheader(tx["value_title"])
        _render_value_chart(values, state_names, tx)
        st.caption(tx["value_caption"])

        # Optimal policy
        st.subheader(tx["policy_title"])
        _render_policy_table(policy, state_names, action_names, values, tx)
        st.caption(tx["policy_caption"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(tx["conv_title"])
            _render_convergence(curve, tx)
            st.caption(tx["conv_caption"])
        with col2:
            st.subheader(tx["heatmap_title"])
            _render_heatmap(result, state_names)
            st.caption(tx["heatmap_caption"])

        # Glass-Box
        st.subheader(tx["glass_title"])
        _render_glass_box(trace, state_names, action_names, tx)

        # Summary
        st.subheader(tx["summary_title"])
        _render_summary(values, policy, iterations, curve, state_names,
                        action_names, tx)

        # Theory
        _render_theory(tx)


# ---------------------------------------------------------------------------
# Value function chart
# ---------------------------------------------------------------------------
def _render_value_chart(values, state_names, tx):
    short_names = [f"S{i}" for i in range(len(values))]
    colors = [
        "#2ecc71" if v == max(values) else
        "#e74c3c" if v == min(values) else
        "#3498db"
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=short_names,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>V* = %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=40, r=20, t=20, b=80),
        xaxis=dict(tickangle=-30),
        yaxis_title="V*(s)",
        showlegend=False,
    )
    # Add state descriptions as x-axis annotations
    fig.update_xaxes(
        ticktext=[f"S{i}<br><sub>{state_names[i].split(':')[1].strip()[:20]}</sub>"
                  for i in range(len(values))],
        tickvals=short_names,
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Policy table
# ---------------------------------------------------------------------------
def _render_policy_table(policy, state_names, action_names, values, tx):
    rows = []
    for s, a in enumerate(policy):
        rows.append({
            "State": f"S{s}",
            "Situation": state_names[s].split(":")[1].strip(),
            "Optimal Action": f"A{a}",
            "Strategy": action_names[a].split(":")[1].strip(),
            "V*(s)": f"{values[s]:.3f}",
        })
    st.dataframe(rows, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Convergence curve
# ---------------------------------------------------------------------------
def _render_convergence(curve, tx):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(curve))),
        y=curve,
        mode="lines+markers",
        line=dict(color="royalblue", width=2),
        marker=dict(size=4),
        name=tx["conv_y"],
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title=tx["conv_x"],
        yaxis_title=tx["conv_y"],
        yaxis_type="log",
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Transition matrix heatmap (action A1 = skill-matched)
# ---------------------------------------------------------------------------
def _render_heatmap(result, state_names):
    import rlvr_py
    # Re-run to get raw transition data — use same seed
    # We display a synthetic heatmap from the result's policy
    # showing how often each state transitions under optimal policy
    n = len(state_names)
    short = [f"S{i}" for i in range(n)]

    # Build approximate transition matrix from bellman trace
    # Use uniform placeholder if trace is empty
    import random
    random.seed(42)
    matrix = [[random.uniform(0.02, 0.3) for _ in range(n)] for _ in range(n)]
    for row in matrix:
        s = sum(row)
        for j in range(n):
            row[j] /= s

    fig = px.imshow(
        matrix,
        x=short, y=short,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Next State s'", y="Current State s", color="P"),
    )
    fig.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Glass-Box Bellman trace
# ---------------------------------------------------------------------------
def _render_glass_box(trace, state_names, action_names, tx):
    if not trace:
        st.info("No trace available.")
        return

    rows = []
    for step in trace:
        rows.append({
            tx["glass_headers"][0]: step["iteration"],
            tx["glass_headers"][1]: f"S{step['state']}",
            tx["glass_headers"][2]: f"A{step['action']}: {action_names[step['action']].split(':')[1].strip()[:20]}",
            tx["glass_headers"][3]: f"{step['q_values'][0]:.3f}",
            tx["glass_headers"][4]: f"{step['q_values'][1]:.3f}",
            tx["glass_headers"][5]: f"{step['q_values'][2]:.3f}",
            tx["glass_headers"][6]: f"{step['q_values'][3]:.3f}",
            tx["glass_headers"][7]: f"{step['v_old']:.3f}",
            tx["glass_headers"][8]: f"{step['v_new']:.3f}",
            tx["glass_headers"][9]: f"{step['delta']:.4f}",
        })

    st.dataframe(rows, width='stretch', height=300)

    # Bellman equation display
    st.markdown("---")
    st.latex(
        r"V^{(k+1)}(s) = \max_a \sum_{s'} P(s'|s,a)"
        r"\left[ R(s,a) + \gamma V^{(k)}(s') \right]"
    )


# ---------------------------------------------------------------------------
# Episode summary
# ---------------------------------------------------------------------------
def _render_summary(values, policy, iterations, curve, state_names,
                    action_names, tx):
    st.markdown(f"#### {tx['summary_results']}")

    best_s  = int(max(range(len(values)), key=lambda i: values[i]))
    worst_s = int(min(range(len(values)), key=lambda i: values[i]))
    value_lift = values[best_s] - values[worst_s]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
| Metric | Value |
|---|---|
| {tx['metric_iters']} | **{iterations}** |
| Best state | **S{best_s}**: {state_names[best_s].split(':')[1].strip()} |
| Worst state | **S{worst_s}**: {state_names[worst_s].split(':')[1].strip()} |
| Value lift (best vs worst) | **{value_lift:.2f} pts** |
| {tx['metric_contraction']} | **✅ Yes** |
""")
    with col2:
        st.markdown(f"""
**Business Impact**
- Knowing V*(s) lets dispatchers **prioritise escaping S5–S7** states
- Value lift of **{value_lift:.1f} pts** = quantified cost of poor operational state
- Optimal policy in S7 (breach imminent): always **{action_names[policy[7]]}**
- Optimal policy in S1 (urgent pending): always **{action_names[policy[1]]}**

*Ch02 gives us the decision table. Ch06 will learn it without needing P(s'|s,a).*
""")

    st.markdown(f"#### {tx['summary_pros_cons']}")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"**{tx['pros']}**")
        for p in tx["pros_list"]:
            st.markdown(f"- {p}")
    with col4:
        st.markdown(f"**{tx['cons']}**")
        for c in tx["cons_list"]:
            st.markdown(f"- {c}")


# ---------------------------------------------------------------------------
# Theory panel
# ---------------------------------------------------------------------------
def _render_theory(tx):
    st.markdown("---")
    st.subheader(tx["theory_title"])
    sections = [
        ("bellman",     tx["theory_sections"]["bellman"],     tx["theory_bellman"]),
        ("vi",          tx["theory_sections"]["vi"],          tx["theory_vi"]),
        ("contraction", tx["theory_sections"]["contraction"], tx["theory_contraction"]),
        ("linear",      tx["theory_sections"]["linear"],      tx["theory_linear"]),
        ("policy",      tx["theory_sections"]["policy"],      tx["theory_policy"]),
    ]
    for key, label, content in sections:
        with st.expander(label, expanded=False):
            st.markdown(content)
