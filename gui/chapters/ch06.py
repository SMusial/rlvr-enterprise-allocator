import streamlit as st
import plotly.graph_objects as go

T = {
    "EN": {
        "title": "Chapter 06 — Temporal Difference Learning",
        "subtitle": "TD(0) · SARSA · Q-Learning · ASP Dispatch · Warsaw Region",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ TD Settings",
        "n_episodes": "Number of episodes",
        "gamma": "γ — Discount factor",
        "alpha": "α — Learning rate",
        "epsilon": "ε — Initial exploration",
        "epsilon_decay": "ε decay rate",
        "seed": "Random seed",
        "run_btn": "▶ Run TD(0), SARSA and Q-Learning",
        "guide_title": "🎓 How to use this chapter",
        "guide": """
**Step 1 — Understand the key difference from Ch05 (MC)**
TD methods update after EVERY step, not after episode end.
They bootstrap: use current estimates V(s') to update V(s).
This makes them faster and applicable to continuing tasks.

**Step 2 — Understand SARSA vs Q-Learning**
Both are TD control methods. The difference is one line:
- SARSA: uses Q(s', a') where a' is chosen by epsilon-greedy (on-policy)
- Q-Learning: uses max_a' Q(s', a') regardless of what action is taken (off-policy)

**Step 3 — Set α (learning rate)**
α=0.1 is a good start. Higher α = faster learning but less stable.
Lower α = slower but more stable convergence.

**Step 4 — Click ▶ Run TD(0), SARSA and Q-Learning**
All three run simultaneously. Results appear side by side.

**Step 5 — Read the TD Error curve**
TD error = R + γV(s') - V(s). Watch it decay toward zero.
This is the learning signal — the agent is "surprised" less over time.

**Step 6 — Compare Q-Learning vs SARSA policies**
Q-Learning finds the optimal policy (off-policy).
SARSA finds the safest policy given epsilon-soft exploration (on-policy).

**Step 7 — Compare with DP reference (Ch04)**
TD methods should converge to the same policy as DP — without needing P(s'|s,a).
""",
        "returns_title": "📈 Episode Returns — TD(0), SARSA, Q-Learning",
        "returns_caption": "Moving average of episode returns. Q-Learning should converge fastest.",
        "td_error_title": "📉 TD Error — |R + γV(s') - V(s)|",
        "td_error_caption": "TD error decays as the agent learns. Smaller = better estimates.",
        "value_title": "📊 Value Function V(s) — TD vs DP Reference",
        "value_caption": "TD estimates should converge toward DP solution (Ch04) with more episodes.",
        "policy_title": "🎯 Optimal Policy — SARSA vs Q-Learning",
        "policy_caption": "Q-Learning finds optimal policy. SARSA finds safest policy under epsilon-soft.",
        "qtable_title": "📊 Q-Table Heatmap",
        "qtable_caption": "Q(s,a) values. Select algorithm to display.",
        "glass_title": "🔬 Glass-Box — TD Update Trace",
        "glass_headers": ["Episode", "Step", "State", "Action", "Reward", "Next State", "TD Error"],
        "summary_title": "📊 Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "TD Algorithms — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "theory_title": "📖 Theory — Chapter 06",
        "theory_sections": {
            "td_intro":  "§6.1 Temporal Difference Learning",
            "td0":       "§6.1 TD(0) Prediction",
            "sarsa":     "§6.2 SARSA — On-Policy TD Control",
            "qlearning": "§6.3 Q-Learning — Off-Policy TD Control",
            "comparison":"§6.4 SARSA vs Q-Learning",
        },
        "theory_td_intro": r"""
**Temporal Difference (TD) Learning** combines ideas from MC and DP:
- Like MC: model-free, learns from experience
- Like DP: bootstraps — updates using current estimates, not waiting for episode end

The TD error (delta):
δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)

This is the "surprise" signal — how much better or worse than expected.
Implemented in `ch06_td.rs`.
""",
        "theory_td0": r"""
**TD(0) Prediction** updates V(s) after every step:

V(S_t) <- V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]

- α = learning rate (step size)
- R_{t+1} + γ V(S_{t+1}) = TD target
- R_{t+1} + γ V(S_{t+1}) - V(S_t) = TD error δ_t

Converges to V^π for any fixed policy π.
Implemented in `td0_prediction()` in `ch06_td.rs`.
""",
        "theory_sarsa": r"""
**SARSA** (State-Action-Reward-State-Action) is on-policy TD control:

Q(S_t, A_t) <- Q(S_t, A_t) + α [R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

Key: A_{t+1} is chosen by the SAME epsilon-greedy policy used for behaviour.
This makes SARSA on-policy — it learns the value of the epsilon-soft policy.

SARSA is conservative: it accounts for the exploration cost in its Q estimates.
Implemented in `sarsa()` in `ch06_td.rs`.
""",
        "theory_qlearning": r"""
**Q-Learning** is off-policy TD control:

Q(S_t, A_t) <- Q(S_t, A_t) + α [R_{t+1} + γ max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]

Key: uses max_{a'} Q(S_{t+1}, a') — the GREEDY action, regardless of what was actually taken.
This makes Q-Learning off-policy — it directly learns Q* (optimal action-value function).

Q-Learning converges to Q* regardless of the behaviour policy (as long as all (s,a) are visited).
Implemented in `q_learning()` in `ch06_td.rs`.
""",
        "theory_comparison": r"""
**SARSA vs Q-Learning — the one-line difference:**

SARSA:      Q(s,a) += α [R + γ Q(s', a') - Q(s,a)]   where a' ~ ε-greedy
Q-Learning: Q(s,a) += α [R + γ max_a' Q(s', a') - Q(s,a)]

**When does it matter?**
In risky environments (like S7 — SLA breach imminent):
- SARSA avoids risky states because it accounts for epsilon-exploration
- Q-Learning ignores exploration cost and finds the theoretically optimal policy

**Cliff Walking analogy:**
- SARSA walks safely away from the cliff (accounts for accidental falls)
- Q-Learning walks along the cliff edge (optimal but risky during learning)

In ASP: SARSA is safer during training, Q-Learning finds the better final policy.
""",
        "algo_labels": {"td0": "TD(0)", "sarsa": "SARSA", "qlearning": "Q-Learning"},
        "pros_list": {
            "td0":       ["Online learning — updates every step", "No model needed", "Lower variance than MC"],
            "sarsa":     ["Safe during learning", "On-policy — consistent with behaviour", "Converges to optimal epsilon-soft policy"],
            "qlearning": ["Directly learns Q*", "Off-policy — can learn from any data", "Converges to optimal greedy policy"],
        },
        "cons_list": {
            "td0":       ["Only predicts V^pi — needs separate policy", "Biased (bootstrapping)", "Sensitive to alpha"],
            "sarsa":     ["Suboptimal if epsilon stays high", "On-policy — needs epsilon > 0", "Slower than Q-Learning in safe environments"],
            "qlearning": ["Can be risky during learning", "Overestimates Q values (maximisation bias)", "Sensitive to alpha"],
        },
    },
        "DE": {
        "title": "Kapitel 06 — Temporale Differenzlernen",
        "subtitle": "TD(0) — SARSA — Q-Learning — ASP Warschau",
        "engine_missing": "Ausführen: `cd rlvr-py && maturin develop`",
        "sidebar_title": "TD-Einstellungen",
        "n_episodes": "Episoden", "gamma": "γ — Diskontierungsfaktor",
        "alpha": "α — Lernrate", "epsilon": "ε — Anfängliche Exploration",
        "epsilon_decay": "ε-Abklingrate", "seed": "Zufallsseed",
        "run_btn": "▶ TD(0), SARSA und Q-Learning starten",
        "guide_title": "Anleitung",
        "guide": """
**Schritt 1**
TD aktualisiert nach JEDEM Schritt (nicht nach Episodenende wie MC).

**Schritt 2**
SARSA = On-Policy.
Q-Learning = Off-Policy.

**Schritt 3**
α (Lernrate) einstellen.
α=0.1 ist ein guter Start.

**Schritt 4**
Klicken, um alle drei Algorithmen zu starten.

**Schritt 5**
TD-Fehlerkurve lesen — sollte gegen null gehen.

**Schritt 6**
Q-Learning vs SARSA-Strategien vergleichen.""",
        "returns_title": "Episodenrückgaben — TD(0), SARSA, Q-Learning",
        "returns_caption": "Gleitender Durchschnitt. Q-Learning sollte am schnellsten konvergieren.",
        "td_error_title": "TD-Fehler — |R + γV(s') - V(s)|",
        "td_error_caption": "TD-Fehler nimmt ab, wenn der Agent lernt.",
        "value_title": "Wertfunktion V(s) — TD vs. DP-Referenz",
        "value_caption": "TD-Schätzungen sollten zur DP-Lösung (Ch04) konvergieren.",
        "policy_title": "Optimale Strategie — SARSA vs. Q-Learning",
        "policy_caption": "Q-Learning findet optimale Strategie. SARSA findet sicherste.",
        "qtable_title": "Q-Tabellen-Heatmap",
        "qtable_caption": "Q(s,a)-Werte. Algorithmus auswählen.",
        "glass_title": "Glass-Box — TD-Update-Protokoll",
        "glass_headers": ["Episode", "Schritt", "Zustand", "Aktion", "Belohnung", "Nächster Zustand", "TD-Fehler"],
        "summary_title": "Zusammenfassung",
        "summary_results": "Algorithmenvergleich",
        "summary_pros_cons": "TD-Algorithmen — Vor- & Nachteile",
        "pros": "Vorteile", "cons": "Nachteile",
        "theory_title": "Theorie — Kapitel 06",
        "theory_sections": {
            "td_intro":   "6.1 Temporales Differenzlernen",
            "td0":        "6.1 TD(0)-Vorhersage",
            "sarsa":      "6.2 SARSA — On-Policy TD-Kontrolle",
            "qlearning":  "6.3 Q-Learning — Off-Policy TD-Kontrolle",
            "comparison": "6.4 SARSA vs. Q-Learning",
        },
        "theory_td_intro": r"""**Temporales Differenzlernen (TD)** kombiniert MC und DP.
TD-Fehler: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$""",
        "theory_td0": r"$V(S_t) \leftarrow V(S_t) + lpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$",
        "theory_sarsa": r"$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + lpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]$",
        "theory_qlearning": r"$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + lpha[R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)]$",
        "theory_comparison": "SARSA: On-Policy, sicher. Q-Learning: Off-Policy, optimal.",
        "algo_labels": {"td0": "TD(0)", "sarsa": "SARSA", "qlearning": "Q-Learning"},
        "pros_list": {
            "td0":       ["Online-Lernen", "Kein Modell nötig", "Geringere Varianz als MC"],
            "sarsa":     ["Sicher während des Lernens", "On-Policy", "Konvergiert zur optimalen ε-weichen Strategie"],
            "qlearning": ["Lernt Q* direkt", "Off-Policy", "Konvergiert zur optimalen gierigen Strategie"],
        },
        "cons_list": {
            "td0":       ["Sagt nur V^π vorher", "Verzerrt (Bootstrapping)", "Empfindlich gegenüber α"],
            "sarsa":     ["Suboptimal bei hohem ε", "On-Policy erfordert ε > 0"],
            "qlearning": ["Kann während des Lernens riskant sein", "Maximierungsverzerrung"],
        },
    },
    "FR": {
        "title": "Chapitre 06 — Apprentissage par Différences Temporelles",
        "subtitle": "TD(0) · SARSA · Q-Learning · ASP · Région de Varsovie",
        "engine_missing": "Exécutez : `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Paramètres TD",
        "n_episodes": "Nombre d'épisodes",
        "gamma": "γ — Facteur d'actualisation",
        "alpha": "α — Taux d'apprentissage",
        "epsilon": "ε — Exploration initiale",
        "epsilon_decay": "Taux de décroissance ε",
        "seed": "Graine aléatoire",
        "run_btn": "▶ Lancer TD(0), SARSA et Q-Learning",
        "guide_title": "🎓 Comment utiliser ce chapitre",
        "guide": "TD met à jour après chaque étape (pas après l'épisode). SARSA = on-policy. Q-Learning = off-policy.",
        "returns_title": "📈 Retours par épisode",
        "returns_caption": "Moyenne mobile. Q-Learning devrait converger le plus vite.",
        "td_error_title": "📉 Erreur TD",
        "td_error_caption": "L'erreur TD décroît au fil de l'apprentissage.",
        "value_title": "📊 Fonction de valeur V(s)",
        "value_caption": "Les estimations TD convergent vers la solution DP (Ch04).",
        "policy_title": "🎯 Politique optimale — SARSA vs Q-Learning",
        "policy_caption": "Q-Learning trouve la politique optimale. SARSA trouve la plus sûre.",
        "qtable_title": "📊 Table Q",
        "qtable_caption": "Valeurs Q(s,a). Sélectionnez l'algorithme.",
        "glass_title": "🔬 Glass-Box — Trace de mise à jour TD",
        "glass_headers": ["Épisode", "Étape", "État", "Action", "Récompense", "État suivant", "Erreur TD"],
        "summary_title": "📊 Résumé",
        "summary_results": "Comparaison des algorithmes",
        "summary_pros_cons": "Algorithmes TD — Avantages & Inconvénients",
        "pros": "✅ Avantages", "cons": "❌ Inconvénients",
        "theory_title": "📖 Théorie — Chapitre 06",
        "theory_sections": {
            "td_intro": "§6.1 Apprentissage par différences temporelles",
            "td0": "§6.1 Prédiction TD(0)",
            "sarsa": "§6.2 SARSA — Contrôle TD on-policy",
            "qlearning": "§6.3 Q-Learning — Contrôle TD off-policy",
            "comparison": "§6.4 SARSA vs Q-Learning",
        },
        "theory_td_intro": "δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)",
        "theory_td0": "V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]",
        "theory_sarsa": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]",
        "theory_qlearning": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ max_a' Q(S_{t+1},a') - Q(S_t,A_t)]",
        "theory_comparison": "SARSA: on-policy, sûr. Q-Learning: off-policy, optimal.",
        "algo_labels": {"td0": "TD(0)", "sarsa": "SARSA", "qlearning": "Q-Learning"},
        "pros_list": {"td0": ["En ligne", "Sans modèle"], "sarsa": ["Sûr pendant l'apprentissage", "On-policy"], "qlearning": ["Apprend Q* directement", "Off-policy"]},
        "cons_list": {"td0": ["Prédit seulement V^pi", "Biaisé"], "sarsa": ["Sous-optimal si ε élevé"], "qlearning": ["Risqué pendant l'apprentissage", "Biais de maximisation"]},
    },
    "ES": {
        "title": "Capítulo 06 — Aprendizaje por Diferencias Temporales",
        "subtitle": "TD(0) · SARSA · Q-Learning · ASP · Región de Varsovia",
        "engine_missing": "Ejecute: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Configuración TD",
        "n_episodes": "Número de episodios",
        "gamma": "γ — Factor de descuento",
        "alpha": "α — Tasa de aprendizaje",
        "epsilon": "ε — Exploración inicial",
        "epsilon_decay": "Tasa de decaimiento ε",
        "seed": "Semilla aleatoria",
        "run_btn": "▶ Ejecutar TD(0), SARSA y Q-Learning",
        "guide_title": "🎓 Cómo usar este capítulo",
        "guide": "TD actualiza después de cada paso. SARSA = on-policy. Q-Learning = off-policy.",
        "returns_title": "📈 Retornos por episodio",
        "returns_caption": "Media móvil. Q-Learning debería converger más rápido.",
        "td_error_title": "📉 Error TD",
        "td_error_caption": "El error TD decrece con el aprendizaje.",
        "value_title": "📊 Función de valor V(s)",
        "value_caption": "Las estimaciones TD convergen hacia la solución DP (Ch04).",
        "policy_title": "🎯 Política óptima — SARSA vs Q-Learning",
        "policy_caption": "Q-Learning encuentra la política óptima. SARSA la más segura.",
        "qtable_title": "📊 Tabla Q",
        "qtable_caption": "Valores Q(s,a). Seleccione el algoritmo.",
        "glass_title": "🔬 Glass-Box — Traza de actualización TD",
        "glass_headers": ["Episodio", "Paso", "Estado", "Acción", "Recompensa", "Siguiente estado", "Error TD"],
        "summary_title": "📊 Resumen",
        "summary_results": "Comparación de algoritmos",
        "summary_pros_cons": "Algoritmos TD — Pros y Contras",
        "pros": "✅ Pros", "cons": "❌ Contras",
        "theory_title": "📖 Teoría — Capítulo 06",
        "theory_sections": {
            "td_intro": "§6.1 Aprendizaje por diferencias temporales",
            "td0": "§6.1 Predicción TD(0)",
            "sarsa": "§6.2 SARSA — Control TD on-policy",
            "qlearning": "§6.3 Q-Learning — Control TD off-policy",
            "comparison": "§6.4 SARSA vs Q-Learning",
        },
        "theory_td_intro": "δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)",
        "theory_td0": "V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]",
        "theory_sarsa": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]",
        "theory_qlearning": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ max_a' Q(S_{t+1},a') - Q(S_t,A_t)]",
        "theory_comparison": "SARSA: on-policy, seguro. Q-Learning: off-policy, óptimo.",
        "algo_labels": {"td0": "TD(0)", "sarsa": "SARSA", "qlearning": "Q-Learning"},
        "pros_list": {"td0": ["En línea", "Sin modelo"], "sarsa": ["Seguro durante aprendizaje", "On-policy"], "qlearning": ["Aprende Q* directamente", "Off-policy"]},
        "cons_list": {"td0": ["Solo predice V^pi", "Sesgado"], "sarsa": ["Subóptimo si ε alto"], "qlearning": ["Arriesgado durante aprendizaje", "Sesgo de maximización"]},
    },
    "PL": {
        "title": "Rozdział 06 — Uczenie przez Różnice Temporalne",
        "subtitle": "TD(0) · SARSA · Q-Learning · Dyspozytura ASP · Region Warszawy",
        "engine_missing": "Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Ustawienia TD",
        "n_episodes": "Liczba epizodów",
        "gamma": "γ — Współczynnik dyskontowania",
        "alpha": "α — Współczynnik uczenia",
        "epsilon": "ε — Eksploracja początkowa",
        "epsilon_decay": "Współczynnik zaniku ε",
        "seed": "Ziarno losowości",
        "run_btn": "▶ Uruchom TD(0), SARSA i Q-Learning",
        "guide_title": "🎓 Jak korzystać z tego rozdziału",
        "guide": """
**Krok 1**
TD aktualizuje po KAŻDYM kroku (nie po epizodzie jak MC).

**Krok 2**
SARSA = on-policy (używa tej samej polityki do uczenia i zachowania).

**Krok 3**
Q-Learning = off-policy (uczy się optymalnej polityki niezależnie od zachowania).

**Krok 4**
Ustaw α (współczynnik uczenia).
α=0.1 to dobry start.

**Krok 5**
Kliknij ▶ aby uruchomić wszystkie trzy algorytmy.

**Krok 6**
Odczytaj krzywą błędu TD — powinna maleć w czasie.

**Krok 7**
Porównaj polityki SARSA vs Q-Learning — Q-Learning powinien znaleźć lepszą.
""",
        "returns_title": "📈 Zwroty epizodów — TD(0), SARSA, Q-Learning",
        "returns_caption": "Średnia krocząca zwrotów. Q-Learning powinien zbiegać najszybciej.",
        "td_error_title": "📉 Błąd TD — |R + γV(s') - V(s)|",
        "td_error_caption": "Błąd TD maleje w miarę uczenia się agenta.",
        "value_title": "📊 Funkcja wartości V(s) — TD vs referencja DP",
        "value_caption": "Estymaty TD zbiegają do rozwiązania DP (Ch04) bez modelu P.",
        "policy_title": "🎯 Optymalna polityka — SARSA vs Q-Learning",
        "policy_caption": "Q-Learning znajduje optymalną politykę. SARSA — najbezpieczniejszą.",
        "qtable_title": "📊 Tabela Q",
        "qtable_caption": "Wartości Q(s,a). Wybierz algorytm.",
        "glass_title": "🔬 Glass-Box — Ślad aktualizacji TD",
        "glass_headers": ["Epizod", "Krok", "Stan", "Akcja", "Nagroda", "Następny stan", "Błąd TD"],
        "summary_title": "📊 Podsumowanie",
        "summary_results": "Porównanie algorytmów",
        "summary_pros_cons": "Algorytmy TD — Zalety i Wady",
        "pros": "✅ Zalety", "cons": "❌ Wady",
        "theory_title": "📖 Teoria — Rozdział 06",
        "theory_sections": {
            "td_intro":  "§6.1 Uczenie przez różnice temporalne",
            "td0":       "§6.1 Predykcja TD(0)",
            "sarsa":     "§6.2 SARSA — On-policy TD Control",
            "qlearning": "§6.3 Q-Learning — Off-policy TD Control",
            "comparison":"§6.4 SARSA vs Q-Learning",
        },
        "theory_td_intro": r"""
**Uczenie przez różnice temporalne (TD)** łączy MC i DP:
- Jak MC: bez modelu, uczy z doświadczenia
- Jak DP: bootstrapping — aktualizuje używając bieżących estymат

Błąd TD (delta): δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)
Implementacja: `ch06_td.rs`
""",
        "theory_td0": "V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]",
        "theory_sarsa": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]",
        "theory_qlearning": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ max_a' Q(S_{t+1},a') - Q(S_t,A_t)]",
        "theory_comparison": r"""
**Jedna linia różnicy:**
SARSA:      Q(s,a) += α [R + γ Q(s', a') - Q(s,a)]   gdzie a' ~ ε-zachłanna
Q-Learning: Q(s,a) += α [R + γ max_a' Q(s', a') - Q(s,a)]

SARSA jest bezpieczniejszy podczas uczenia (uwzględnia eksplorację).
Q-Learning znajduje lepszą politykę końcową (bezpośrednio optymalizuje Q*).
""",
        "algo_labels": {"td0": "TD(0)", "sarsa": "SARSA", "qlearning": "Q-Learning"},
        "pros_list": {
            "td0":       ["Uczenie online — aktualizacja po każdym kroku", "Bez modelu", "Niższa wariancja niż MC"],
            "sarsa":     ["Bezpieczny podczas uczenia", "On-policy — spójny z zachowaniem", "Zbiega do optymalnej polityki epsilon-soft"],
            "qlearning": ["Bezpośrednio uczy Q*", "Off-policy — może uczyć z dowolnych danych", "Zbiega do optymalnej zachłannej polityki"],
        },
        "cons_list": {
            "td0":       ["Tylko predykcja V^pi", "Obciążony (bootstrapping)", "Wrażliwy na α"],
            "sarsa":     ["Suboptymalny gdy ε wysokie", "On-policy — wymaga ε > 0"],
            "qlearning": ["Ryzykowny podczas uczenia", "Przeszacowuje Q (bias maksymalizacji)", "Wrażliwy na α"],
        },
    },
}

COLORS = {"td0": "#0082F0", "sarsa": "#FF8C0A", "qlearning": "#0FC373"}

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
    n_episodes    = st.sidebar.slider(tx["n_episodes"],    50, 5000, 1000, 50)
    gamma         = st.sidebar.slider(tx["gamma"],         0.5, 0.999, 0.95, 0.005)
    alpha         = st.sidebar.slider(tx["alpha"],         0.01, 1.0, 0.1, 0.01)
    epsilon       = st.sidebar.slider(tx["epsilon"],       0.0, 1.0, 0.3, 0.05)
    epsilon_decay = st.sidebar.slider(tx["epsilon_decay"], 0.0, 0.1, 0.01, 0.001, format="%.3f")
    seed          = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    with st.expander(tx["guide_title"], expanded=False):
        st.markdown(tx["guide"])

    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running Rust TD engine..."):
            result = rlvr_py.run_ch06_td(
                int(seed), int(n_episodes), float(gamma),
                float(alpha), float(epsilon), float(epsilon_decay)
            )
        st.session_state["ch06_result"] = result

    if "ch06_result" not in st.session_state:
        st.info("Configure settings and click **▶ Run TD(0), SARSA and Q-Learning**.")
        _render_theory(tx)
        return

    result       = st.session_state["ch06_result"]
    state_names  = result["state_names"]
    action_names = result["action_names"]
    algos        = ["td0", "sarsa", "qlearning"]

    # KPI
    cols = st.columns(3)
    for i, key in enumerate(algos):
        r = result[key]
        avg = sum(r["returns_curve"][-50:]) / min(50, len(r["returns_curve"]))
        cols[i].metric(tx["algo_labels"][key],
                       f"Avg return: {avg:.2f}",
                       f"Steps: {r['total_steps']:,}")

    # Returns
    st.subheader(tx["returns_title"])
    fig = go.Figure()
    for key in algos:
        ma = _moving_avg(result[key]["returns_curve"], 30)
        fig.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
            mode="lines", name=tx["algo_labels"][key],
            line=dict(color=COLORS[key], width=2)))
    fig.update_layout(height=300, margin=dict(l=40,r=20,t=20,b=40),
                      xaxis_title="Episode", yaxis_title="Return (MA-30)",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch')
    st.caption(tx["returns_caption"])

    # TD Error
    st.subheader(tx["td_error_title"])
    fig2 = go.Figure()
    for key in ["sarsa", "qlearning"]:
        ma = _moving_avg(result[key]["td_error_curve"], 30)
        fig2.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
            mode="lines", name=tx["algo_labels"][key],
            line=dict(color=COLORS[key], width=2)))
    fig2.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
                       xaxis_title="Episode", yaxis_title="Avg TD Error",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig2, width='stretch')
    st.caption(tx["td_error_caption"])

    # Value function
    st.subheader(tx["value_title"])
    short = [f"S{i}" for i in range(result["n_states"])]
    fig3 = go.Figure()
    for key in algos:
        fig3.add_trace(go.Bar(x=short, y=result[key]["values"],
            name=tx["algo_labels"][key], marker_color=COLORS[key], opacity=0.8))
    fig3.update_layout(height=300, barmode="group",
                       margin=dict(l=40,r=20,t=20,b=40),
                       legend=dict(orientation="h"))
    st.plotly_chart(fig3, width='stretch')
    st.caption(tx["value_caption"])

    # Policy comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(tx["policy_title"])
        rows = []
        for s in range(result["n_states"]):
            sa = result["sarsa"]["policy"][s]
            ql = result["qlearning"]["policy"][s]
            rows.append({
                "State": f"S{s}",
                "SARSA": f"A{sa}",
                "Q-Learning": f"A{ql}",
                "Match": "✅" if sa == ql else "🔄",
            })
        st.dataframe(rows, hide_index=True)
        st.caption(tx["policy_caption"])

    with col2:
        st.subheader(tx["qtable_title"])
        algo_sel = st.selectbox("Algorithm", [tx["algo_labels"][k] for k in ["sarsa","qlearning"]])
        key_sel = "sarsa" if "SARSA" in algo_sel or "sarsa" in algo_sel.lower() else "qlearning"
        qt = result[key_sel]["q_table"]
        action_short = [f"A{i}" for i in range(result["n_actions"])]
        fig4 = go.Figure(go.Heatmap(
            z=qt, x=action_short, y=short,
            colorscale="Blues",
            text=[[f"{qt[s][a]:.2f}" for a in range(result["n_actions"])]
                  for s in range(result["n_states"])],
            texttemplate="%{text}",
        ))
        fig4.update_layout(height=320, margin=dict(l=60,r=20,t=20,b=40))
        st.plotly_chart(fig4, width='stretch')
        st.caption(tx["qtable_caption"])

    # Glass-Box
    st.subheader(tx["glass_title"])
    _render_glass_box(result, tx)

    # Summary
    st.subheader(tx["summary_title"])
    _render_summary(result, tx, algos)

    _render_theory(tx)


def _render_glass_box(result, tx):
    algo_options = {tx["algo_labels"][k]: k for k in ["td0","sarsa","qlearning"]}
    selected = st.selectbox("Algorithm", list(algo_options.keys()), key="gb_algo")
    key = algo_options[selected]
    r = result[key]
    ep_idx = st.slider("Episode", 0, max(len(r["returns_curve"])-1, 0),
                       max(len(r["returns_curve"])-1, 0), key="gb_ep")
    st.metric("Episode return", f"{r['returns_curve'][ep_idx]:.3f}")
    st.metric("Avg TD error this episode", f"{r['td_error_curve'][ep_idx]:.4f}")
    if key == "sarsa":
        st.latex(r"Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]")
    elif key == "qlearning":
        st.latex(r"Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)]")
    else:
        st.latex(r"V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]")


def _render_summary(result, tx, algos):
    st.markdown(f"#### {tx['summary_results']}")
    rows = []
    for key in algos:
        r = result[key]
        avg = sum(r["returns_curve"][-100:]) / min(100, len(r["returns_curve"]))
        rows.append({
            "Algorithm":           tx["algo_labels"][key],
            "Avg return (last 100)": f"{avg:.3f}",
            "Total steps":         str(r["total_steps"]),
            "V*(S0)":              f"{r['values'][0]:.3f}",
            "V*(S7)":              f"{r['values'][7]:.3f}",
            "Policy S7":           f"A{r['policy'][7]}",
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
    for key in ["td_intro","td0","sarsa","qlearning","comparison"]:
        with st.expander(tx["theory_sections"][key], expanded=False):
            st.markdown(tx[f"theory_{key}"])
