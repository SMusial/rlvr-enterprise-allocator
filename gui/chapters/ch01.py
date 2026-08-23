
import streamlit as st
import plotly.graph_objects as go
import json

# ---------------------------------------------------------------------------
# Language strings — EN, PL, FR, DE, ES
# ---------------------------------------------------------------------------
T = {
    "EN": {
        "title": "Chapter 01 — ASP Dispatch: Introduction to RL",
        "subtitle": "Field Service Optimisation via Reinforcement Learning · Warsaw Region",
        "engine_ok": "⚙️ Rust engine active",
        "engine_missing": "⚙️ Rust engine not found. Run: `cd rlvr-py && maturin develop`",
        "lang_label": "🌐 Language",
        "sidebar_title": "⚙️ Episode Settings",
        "n_tech": "Technicians",
        "n_orders": "Work Orders",
        "epsilon": "ε — Exploration rate",
        "gamma": "γ — Discount factor",
        "seed": "Random seed",
        "n_episodes": "Episodes (learning curve)",
        "run_btn": "▶ Run Episode",
        "guide_title": "🎓 How to use this chapter",
        "guide": """
**Step 1 — Set ε (exploration rate)**
Move the slider. ε=1.0 means the agent picks randomly every time (pure exploration).
ε=0.0 means it always picks the best known action (pure exploitation — but since the
Q-table is all zeros in Ch01, this is also random). Try ε=0.5 to start.

**Step 2 — Set technicians and work orders**
5 technicians / 10 work orders is a good starting point. More orders = longer episode.

**Step 3 — Click ▶ Run Episode**
The Rust engine runs the full MDP loop and returns every step.

**Step 4 — Read the Warsaw map**
Blue markers = technicians (T0–T4). Coloured markers = work orders (W0–W9).
Green lines = SLA met. Red lines = SLA breached. Click any marker for details.

**Step 5 — Use the Step slider**
Move it to highlight a specific dispatch decision on the map and in the Glass-Box.

**Step 6 — Read the Glass-Box**
Every row shows the full MDP tuple: Sₜ (state), Aₜ (action), Rₜ (reward), Gₜ (return).
The Bellman equation is shown greyed out — it activates in Chapter 02.

**Step 7 — Read the Episode Summary**
Quantified business results + pros/cons of the ε-greedy method used in this chapter.
""",
        "map_title": "📍 Warsaw Dispatch Map",
        "map_caption": "Blue = Technicians · Amber/Red = Work Orders · Green = SLA met · Red = SLA breach",
        "step_slider": "🔍 Highlight step",
        "glass_title": "🔬 Glass-Box Inspector — MDP Step Trace",
        "glass_headers": ["Step", "Tech", "Order", "Skill Match", "Distance", "Urgency", "Reward Rₜ", "Return Gₜ", "SLA", "Mode"],
        "sla_met": "✅ Met",
        "sla_breach": "❌ Breach",
        "skill_ok": "✅ Match",
        "skill_no": "⚠️ Mismatch",
        "explore": "🔍 Explore",
        "exploit": "🎯 Exploit",
        "bellman_caption": "Bellman equation — activates in Chapter 02 when Q-table updates are introduced",
        "curve_title": "📈 Learning Curve — Gₜ over Episodes",
        "curve_x": "Episode",
        "curve_y": "Total Discounted Return Gₜ",
        "curve_mean": "Rolling mean (5 ep)",
        "theory_title": "📖 Theory — Chapter 01",
        "theory_sections": {
            "mdp": "§1.1 The MDP Framework",
            "egreedy": "§1.1 ε-Greedy Policy",
            "gt": "§1.1 Discounted Return Gₜ",
            "ndarray": "§1.2.1 ndarray Q-Table",
            "reward": "§1.1 Reward Design",
        },
        "theory_mdp": r"""
**The MDP tuple (S, A, P, R, γ)** is the mathematical foundation of every RL system.

- **S** — State space: technician positions, skills, availability; work order locations, urgency
- **A** — Action space: assign technician i to work order j
- **P(s'|s,a)** — Transition: next state depends *only* on current state + action (Markov property)
- **R(s,a)** — Reward: +10 SLA met, −5 breach, −2 skill mismatch, −0.1×km distance
- **γ** — Discount factor: how much future rewards are worth vs immediate ones

**Markov property** (implemented in `transition()` in `ch01_asp_dispatch.rs`):
$$P(s_{t+1} | s_t, a_t, s_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)$$

The future depends only on *now* — not on history. This makes the problem tractable.
""",
        "theory_egreedy": r"""
**ε-greedy** is the simplest exploration strategy — the one used in this chapter.

$$a_t = \begin{cases} \text{random action} & \text{with probability } \varepsilon \\ \arg\max_a Q(s,a) & \text{with probability } 1-\varepsilon \end{cases}$$

- High ε → agent explores (tries new dispatches, gathers information)
- Low ε → agent exploits (uses best known dispatch from Q-table)
- In Ch01 the Q-table is all zeros, so exploit = random too
- In Ch02 the Q-table gets trained — then exploitation becomes meaningful

Implemented in `epsilon_greedy()` in `ch01_asp_dispatch.rs`.
""",
        "theory_gt": r"""
**Discounted return Gₜ** measures the total value of a decision, accounting for future consequences.

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k} = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \ldots$$

- γ close to 1 → agent is *farsighted* (values future rewards almost as much as immediate)
- γ close to 0 → agent is *myopic* (only cares about immediate reward)
- The Glass-Box shows Gₜ for every step — computed backward from episode end

Implemented in `discounted_return()` in `ch01_asp_dispatch.rs`.
""",
        "theory_ndarray": r"""
**ndarray Q-table** — the data structure that stores learned action values.

```rust
let mut q_table = Array2::<f64>::zeros((n_tech, n_orders));
```

- Rows = technicians (states), Columns = work orders (actions)
- Value Q(s,a) = expected cumulative reward for assigning tech s to order a
- In Ch01: all zeros (untrained) — the agent has no prior knowledge
- In Ch02: Q-values update via the Bellman equation after every step

The `ndarray` crate (`Array2<f64>`) is the Rust equivalent of a NumPy 2D array.
""",
        "theory_reward": r"""
**Reward design** directly shapes agent behaviour — poorly designed rewards lead to unintended strategies.

Our reward function in `ch01_asp_dispatch.rs`:

| Condition | Reward |
|---|---|
| SLA met | +10.0 |
| SLA breach | −5.0 |
| Skill mismatch | −2.0 |
| Distance penalty | −0.1 × km |

**SLA failure probability** is realistic (not always 100%):
- Base failure rate: 8%
- Skill mismatch adds: +35%
- High urgency (>0.7) adds: +20%
- Distance >15km adds: +18%
- Distance >25km adds: +12%

This produces realistic SLA rates of 77–93% depending on dispatch quality.
""",
        "summary_title": "📊 Episode Summary",
        "summary_results": "Quantified Results",
        "summary_pros_cons": "ε-Greedy Method — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "pros_list": [
            "Simple to implement — one parameter ε controls everything",
            "Guaranteed exploration — never gets permanently stuck",
            "Works with zero prior knowledge (all-zero Q-table)",
            "Computationally trivial — O(1) per decision",
            "Good baseline to compare against smarter algorithms (Ch02–Ch09)",
        ],
        "cons_list": [
            "Explores uniformly — wastes time on obviously bad actions",
            "No memory — ignores what it learned in previous steps",
            "Q-table is untrained in Ch01 — exploit = random (no real advantage yet)",
            "Does not scale to large state spaces (Ch15 solves this with neural networks)",
            "ε decay must be tuned manually — wrong decay = premature exploitation",
        ],
        "metric_gt": "Total Return Gₜ",
        "metric_sla": "SLA Rate",
        "metric_skill": "Skill Match Rate",
        "metric_explore": "Exploration Rate",
        "metric_sla_saved": "SLA Penalties Avoided",
        "metric_dist": "Avg Dispatch Distance",
        "metric_reward": "Avg Step Reward",
    },
    "PL": {
        "title": "Rozdział 01 — Dyspozytura ASP: Wprowadzenie do RL",
        "subtitle": "Optymalizacja serwisu terenowego przez uczenie ze wzmocnieniem · Region Warszawy",
        "engine_ok": "⚙️ Silnik Rust aktywny",
        "engine_missing": "⚙️ Silnik Rust nie znaleziony. Uruchom: `cd rlvr-py && maturin develop`",
        "lang_label": "🌐 Język",
        "sidebar_title": "⚙️ Ustawienia epizodu",
        "n_tech": "Technicy",
        "n_orders": "Zlecenia robocze",
        "epsilon": "ε — Współczynnik eksploracji",
        "gamma": "γ — Współczynnik dyskontowania",
        "seed": "Ziarno losowości",
        "n_episodes": "Epizody (krzywa uczenia)",
        "run_btn": "▶ Uruchom epizod",
        "guide_title": "🎓 Jak korzystać z tego rozdziału",
        "guide": """
**Krok 1 — Ustaw ε (współczynnik eksploracji)**
Przesuń suwak. ε=1.0 oznacza losowy wybór (czysta eksploracja).
ε=0.0 oznacza zawsze najlepszą znaną akcję (eksploatacja — ale w Ch01 tabela Q jest zerowa,
więc to też jest losowe). Zacznij od ε=0.5.

**Krok 2 — Ustaw techników i zlecenia**
5 techników / 10 zleceń to dobry punkt startowy.

**Krok 3 — Kliknij ▶ Uruchom epizod**
Silnik Rust wykonuje pełną pętlę MDP i zwraca każdy krok.

**Krok 4 — Odczytaj mapę Warszawy**
Niebieskie markery = technicy (T0–T4). Kolorowe markery = zlecenia (Z0–Z9).
Zielone linie = SLA spełnione. Czerwone linie = naruszenie SLA.

**Krok 5 — Użyj suwaka kroków**
Przesuń, aby podświetlić konkretną decyzję dyspozytury na mapie i w Glass-Box.

**Krok 6 — Odczytaj Glass-Box**
Każdy wiersz pokazuje pełną krotkę MDP: Sₜ, Aₜ, Rₜ, Gₜ.

**Krok 7 — Odczytaj podsumowanie epizodu**
Wymierne wyniki biznesowe + zalety i wady metody ε-zachłannej.
""",
        "map_title": "📍 Mapa dyspozytury Warszawa",
        "map_caption": "Niebieski = Technicy · Bursztynowy/Czerwony = Zlecenia · Zielony = SLA OK · Czerwony = Naruszenie SLA",
        "step_slider": "🔍 Podświetl krok",
        "glass_title": "🔬 Inspektor Glass-Box — Ślad kroków MDP",
        "glass_headers": ["Krok", "Tech", "Zlecenie", "Dopasowanie", "Odległość", "Pilność", "Nagroda Rₜ", "Zwrot Gₜ", "SLA", "Tryb"],
        "sla_met": "✅ Spełnione",
        "sla_breach": "❌ Naruszenie",
        "skill_ok": "✅ Dopasowanie",
        "skill_no": "⚠️ Niedopasowanie",
        "explore": "🔍 Eksploracja",
        "exploit": "🎯 Eksploatacja",
        "bellman_caption": "Równanie Bellmana — aktywuje się w Rozdziale 02 po wprowadzeniu aktualizacji tabeli Q",
        "curve_title": "📈 Krzywa uczenia — Gₜ w kolejnych epizodach",
        "curve_x": "Epizod",
        "curve_y": "Łączny zdyskontowany zwrot Gₜ",
        "curve_mean": "Średnia krocząca (5 ep)",
        "theory_title": "📖 Teoria — Rozdział 01",
        "theory_sections": {
            "mdp": "§1.1 Framework MDP",
            "egreedy": "§1.1 Polityka ε-zachłanna",
            "gt": "§1.1 Zdyskontowany zwrot Gₜ",
            "ndarray": "§1.2.1 Tabela Q ndarray",
            "reward": "§1.1 Projektowanie nagrody",
        },
        "theory_mdp": r"""
**Krotka MDP (S, A, P, R, γ)** to matematyczna podstawa każdego systemu RL.

- **S** — Przestrzeń stanów: pozycje techników, umiejętności, dostępność; lokalizacje zleceń, pilność
- **A** — Przestrzeń akcji: przypisz technika i do zlecenia j
- **P(s'|s,a)** — Przejście: następny stan zależy *tylko* od bieżącego stanu + akcji (własność Markowa)
- **R(s,a)** — Nagroda: +10 SLA spełnione, −5 naruszenie, −2 niedopasowanie, −0.1×km odległość
- **γ** — Współczynnik dyskontowania: jak bardzo przyszłe nagrody są warte w porównaniu z natychmiastowymi

**Własność Markowa** (zaimplementowana w `transition()` w `ch01_asp_dispatch.rs`):
$$P(s_{t+1} | s_t, a_t, s_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)$$
""",
        "theory_egreedy": r"""
**ε-zachłanna** to najprostsza strategia eksploracji — używana w tym rozdziale.

$$a_t = \begin{cases} \text{losowa akcja} & \text{z prawdopodobieństwem } \varepsilon \\ \arg\max_a Q(s,a) & \text{z prawdopodobieństwem } 1-\varepsilon \end{cases}$$

- Wysokie ε → agent eksploruje (próbuje nowych dyspozycji)
- Niskie ε → agent eksploatuje (używa najlepszej znanej dyspozycji z tabeli Q)
- W Ch01 tabela Q jest zerowa — eksploatacja = losowa
- W Ch02 tabela Q jest trenowana — eksploatacja staje się sensowna
""",
        "theory_gt": r"""
**Zdyskontowany zwrot Gₜ** mierzy całkowitą wartość decyzji uwzględniając przyszłe konsekwencje.

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k} = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \ldots$$

- γ bliskie 1 → agent jest *dalekowzroczny*
- γ bliskie 0 → agent jest *krótkowzroczny*
""",
        "theory_ndarray": r"""
**Tabela Q ndarray** — struktura danych przechowująca wyuczone wartości akcji.

```rust
let mut q_table = Array2::<f64>::zeros((n_tech, n_orders));
```

- Wiersze = technicy (stany), Kolumny = zlecenia (akcje)
- W Ch01: same zera (niewyuczona) — agent nie ma wiedzy a priori
- W Ch02: wartości Q aktualizują się przez równanie Bellmana
""",
        "theory_reward": r"""
**Projektowanie nagrody** bezpośrednio kształtuje zachowanie agenta.

| Warunek | Nagroda |
|---|---|
| SLA spełnione | +10.0 |
| Naruszenie SLA | −5.0 |
| Niedopasowanie umiejętności | −2.0 |
| Kara za odległość | −0.1 × km |

Realistyczne wskaźniki SLA: 77–93% w zależności od jakości dyspozycji.
""",
        "summary_title": "📊 Podsumowanie epizodu",
        "summary_results": "Wymierne wyniki",
        "summary_pros_cons": "Metoda ε-zachłanna — Zalety i Wady",
        "pros": "✅ Zalety",
        "cons": "❌ Wady",
        "pros_list": [
            "Prosta implementacja — jeden parametr ε kontroluje wszystko",
            "Gwarantowana eksploracja — nigdy nie utknięcie na stałe",
            "Działa bez wiedzy a priori (zerowa tabela Q)",
            "Trywialnie obliczeniowa — O(1) na decyzję",
            "Dobry punkt odniesienia do porównania z lepszymi algorytmami (Ch02–Ch09)",
        ],
        "cons_list": [
            "Eksploruje równomiernie — marnuje czas na oczywiste złe akcje",
            "Brak pamięci — ignoruje to, czego nauczyła się w poprzednich krokach",
            "Tabela Q niewyuczona w Ch01 — eksploatacja = losowa",
            "Nie skaluje się do dużych przestrzeni stanów (Ch15 rozwiązuje to sieciami neuronowymi)",
            "Zanik ε musi być ręcznie dostrojony",
        ],
        "metric_gt": "Łączny zwrot Gₜ",
        "metric_sla": "Wskaźnik SLA",
        "metric_skill": "Wskaźnik dopasowania umiejętności",
        "metric_explore": "Wskaźnik eksploracji",
        "metric_sla_saved": "Uniknięte kary SLA",
        "metric_dist": "Śr. odległość dyspozycji",
        "metric_reward": "Śr. nagroda za krok",
    },
    "FR": {
        "title": "Chapitre 01 — Dispatch ASP : Introduction au RL",
        "subtitle": "Optimisation du service terrain par apprentissage par renforcement · Région de Varsovie",
        "engine_ok": "⚙️ Moteur Rust actif",
        "engine_missing": "⚙️ Moteur Rust introuvable. Exécutez : `cd rlvr-py && maturin develop`",
        "lang_label": "🌐 Langue",
        "sidebar_title": "⚙️ Paramètres d'épisode",
        "n_tech": "Techniciens",
        "n_orders": "Ordres de travail",
        "epsilon": "ε — Taux d'exploration",
        "gamma": "γ — Facteur d'actualisation",
        "seed": "Graine aléatoire",
        "n_episodes": "Épisodes (courbe d'apprentissage)",
        "run_btn": "▶ Lancer l'épisode",
        "guide_title": "🎓 Comment utiliser ce chapitre",
        "guide": """
**Étape 1 — Réglez ε** : ε=1.0 = exploration pure, ε=0.0 = exploitation pure.
**Étape 2 — Réglez techniciens et ordres** : 5/10 est un bon point de départ.
**Étape 3 — Cliquez ▶ Lancer l'épisode** : le moteur Rust exécute la boucle MDP complète.
**Étape 4 — Lisez la carte de Varsovie** : bleu = techniciens, couleur = ordres, vert = SLA respecté.
**Étape 5 — Utilisez le curseur d'étape** pour mettre en évidence une décision spécifique.
**Étape 6 — Lisez le Glass-Box** : chaque ligne montre le tuple MDP complet Sₜ, Aₜ, Rₜ, Gₜ.
**Étape 7 — Lisez le résumé** : résultats quantifiés + avantages/inconvénients de la méthode.
""",
        "map_title": "📍 Carte de dispatch Varsovie",
        "map_caption": "Bleu = Techniciens · Ambre/Rouge = Ordres · Vert = SLA respecté · Rouge = Violation SLA",
        "step_slider": "🔍 Mettre en évidence l'étape",
        "glass_title": "🔬 Inspecteur Glass-Box — Trace des étapes MDP",
        "glass_headers": ["Étape", "Tech", "Ordre", "Compétence", "Distance", "Urgence", "Récompense Rₜ", "Retour Gₜ", "SLA", "Mode"],
        "sla_met": "✅ Respecté",
        "sla_breach": "❌ Violation",
        "skill_ok": "✅ Correspondance",
        "skill_no": "⚠️ Inadéquation",
        "explore": "🔍 Explorer",
        "exploit": "🎯 Exploiter",
        "bellman_caption": "Équation de Bellman — s'active au Chapitre 02",
        "curve_title": "📈 Courbe d'apprentissage — Gₜ par épisode",
        "curve_x": "Épisode",
        "curve_y": "Retour actualisé total Gₜ",
        "curve_mean": "Moyenne mobile (5 ép)",
        "theory_title": "📖 Théorie — Chapitre 01",
        "theory_sections": {
            "mdp": "§1.1 Le cadre MDP",
            "egreedy": "§1.1 Politique ε-greedy",
            "gt": "§1.1 Retour actualisé Gₜ",
            "ndarray": "§1.2.1 Table Q ndarray",
            "reward": "§1.1 Conception de la récompense",
        },
        "theory_mdp": r"""
**Le tuple MDP (S, A, P, R, γ)** est le fondement mathématique de tout système RL.
$$P(s_{t+1} | s_t, a_t) \text{ — propriété de Markov}$$
Implémenté dans `transition()` dans `ch01_asp_dispatch.rs`.
""",
        "theory_egreedy": r"""
**ε-greedy** est la stratégie d'exploration la plus simple.
$$a_t = \begin{cases} \text{action aléatoire} & \text{avec probabilité } \varepsilon \\ \arg\max_a Q(s,a) & \text{avec probabilité } 1-\varepsilon \end{cases}$$
""",
        "theory_gt": r"""
**Retour actualisé Gₜ** mesure la valeur totale d'une décision.
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}$$
""",
        "theory_ndarray": r"""
**Table Q ndarray** — structure de données pour les valeurs d'action apprises.
```rust
let mut q_table = Array2::<f64>::zeros((n_tech, n_orders));
```
""",
        "theory_reward": r"""
**Conception de la récompense** : +10 SLA respecté, −5 violation, −2 inadéquation, −0.1×km distance.
Taux SLA réaliste : 77–93%.
""",
        "summary_title": "📊 Résumé de l'épisode",
        "summary_results": "Résultats quantifiés",
        "summary_pros_cons": "Méthode ε-greedy — Avantages & Inconvénients",
        "pros": "✅ Avantages",
        "cons": "❌ Inconvénients",
        "pros_list": [
            "Simple à implémenter — un seul paramètre ε",
            "Exploration garantie — jamais bloqué définitivement",
            "Fonctionne sans connaissance préalable",
            "Calcul trivial — O(1) par décision",
            "Bonne référence pour comparer avec des algorithmes plus intelligents",
        ],
        "cons_list": [
            "Explore uniformément — perd du temps sur de mauvaises actions",
            "Pas de mémoire — ignore les apprentissages précédents",
            "Table Q non entraînée en Ch01 — exploitation = aléatoire",
            "Ne passe pas à l'échelle pour de grands espaces d'états",
            "La décroissance de ε doit être réglée manuellement",
        ],
        "metric_gt": "Retour total Gₜ",
        "metric_sla": "Taux SLA",
        "metric_skill": "Taux de correspondance",
        "metric_explore": "Taux d'exploration",
        "metric_sla_saved": "Pénalités SLA évitées",
        "metric_dist": "Distance moy. dispatch",
        "metric_reward": "Récompense moy. par étape",
    },
    "DE": {
        "title": "Kapitel 01 — ASP-Disposition: Einführung in RL",
        "subtitle": "Außendienstoptimierung durch Bestärkendes Lernen · Region Warschau",
        "engine_ok": "⚙️ Rust-Engine aktiv",
        "engine_missing": "⚙️ Rust-Engine nicht gefunden. Ausführen: `cd rlvr-py && maturin develop`",
        "lang_label": "🌐 Sprache",
        "sidebar_title": "⚙️ Episodeneinstellungen",
        "n_tech": "Techniker",
        "n_orders": "Arbeitsaufträge",
        "epsilon": "ε — Explorationsrate",
        "gamma": "γ — Diskontierungsfaktor",
        "seed": "Zufallsseed",
        "n_episodes": "Episoden (Lernkurve)",
        "run_btn": "▶ Episode starten",
        "guide_title": "🎓 Anleitung",
        "guide": """
**Schritt 1 — ε einstellen**: ε=1.0 = reine Exploration, ε=0.0 = reine Exploitation.
**Schritt 2 — Techniker und Aufträge einstellen**: 5/10 ist ein guter Ausgangspunkt.
**Schritt 3 — ▶ Episode starten klicken**: Rust-Engine führt die vollständige MDP-Schleife aus.
**Schritt 4 — Warschau-Karte lesen**: Blau = Techniker, Farbe = Aufträge, Grün = SLA erfüllt.
**Schritt 5 — Schritt-Schieberegler verwenden** um eine bestimmte Entscheidung hervorzuheben.
**Schritt 6 — Glass-Box lesen**: jede Zeile zeigt das vollständige MDP-Tupel Sₜ, Aₜ, Rₜ, Gₜ.
**Schritt 7 — Zusammenfassung lesen**: quantifizierte Ergebnisse + Vor-/Nachteile der Methode.
""",
        "map_title": "📍 Warschau Dispositionskarte",
        "map_caption": "Blau = Techniker · Bernstein/Rot = Aufträge · Grün = SLA erfüllt · Rot = SLA-Verletzung",
        "step_slider": "🔍 Schritt hervorheben",
        "glass_title": "🔬 Glass-Box-Inspektor — MDP-Schrittprotokoll",
        "glass_headers": ["Schritt", "Tech", "Auftrag", "Qualifikation", "Entfernung", "Dringlichkeit", "Belohnung Rₜ", "Ertrag Gₜ", "SLA", "Modus"],
        "sla_met": "✅ Erfüllt",
        "sla_breach": "❌ Verletzung",
        "skill_ok": "✅ Übereinstimmung",
        "skill_no": "⚠️ Nichtübereinstimmung",
        "explore": "🔍 Erkunden",
        "exploit": "🎯 Ausnutzen",
        "bellman_caption": "Bellman-Gleichung — aktiviert sich in Kapitel 02",
        "curve_title": "📈 Lernkurve — Gₜ über Episoden",
        "curve_x": "Episode",
        "curve_y": "Gesamter diskontierter Ertrag Gₜ",
        "curve_mean": "Gleitender Durchschnitt (5 Ep)",
        "theory_title": "📖 Theorie — Kapitel 01",
        "theory_sections": {
            "mdp": "§1.1 Das MDP-Framework",
            "egreedy": "§1.1 ε-greedy-Strategie",
            "gt": "§1.1 Diskontierter Ertrag Gₜ",
            "ndarray": "§1.2.1 ndarray Q-Tabelle",
            "reward": "§1.1 Belohnungsdesign",
        },
        "theory_mdp": r"""
**Das MDP-Tupel (S, A, P, R, γ)** ist das mathematische Fundament jedes RL-Systems.
$$P(s_{t+1} | s_t, a_t) \text{ — Markov-Eigenschaft}$$
Implementiert in `transition()` in `ch01_asp_dispatch.rs`.
""",
        "theory_egreedy": r"""
**ε-greedy** ist die einfachste Explorationsstrategie.
$$a_t = \begin{cases} \text{zufällige Aktion} & \text{mit Wahrscheinlichkeit } \varepsilon \\ \arg\max_a Q(s,a) & \text{mit Wahrscheinlichkeit } 1-\varepsilon \end{cases}$$
""",
        "theory_gt": r"""
**Diskontierter Ertrag Gₜ** misst den Gesamtwert einer Entscheidung.
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}$$
""",
        "theory_ndarray": r"""
**ndarray Q-Tabelle** — Datenstruktur für gelernte Aktionswerte.
```rust
let mut q_table = Array2::<f64>::zeros((n_tech, n_orders));
```
""",
        "theory_reward": r"""
**Belohnungsdesign**: +10 SLA erfüllt, −5 Verletzung, −2 Nichtübereinstimmung, −0.1×km Entfernung.
Realistische SLA-Rate: 77–93%.
""",
        "summary_title": "📊 Episodenzusammenfassung",
        "summary_results": "Quantifizierte Ergebnisse",
        "summary_pros_cons": "ε-greedy-Methode — Vor- & Nachteile",
        "pros": "✅ Vorteile",
        "cons": "❌ Nachteile",
        "pros_list": [
            "Einfach zu implementieren — ein Parameter ε steuert alles",
            "Garantierte Exploration — nie dauerhaft feststeckend",
            "Funktioniert ohne Vorwissen (Null-Q-Tabelle)",
            "Rechnerisch trivial — O(1) pro Entscheidung",
            "Gute Basislinie zum Vergleich mit intelligenteren Algorithmen",
        ],
        "cons_list": [
            "Erkundet gleichmäßig — verschwendet Zeit mit offensichtlich schlechten Aktionen",
            "Kein Gedächtnis — ignoriert frühere Erkenntnisse",
            "Q-Tabelle in Ch01 untrainiert — Exploitation = zufällig",
            "Skaliert nicht für große Zustandsräume",
            "ε-Abfall muss manuell abgestimmt werden",
        ],
        "metric_gt": "Gesamtertrag Gₜ",
        "metric_sla": "SLA-Rate",
        "metric_skill": "Qualifikationsübereinstimmung",
        "metric_explore": "Explorationsrate",
        "metric_sla_saved": "Vermiedene SLA-Strafen",
        "metric_dist": "Durchschn. Dispositionsentfernung",
        "metric_reward": "Durchschn. Belohnung pro Schritt",
    },
    "ES": {
        "title": "Capítulo 01 — Despacho ASP: Introducción al RL",
        "subtitle": "Optimización del servicio de campo mediante aprendizaje por refuerzo · Región de Varsovia",
        "engine_ok": "⚙️ Motor Rust activo",
        "engine_missing": "⚙️ Motor Rust no encontrado. Ejecute: `cd rlvr-py && maturin develop`",
        "lang_label": "🌐 Idioma",
        "sidebar_title": "⚙️ Configuración del episodio",
        "n_tech": "Técnicos",
        "n_orders": "Órdenes de trabajo",
        "epsilon": "ε — Tasa de exploración",
        "gamma": "γ — Factor de descuento",
        "seed": "Semilla aleatoria",
        "n_episodes": "Episodios (curva de aprendizaje)",
        "run_btn": "▶ Ejecutar episodio",
        "guide_title": "🎓 Cómo usar este capítulo",
        "guide": """
**Paso 1 — Ajuste ε**: ε=1.0 = exploración pura, ε=0.0 = explotación pura.
**Paso 2 — Ajuste técnicos y órdenes**: 5/10 es un buen punto de partida.
**Paso 3 — Haga clic en ▶ Ejecutar episodio**: el motor Rust ejecuta el bucle MDP completo.
**Paso 4 — Lea el mapa de Varsovia**: azul = técnicos, color = órdenes, verde = SLA cumplido.
**Paso 5 — Use el control deslizante de pasos** para resaltar una decisión específica.
**Paso 6 — Lea el Glass-Box**: cada fila muestra la tupla MDP completa Sₜ, Aₜ, Rₜ, Gₜ.
**Paso 7 — Lea el resumen**: resultados cuantificados + pros/contras del método.
""",
        "map_title": "📍 Mapa de despacho Varsovia",
        "map_caption": "Azul = Técnicos · Ámbar/Rojo = Órdenes · Verde = SLA cumplido · Rojo = Violación SLA",
        "step_slider": "🔍 Resaltar paso",
        "glass_title": "🔬 Inspector Glass-Box — Traza de pasos MDP",
        "glass_headers": ["Paso", "Téc", "Orden", "Habilidad", "Distancia", "Urgencia", "Recompensa Rₜ", "Retorno Gₜ", "SLA", "Modo"],
        "sla_met": "✅ Cumplido",
        "sla_breach": "❌ Violación",
        "skill_ok": "✅ Coincidencia",
        "skill_no": "⚠️ Desajuste",
        "explore": "🔍 Explorar",
        "exploit": "🎯 Explotar",
        "bellman_caption": "Ecuación de Bellman — se activa en el Capítulo 02",
        "curve_title": "📈 Curva de aprendizaje — Gₜ por episodio",
        "curve_x": "Episodio",
        "curve_y": "Retorno total descontado Gₜ",
        "curve_mean": "Media móvil (5 ep)",
        "theory_title": "📖 Teoría — Capítulo 01",
        "theory_sections": {
            "mdp": "§1.1 El marco MDP",
            "egreedy": "§1.1 Política ε-greedy",
            "gt": "§1.1 Retorno descontado Gₜ",
            "ndarray": "§1.2.1 Tabla Q ndarray",
            "reward": "§1.1 Diseño de recompensa",
        },
        "theory_mdp": r"""
**La tupla MDP (S, A, P, R, γ)** es el fundamento matemático de todo sistema RL.
$$P(s_{t+1} | s_t, a_t) \text{ — propiedad de Markov}$$
Implementado en `transition()` en `ch01_asp_dispatch.rs`.
""",
        "theory_egreedy": r"""
**ε-greedy** es la estrategia de exploración más simple.
$$a_t = \begin{cases} \text{acción aleatoria} & \text{con probabilidad } \varepsilon \\ \arg\max_a Q(s,a) & \text{con probabilidad } 1-\varepsilon \end{cases}$$
""",
        "theory_gt": r"""
**Retorno descontado Gₜ** mide el valor total de una decisión.
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}$$
""",
        "theory_ndarray": r"""
**Tabla Q ndarray** — estructura de datos para valores de acción aprendidos.
```rust
let mut q_table = Array2::<f64>::zeros((n_tech, n_orders));
```
""",
        "theory_reward": r"""
**Diseño de recompensa**: +10 SLA cumplido, −5 violación, −2 desajuste, −0.1×km distancia.
Tasa SLA realista: 77–93%.
""",
        "summary_title": "📊 Resumen del episodio",
        "summary_results": "Resultados cuantificados",
        "summary_pros_cons": "Método ε-greedy — Pros y Contras",
        "pros": "✅ Pros",
        "cons": "❌ Contras",
        "pros_list": [
            "Simple de implementar — un parámetro ε controla todo",
            "Exploración garantizada — nunca atascado permanentemente",
            "Funciona sin conocimiento previo (tabla Q cero)",
            "Computacionalmente trivial — O(1) por decisión",
            "Buena línea base para comparar con algoritmos más inteligentes",
        ],
        "cons_list": [
            "Explora uniformemente — desperdicia tiempo en acciones malas",
            "Sin memoria — ignora aprendizajes anteriores",
            "Tabla Q no entrenada en Ch01 — explotación = aleatoria",
            "No escala para grandes espacios de estados",
            "La decadencia de ε debe ajustarse manualmente",
        ],
        "metric_gt": "Retorno total Gₜ",
        "metric_sla": "Tasa SLA",
        "metric_skill": "Tasa de coincidencia",
        "metric_explore": "Tasa de exploración",
        "metric_sla_saved": "Penalizaciones SLA evitadas",
        "metric_dist": "Dist. media despacho",
        "metric_reward": "Recompensa media por paso",
    },
}

# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def _tx(lang):
    """Return translation dict for lang, filling missing keys from EN."""
    base = dict(T.get("EN", {}))
    over = T.get(lang, {})
    for k, v in over.items():
        base[k] = v
    return base

def _render_handbook():
    st.iframe(
        """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Podr&#x0119;cznik &#x2014; Rozdzia&#x0142; 01: MDP i Dyspozycja ASP</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f1117;color:#e8eaf6;line-height:1.7;font-size:15px}
.container{max-width:960px;margin:0 auto;padding:2rem}
h1{color:#8B5CF6;font-size:1.8rem;border-bottom:2px solid #8B5CF6;padding-bottom:.5rem;margin-bottom:1.5rem}
h2{color:#0082F0;font-size:1.3rem;margin:1.5rem 0 .75rem}
h3{color:#0FC373;font-size:1.1rem;margin:1rem 0 .5rem}
p{margin:.5rem 0}
ul,ol{margin:.5rem 0 .5rem 1.5rem}
li{margin:.25rem 0}
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
.card.purple{border-left-color:#8B5CF6}
table{width:100%;border-collapse:collapse;margin:.75rem 0;font-size:.9rem}
th{background:#252840;color:#8B5CF6;padding:.6rem .75rem;text-align:left;font-weight:600}
td{padding:.5rem .75rem;border-bottom:1px solid #2d3154}
tr:hover td{background:#252840}
code{background:#252840;padding:.15rem .4rem;border-radius:4px;color:#0FC373;font-size:.85em;font-family:monospace}
pre{background:#252840;padding:1rem;border-radius:8px;overflow-x:auto;margin:.75rem 0}
pre code{background:none;padding:0;font-size:.85rem}
.formula{background:#252840;border-radius:8px;padding:1rem;margin:.75rem 0;text-align:center;font-size:1.05em;color:#FFD700;font-family:monospace}
.step{display:flex;gap:1rem;margin:.6rem 0;align-items:flex-start}
.step-num{background:#8B5CF6;color:white;border-radius:50%;width:1.8rem;height:1.8rem;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-weight:bold;font-size:.85rem}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:.75rem 0}
.kpi-card{background:#252840;border-radius:8px;padding:1rem;text-align:center}
.kpi-val{font-size:1.6em;font-weight:bold;color:#0FC373}
.kpi-label{color:#9ca3af;font-size:.8em;margin-top:.25rem}
.tag{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.8em;margin:.15rem}
.tag.green{background:#0FC37322;color:#0FC373;border:1px solid #0FC37344}
.tag.red{background:#FF4B4B22;color:#FF4B4B;border:1px solid #FF4B4B44}
.tag.blue{background:#0082F022;color:#0082F0;border:1px solid #0082F044}
.tag.purple{background:#8B5CF622;color:#8B5CF6;border:1px solid #8B5CF644}
.tag.orange{background:#FF8C0A22;color:#FF8C0A;border:1px solid #FF8C0A44}
.badge{display:inline-flex;align-items:center;gap:.4rem;background:#1e2235;border:1px solid #2d3154;border-radius:6px;padding:.3rem .7rem;font-size:.85rem;margin:.2rem}
.highlight{background:#8B5CF622;border:1px solid #8B5CF644;border-radius:6px;padding:.75rem 1rem;margin:.5rem 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:.75rem 0}
@media(max-width:600px){.grid2{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<h1>&#x1F4D8; Podr&#x0119;cznik &#x2014; Rozdzia&#x0142; 01</h1>
<p style="color:#9ca3af;margin-bottom:1.5rem"><em>MDP i Dyspozycja ASP &middot; Warszawa &middot; Silnik Rust &middot; Interaktywne Laboratorium</em></p>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('intro')">&#x1F4D6; Wprowadzenie</button>
  <button class="tab-btn" onclick="showTab('what')">&#x2753; Czym jest Ch01?</button>
  <button class="tab-btn" onclick="showTab('theory')">&#x1F9EE; Teoria RL</button>
  <button class="tab-btn" onclick="showTab('env')">&#x1F5FA;&#xFE0F; &#x15A;rodowisko</button>
  <button class="tab-btn" onclick="showTab('ui')">&#x1F3AE; Jak u&#x17C;ywa&#x107; UI</button>
  <button class="tab-btn" onclick="showTab('interp')">&#x1F4CA; Interpretacja</button>
  <button class="tab-btn" onclick="showTab('exercises')">&#x1F9EA; &#x106;wiczenia</button>
  <button class="tab-btn" onclick="showTab('summary')">&#x1F4CB; Podsumowanie</button>
</div>

<!-- TAB: Wprowadzenie -->
<div id="intro" class="tab-content active">
<h2>&#x1F3AF; Cele nauki</h2>
<div class="card green">
Po uko&#x0144;czeniu tego rozdzia&#x0142;u b&#x0119;dziesz potrafi&#x0142;:
<ul>
<li>Zdefiniowa&#x107; 5 sk&#x0142;adnik&#xF3;w MDP: S, A, P, R, &#x3B3;</li>
<li>Wyja&#x15B;ni&#x107; czym jest stan, akcja, nagroda i zwrot w kontek&#x15B;cie ASP</li>
<li>Zaimplementowa&#x107; polityk&#x0119; &#x3B5;-zachlann&#x0105; i wyja&#x15B;ni&#x107; kompromis eksploracja-eksploatacja</li>
<li>Odczyta&#x107; map&#x0119; Warszawy i zinterpretowa&#x107; decyzje dyspozycji jako przej&#x15B;cia MDP</li>
<li>Odczyta&#x107; Glass-Box i prze&#x15B;ledzi&#x107; pe&#x0142;n&#x0105; krotkMDP: (S&#x209C;, A&#x209C;, R&#x209C;, G&#x209C;)</li>
<li>Wyja&#x15B;ni&#x107; dlaczego r&#xF3;wnanie Bellmana jest wyszarzone w Ch01 (aktywuje si&#x0119; w Ch02)</li>
</ul>
</div>

<h2>&#x1F3E2; Problem biznesowy</h2>
<div class="card blue">
<strong>Centrum Dyspozycji ASP Warszawa</strong> &#x2014; 5 technik&#xF3;w, do 20 zlece&#x0144; na zmian&#x0119;.<br><br>
Ka&#x017C;da decyzja dyspozycji to <strong>akcja MDP</strong>. Wynik (SLA spe&#x0142;nione lub naruszone) to <strong>nagroda</strong>.
Sekwencja wszystkich decyzji w jednej zmianie to <strong>epizod</strong>.<br><br>
W Ch01 tabela Q ma same zera &#x2014; agent nie ma jeszcze &#x17C;adnej wiedzy.
Wszystkie decyzje s&#x0105; podejmowane wy&#x0142;&#x0105;cznie przez losow&#x0105; eksploracj&#x0119; &#x3B5;-zachlann&#x0105;.
To jest <em>punkt bazowy</em> wzgl&#x0119;dem kt&#xF3;rego mierzone s&#x0105; wszystkie przysz&#x0142;e rozdzia&#x0142;y.
</div>

<h2>&#x1F5FA;&#xFE0F; Mapa my&#x15B;li: od MDP do optymalnej polityki</h2>
<div class="card">
<div style="text-align:center;font-family:monospace;color:#8B5CF6;font-size:.9rem">
Ch01: MDP + &#x3B5;-greedy (Q=0)<br>
&#x2193;<br>
Ch02: Bellman + Value Iteration (model znany)<br>
&#x2193;<br>
Ch03&#x2013;05: Bandit / MC / TD (model nieznany)<br>
&#x2193;<br>
Ch06&#x2013;09: Q-Learning / SARSA / Policy Gradient<br>
&#x2193;<br>
Ch10&#x2013;13: Model-Based / MARL / QMIX
</div>
</div>

<div class="kpi">
<div class="kpi-card"><div class="kpi-val">5</div><div class="kpi-label">Sk&#x0142;adnik&#xF3;w MDP</div></div>
<div class="kpi-card"><div class="kpi-val">&#x3B5;</div><div class="kpi-label">Wsp&#xF3;&#x0142;czynnik eksploracji</div></div>
<div class="kpi-card"><div class="kpi-val">G&#x209C;</div><div class="kpi-label">Zdyskontowany zwrot</div></div>
<div class="kpi-card"><div class="kpi-val">Q=0</div><div class="kpi-label">Tabela Q w Ch01</div></div>
</div>
</div>

<!-- TAB: Czym jest Ch01 -->
<div id="what" class="tab-content">
<h2>&#x2753; Czym jest Rozdzia&#x0142; 01?</h2>
<div class="card purple">
Ch01 to <strong>punkt startowy ca&#x0142;ego kursu RL</strong>. Nie ma tu uczenia &#x2014; agent dzia&#x0142;a losowo.
Celem jest zrozumienie formalizmu MDP i narz&#x0119;dzi wizualizacji zanim wprowadzone zostanie uczenie.
</div>

<h2>Co si&#x0119; dzieje w silniku Rust?</h2>
<div class="card">
Funkcja <code>run_ch01_episode()</code> w <code>ch01_mdp.rs</code>:
<ol>
<li>Generuje losowych 5 technik&#xF3;w i N zlece&#x0144; na mapie Warszawy</li>
<li>Dla ka&#x017C;dego zlecenia wybiera technika przez polityk&#x0119; &#x3B5;-zachlann&#x0105; (Q=0 &#x2192; zawsze losowo)</li>
<li>Oblicza nagrod&#x0119; R&#x209C; na podstawie: dopasowania umiej&#x0119;tno&#x15B;ci, odleg&#x0142;o&#x15B;ci, pilno&#x15B;ci</li>
<li>Oblicza zdyskontowany zwrot G&#x209C; wstecz przez ca&#x0142;y epizod</li>
<li>Zwraca pe&#x0142;ny &#x15B;lad MDP: ka&#x017C;dy krok (S&#x209C;, A&#x209C;, R&#x209C;, G&#x209C;, explored)</li>
</ol>
</div>

<h2>Czego Ch01 NIE robi</h2>
<div class="card red">
<ul>
<li>&#x274C; Nie aktualizuje tabeli Q &#x2014; Q pozostaje zerowe przez ca&#x0142;y epizod</li>
<li>&#x274C; Nie u&#x017C;ywa r&#xF3;wnania Bellmana &#x2014; aktywuje si&#x0119; w Ch02</li>
<li>&#x274C; Nie uczy si&#x0119; mi&#x0119;dzy epizodami &#x2014; ka&#x017C;dy epizod zaczyna od zera</li>
<li>&#x274C; Nie optymalizuje polityki &#x2014; to zadanie Ch06 (Q-Learning)</li>
</ul>
</div>

<h2>Dlaczego to wa&#x017C;ne?</h2>
<div class="card green">
Ch01 ustanawia <strong>losowy punkt bazowy</strong>. &#x15A;redni G&#x209C; z Ch01 to minimum kt&#xF3;re ka&#x017C;dy algorytm RL musi pokona&#x107;.
Je&#x15B;li algorytm z Ch06 nie bije Ch01 &#x2014; co&#x15B; jest nie tak z hiperparametrami.
</div>
</div>

<!-- TAB: Teoria RL -->
<div id="theory" class="tab-content">
<h2>&#x1F9EE; Formalizm MDP</h2>
<table>
<tr><th>Sk&#x0142;adnik</th><th>Symbol</th><th>Znaczenie w ASP</th><th>Przyk&#x0142;ad</th></tr>
<tr><td>Przestrze&#x0144; stan&#xF3;w</td><td><strong>S</strong></td><td>Sytuacja operacyjna centrum dyspozycji</td><td>S3: cz&#x0119;&#x15B;ciowa dost&#x0119;pno&#x15B;&#x107;, wysokie obci&#x0105;&#x017C;enie</td></tr>
<tr><td>Przestrze&#x0144; akcji</td><td><strong>A</strong></td><td>Kt&#xF3;rego technika wys&#x0142;a&#x107; do kt&#xF3;rego zlecenia</td><td>Wy&#x15B;lij T2 do W5</td></tr>
<tr><td>Model przej&#x15B;&#x107;</td><td><strong>P(s'|s,a)</strong></td><td>Prawdopodobie&#x0144;stwo nast&#x0119;pnego stanu</td><td>Po dyspozycji T2 staje si&#x0119; niedost&#x0119;pny</td></tr>
<tr><td>Funkcja nagrody</td><td><strong>R(s,a)</strong></td><td>Natychmiastowa informacja zwrotna</td><td>+10 SLA spe&#x0142;nione, &#x2212;50 SLA naruszone</td></tr>
<tr><td>Wsp&#xF3;&#x0142;czynnik dyskonta</td><td><strong>&#x3B3;</strong></td><td>Jak bardzo cenimy przysz&#x0142;e nagrody</td><td>&#x3B3;=0.95: przysz&#x0142;e nagrody warte 95% bie&#x017C;&#x0105;cych</td></tr>
</table>

<h2>Zdyskontowany zwrot G&#x209C;</h2>
<div class="formula">G&#x209C; = R&#x209C;&#x208A;&#x2081; + &#x3B3; R&#x209C;&#x208A;&#x2082; + &#x3B3;&#xB2; R&#x209C;&#x208A;&#x2083; + &hellip; = &sum;<sub>k=0</sub><sup>&infin;</sup> &#x3B3;<sup>k</sup> R&#x209C;&#x208A;&#x2081;&#x208A;<sub>k</sub></div>
<div class="card">
<strong>Przyk&#x0142;ad liczbowy</strong> (&#x3B3;=0.95, epizod 3-krokowy):
<ul>
<li>Krok 1: wy&#x15B;lij T0 &#x2192; R = +10 (SLA spe&#x0142;nione)</li>
<li>Krok 2: wy&#x15B;lij T2 &#x2192; R = &#x2212;5 (z&#x0142;e umiej&#x0119;tno&#x15B;ci)</li>
<li>Krok 3: wy&#x15B;lij T1 &#x2192; R = +10 (SLA spe&#x0142;nione)</li>
</ul>
G&#x2080; = 10 + 0.95&times;(&#x2212;5) + 0.95&#xB2;&times;10 = 10 &#x2212; 4.75 + 9.025 = <strong>14.275</strong>
</div>

<h2>Polityka &#x3B5;-zachlanna</h2>
<div class="card orange">
<strong>Z prawdopodobie&#x0144;stwem &#x3B5;:</strong> wybierz losow&#x0105; akcj&#x0119; (eksploracja)<br>
<strong>Z prawdopodobie&#x0144;stwem 1&#x2212;&#x3B5;:</strong> wybierz najlepsz&#x0105; znan&#x0105; akcj&#x0119; (eksploatacja)<br><br>
W Ch01 tabela Q ma same zera &#x2014; eksploatacja = losowo.<br>
&#x3B5; ma znaczenie dopiero od Ch06 gdy warto&#x15B;ci Q s&#x0105; niezerowe.
</div>
<table>
<tr><th>Warto&#x15B;&#x107; &#x3B5;</th><th>Zachowanie</th><th>Kiedy u&#x017C;ywa&#x107;</th></tr>
<tr><td>1.0</td><td>Zawsze losowo</td><td>Pocz&#x0105;tek treningu &#x2014; nic nie wiemy</td></tr>
<tr><td>0.5</td><td>50/50 eksploracja/eksploatacja</td><td>&#x15A;rodek treningu</td></tr>
<tr><td>0.1</td><td>G&#x0142;&#xF3;wnie eksploatacja</td><td>P&#xF3;&#x017A;ny trening &#x2014; polityka prawie optymalna</td></tr>
<tr><td>0.0</td><td>Zawsze zachlanna</td><td>Tylko ewaluacja (bez uczenia)</td></tr>
</table>

<h2>Funkcja warto&#x15B;ci V(s) i Q(s,a)</h2>
<div class="card">
<strong>V(s)</strong> &#x2014; oczekiwany zdyskontowany zwrot startuj&#x0105;c ze stanu s pod polityk&#x0105; &#x3C0;:<br>
<div class="formula">V&#x3C0;(s) = E&#x3C0;[G&#x209C; | S&#x209C;=s]</div>
<strong>Q(s,a)</strong> &#x2014; oczekiwany zdyskontowany zwrot podejmuj&#x0105;c akcj&#x0119; a w stanie s:<br>
<div class="formula">Q&#x3C0;(s,a) = E&#x3C0;[G&#x209C; | S&#x209C;=s, A&#x209C;=a]</div>
W Ch01: Q(s,a) = 0 dla wszystkich (s,a). Aktualizacja Q zaczyna si&#x0119; w Ch06.
</div>

<h2>R&#xF3;wnanie Bellmana (wyszarzone w Ch01)</h2>
<div class="card red">
R&#xF3;wnanie Bellmana wymaga znajomo&#x15B;ci modelu P(s'|s,a):<br>
<div class="formula">V*(s) = max<sub>a</sub> &sum;<sub>s'</sub> P(s'|s,a) [ R(s,a) + &#x3B3; V*(s') ]</div>
W Ch01 nie znamy P(s'|s,a) &#x2014; dlatego r&#xF3;wnanie jest wyszarzone.<br>
Ch02 buduje macierz przej&#x15B;&#x107; ASP i rozwi&#x0105;zuje to r&#xF3;wnanie przez iteracj&#x0119; warto&#x15B;ci.
</div>
</div>

<!-- TAB: Środowisko -->
<div id="env" class="tab-content">
<h2>&#x1F5FA;&#xFE0F; Mapa Warszawy &#x2014; &#x15A;rodowisko ASP</h2>
<div class="card blue">
Symulacja obejmuje rzeczywiste wsp&#xF3;&#x0142;rz&#x0119;dne geograficzne Warszawy.<br>
Technicy i zlecenia s&#x0105; losowo rozmieszczeni w obszarze miejskim.<br>
Odleg&#x0142;o&#x15B;&#x107; jest obliczana jako odleg&#x0142;o&#x15B;&#x107; euklidesowa w stopniach geograficznych.
</div>

<h2>Technicy (T0&#x2013;T4)</h2>
<table>
<tr><th>Atrybut</th><th>Opis</th><th>Wp&#x0142;yw na nagrod&#x0119;</th></tr>
<tr><td>Pozycja (lat, lon)</td><td>Aktualna lokalizacja na mapie Warszawy</td><td>Odleg&#x0142;o&#x15B;&#x107; do zlecenia &#x2192; czas dojazdu</td></tr>
<tr><td>Umiej&#x0119;tno&#x15B;&#x107; (skill)</td><td>HVAC / Elektryka / Hydraulika / Sie&#x107; / Mechanika</td><td>Dopasowanie do zlecenia &#x2192; +bonus lub -kara</td></tr>
<tr><td>Dost&#x0119;pno&#x15B;&#x107;</td><td>Czy technik jest wolny</td><td>Niedost&#x0119;pny technik nie mo&#x017C;e by&#x107; wys&#x0142;any</td></tr>
</table>

<h2>Zlecenia (W0&#x2013;W9)</h2>
<table>
<tr><th>Atrybut</th><th>Opis</th><th>Wp&#x0142;yw na nagrod&#x0119;</th></tr>
<tr><td>Pozycja (lat, lon)</td><td>Lokalizacja zlecenia na mapie</td><td>Odleg&#x0142;o&#x15B;&#x107; od technika</td></tr>
<tr><td>Wymagana umiej&#x0119;tno&#x15B;&#x107;</td><td>Jaki skill jest potrzebny</td><td>Niedopasowanie &#x2192; kara</td></tr>
<tr><td>Pilno&#x15B;&#x107; (urgency)</td><td>0.0&#x2013;1.0 (1.0 = krytyczne)</td><td>Wysoka pilno&#x15B;&#x107; + op&#xF3;&#x017A;nienie &#x2192; du&#x017C;a kara</td></tr>
</table>

<h2>Funkcja nagrody R(s,a)</h2>
<div class="card">
Nagroda za dyspozycj&#x0119; technika T do zlecenia W:
<pre><code>R = base_reward
  + skill_bonus    (je&#x15B;li skill T == skill W: +5.0)
  - distance_penalty (odleg&#x0142;o&#x15B;&#x107; * 2.0)
  - urgency_penalty  (urgency * op&#xF3;&#x017A;nienie * 10.0)
  + sla_bonus      (je&#x15B;li SLA spe&#x0142;nione: +10.0)
  - sla_penalty    (je&#x15B;li SLA naruszone: -50.0)</code></pre>
</div>

<h2>Stany operacyjne (S0&#x2013;S7)</h2>
<p>Ch01 u&#x017C;ywa uproszczonego kodowania stanu opartego na dost&#x0119;pno&#x15B;ci technik&#xF3;w i obci&#x0105;&#x017C;eniu zleceniami. Pe&#x0142;ne 8 stan&#xF3;w z nazwami jest zdefiniowanych w Ch02.</p>
<table>
<tr><th>Stan</th><th>Opis</th></tr>
<tr><td><code>S0</code></td><td>Wszyscy dost&#x0119;pni, brak pilnych zlece&#x0144;</td></tr>
<tr><td><code>S1</code></td><td>Wszyscy dost&#x0119;pni, pilne zlecenie oczekuje</td></tr>
<tr><td><code>S2</code></td><td>Cz&#x0119;&#x15B;ciowa dost&#x0119;pno&#x15B;&#x107;, niskie obci&#x0105;&#x017C;enie</td></tr>
<tr><td><code>S3</code></td><td>Cz&#x0119;&#x15B;ciowa dost&#x0119;pno&#x15B;&#x107;, wysokie obci&#x0105;&#x017C;enie</td></tr>
<tr><td><code>S4</code></td><td>Niska dost&#x0119;pno&#x15B;&#x107;, znos&#x0105;ce obci&#x0105;&#x017C;enie</td></tr>
<tr><td><code>S5</code></td><td>Niska dost&#x0119;pno&#x15B;&#x107;, wysokie obci&#x0105;&#x017C;enie</td></tr>
<tr><td><code>S6</code></td><td>Krytyczna &#x2014; wi&#x0119;kszo&#x15B;&#x107; technik&#xF3;w zaj&#x0119;ta</td></tr>
<tr><td><code>S7</code></td><td>Wszyscy zaj&#x0119;ci, naruszenie SLA bliskie</td></tr>
</table>
</div>

<!-- TAB: Jak używać UI -->
<div id="ui" class="tab-content">
<h2>&#x1F3AE; Jak u&#x017C;ywa&#x107; interfejsu Ch01</h2>

<div class="step"><div class="step-num">1</div><div><strong>Ustaw &#x3B5; (wsp&#xF3;&#x0142;czynnik eksploracji)</strong><br>Przesu&#x0144; suwak. &#x3B5;=1.0: agent zawsze losowy. &#x3B5;=0.0: agent zawsze zachlanny (w Ch01 = te&#x017C; losowy bo Q=0). Zacznij od &#x3B5;=0.5.</div></div>

<div class="step"><div class="step-num">2</div><div><strong>Ustaw liczb&#x0119; technik&#xF3;w i zlece&#x0144;</strong><br>5 technik&#xF3;w / 10 zlece&#x0144; to dobry punkt startowy. Wi&#x0119;cej zlece&#x0144; = d&#x0142;u&#x017C;szy epizod.</div></div>

<div class="step"><div class="step-num">3</div><div><strong>Kliknij &#x25B6; Uruchom epizod</strong><br>Silnik Rust wykonuje pe&#x0142;n&#x0105; p&#x0119;tl&#x0119; MDP i zwraca ka&#x017C;dy krok.</div></div>

<div class="step"><div class="step-num">4</div><div><strong>Odczytaj map&#x0119; Warszawy</strong><br>Niebieskie markery = technicy (T0&#x2013;T4). Kolorowe markery = zlecenia (W0&#x2013;W9). Zielone linie = SLA spe&#x0142;nione. Czerwone linie = SLA naruszone. Kliknij marker aby zobaczy&#x107; szczeg&#xF3;&#x0142;y.</div></div>

<div class="step"><div class="step-num">5</div><div><strong>U&#x017C;yj suwaka krok&#xF3;w</strong><br>Przesu&#x0144; aby pod&#x15B;wietli&#x107; konkretn&#x0105; decyzj&#x0119; dyspozycji na mapie i w Glass-Box.</div></div>

<div class="step"><div class="step-num">6</div><div><strong>Odczytaj Glass-Box</strong><br>Ka&#x017C;dy wiersz pokazuje pe&#x0142;n&#x0105; krotkMDP: S&#x209C; (stan), A&#x209C; (akcja), R&#x209C; (nagroda), G&#x209C; (zwrot). R&#xF3;wnanie Bellmana jest wyszarzone &#x2014; aktywuje si&#x0119; w Rozdziale 02.</div></div>

<div class="step"><div class="step-num">7</div><div><strong>Odczytaj podsumowanie epizodu</strong><br>Skwantyfikowane wyniki biznesowe + zalety/wady metody &#x3B5;-zachlannej.</div></div>

<h2>&#x1F4A1; Wskaz&#xF3;wki</h2>
<div class="card green">
<ul>
<li>Uruchom kilka epizod&#xF3;w z tym samym &#x3B5; &#x2014; wyniki b&#x0119;d&#x0105; si&#x0119; r&#xF3;&#x017C;ni&#x107; (losowo&#x15B;&#x107;)</li>
<li>Zmie&#x0144; seed aby zobaczy&#x107; inn&#x0105; konfiguracj&#x0119; mapy</li>
<li>Por&#xF3;wnaj G&#x209C; dla &#x3B5;=1.0 vs &#x3B5;=0.0 &#x2014; powinny by&#x107; podobne (Q=0)</li>
</ul>
</div>
</div>

<!-- TAB: Interpretacja -->
<div id="interp" class="tab-content">
<h2>&#x1F4CA; Jak interpretowa&#x107; wyniki</h2>

<h3>Mapa Warszawy</h3>
<div class="card">
<span class="tag blue">Niebieskie markery</span> = Technicy T0&#x2013;T4 (aktualna pozycja)<br>
<span class="tag green">Zielone linie</span> = SLA spe&#x0142;nione &#x2014; dyspozycja by&#x0142;a na czas i z w&#x0142;a&#x15B;ciwymi umiej&#x0119;tno&#x15B;ciami<br>
<span class="tag red">Czerwone linie</span> = SLA naruszone &#x2014; za daleko, z&#x0142;e umiej&#x0119;tno&#x15B;ci lub za p&#xF3;&#x017A;no<br><br>
Im wi&#x0119;cej zielonych linii tym lepszy epizod. W Ch01 proporcja jest losowa.
</div>

<h3>Glass-Box &#x2014; tabela MDP</h3>
<table>
<tr><th>Kolumna</th><th>Znaczenie</th><th>Przyk&#x0142;ad</th></tr>
<tr><td><code>S&#x209C;</code></td><td>Stan w chwili t</td><td>S3: cz&#x0119;&#x15B;ciowa dost&#x0119;pno&#x15B;&#x107;</td></tr>
<tr><td><code>A&#x209C;</code></td><td>Podj&#x0119;ta akcja</td><td>Wy&#x15B;lij T2 &#x2192; W5</td></tr>
<tr><td><code>R&#x209C;</code></td><td>Natychmiastowa nagroda</td><td>+10.0 (SLA spe&#x0142;nione)</td></tr>
<tr><td><code>G&#x209C;</code></td><td>Zdyskontowany zwrot od tego kroku</td><td>14.275</td></tr>
<tr><td>Eksploracja</td><td>Czy akcja by&#x0142;a losowa (&#x3B5;) czy zachlanna</td><td>&#x1F3B2; Losowa</td></tr>
</table>

<h3>Krzywa uczenia</h3>
<div class="card blue">
W Ch01 krzywa uczenia jest <strong>p&#x0142;aska</strong> &#x2014; agent nie uczy si&#x0119; mi&#x0119;dzy epizodami bo Q=0.<br><br>
To jest zamierzone. Ch01 ustanawia <em>losowy punkt bazowy</em>.<br>
Od Ch06 zobaczysz krzyw&#x0105; rosn&#x0105;c&#x0105; w miar&#x0119; jak agent si&#x0119; uczy.
</div>

<h3>KPI podsumowania epizodu</h3>
<div class="kpi">
<div class="kpi-card"><div class="kpi-val">G&#x209C;</div><div class="kpi-label">Ca&#x0142;kowity zwrot epizodu</div></div>
<div class="kpi-card"><div class="kpi-val">SLA%</div><div class="kpi-label">% zlece&#x0144; spe&#x0142;niaj&#x0105;cych SLA</div></div>
<div class="kpi-card"><div class="kpi-val">&#x3B5;</div><div class="kpi-label">U&#x017C;yty wsp&#xF3;&#x0142;czynnik eksploracji</div></div>
<div class="kpi-card"><div class="kpi-val">T</div><div class="kpi-label">D&#x0142;ugo&#x15B;&#x107; epizodu (kroki)</div></div>
</div>

<h3>Bellman wyszarzony &#x2014; dlaczego?</h3>
<div class="card red">
R&#xF3;wnanie Bellmana wymaga <strong>modelu przej&#x15B;&#x107; P(s'|s,a)</strong> kt&#xF3;rego w Ch01 nie mamy.<br>
Ch02 buduje t&#x0119; macierz i aktywuje kolumn&#x0119; Bellmana w Glass-Box.
</div>
</div>

<!-- TAB: Ćwiczenia -->
<div id="exercises" class="tab-content">
<h2>&#x1F9EA; &#x106;wiczenia Hands-On</h2>

<div class="card">
<h3>&#x106;wiczenie 1 &#x2014; Pomiar punktu bazowego</h3>
Uruchom 5 epizod&#xF3;w z &#x3B5;=1.0 (czysto losowe). Zapisz &#x15B;redni G&#x209C;.<br>
To jest Tw&#xF3;j punkt bazowy Ch01. Ka&#x017C;dy przysz&#x0142;y rozdzia&#x0142; powinien pobija&#x107; t&#x0119; liczb&#x0119;.<br><br>
<strong>Oczekiwany wynik:</strong> G&#x209C; &#x2248; 20&#x2013;40 (zale&#x017C;y od konfiguracji mapy)
</div>

<div class="card blue">
<h3>&#x106;wiczenie 2 &#x2014; Wra&#x017C;liwo&#x15B;&#x107; na &#x3B5;</h3>
Uruchom z &#x3B5;=0.0 (czysto zachlanne). Czy wynik jest lepszy czy gorszy ni&#x017C; &#x3B5;=1.0?<br>
Dlaczego? (Wskaz&#xF3;wka: Q=0 &#x2014; zachlanne = losowe w Ch01)<br><br>
<strong>Oczekiwany wynik:</strong> Podobny G&#x209C; &#x2014; bo Q=0 czyni eksploatacj&#x0119; r&#xF3;wnowa&#x017C;n&#x0105; eksploracji
</div>

<div class="card orange">
<h3>&#x106;wiczenie 3 &#x2014; Czytanie mapy</h3>
Znajd&#x017A; krok z najwi&#x0119;ksz&#x0105; ujemn&#x0105; nagrod&#x0105; w Glass-Box.<br>
Kliknij ten krok na mapie. Co posz&#x0142;o nie tak?<br>
Z&#x0142;e umiej&#x0119;tno&#x15B;ci? Za daleko? Za p&#xF3;&#x017A;no?<br><br>
<strong>Cel:</strong> Zrozumie&#x107; sk&#x0142;adniki funkcji nagrody R(s,a)
</div>

<div class="card green">
<h3>&#x106;wiczenie 4 &#x2014; R&#x0119;czne obliczenie zwrotu</h3>
We&#x017A; pierwsze 3 nagrody z Glass-Box i r&#x0119;cznie oblicz G&#x2080; u&#x017C;ywaj&#x0105;c &#x3B3;=0.95.<br>
Zweryfikuj czy Tw&#xF3;j wynik zgadza si&#x0119; z kolumn&#x0105; G&#x209C;.<br><br>
<strong>Wz&#xF3;r:</strong> G&#x2080; = R&#x2081; + 0.95&times;R&#x2082; + 0.95&#xB2;&times;R&#x2083;
</div>

<div class="card purple">
<h3>&#x106;wiczenie 5 &#x2014; Wp&#x0142;yw &#x3B3; na zwrot</h3>
Uruchom ten sam epizod (ten sam seed) z &#x3B3;=0.99 i &#x3B3;=0.5.<br>
Jak zmienia si&#x0119; G&#x2080;? Kt&#xF3;ry agent jest bardziej "dalekowzroczny"?<br><br>
<strong>Oczekiwany wynik:</strong> &#x3B3;=0.99 daje wy&#x017C;szy G&#x2080; gdy nagrody s&#x0105; pozytywne
</div>
</div>

<!-- TAB: Podsumowanie -->
<div id="summary" class="tab-content">
<h2>&#x1F4CB; Podsumowanie Rozdzia&#x0142;u 01</h2>

<div class="kpi">
<div class="kpi-card"><div class="kpi-val">5</div><div class="kpi-label">Sk&#x0142;adnik&#xF3;w MDP</div></div>
<div class="kpi-card"><div class="kpi-val">Q=0</div><div class="kpi-label">Tabela Q (brak uczenia)</div></div>
<div class="kpi-card"><div class="kpi-val">&#x3B5;</div><div class="kpi-label">Jedyny hiperparametr</div></div>
<div class="kpi-card"><div class="kpi-val">Ch02</div><div class="kpi-label">Nast&#x0119;pny: Bellman + VI</div></div>
</div>

<h2>Kluczowe wnioski</h2>
<div class="card green">
<ul>
<li>&#x2705; MDP to formalny j&#x0119;zyk opisu problem&#xF3;w decyzyjnych w czasie</li>
<li>&#x2705; G&#x209C; = zdyskontowana suma nagród &#x2014; to co agent maksymalizuje</li>
<li>&#x2705; &#x3B5;-zachlanna balansuje eksploracj&#x0119; i eksploatacj&#x0119;</li>
<li>&#x2705; Ch01 ustanawia losowy punkt bazowy &#x2014; Q=0, brak uczenia</li>
<li>&#x2705; Bellman aktywuje si&#x0119; w Ch02 gdy znamy P(s'|s,a)</li>
</ul>
</div>

<h2>Zalety i wady podej&#x15B;cia Ch01</h2>
<div class="grid2">
<div class="card green">
<strong>&#x2705; Zalety</strong>
<ul>
<li>Prosta implementacja</li>
<li>Ustanawia punkt bazowy</li>
<li>Wizualizuje formalizm MDP</li>
<li>Dzia&#x0142;a bez modelu P(s'|s,a)</li>
</ul>
</div>
<div class="card red">
<strong>&#x274C; Wady</strong>
<ul>
<li>Brak uczenia &#x2014; Q=0 zawsze</li>
<li>Nie optymalizuje polityki</li>
<li>Wyniki s&#x0105; czysto losowe</li>
<li>Nie u&#x017C;ywa Bellmana</li>
</ul>
</div>
</div>

<h2>Co dalej &#x2014; Rozdzia&#x0142; 02</h2>
<div class="card blue">
Ch02 wprowadza <strong>Iteracj&#x0119; Warto&#x15B;ci</strong>:
<ul>
<li>Buduje macierz przej&#x15B;&#x107; P(s'|s,a) dla ASP Warszawa</li>
<li>Rozwi&#x0105;zuje r&#xF3;wnanie Bellmana iteracyjnie</li>
<li>Oblicza V*(s) dla wszystkich 8 stan&#xF3;w operacyjnych</li>
<li>Wyznacza optymaln&#x0105; polityk&#x0119; &#x3C0;*(s) bez symulacji</li>
<li>Aktywuje kolumn&#x0119; Bellmana w Glass-Box</li>
</ul>
</div>
</div>

</div>

<script>
function showTab(id) {
  document.querySelectorAll('.tab-content').forEach(function(el) {
    el.classList.remove('active');
  });
  document.querySelectorAll('.tab-btn').forEach(function(el) {
    el.classList.remove('active');
  });
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}
</script>
</body>
</html>""",
        height=4000,
    )

def render():
    lang = st.session_state.get("lang", "EN")
    # --- language selector (top of sidebar) ---
    # --- language selector (radio, sidebar — top) ---

    tx = _tx(lang)

    st.title(tx["title"])
    st.caption(tx["subtitle"])

    tab1, tab2 = st.tabs(["🧪 Interactive Lab", "📘 Hands-On Guide EN"])
    with tab2:
        _render_handbook()
    with tab1:

        # --- engine check ---
        try:
            import rlvr_py
            st.sidebar.success(tx["engine_ok"])
        except ImportError:
            st.error(tx["engine_missing"])
            return

        # --- sidebar controls ---
        st.sidebar.header(tx["sidebar_title"])
        n_tech    = st.sidebar.slider(tx["n_tech"],    2, 10, 5)
        n_orders  = st.sidebar.slider(tx["n_orders"],  4, 20, 10)
        epsilon   = st.sidebar.slider(tx["epsilon"],   0.0, 1.0, 0.5, 0.05)
        gamma     = st.sidebar.slider(tx["gamma"],     0.5, 1.0, 0.95, 0.01)
        seed      = st.sidebar.number_input(tx["seed"], 0, 9999, 42)
        n_ep      = st.sidebar.slider(tx["n_episodes"], 5, 100, 30)

        # --- guide ---
        with st.expander(tx["guide_title"], expanded=False):
            st.markdown(tx["guide"])

        # --- run button ---
        run = st.button(tx["run_btn"], type="primary")

        if run:
            with st.spinner("Running Rust engine..."):
                raw = rlvr_py.run_ch01_episode(
                    int(seed), int(n_tech), int(n_orders),
                    float(epsilon), float(gamma)
                )
            result = json.loads(raw) if isinstance(raw, str) else raw
            steps  = result["steps"]
            st.session_state["ch01_steps"]  = steps
            st.session_state["ch01_result"] = result
            st.session_state["ch01_lang"]   = lang

            # learning curve
            curve_data = []
            for ep in range(n_ep):
                ep_raw = rlvr_py.run_ch01_episode(
                    int(seed) + ep, int(n_tech), int(n_orders),
                    float(epsilon), float(gamma)
                )
                ep_res = json.loads(ep_raw) if isinstance(ep_raw, str) else ep_raw
                curve_data.append(ep_res["total_gt"])
            st.session_state["ch01_curve"] = curve_data

        # --- render if data available ---
        if "ch01_steps" not in st.session_state:
            st.info("Configure settings in the sidebar and click **▶ Run Episode**.")
            _render_theory(tx, lang)
            return

        steps  = st.session_state["ch01_steps"]
        result = st.session_state["ch01_result"]
        curve  = st.session_state.get("ch01_curve", [])

        n_steps = len(steps)
        sla_count   = sum(1 for s in steps if s["sla_met"])
        skill_count = sum(1 for s in steps if s["skill_match"])
        exp_count   = sum(1 for s in steps if s["explored"])
        avg_dist    = sum(s["distance_km"] for s in steps) / max(n_steps, 1)
        avg_reward  = sum(s["reward"] for s in steps) / max(n_steps, 1)
        sla_rate    = sla_count / max(n_steps, 1)
        skill_rate  = skill_count / max(n_steps, 1)
        exp_rate    = exp_count / max(n_steps, 1)
        total_gt    = result["total_gt"]
        sla_saved   = sla_count  # each SLA met = 1 penalty avoided

        # --- KPI row ---
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(tx["metric_gt"],      f"{total_gt:.1f}")
        c2.metric(tx["metric_sla"],     f"{sla_rate*100:.1f}%")
        c3.metric(tx["metric_skill"],   f"{skill_rate*100:.1f}%")
        c4.metric(tx["metric_explore"], f"{exp_rate*100:.1f}%")
        c5.metric(tx["metric_dist"],    f"{avg_dist:.1f} km")

        # --- step slider ---
        sel = st.slider(tx["step_slider"], 0, max(n_steps - 1, 0), 0)

        # --- map ---
        st.subheader(tx["map_title"])
        _render_map(steps, sel, tx)
        st.caption(tx["map_caption"])

        # --- glass-box ---
        st.subheader(tx["glass_title"])
        _render_glass_box(steps, sel, tx, gamma)

        # --- learning curve ---
        if curve:
            st.subheader(tx["curve_title"])
            _render_curve(curve, tx)

        # --- episode summary ---
        st.subheader(tx["summary_title"])
        _render_summary(steps, result, sla_rate, skill_rate, exp_rate,
                        avg_dist, avg_reward, sla_saved, total_gt, tx)

        # --- theory ---
        _render_theory(tx, lang)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
def _render_map(steps, sel, tx):
    if not steps:
        return

    fig = go.Figure()

    # collect unique tech and order positions
    techs  = {}
    orders = {}
    for s in steps:
        techs[s["tech_idx"]]   = (s["tech_x"],  s["tech_y"])
        orders[s["order_idx"]] = (s["order_x"], s["order_y"])

    # dispatch lines
    for i, s in enumerate(steps):
        color = "rgba(0,200,80,0.6)" if s["sla_met"] else "rgba(220,50,50,0.6)"
        width = 4 if i == sel else 1.5
        if i == sel:
            color = "rgba(255,165,0,0.95)"
        fig.add_trace(go.Scattermapbox(
            lat=[s["tech_y"], s["order_y"]],
            lon=[s["tech_x"], s["order_x"]],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="skip",
            showlegend=False,
        ))

    # technicians
    t_lats = [v[1] for v in techs.values()]
    t_lons = [v[0] for v in techs.values()]
    t_text = [f"T{k}" for k in techs.keys()]
    fig.add_trace(go.Scattermapbox(
        lat=t_lats, lon=t_lons,
        mode="markers+text",
        marker=dict(size=14, color="royalblue"),
        text=t_text, textposition="top right",
        name="Technicians",
        hovertemplate="<b>%{text}</b><br>lat: %{lat:.4f}<br>lon: %{lon:.4f}<extra></extra>",
    ))

    # work orders
    o_lats  = [v[1] for v in orders.values()]
    o_lons  = [v[0] for v in orders.values()]
    o_text  = [f"W{k}" for k in orders.keys()]
    o_color = []
    for k in orders.keys():
        matched = next((s for s in steps if s["order_idx"] == k), None)
        if matched:
            o_color.append("red" if not matched["sla_met"] else "orange")
        else:
            o_color.append("grey")
    fig.add_trace(go.Scattermapbox(
        lat=o_lats, lon=o_lons,
        mode="markers+text",
        marker=dict(size=12, color=o_color),
        text=o_text, textposition="top left",
        name="Work Orders",
        hovertemplate="<b>%{text}</b><br>lat: %{lat:.4f}<br>lon: %{lon:.4f}<extra></extra>",
    ))

    # highlight selected step
    s = steps[sel]
    fig.add_trace(go.Scattermapbox(
        lat=[s["tech_y"], s["order_y"]],
        lon=[s["tech_x"], s["order_x"]],
        mode="markers",
        marker=dict(size=18, color="orange", symbol="circle"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=52.23, lon=21.01), zoom=10),
        margin=dict(l=0, r=0, t=0, b=0),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Glass-Box
# ---------------------------------------------------------------------------
def _render_glass_box(steps, sel, tx, gamma):
    rows = []
    for i, s in enumerate(steps):
        highlight = "**" if i == sel else ""
        rows.append({
            tx["glass_headers"][0]: f"{highlight}{i}{highlight}",
            tx["glass_headers"][1]: f"T{s['tech_idx']}",
            tx["glass_headers"][2]: f"W{s['order_idx']}",
            tx["glass_headers"][3]: tx["skill_ok"] if s["skill_match"] else tx["skill_no"],
            tx["glass_headers"][4]: f"{s['distance_km']:.1f} km",
            tx["glass_headers"][5]: f"{s['urgency']:.2f}",
            tx["glass_headers"][6]: f"{s['reward']:+.1f}",
            tx["glass_headers"][7]: f"{s['gt']:.2f}",
            tx["glass_headers"][8]: tx["sla_met"] if s["sla_met"] else tx["sla_breach"],
            tx["glass_headers"][9]: tx["explore"] if s["explored"] else tx["exploit"],
        })

    st.dataframe(rows, width='stretch', height=300)

    # selected step detail
    s = steps[sel]
    st.markdown(f"""
**Step {sel} detail:**
- Tech **T{s['tech_idx']}** (skill: `{s['tech_skill']}`) at ({s['tech_y']:.4f}°N, {s['tech_x']:.4f}°E)
- Order **W{s['order_idx']}** (required: `{s['order_skill']}`) at ({s['order_y']:.4f}°N, {s['order_x']:.4f}°E)
- Distance: **{s['distance_km']:.2f} km** · Urgency: **{s['urgency']:.2f}**
- Reward Rₜ = **{s['reward']:+.2f}** · Return Gₜ = **{s['gt']:.3f}**
- Mode: {tx['explore'] if s['explored'] else tx['exploit']} · ε = {s['epsilon']:.2f}
""")

    # Bellman equation (greyed out)
    st.markdown("---")
    st.caption(tx["bellman_caption"])
    st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha \left[ R + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]")
    st.markdown(
        "<p style='color:grey;font-size:0.8em'>Q-table all zeros in Ch01 — "
        "Bellman updates activate in Chapter 02</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------
def _render_curve(curve, tx):
    import plotly.graph_objects as go
    n = len(curve)
    rolling = []
    w = 5
    for i in range(n):
        window = curve[max(0, i - w + 1): i + 1]
        rolling.append(sum(window) / len(window))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=curve,
        mode="lines+markers",
        name=tx["curve_y"],
        line=dict(color="royalblue", width=1.5),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=rolling,
        mode="lines",
        name=tx["curve_mean"],
        line=dict(color="orange", width=2.5, dash="dash"),
    ))
    fig.add_hrect(y0=min(curve) - 1, y1=0,
                  fillcolor="red", opacity=0.05, line_width=0)
    fig.update_layout(
        xaxis_title=tx["curve_x"],
        yaxis_title=tx["curve_y"],
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Episode summary
# ---------------------------------------------------------------------------
def _render_summary(steps, result, sla_rate, skill_rate, exp_rate,
                    avg_dist, avg_reward, sla_saved, total_gt, tx):
    n = len(steps)

    st.markdown(f"#### {tx['summary_results']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
| Metric | Value |
|---|---|
| {tx['metric_gt']} | **{total_gt:.2f}** |
| {tx['metric_sla']} | **{sla_rate*100:.1f}%** ({int(sla_rate*n)}/{n} orders) |
| {tx['metric_skill']} | **{skill_rate*100:.1f}%** |
| {tx['metric_explore']} | **{exp_rate*100:.1f}%** |
| {tx['metric_dist']} | **{avg_dist:.1f} km** |
| {tx['metric_reward']} | **{avg_reward:+.2f}** |
| {tx['metric_sla_saved']} | **{sla_saved}** × 5.0 = **{sla_saved*5.0:.0f} pts** |
""")
    with col2:
        # business impact framing
        penalty_avoided = sla_saved * 500  # €500 per SLA breach avoided (illustrative)
        fuel_saved = max(0, (20.0 - avg_dist) * n * 0.15)  # €0.15/km delta
        st.markdown(f"""
**Business Impact (illustrative)**
- SLA penalties avoided: **€{penalty_avoided:,.0f}**
- Fuel savings vs random baseline: **€{fuel_saved:,.0f}**
- Avg dispatch quality score: **{(sla_rate*0.6 + skill_rate*0.4)*100:.1f}/100**

*Note: Ch01 uses untrained Q-table (ε-greedy on zeros).
Ch02 will train the Q-table and improve these numbers.*
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
def _render_theory(tx, lang):
    st.markdown("---")
    st.subheader(tx["theory_title"])

    active = st.session_state.get("theory_active", None)

    sections = [
        ("mdp",     tx["theory_sections"]["mdp"],     tx["theory_mdp"]),
        ("egreedy", tx["theory_sections"]["egreedy"],  tx["theory_egreedy"]),
        ("gt",      tx["theory_sections"]["gt"],       tx["theory_gt"]),
        ("ndarray", tx["theory_sections"]["ndarray"],  tx["theory_ndarray"]),
        ("reward",  tx["theory_sections"]["reward"],   tx["theory_reward"]),
    ]

    for key, label, content in sections:
        expanded = (key == active)
        with st.expander(label, expanded=expanded):
            st.markdown(content)
