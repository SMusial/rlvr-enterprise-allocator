
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
    _hcol1, _hcol2 = st.columns([8, 1])
    with _hcol1:
        st.subheader("Hands-On Guide — Chapter 01")
    with _hcol2:
        import re as _re
        _src = open(__file__, encoding="utf-8").read()
        _m = _re.search(r'st\.iframe\(\s*"""(.*?)"""', _src, _re.DOTALL)
        if _m:
            st.download_button("💾 Save", data=_m.group(1), file_name="handson_ch01_en.html", mime="text/html")
    st.iframe(
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hands-On Guide &mdash; Chapter 01</title>
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
.card.purple{border-left-color:#8B5CF6}
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
<h1>&#x1F4D8; Hands-On Guide &mdash; Chapter 01</h1>
<p style="color:#9ca3af;margin-bottom:1.5rem"><em>MDP &amp; ASP Dispatch &middot; Warsaw &middot; Rust Engine &middot; Interactive Lab</em></p>
<div class="tabs">
  <button class="tab-btn active" onclick="showTab('intro')">&#x1F4D6; Introduction</button>
  <button class="tab-btn" onclick="showTab('what')">&#x2753; What is Ch01?</button>
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
<li>Define the 5 components of an MDP: S, A, P, R, &gamma;</li>
<li>Explain what a state, action, reward and return are in the ASP context</li>
<li>Implement &epsilon;-greedy exploration and explain the exploration-exploitation trade-off</li>
<li>Read the Warsaw map and interpret dispatch decisions as MDP transitions</li>
<li>Read the Glass-Box and trace a full MDP tuple: (S&#x209C;, A&#x209C;, R&#x209C;, G&#x209C;)</li>
<li>Explain why the Bellman equation is greyed out in Ch01 (activates in Ch02)</li>
</ul>
</div>
<h2>&#x1F3E2; Business Problem</h2>
<div class="card blue"><strong>Warsaw ASP Dispatch Centre</strong> &mdash; 5 technicians, up to 20 work orders per shift.<br><br>
Every dispatch decision is an <strong>MDP action</strong>. The outcome is the <strong>reward</strong>. The sequence of decisions is an <strong>episode</strong>.<br><br>
In Ch01 the Q-table is all zeros &mdash; this is the <em>random baseline</em> against which all future chapters are measured.</div>
<div class="kpi">
<div class="kpi-card"><div class="kpi-val">5</div><div class="kpi-label">MDP Components</div></div>
<div class="kpi-card"><div class="kpi-val">&epsilon;</div><div class="kpi-label">Exploration rate</div></div>
<div class="kpi-card"><div class="kpi-val">G&#x209C;</div><div class="kpi-label">Discounted return</div></div>
<div class="kpi-card"><div class="kpi-val">Q=0</div><div class="kpi-label">Q-table in Ch01</div></div>
</div>
</div>
<div id="what" class="tab-content">
<h2>&#x2753; What is Chapter 01?</h2>
<div class="card purple">Ch01 is the <strong>starting point of the entire RL course</strong>. There is no learning here &mdash; the agent acts randomly. The goal is to understand the MDP formalism and visualisation tools before learning is introduced.</div>
<h2>What happens in the Rust engine?</h2>
<div class="card">Function <code>run_ch01_episode()</code> in <code>ch01_mdp.rs</code>:
<ol>
<li>Generates 5 random technicians and N work orders on the Warsaw map</li>
<li>For each order, selects a technician via &epsilon;-greedy policy (Q=0 &rarr; always random)</li>
<li>Computes reward R&#x209C; based on: skill match, distance, urgency</li>
<li>Computes discounted return G&#x209C; backwards through the episode</li>
<li>Returns full MDP trace: each step (S&#x209C;, A&#x209C;, R&#x209C;, G&#x209C;, explored)</li>
</ol>
</div>
<div class="card red"><strong>Ch01 does NOT:</strong>
<ul>
<li>&#x274C; Update Q-table &mdash; Q stays at zero throughout the episode</li>
<li>&#x274C; Use the Bellman equation &mdash; activates in Ch02</li>
<li>&#x274C; Learn between episodes &mdash; each episode starts fresh</li>
<li>&#x274C; Optimise the policy &mdash; that is Ch06 (Q-Learning)</li>
</ul>
</div>
</div>
<div id="theory" class="tab-content">
<h2>&#x1F9EE; MDP Formalism</h2>
<table>
<tr><th>Component</th><th>Symbol</th><th>ASP meaning</th><th>Example</th></tr>
<tr><td>State space</td><td><strong>S</strong></td><td>Operational situation of the dispatch centre</td><td>S3: partial availability, high load</td></tr>
<tr><td>Action space</td><td><strong>A</strong></td><td>Which technician to dispatch to which order</td><td>Send T2 to W5</td></tr>
<tr><td>Transition model</td><td><strong>P(s'|s,a)</strong></td><td>Probability of the next state</td><td>After dispatch, T2 becomes unavailable</td></tr>
<tr><td>Reward function</td><td><strong>R(s,a)</strong></td><td>Immediate feedback for the dispatch decision</td><td>+10 SLA met, &minus;50 SLA breached</td></tr>
<tr><td>Discount factor</td><td><strong>&gamma;</strong></td><td>How much future rewards are valued</td><td>&gamma;=0.95: future rewards worth 95% of immediate</td></tr>
</table>
<h2>Discounted Return G&#x209C;</h2>
<div class="formula">G&#x209C; = R&#x209C;&#x208A;&#x2081; + &gamma; R&#x209C;&#x208A;&#x2082; + &gamma;&sup2; R&#x209C;&#x208A;&#x2083; + &hellip;</div>
<div class="card"><strong>Worked example</strong> (&gamma;=0.95): G&#x2080; = 10 + 0.95&times;(&minus;5) + 0.95&sup2;&times;10 = <strong>14.275</strong></div>
<h2>&epsilon;-Greedy Policy</h2>
<div class="card orange"><strong>With probability &epsilon;:</strong> choose a random action (explore)<br>
<strong>With probability 1&minus;&epsilon;:</strong> choose the best known action (exploit)<br><br>
In Ch01 Q=0 &mdash; exploit = random too. &epsilon; only matters from Ch06 onwards.</div>
<table>
<tr><th>&epsilon; value</th><th>Behaviour</th><th>When to use</th></tr>
<tr><td>1.0</td><td>Always random</td><td>Start of training &mdash; know nothing</td></tr>
<tr><td>0.5</td><td>50/50 explore/exploit</td><td>Mid-training</td></tr>
<tr><td>0.1</td><td>Mostly exploit</td><td>Late training &mdash; policy nearly optimal</td></tr>
<tr><td>0.0</td><td>Always greedy</td><td>Evaluation only (no learning)</td></tr>
</table>
<h2>Bellman Equation (greyed out in Ch01)</h2>
<div class="card red">The Bellman equation requires the transition model P(s'|s,a):<br>
<div class="formula">V*(s) = max_a &sum; P(s'|s,a) [ R(s,a) + &gamma; V*(s') ]</div>
In Ch01 we don't know P(s'|s,a) &mdash; that's why it's greyed out. Ch02 builds the transition matrix and activates it.</div>
</div>
<div id="env" class="tab-content">
<h2>&#x1F5FA; Warsaw ASP Environment</h2>
<h3>Technicians (T0&ndash;T4)</h3>
<table>
<tr><th>Attribute</th><th>Effect on reward</th></tr>
<tr><td>Position (lat, lon)</td><td>Distance to order &rarr; travel time</td></tr>
<tr><td>Skill</td><td>Match to order = bonus, mismatch = penalty</td></tr>
<tr><td>Availability</td><td>Unavailable technician cannot be dispatched</td></tr>
</table>
<h3>Work Orders (W0&ndash;W9)</h3>
<table>
<tr><th>Attribute</th><th>Effect on reward</th></tr>
<tr><td>Position (lat, lon)</td><td>Distance from technician</td></tr>
<tr><td>Required skill</td><td>Mismatch &rarr; penalty</td></tr>
<tr><td>Urgency (0.0&ndash;1.0)</td><td>High urgency + delay &rarr; large penalty</td></tr>
</table>
<h3>Reward Function R(s,a)</h3>
<div class="card"><code>R = base + skill_bonus - distance_penalty - urgency_penalty + sla_bonus/penalty</code></div>
</div>
<div id="ui" class="tab-content">
<h2>&#x1F3AE; How to use the Ch01 interface</h2>
<div class="step"><div class="step-num">1</div><div><strong>Set &epsilon; (exploration rate)</strong><br>Move the slider. &epsilon;=1.0: always random. &epsilon;=0.0: always greedy (in Ch01 = random too since Q=0). Start with &epsilon;=0.5.</div></div>
<div class="step"><div class="step-num">2</div><div><strong>Set number of technicians and orders</strong><br>5 technicians / 10 orders is a good starting point. More orders = longer episode.</div></div>
<div class="step"><div class="step-num">3</div><div><strong>Click &#x25B6; Run Episode</strong><br>The Rust engine executes the full MDP loop and returns every step.</div></div>
<div class="step"><div class="step-num">4</div><div><strong>Read the Warsaw map</strong><br>Blue markers = technicians (T0&ndash;T4). Coloured markers = orders (W0&ndash;W9). Green lines = SLA met. Red lines = SLA breached. Click any marker for details.</div></div>
<div class="step"><div class="step-num">5</div><div><strong>Use the step slider</strong><br>Move to highlight a specific dispatch decision on the map and in the Glass-Box.</div></div>
<div class="step"><div class="step-num">6</div><div><strong>Read the Glass-Box</strong><br>Each row shows the full MDP tuple: S&#x209C;, A&#x209C;, R&#x209C;, G&#x209C;. The Bellman column is greyed out &mdash; activates in Ch02.</div></div>
<div class="step"><div class="step-num">7</div><div><strong>Read the episode summary</strong><br>Quantified business results + pros/cons of the &epsilon;-greedy method.</div></div>
</div>
<div id="interp" class="tab-content">
<h2>&#x1F4CA; Interpreting results</h2>
<h3>Warsaw Map</h3>
<div class="card">Green lines = SLA met &mdash; dispatch was on time with the right skills.<br>
Red lines = SLA breached &mdash; too far, wrong skills, or too slow.<br><br>
More green lines = better episode. In Ch01 the ratio is random.</div>
<h3>Glass-Box &mdash; MDP table</h3>
<table>
<tr><th>Column</th><th>Meaning</th><th>Example</th></tr>
<tr><td><code>S&#x209C;</code></td><td>State at time t</td><td>S3: partial availability</td></tr>
<tr><td><code>A&#x209C;</code></td><td>Action taken</td><td>Send T2 &rarr; W5</td></tr>
<tr><td><code>R&#x209C;</code></td><td>Immediate reward</td><td>+10.0 (SLA met)</td></tr>
<tr><td><code>G&#x209C;</code></td><td>Discounted return from this step</td><td>14.275</td></tr>
</table>
<div class="card red"><strong>Bellman greyed out</strong> &mdash; requires P(s'|s,a) which we don't have in Ch01. Activates in Ch02.</div>
<h3>Learning Curve</h3>
<div class="card blue">In Ch01 the learning curve is <strong>flat</strong> &mdash; the agent does not learn between episodes because Q=0.<br>This is intentional. Ch01 establishes the <em>random baseline</em>. From Ch06 onwards you will see the curve rise.</div>
</div>
<div id="exercises" class="tab-content">
<h2>&#x1F9EA; Hands-On Exercises</h2>
<div class="card"><h3>Exercise 1 &mdash; Baseline measurement</h3>Run 5 episodes with &epsilon;=1.0 (pure random). Record the average G&#x209C;. This is your Ch01 baseline. Every future chapter should beat this number.</div>
<div class="card blue"><h3>Exercise 2 &mdash; &epsilon; sensitivity</h3>Run with &epsilon;=0.0 (pure greedy). Is the result better or worse than &epsilon;=1.0? Why? (Hint: Q=0 &mdash; greedy = random in Ch01)</div>
<div class="card orange"><h3>Exercise 3 &mdash; Map reading</h3>Find the step with the largest negative reward in the Glass-Box. Click that step on the map. What went wrong? Wrong skill? Too far? Too slow?</div>
<div class="card green"><h3>Exercise 4 &mdash; Return calculation</h3>Take the first 3 rewards from the Glass-Box and manually compute G&#x2080; using &gamma;=0.95. Verify your answer matches the G&#x209C; column.</div>
<div class="card purple"><h3>Exercise 5 &mdash; &gamma; effect</h3>Run the same episode (same seed) with &gamma;=0.99 and &gamma;=0.5. How does G&#x2080; change? Which agent is more &ldquo;far-sighted&rdquo;?</div>
</div>
<div id="summary" class="tab-content">
<h2>&#x1F4CB; Chapter 01 Summary</h2>
<div class="kpi">
<div class="kpi-card"><div class="kpi-val">5</div><div class="kpi-label">MDP Components</div></div>
<div class="kpi-card"><div class="kpi-val">Q=0</div><div class="kpi-label">No learning</div></div>
<div class="kpi-card"><div class="kpi-val">&epsilon;</div><div class="kpi-label">Only hyperparameter</div></div>
<div class="kpi-card"><div class="kpi-val">Ch02</div><div class="kpi-label">Next: Bellman + VI</div></div>
</div>
<div class="grid2">
<div class="card green"><strong>&#x2705; Pros</strong><ul><li>Simple implementation</li><li>Establishes baseline</li><li>Visualises MDP formalism</li><li>Works without P(s'|s,a)</li></ul></div>
<div class="card red"><strong>&#x274C; Cons</strong><ul><li>No learning &mdash; Q=0 always</li><li>Does not optimise policy</li><li>Results are purely random</li><li>Does not use Bellman</li></ul></div>
</div>
<div class="card green">Ch01 establishes the <strong>random baseline</strong>. The Q-table is all zeros. Every algorithm from Ch02 onwards will learn to beat this baseline by updating Q(s,a) after each step.</div>
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
        f"&#x1F1F5;&#x1F1F1; **Wersja polska:** [Podr\u0119cznik Rozdzia&#x142; 01 (PL)](https://raw.githubusercontent.com/SMusial/rlvr-enterprise-allocator/main/docs/handson_ch01_pl.html)"
        " &#x2014; otwiera si\u0119 w osobnym oknie przegl\u0105darki",
        unsafe_allow_html=False,
    )

def _render_handbook_pl():
    _plcol1, _plcol2 = st.columns([8, 1])
    with _plcol1:
        st.subheader("Hands-On Guide — Chapter 01 (PL)")
    with _plcol2:
        import re as _re2
        _src2 = open(__file__, encoding="utf-8").read()
        _m2 = _re2.search(r'def _render_handbook_pl.*?st\.iframe\(\s*"""(.*?)"""', _src2, _re2.DOTALL)
        if _m2:
            st.download_button("💾 Save", data=_m2.group(1), file_name="handson_ch01_pl.html", mime="text/html")
    st.iframe(
        """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Podrecznik - Rozdzial 01</title>
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
.card.purple{border-left-color:#8B5CF6}
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
<h1>&#x1F4D8; Podr&#x119;cznik &#x2014; Rozdzia&#x142; 01</h1>
<p style="color:#9ca3af;margin-bottom:1.5rem"><em>MDP i Dyspozycja ASP &middot; Warszawa &middot; Silnik Rust</em></p>
<div class="tabs">
  <button class="tab-btn active" onclick="showTab('intro')">&#x1F4D6; Wprowadzenie</button>
  <button class="tab-btn" onclick="showTab('what')">&#x2753; Czym jest Ch01?</button>
  <button class="tab-btn" onclick="showTab('theory')">&#x1F9EE; Teoria RL</button>
  <button class="tab-btn" onclick="showTab('env')">&#x1F5FA; &#x15A;rodowisko</button>
  <button class="tab-btn" onclick="showTab('ui')">&#x1F3AE; Jak u&#x17C;ywa&#x107; UI</button>
  <button class="tab-btn" onclick="showTab('interp')">&#x1F4CA; Interpretacja</button>
  <button class="tab-btn" onclick="showTab('exercises')">&#x1F9EA; &#x106;wiczenia</button>
  <button class="tab-btn" onclick="showTab('summary')">&#x1F4CB; Podsumowanie</button>
</div>
<div id="intro" class="tab-content active">
<h2>&#x1F3AF; Cele nauki</h2>
<div class="card green">Po uko&#x144;czeniu tego rozdzia&#x142;u b&#x119;dziesz potrafi&#x142;:
<ul>
<li>Zdefiniowa&#x107; 5 sk&#x142;adnik&#xF3;w MDP: S, A, P, R, &#x3B3;</li>
<li>Wyja&#x15B;ni&#x107; czym jest stan, akcja, nagroda i zwrot w kontek&#x15B;cie ASP</li>
<li>Zaimplementowa&#x107; polityk&#x119; &#x3B5;-zachlann&#x105; i wyja&#x15B;ni&#x107; kompromis eksploracja-eksploatacja</li>
<li>Odczyta&#x107; map&#x119; Warszawy i zinterpretowa&#x107; decyzje dyspozycji jako przej&#x15B;cia MDP</li>
<li>Odczyta&#x107; Glass-Box i prze&#x15B;ledzi&#x107; pe&#x142;n&#x105; krotkMDP: (S&#x209C;, A&#x209C;, R&#x209C;, G&#x209C;)</li>
<li>Wyja&#x15B;ni&#x107; dlaczego r&#xF3;wnanie Bellmana jest wyszarzone w Ch01</li>
</ul>
</div>
<h2>&#x1F3E2; Problem biznesowy</h2>
<div class="card blue"><strong>Centrum Dyspozycji ASP Warszawa</strong> &#x2014; 5 technik&#xF3;w, do 20 zlece&#x144; na zmian&#x119;.<br><br>
Ka&#x17C;da decyzja dyspozycji to <strong>akcja MDP</strong>. Wynik to <strong>nagroda</strong>. Sekwencja decyzji to <strong>epizod</strong>.<br><br>
W Ch01 tabela Q ma same zera &#x2014; to <em>punkt bazowy</em> wzgl&#x119;dem kt&#xF3;rego mierzone s&#x105; wszystkie przysz&#x142;e rozdzia&#x142;y.</div>
<div class="kpi">
<div class="kpi-card"><div class="kpi-val">5</div><div class="kpi-label">Sk&#x142;adnik&#xF3;w MDP</div></div>
<div class="kpi-card"><div class="kpi-val">&#x3B5;</div><div class="kpi-label">Wsp&#xF3;&#x142;. eksploracji</div></div>
<div class="kpi-card"><div class="kpi-val">G&#x209C;</div><div class="kpi-label">Zdyskontowany zwrot</div></div>
<div class="kpi-card"><div class="kpi-val">Q=0</div><div class="kpi-label">Tabela Q w Ch01</div></div>
</div>
</div>
<div id="what" class="tab-content">
<h2>&#x2753; Czym jest Rozdzia&#x142; 01?</h2>
<div class="card purple">Ch01 to <strong>punkt startowy ca&#x142;ego kursu RL</strong>. Nie ma tu uczenia &#x2014; agent dzia&#x142;a losowo.</div>
<h2>Co si&#x119; dzieje w silniku Rust?</h2>
<div class="card">Funkcja <code>run_ch01_episode()</code> w <code>ch01_mdp.rs</code>:
<ol>
<li>Generuje 5 losowych technik&#xF3;w i N zlece&#x144; na mapie Warszawy</li>
<li>Wybiera technika przez polityk&#x119; &#x3B5;-zachlann&#x105; (Q=0 &#x2192; zawsze losowo)</li>
<li>Oblicza nagrod&#x119; R&#x209C; na podstawie dopasowania umiej&#x119;tno&#x15B;ci, odleg&#x142;o&#x15B;ci, pilno&#x15B;ci</li>
<li>Oblicza zdyskontowany zwrot G&#x209C; dla ca&#x142;ego epizodu</li>
<li>Zwraca pe&#x142;ny &#x15B;lad MDP: (S&#x209C;, A&#x209C;, R&#x209C;, G&#x209C;, explored)</li>
</ol>
</div>
<div class="card red"><strong>Ch01 NIE robi:</strong>
<ul>
<li>&#x274C; Nie aktualizuje Q &#x2014; Q=0 przez ca&#x142;y epizod</li>
<li>&#x274C; Nie u&#x17C;ywa Bellmana &#x2014; aktywuje si&#x119; w Ch02</li>
<li>&#x274C; Nie uczy si&#x119; mi&#x119;dzy epizodami</li>
</ul>
</div>
</div>
<div id="theory" class="tab-content">
<h2>&#x1F9EE; Formalizm MDP</h2>
<table><tr><th>Sk&#x142;adnik</th><th>Symbol</th><th>Znaczenie w ASP</th></tr>
<tr><td>Przestrze&#x144; stan&#xF3;w</td><td><strong>S</strong></td><td>Sytuacja operacyjna centrum dyspozycji</td></tr>
<tr><td>Przestrze&#x144; akcji</td><td><strong>A</strong></td><td>Kt&#xF3;rego technika wys&#x142;a&#x107; do kt&#xF3;rego zlecenia</td></tr>
<tr><td>Model przej&#x15B;&#x107;</td><td><strong>P(s'|s,a)</strong></td><td>Prawdopodobie&#x144;stwo nast&#x119;pnego stanu</td></tr>
<tr><td>Funkcja nagrody</td><td><strong>R(s,a)</strong></td><td>Natychmiastowa informacja zwrotna</td></tr>
<tr><td>Wsp&#xF3;&#x142;. dyskonta</td><td><strong>&#x3B3;</strong></td><td>Jak bardzo cenimy przysz&#x142;e nagrody</td></tr></table>
<h2>Zdyskontowany zwrot G&#x209C;</h2>
<div class="formula">G&#x209C; = R&#x209C;+1 + &#x3B3; R&#x209C;+2 + &#x3B3;^2 R&#x209C;+3 + ...</div>
<div class="card"><strong>Przyk&#x142;ad</strong> (&#x3B3;=0.95, 3 kroki):<br>
G0 = 10 + 0.95x(-5) + 0.95^2 x 10 = 10 - 4.75 + 9.025 = <strong>14.275</strong></div>
<h2>Polityka &#x3B5;-zachlanna</h2>
<div class="card orange"><strong>Z prawdopodobie&#x144;stwem &#x3B5;:</strong> losowa akcja (eksploracja)<br>
<strong>Z prawdopodobie&#x144;stwem 1&#x2212;&#x3B5;:</strong> najlepsza znana akcja (eksploatacja)</div>
<table><tr><th>Warto&#x15B;&#x107; &#x3B5;</th><th>Zachowanie</th><th>Kiedy u&#x17C;ywa&#x107;</th></tr>
<tr><td>1.0</td><td>Zawsze losowo</td><td>Pocz&#x105;tek treningu</td></tr>
<tr><td>0.5</td><td>50/50</td><td>&#x15A;rodek treningu</td></tr>
<tr><td>0.1</td><td>G&#x142;&#xF3;wnie eksploatacja</td><td>P&#xF3;&#x17A;ny trening</td></tr>
<tr><td>0.0</td><td>Zawsze zachlanna</td><td>Tylko ewaluacja</td></tr></table>
</div>
<div id="env" class="tab-content">
<h2>&#x1F5FA; &#x15A;rodowisko ASP Warszawa</h2>
<h3>Technicy (T0&#x2013;T4)</h3>
<table><tr><th>Atrybut</th><th>Wp&#x142;yw na nagrod&#x119;</th></tr>
<tr><td>Pozycja (lat, lon)</td><td>Odleg&#x142;o&#x15B;&#x107; do zlecenia</td></tr>
<tr><td>Umiej&#x119;tno&#x15B;&#x107;</td><td>Dopasowanie = bonus, brak = kara</td></tr>
<tr><td>Dost&#x119;pno&#x15B;&#x107;</td><td>Niedost&#x119;pny nie mo&#x17C;e by&#x107; wys&#x142;any</td></tr></table>
<h3>Zlecenia (W0&#x2013;W9)</h3>
<table><tr><th>Atrybut</th><th>Wp&#x142;yw na nagrod&#x119;</th></tr>
<tr><td>Pozycja (lat, lon)</td><td>Odleg&#x142;o&#x15B;&#x107; od technika</td></tr>
<tr><td>Wymagana umiej&#x119;tno&#x15B;&#x107;</td><td>Niedopasowanie = kara</td></tr>
<tr><td>Pilno&#x15B;&#x107; (0.0&#x2013;1.0)</td><td>Wysoka + op&#xF3;&#x17A;nienie = du&#x17C;a kara</td></tr></table>
<h3>Funkcja nagrody</h3>
<div class="card"><code>R = base + skill_bonus - distance_penalty - urgency_penalty + sla_bonus/penalty</code></div>
</div>
<div id="ui" class="tab-content">
<h2>&#x1F3AE; Jak u&#x17C;ywa&#x107; interfejsu Ch01</h2>
<div class="step"><div class="step-num">1</div><div><strong>Ustaw &#x3B5;</strong> &#x2014; zacznij od 0.5</div></div>
<div class="step"><div class="step-num">2</div><div><strong>Ustaw liczb&#x119; technik&#xF3;w i zlece&#x144;</strong> &#x2014; 5/10 to dobry start</div></div>
<div class="step"><div class="step-num">3</div><div><strong>Kliknij &#x25B6; Uruchom epizod</strong></div></div>
<div class="step"><div class="step-num">4</div><div><strong>Odczytaj map&#x119;</strong> &#x2014; zielone = SLA spe&#x142;nione, czerwone = naruszone</div></div>
<div class="step"><div class="step-num">5</div><div><strong>U&#x17C;yj suwaka krok&#xF3;w</strong> &#x2014; prze&#x15B;led&#x17A; ka&#x17C;d&#x105; decyzj&#x119;</div></div>
<div class="step"><div class="step-num">6</div><div><strong>Odczytaj Glass-Box</strong> &#x2014; pe&#x142;na krotka MDP per krok</div></div>
<div class="step"><div class="step-num">7</div><div><strong>Odczytaj podsumowanie</strong> &#x2014; KPI biznesowe epizodu</div></div>
</div>
<div id="interp" class="tab-content">
<h2>&#x1F4CA; Interpretacja wynik&#xF3;w</h2>
<h3>Glass-Box</h3>
<table><tr><th>Kolumna</th><th>Znaczenie</th></tr>
<tr><td><code>S&#x209C;</code></td><td>Stan w chwili t</td></tr>
<tr><td><code>A&#x209C;</code></td><td>Podj&#x119;ta akcja</td></tr>
<tr><td><code>R&#x209C;</code></td><td>Natychmiastowa nagroda</td></tr>
<tr><td><code>G&#x209C;</code></td><td>Zdyskontowany zwrot od tego kroku</td></tr></table>
<div class="card red"><strong>Bellman wyszarzony</strong> &#x2014; wymaga P(s'|s,a). Aktywuje si&#x119; w Ch02.</div>
<div class="card blue"><strong>Krzywa uczenia p&#x142;aska</strong> &#x2014; Q=0, agent nie uczy si&#x119;. Baseline dla przysz&#x142;ych rozdzia&#x142;&#xF3;w.</div>
</div>
<div id="exercises" class="tab-content">
<h2>&#x1F9EA; &#x106;wiczenia</h2>
<div class="card"><h3>&#x106;wiczenie 1 &#x2014; Punkt bazowy</h3>Uruchom 5 epizod&#xF3;w z &#x3B5;=1.0. Zapisz &#x15B;redni G&#x209C;. To Tw&#xF3;j baseline Ch01.</div>
<div class="card blue"><h3>&#x106;wiczenie 2 &#x2014; Wra&#x17C;liwo&#x15B;&#x107; na &#x3B5;</h3>Por&#xF3;wnaj &#x3B5;=1.0 vs &#x3B5;=0.0. Dlaczego wyniki s&#x105; podobne? (Q=0)</div>
<div class="card orange"><h3>&#x106;wiczenie 3 &#x2014; Czytanie mapy</h3>Znajd&#x17A; krok z najwi&#x119;ksz&#x105; ujemn&#x105; nagrod&#x105;. Co posz&#x142;o nie tak?</div>
<div class="card green"><h3>&#x106;wiczenie 4 &#x2014; R&#x119;czne G</h3>We&#x17A; pierwsze 3 nagrody i oblicz G0 r&#x119;cznie (&#x3B3;=0.95). Zweryfikuj z Glass-Box.</div>
</div>
<div id="summary" class="tab-content">
<h2>&#x1F4CB; Podsumowanie</h2>
<div class="kpi">
<div class="kpi-card"><div class="kpi-val">5</div><div class="kpi-label">Sk&#x142;adnik&#xF3;w MDP</div></div>
<div class="kpi-card"><div class="kpi-val">Q=0</div><div class="kpi-label">Brak uczenia</div></div>
<div class="kpi-card"><div class="kpi-val">&#x3B5;</div><div class="kpi-label">Jedyny hiperparametr</div></div>
<div class="kpi-card"><div class="kpi-val">Ch02</div><div class="kpi-label">Nast&#x119;pny: Bellman</div></div>
</div>
<div class="grid2">
<div class="card green"><strong>&#x2705; Zalety</strong><ul><li>Prosta implementacja</li><li>Ustanawia baseline</li><li>Wizualizuje MDP</li></ul></div>
<div class="card red"><strong>&#x274C; Wady</strong><ul><li>Brak uczenia</li><li>Czysto losowe</li><li>Nie u&#x17C;ywa Bellmana</li></ul></div>
</div>
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

def render():
    lang = st.session_state.get("lang", "EN")
    # --- language selector (top of sidebar) ---
    # --- language selector (radio, sidebar — top) ---

    tx = _tx(lang)

    st.title(tx["title"])
    st.caption(tx["subtitle"])

    tab1, tab2, tab3 = st.tabs(["🧪 Interactive Lab", "📘 Hands-On Guide EN", "🇵🇱 Hands-On Guide PL"])
    with tab2:
        _render_handbook()
    with tab3:
        _render_handbook_pl()
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
