
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
Przesuń suwak.
ε=1.0: agent zawsze wybiera losowo.
ε=0.0: agent zawsze wybiera najlepszą znaną akcję (Q=0 w Ch01, również losowe).
Zacznij od ε=0.5.

**Krok 2 — Ustaw techników i zlecenia**
5 techników / 10 zleceń to dobry punkt startowy.
Więcej zleceń = dłuższy epizod.

**Krok 3 — Kliknij ▶ Uruchom epizod**
Silnik Rust wykonuje pełną pętlę MDP i zwraca każdy krok.

**Krok 4 — Odczytaj mapę Warszawy**
Niebieskiemarker = technicy (T0–T4).
Kolorowe markery = zlecenia (W0–W9).
Zielone linie = SLA spełnione. Czerwone linie = SLA naruszone.
Kliknij marker aby zobaczyć szczegóły.

**Krok 5 — Użyj suwaka kroków**
Przesuń aby podświetlić konkretna decyzję dyspozycji na mapie i w Glass-Box.

**Krok 6 — Odczytaj Glass-Box**
Każdy wiersz pokazuje pełną krotkMDP: Sₜ, Aₜ, Rₜ, Gₜ.
Równanie Bellmana jest wyszarzone — aktywuje się w Rozdziale 02.

**Krok 7 — Odczytaj podsumowanie epizodu**
Skwantyfikowane wyniki biznesowe + zalety/wady metody ε-zachlannej.
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
**Étape 1 — Régler ε (taux d'exploration)**
Déplacer le curseur.
ε=1.0 : l'agent choisit toujours aléatoirement.
ε=0.0 : l'agent choisit toujours la meilleure action connue (Q-table nulle en Ch01, aussi aléatoire).
Commencer avec ε=0.5.

**Étape 2 — Régler techniciens et ordres de travail**
5 techniciens / 10 ordres de travail est un bon point de départ.
Plus d'ordres = épisode plus long.

**Étape 3 — Cliquer ▶ Lancer l'épisode**
Le moteur Rust exécute la boucle MDP complète et retourne chaque étape.

**Étape 4 — Lire la carte de Varsovie**
Marqueurs bleus = techniciens (T0–T4).
Marqueurs colorés = ordres de travail (W0–W9).
Lignes vertes = SLA respecté. Lignes rouges = SLA violé.
Cliquer sur un marqueur pour les détails.

**Étape 5 — Utiliser le curseur d'étape**
Déplacer pour mettre en évidence une décision de dispatch sur la carte et dans le Glass-Box.

**Étape 6 — Lire le Glass-Box**
Chaque ligne montre le tuple MDP complet : Sₜ, Aₜ, Rₜ, Gₜ.
L'équation de Bellman est grisée — elle s'active au Chapitre 02.

**Étape 7 — Lire le résumé de l'épisode**
Résultats commerciaux quantifiés + avantages/inconvénients de la méthode ε-greedy.
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
**Schritt 1 — ε einstellen (Explorationsrate)**
Schieberegler bewegen.
ε=1.0: Agent wählt immer zufällig (reine Exploration).
ε=0.0: Agent wählt immer die beste bekannte Aktion (da Q-Tabelle null ist, auch zufällig).
Mit ε=0.5 beginnen.

**Schritt 2 — Techniker und Aufträge einstellen**
5 Techniker / 10 Aufträge ist ein guter Ausgangspunkt.
Mehr Aufträge = längere Episode.

**Schritt 3 — ▶ Episode starten klicken**
Die Rust-Engine führt die vollständige MDP-Schleife aus und gibt jeden Schritt zurück.

**Schritt 4 — Warschau-Karte lesen**
Blaue Marker = Techniker (T0–T4).
Farbige Marker = Aufträge (W0–W9).
Grüne Linien = SLA erfüllt. Rote Linien = SLA verletzt.
Auf einen Marker klicken für Details.

**Schritt 5 — Schritt-Schieberegler verwenden**
Verschieben, um eine bestimmte Dispatchentscheidung auf der Karte und im Glass-Box hervorzuheben.

**Schritt 6 — Glass-Box lesen**
Jede Zeile zeigt das vollständige MDP-Tupel: Sₜ (Zustand), Aₜ (Aktion), Rₜ (Belohnung), Gₜ (Gesamtbelohnung).
Die Bellman-Gleichung ist ausgegraut — sie aktiviert sich in Kapitel 02.

**Schritt 7 — Zusammenfassung lesen**
Quantifizierte Geschäftsergebnisse + Vor- und Nachteile der ε-greedy-Methode.
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
**Paso 1 — Ajustar ε (tasa de exploración)**
Mover el control deslizante.
ε=1.0: el agente elige siempre aleatoriamente.
ε=0.0: el agente elige siempre la mejor acción conocida (Q-table cero en Ch01, también aleatorio).
Comenzar con ε=0.5.

**Paso 2 — Ajustar técnicos y órdenes de trabajo**
5 técnicos / 10 órdenes de trabajo es un buen punto de partida.
Más órdenes = episodio más largo.

**Paso 3 — Hacer clic en ▶ Ejecutar episodio**
El motor Rust ejecuta el bucle MDP completo y devuelve cada paso.

**Paso 4 — Leer el mapa de Varsovia**
Marcadores azules = técnicos (T0–T4).
Marcadores de colores = órdenes de trabajo (W0–W9).
Líneas verdes = SLA cumplido. Líneas rojas = SLA incumplido.
Hacer clic en cualquier marcador para ver detalles.

**Paso 5 — Usar el control deslizante de pasos**
Mover para resaltar una decisión de despacho específica en el mapa y en el Glass-Box.

**Paso 6 — Leer el Glass-Box**
Cada fila muestra la tupla MDP completa: Sₜ, Aₜ, Rₜ, Gₜ.
La ecuación de Bellman aparece atenuada — se activa en el Capítulo 02.

**Paso 7 — Leer el resumen del episodio**
Resultados comerciales cuantificados + pros/contras del método ε-greedy.
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
    # --- language selector (top of sidebar) ---
    # --- language selector (radio, sidebar — top) ---

    tx = _tx(lang)

    st.title(tx["title"])
    st.caption(tx["subtitle"])

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
