
import os
import streamlit as st
import plotly.graph_objects as go
import json

# ---------------------------------------------------------------------------
# Language strings — EN, PL, FR, DE, ES
# ---------------------------------------------------------------------------
T = {"EN": {
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
    }}

# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def _tx(lang=None):
    import copy
    return copy.deepcopy(T.get("EN", {}))

def _render_handbook_pl():
    _plcol1, _plcol2 = st.columns([8, 1])
    with _plcol1:
        st.subheader("Hands-On Guide — Rozdział 01 (PL)")
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

def _render_handbook():
    import os as _os
    _col1, _col2 = st.columns([8, 1])
    with _col1:
        st.subheader("Hands-On Guide — Chapter 01 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch01_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("💾 Save", data=_html, file_name="handson_ch01_en.html", mime="text/html")
    st.iframe(_html, height=4000)

def render():
    lang = "EN"
    # --- language selector (top of sidebar) ---
    # --- language selector (radio, sidebar — top) ---

    tx = _tx(lang)

    st.title(tx["title"])
    st.caption(tx["subtitle"])

    tab1, tab2, tab3 = st.tabs(["🧪 Interactive Lab", "📘 Hands-On Guide EN", "📙 Hands-On Guide PL"])
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

    # Auto-fit bounds to all technicians and work orders
    all_lats = [v[1] for v in techs.values()] + [v[1] for v in orders.values()]
    all_lons = [v[0] for v in techs.values()] + [v[0] for v in orders.values()]
    lat_center = (min(all_lats) + max(all_lats)) / 2
    lon_center = (min(all_lons) + max(all_lons)) / 2
    lat_range  = max(all_lats) - min(all_lats)
    lon_range  = max(all_lons) - min(all_lons)
    # Add 30% padding and compute zoom level
    padded = max(lat_range, lon_range, 0.001) * 1.3
    import math
    zoom = round(8.0 - math.log2(padded))
    zoom = max(10, min(12, zoom))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat_center, lon=lon_center),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Glass-Box
# ---------------------------------------------------------------------------
def _render_glass_box(steps, sel, tx, gamma):
    # Episode selector info
    st.caption(f"Showing episode step trace — {len(steps)} steps total. Use the Step slider above to highlight a specific dispatch decision.")
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
    st.plotly_chart(fig, use_container_width=True)


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