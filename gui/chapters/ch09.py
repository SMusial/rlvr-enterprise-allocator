import streamlit as st
import plotly.graph_objects as go

T = {
    "EN": {
        "title": "Chapter 09 — Policy Gradient: REINFORCE & Softmax",
        "subtitle": "REINFORCE · REINFORCE+Baseline · Actor-Critic TD(0) · Temperature · Warsaw ASP",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Settings",
        "n_episodes": "Episodes", "gamma": "γ Discount", "alpha": "α Policy LR",
        "alpha_baseline": "α_v Baseline LR", "temperature": "τ Temperature", "seed": "Seed",
        "run_btn": "▶ Run All Four Algorithms",
        "guide_title": "📖 Guide",
        "guide": ("**Step 1 - Policy Gradient vs Q-learning**\n""REINFORCE directly optimises pi(a|s)=softmax(theta[s][a]). No Q-table.\n""Gradient ascent on expected return J(theta).\n\n""**Step 2 - REINFORCE vs Actor-Critic**\n""REINFORCE: Monte Carlo - waits for full episode. Unbiased, high variance.\n""Actor-Critic: TD(0) critic - online per step. Lower variance, biased.\n\n""**Step 3 - Baseline**\n""Subtracting b(s) from G_t reduces variance without bias.\n""Watch REINFORCE+Baseline curve stabilise faster.\n\n""**Step 4 - Temperature tau**\n""tau=1.0 standard. tau<1 sharper policy. tau>1 more uniform.\n\n""**Step 5 - Entropy chart**\n""High entropy = exploring. Low entropy = exploiting.\n""Healthy: entropy decreases gradually."),
        "returns_title": "📈 Episode Returns", "returns_caption": "MA-30. REINFORCE+Baseline most stable.",
        "pg_loss_title": "📉 PG Magnitude", "pg_loss_caption": "Mean |Δθ| per episode.",
        "entropy_title": "🌡️ Policy Entropy", "entropy_caption": "H(π(·|s)). Decreases as policy sharpens.",
        "value_title": "🏛️ V(s)", "value_caption": "S7 (SLA breach) should be lowest.",
        "theta_title": "🗺️ θ[s][a] Heatmap", "theta_caption": "Softmax logits. Higher = more likely.",
        "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Summary", "summary_results": "Comparison",
        "summary_pros_cons": "Pros & Cons", "pros": "✅ Pros", "cons": "❌ Cons",
        "theory_title": "📚 Theory",
        "theory_sections": {"pg": "9.1 Policy Gradient Theorem", "reinf": "9.2 REINFORCE", "baseline": "9.3 Baseline", "ac": "9.4 Actor-Critic"},
        "theory_pg": "∇J(θ) ∝ Σ_s μ(s) Σ_a Q^π(s,a) ∇π(a|s,θ)\n∇log π(a|s) = (𝟙[a=A_t]−π(a|s))/τ",
        "theory_reinf": "G_t = Σ γ^{k-t} R_{k+1}\nθ ← θ + α γ^t G_t ∇log π(A_t|S_t,θ)",
        "theory_baseline": "θ ← θ + α γ^t (G_t−b(S_t)) ∇log π\nb(s) does not introduce bias.",
        "theory_ac": "δ_t = R+γV(S')−V(S)\nV(S) ← V(S)+α_v δ_t\nθ ← θ+α γ^t δ_t ∇log π",
        "algo_labels": {"reinforce": "REINFORCE", "reinforce_baseline": "REINFORCE+Baseline", "softmax_td0": "Actor-Critic (TD0)", "reinforce_temp": "REINFORCE τ=0.5"},
        "pros_list": {"reinforce": ["Unbiased","Simple","No critic"], "reinforce_baseline": ["Lower variance","Still unbiased","Recommended"], "softmax_td0": ["Online updates","Lower variance than MC","Foundation A2C/PPO"], "reinforce_temp": ["Sharper policy","Less exploration needed","Good deterministic envs"]},
        "cons_list": {"reinforce": ["High variance","Slow","Full episode needed"], "reinforce_baseline": ["Needs α_v","Slightly complex","Still MC"], "softmax_td0": ["Biased (TD)","Two LRs to tune","Can be unstable"], "reinforce_temp": ["Less exploration","Sensitive to τ","May miss global opt"]},
    },
    "PL": {
        "title": "Rozdział 09 — Gradient Polityki: REINFORCE & Softmax",
        "subtitle": "REINFORCE · REINFORCE+Baseline · Actor-Critic TD(0) · Temperatura · ASP Warszawa",
        "engine_missing": "Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Ustawienia",
        "n_episodes": "Epizody", "gamma": "γ Dyskonto", "alpha": "α Uczenie polityki",
        "alpha_baseline": "α_v Uczenie baseline", "temperature": "τ Temperatura", "seed": "Ziarno",
        "run_btn": "▶ Uruchom wszystkie cztery algorytmy",
        "guide_title": "📖 Przewodnik",
        "guide": """
**Krok 1 — Policy Gradient vs Q-Learning**
REINFORCE optymalizuje bezpośrednio π(a|s)=softmax(θ[s][a]). Brak tabeli Q. Wspinaczka gradientowa na J(θ).

**Krok 2 — REINFORCE vs Actor-Critic**
REINFORCE: Monte Carlo — czeka na pełny epizod. Nieobciążone, wysoka wariancja.
Actor-Critic: krytyk TD(0) — online per krok. Niższa wariancja, obciążone.

**Krok 3 — Baseline**
Odejmowanie b(s) od G_t redukuje wariancję bez biasu.
Krzywa REINFORCE+Baseline stabilizuje się szybciej.

**Krok 4 — Ustaw temperaturę τ**
τ=1.0 standardowe. τ<1 ostrzejsza polityka. τ>1 bardziej jednorodna dystrybucja.

**Krok 5 — Odczytaj wykres entropii**
Wysoka entropia = eksploracja. Niska entropia = eksploatacja.
Zdrowe: entropia maleje stopniowo.
""",
        "returns_title": "📈 Zwroty epizodów", "returns_caption": "MA-30. REINFORCE+Baseline najbardziej stabilny.",
        "pg_loss_title": "📉 Wielkość gradientu", "pg_loss_caption": "",
        "entropy_title": "🌡️ Entropia polityki", "entropy_caption": "",
        "value_title": "🏛️ V(s)", "value_caption": "S7 powinno być najniższe.",
        "theta_title": "🗺️ θ[s][a]", "theta_caption": "",
        "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Podsumowanie", "summary_results": "Porównanie",
        "summary_pros_cons": "Zalety i Wady", "pros": "✅ Zalety", "cons": "❌ Wady",
        "theory_title": "📚 Teoria",
        "theory_sections": {"pg": "9.1 Twierdzenie o gradiencie", "reinf": "9.2 REINFORCE", "baseline": "9.3 Baseline", "ac": "9.4 Actor-Critic"},
        "theory_pg": "∇J(θ) ∝ Σ_s μ(s) Σ_a Q^π(s,a) ∇π(a|s,θ)",
        "theory_reinf": "θ ← θ + α γ^t G_t ∇log π(A_t|S_t,θ)",
        "theory_baseline": "θ ← θ + α γ^t (G_t−b(S_t)) ∇log π",
        "theory_ac": "δ_t=R+γV(S')−V(S); θ←θ+α γ^t δ_t ∇log π",
        "algo_labels": {"reinforce": "REINFORCE", "reinforce_baseline": "REINFORCE+Baseline", "softmax_td0": "Actor-Critic (TD0)", "reinforce_temp": "REINFORCE τ=0.5"},
        "pros_list": {"reinforce": ["Nieobciążony","Prosty","Brak krytyka"], "reinforce_baseline": ["Niższa wariancja","Nadal nieobciążony","Zalecany"], "softmax_td0": ["Online","Niższa wariancja","Podstawa A2C/PPO"], "reinforce_temp": ["Ostrzejsza polityka","Mniej eksploracji","Dobre dla deterministycznych"]},
        "cons_list": {"reinforce": ["Wysoka wariancja","Wolny","Wymaga epizodu"], "reinforce_baseline": ["Wymaga α_v","Nieco złożony","Nadal MC"], "softmax_td0": ["Obciążony","Dwa LR","Może być niestabilny"], "reinforce_temp": ["Mniej eksploracji","Wrażliwy na τ","Może pominąć optimum"]},
    },
        "DE": {
        "title": "Kapitel 09 — Policy-Gradient-Methoden",
        "subtitle": "REINFORCE — REINFORCE+Baseline — Actor-Critic — ASP Warschau",
        "engine_missing": "Ausführen: `cd rlvr-py && maturin develop`",
        "sidebar_title": "Einstellungen",
        "n_episodes": "Episoden", "gamma": "Gamma", "alpha": "Alpha (Akteur)",
        "alpha_critic": "Alpha (Kritiker)", "epsilon": "Epsilon", "epsilon_decay": "Epsilon-Abklingrate",
        "seed": "Zufallsseed",
        "run_btn": "▶ Alle Algorithmen starten",
        "guide_title": "Anleitung",
        "guide": """
**Krok 1 — Policy Gradient vs Q-Learning**
REINFORCE optymalizuje bezpośrednio π(a|s)=softmax(θ[s][a]).
Brak tabeli Q. Wspinaczka gradientowa na J(θ).

**Krok 2 — REINFORCE vs Actor-Critic**
REINFORCE: Monte Carlo — czeka na pełny epizod. Nieobciążone, wysoka wariancja.
Actor-Critic: krytyk TD(0) — online per krok. Niższa wariancja, obciążone.

**Krok 3 — Baseline**
Odejmowanie b(s) od G_t redukuje wariancję bez biasu.
Krzywa REINFORCE+Baseline stabilizuje się szybciej.

**Krok 4 — Ustaw temperaturę τ**
τ=1.0 standardowe. τ<1 ostrzejsza polityka. τ>1 bardziej jednorodna dystrybucja.

**Krok 5 — Odczytaj wykres entropii**
Wysoka entropia = eksploracja. Niska entropia = eksploatacja.
Zdrowe: entropia maleje stopniowo.
""",
        "returns_title": "Episodenrückgaben",
        "returns_caption": "Gleitender Durchschnitt.",
        "value_title": "Wertfunktion V(s)",
        "value_caption": "",
        "glass_title": "Glass-Box",
        "summary_title": "Zusammenfassung", "summary_results": "Vergleich",
        "summary_pros_cons": "Vor- & Nachteile",
        "pros": "Vorteile", "cons": "Nachteile",
        "theory_title": "Theorie — Kapitel 09",
        "theory_sections": {"pg": "9.1 Policy-Gradient-Theorem", "reinforce": "9.2 REINFORCE", "baseline": "9.3 Baseline", "ac": "9.4 Actor-Critic"},
        "algo_labels": {"reinforce": "REINFORCE", "reinforce_baseline": "REINFORCE+Baseline", "actor_critic": "Actor-Critic"},
        "pros_list": {
            "reinforce": ["Direkte Strategieoptimierung", "Funktioniert mit stochastischen Strategien"],
            "reinforce_baseline": ["Geringere Varianz", "Stabileres Lernen"],
            "actor_critic": ["Online-Lernen", "Geringere Varianz als REINFORCE"],
        },
        "cons_list": {
            "reinforce": ["Hohe Varianz", "Langsame Konvergenz"],
            "reinforce_baseline": ["Baseline muss gelernt werden"],
            "actor_critic": ["Zwei Lernraten", "Empfindlich gegenüber Hyperparametern"],
        },
        "theory_pg": "nabla J(theta) = E_pi[nabla log pi(a|s,theta) G_t]",
        "theory_reinforce": "theta <- theta + alpha*gamma^t*G_t*nabla log pi(A_t|S_t,theta)",
        "theory_baseline": "theta <- theta + alpha*gamma^t*(G_t-b(S_t))*nabla log pi",
        "theory_ac": "delta_t = R+gamma*V(S')-V(S); theta <- theta+alpha*delta_t*nabla log pi",
    },
    "FR": {
        "title": "Chapitre 09 — Gradient de Politique", "subtitle": "REINFORCE · Actor-Critic · ASP Varsovie",
        "engine_missing": "Exécutez: `cd rlvr-py && maturin develop`", "sidebar_title": "⚙️ Paramètres",
        "n_episodes": "Épisodes", "gamma": "γ", "alpha": "α politique", "alpha_baseline": "α_v baseline",
        "temperature": "τ", "seed": "Graine", "run_btn": "▶ Lancer",
        "guide_title": "📖 Guide", "guide": """
**Étape 1 — Policy Gradient vs Q-Learning**
REINFORCE optimise directement π(a|s)=softmax(θ[s][a]). Pas de Q-table. Montée de gradient sur J(θ).

**Étape 2 — REINFORCE vs Actor-Critic**
REINFORCE : Monte Carlo — attend l'épisode complet. Non biaisé, haute variance.
Actor-Critic : critique TD(0) — en ligne par étape. Variance plus faible, biaisé.

**Étape 3 — Baseline**
Soustraire b(s) de G_t réduit la variance sans biais.
La courbe REINFORCE+Baseline se stabilise plus vite.

**Étape 4 — Régler la température τ**
τ=1.0 standard. τ<1 politique plus nette. τ>1 distribution plus uniforme.

**Étape 5 — Lire le graphique d'entropie**
Haute entropie = exploration. Faible entropie = exploitation.
Sain : l'entropie diminue progressivement.
""",
        "returns_title": "📈 Retours", "returns_caption": "", "pg_loss_title": "📉 Gradient", "pg_loss_caption": "",
        "entropy_title": "🌡️ Entropie", "entropy_caption": "", "value_title": "🏛️ V(s)", "value_caption": "",
        "theta_title": "🗺️ θ[s][a]", "theta_caption": "", "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Résumé", "summary_results": "Comparaison", "summary_pros_cons": "Avantages & Inconvénients",
        "pros": "✅ Pros", "cons": "❌ Cons", "theory_title": "📚 Théorie",
        "theory_sections": {"pg": "9.1 Théorème PG", "reinf": "9.2 REINFORCE", "baseline": "9.3 Baseline", "ac": "9.4 Actor-Critic"},
        "theory_pg": "∇J(θ)∝Σμ(s)ΣQ∇π", "theory_reinf": "θ←θ+αγ^t G_t ∇log π",
        "theory_baseline": "θ←θ+αγ^t(G_t−b)∇log π", "theory_ac": "δ=R+γV'−V; θ←θ+αδ∇log π",
        "algo_labels": {"reinforce": "REINFORCE", "reinforce_baseline": "REINFORCE+Baseline", "softmax_td0": "Actor-Critic", "reinforce_temp": "τ=0.5"},
        "pros_list": {"reinforce": ["Non biaisé"], "reinforce_baseline": ["Variance réduite"], "softmax_td0": ["En ligne"], "reinforce_temp": ["Politique nette"]},
        "cons_list": {"reinforce": ["Variance élevée"], "reinforce_baseline": ["α_v requis"], "softmax_td0": ["Biaisé"], "reinforce_temp": ["Moins d'exploration"]},
    },
    "ES": {
        "title": "Capítulo 09 — Gradiente de Política", "subtitle": "REINFORCE · Actor-Critic · ASP Varsovia",
        "engine_missing": "Ejecute: `cd rlvr-py && maturin develop`", "sidebar_title": "⚙️ Configuración",
        "n_episodes": "Episodios", "gamma": "γ", "alpha": "α política", "alpha_baseline": "α_v baseline",
        "temperature": "τ", "seed": "Semilla", "run_btn": "▶ Ejecutar",
        "guide_title": "📖 Guía", "guide": """
**Paso 1 — Policy Gradient vs Q-Learning**
REINFORCE optimiza directamente π(a|s)=softmax(θ[s][a]). Sin Q-table. Ascenso de gradiente en J(θ).

**Paso 2 — REINFORCE vs Actor-Critic**
REINFORCE: Monte Carlo — espera el episodio completo. Sin sesgo, alta varianza.
Actor-Critic: crítico TD(0) — en línea por paso. Varianza más baja, sesgado.

**Paso 3 — Baseline**
Restar b(s) de G_t reduce la varianza sin sesgo.
La curva REINFORCE+Baseline se estabiliza más rápido.

**Paso 4 — Ajustar la temperatura τ**
τ=1.0 estándar. τ<1 política más nítida. τ>1 distribución más uniforme.

**Paso 5 — Leer el gráfico de entropía**
Alta entropía = explorando. Baja entropía = explotando.
Saludable: la entropía disminuye gradualmente.
""",
        "returns_title": "📈 Retornos", "returns_caption": "", "pg_loss_title": "📉 Gradiente", "pg_loss_caption": "",
        "entropy_title": "🌡️ Entropía", "entropy_caption": "", "value_title": "🏛️ V(s)", "value_caption": "",
        "theta_title": "🗺️ θ[s][a]", "theta_caption": "", "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Resumen", "summary_results": "Comparación", "summary_pros_cons": "Pros y Contras",
        "pros": "✅ Pros", "cons": "❌ Cons", "theory_title": "📚 Teoría",
        "theory_sections": {"pg": "9.1 Teorema PG", "reinf": "9.2 REINFORCE", "baseline": "9.3 Baseline", "ac": "9.4 Actor-Critic"},
        "theory_pg": "∇J(θ)∝Σμ(s)ΣQ∇π", "theory_reinf": "θ←θ+αγ^t G_t ∇log π",
        "theory_baseline": "θ←θ+αγ^t(G_t−b)∇log π", "theory_ac": "δ=R+γV'−V; θ←θ+αδ∇log π",
        "algo_labels": {"reinforce": "REINFORCE", "reinforce_baseline": "REINFORCE+Baseline", "softmax_td0": "Actor-Critic", "reinforce_temp": "τ=0.5"},
        "pros_list": {"reinforce": ["No sesgado"], "reinforce_baseline": ["Varianza reducida"], "softmax_td0": ["En línea"], "reinforce_temp": ["Política nítida"]},
        "cons_list": {"reinforce": ["Alta varianza"], "reinforce_baseline": ["Requiere α_v"], "softmax_td0": ["Sesgado"], "reinforce_temp": ["Menos exploración"]},
    },
}
COLORS = {"reinforce": "#8B5CF6", "reinforce_baseline": "#0082F0", "softmax_td0": "#0FC373", "reinforce_temp": "#FF8C0A"}
ALGOS  = ["reinforce", "reinforce_baseline", "softmax_td0", "reinforce_temp"]

def _ma(data, w=30):
    r = []
    for i in range(len(data)):
        s = max(0, i-w+1); r.append(sum(data[s:i+1])/(i-s+1))
    return r


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
    lang = st.session_state.get("lang", "EN"); tx = _tx(lang)
    st.title(tx["title"]); st.caption(tx["subtitle"])
    try: import rlvr_py
    except ImportError: st.error(tx["engine_missing"]); return
    st.sidebar.header(tx["sidebar_title"])
    n_ep  = st.sidebar.slider(tx["n_episodes"],    50, 3000, 500, 50)
    gamma = st.sidebar.slider(tx["gamma"],         0.5, 0.999, 0.95, 0.005)
    alpha = st.sidebar.slider(tx["alpha"],         0.001, 0.1, 0.01, 0.001, format="%.3f")
    ab    = st.sidebar.slider(tx["alpha_baseline"],0.01, 0.5, 0.1, 0.01)
    tau   = st.sidebar.slider(tx["temperature"],   0.1, 3.0, 1.0, 0.1)
    seed  = st.sidebar.number_input(tx["seed"], 0, 9999, 42)
    with st.expander(tx["guide_title"], expanded=False): st.markdown(tx["guide"])
    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running..."):
            res = rlvr_py.run_ch09_policy_gradient(int(seed), int(n_ep), float(gamma), float(alpha), float(ab), float(tau))
        st.session_state["ch09_result"] = res
    if "ch09_result" not in st.session_state:
        st.info("Click ▶ to run."); _theory(tx); return
    res = st.session_state["ch09_result"]
    short = [f"S{i}" for i in range(res["n_states"])]
    cols = st.columns(4)
    for i, k in enumerate(ALGOS):
        avg = sum(res[k]["returns_curve"][-50:]) / min(50, len(res[k]["returns_curve"]))
        aen = sum(res[k]["entropy_curve"][-50:]) / max(1, min(50, len(res[k]["entropy_curve"])))
        cols[i].metric(tx["algo_labels"][k], f"Avg:{avg:.2f}", f"H:{aen:.2f}")
    st.subheader(tx["returns_title"])
    fig = go.Figure()
    for k in ALGOS: fig.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["returns_curve"]), mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
    fig.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40), xaxis_title="Episode", yaxis_title="Return (MA-30)", legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch'); st.caption(tx["returns_caption"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(tx["pg_loss_title"])
        f2 = go.Figure()
        for k in ALGOS: f2.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["pg_loss_curve"]), mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
        f2.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40), xaxis_title="Episode", yaxis_title="Mean |Δθ|", legend=dict(orientation="h"))
        st.plotly_chart(f2, width='stretch'); st.caption(tx["pg_loss_caption"])
    with c2:
        st.subheader(tx["entropy_title"])
        f3 = go.Figure()
        for k in ALGOS: f3.add_trace(go.Scatter(x=list(range(n_ep)), y=_ma(res[k]["entropy_curve"]), mode="lines", name=tx["algo_labels"][k], line=dict(color=COLORS[k], width=2)))
        f3.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40), xaxis_title="Episode", yaxis_title="H(π(·|s))", legend=dict(orientation="h"))
        st.plotly_chart(f3, width='stretch'); st.caption(tx["entropy_caption"])
    st.subheader(tx["value_title"])
    f4 = go.Figure()
    for k in ALGOS: f4.add_trace(go.Bar(x=short, y=res[k]["values"], name=tx["algo_labels"][k], marker_color=COLORS[k], opacity=0.8))
    f4.update_layout(height=260, barmode="group", margin=dict(l=40,r=20,t=20,b=40), legend=dict(orientation="h"))
    st.plotly_chart(f4, width='stretch'); st.caption(tx["value_caption"])
    st.subheader(tx["theta_title"])
    sel  = st.selectbox("Algorithm", [tx["algo_labels"][k] for k in ALGOS])
    ks   = {tx["algo_labels"][k]: k for k in ALGOS}.get(sel, "reinforce")
    th   = res[ks]["theta"]; ash = [f"A{i}" for i in range(res["n_actions"])]
    f5   = go.Figure(go.Heatmap(z=th, x=ash, y=short, colorscale="Purples", text=[[f"{th[s][a]:.3f}" for a in range(res["n_actions"])] for s in range(res["n_states"])], texttemplate="%{text}"))
    f5.update_layout(height=280, margin=dict(l=60,r=20,t=20,b=40)); st.plotly_chart(f5, width='stretch')
    st.subheader(tx["glass_title"]); _glass(res, tx)
    st.subheader(tx["summary_title"]); _summary(res, tx)
    _theory(tx)

def _glass(res, tx):
    opts = {tx["algo_labels"][k]: k for k in ALGOS}
    sel  = st.selectbox("Algorithm", list(opts.keys()), key="gb9")
    k    = opts[sel]; r = res[k]
    ep   = st.slider("Episode", 0, max(len(r["returns_curve"])-1,0), max(len(r["returns_curve"])-1,0), key="gb9ep")
    c1,c2,c3 = st.columns(3)
    c1.metric("Return", f"{r['returns_curve'][ep]:.3f}")
    c2.metric("|Δθ|",   f"{r['pg_loss_curve'][ep]:.4f}")
    c3.metric("H(π)",   f"{r['entropy_curve'][ep]:.4f}")
    if "actor_critic" in k:
        st.latex(r"\delta_t=R_{t+1}+\gamma V(S')-V(S)")
        st.latex(r"V(S)\leftarrow V(S)+\alpha_v\delta_t")
        st.latex(r"\theta\leftarrow\theta+\alpha\gamma^t\delta_t\nabla\log\pi(A_t|S_t,\theta)")
    else:
        st.latex(r"G_t=\sum_{k=t}^T\gamma^{k-t}R_{k+1}")
        st.latex(r"\theta\leftarrow\theta+\alpha\gamma^t G_t\nabla\log\pi(A_t|S_t,\theta)")
        st.latex(r"\nabla\log\pi(a|s)=\mathbf{1}[a=A_t]-\pi(a|s,\theta)")

def _summary(res, tx):
    rows = []
    for k in ALGOS:
        r = res[k]; avg = sum(r["returns_curve"][-100:])/min(100,len(r["returns_curve"]))
        aen = sum(r["entropy_curve"])/max(1,len(r["entropy_curve"]))
        rows.append({"Algorithm": tx["algo_labels"][k], "Avg return (last 100)": f"{avg:.3f}", "Steps": str(r["total_steps"]), "Avg H": f"{aen:.3f}", "V*(S0)": f"{r['values'][0]:.3f}", "V*(S7)": f"{r['values'][7]:.3f}"})
    st.dataframe(rows, hide_index=True)
    for k in ALGOS:
        label = tx["algo_labels"][k]; c1,c2 = st.columns(2)
        with c1: st.markdown(f"**{label} — {tx['pros']}**"); [st.markdown(f"- {p}") for p in tx["pros_list"][k]]
        with c2: st.markdown(f"**{label} — {tx['cons']}**"); [st.markdown(f"- {c}") for c in tx["cons_list"][k]]
        st.markdown("---")

def _theory(tx):
    st.markdown("---"); st.subheader(tx["theory_title"])
    for k in ["pg","reinf","baseline","ac"]:
        with st.expander(tx["theory_sections"][k], expanded=False): st.markdown(tx[f"theory_{k}"])
