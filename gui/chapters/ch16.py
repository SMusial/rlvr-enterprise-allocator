import streamlit as st
import os as _os


def _render_handbook():
    _col1, _col2 = st.columns([8, 1])
    with _col1:
        st.subheader("Hands-On Guide \u2014 Chapter 16 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch16_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("\U0001f4be Save", data=_html,
                           file_name="handson_ch16_en.html", mime="text/html")
    st.components.v1.html(_html, height=4000, scrolling=True)


def _tx():
    return {
        "title":    "Chapter 16 \u2014 Deep Reinforcement Learning Models",
        "subtitle": "DQN \u00b7 Double DQN \u00b7 Dueling DQN \u00b7 PPO \u00b7 Warsaw ASP 8-state MDP \u00b7 tch/libtorch",
        "engine_missing": "\u274c Rust engine not found. Run: cd rlvr-py && maturin develop --release",
        "sidebar_title": "Ch16 Settings",
        "episodes":      "Episodes",
        "lr":            "\u03b1 (learning rate)",
        "epsilon_start": "\u03b5 start",
        "epsilon_end":   "\u03b5 end",
        "epsilon_decay": "\u03b5 decay",
        "batch_size":    "Batch size",
        "target_update": "Target update (episodes)",
        "buffer_size":   "Replay buffer size",
        "hidden_units":  "Hidden units",
        "seed":          "Random seed",
        "run_btn":       "\u25b6 Run All Four Algorithms",
        "reward_title":  "Episode Returns",
        "loss_title":    "Training Loss",
        "epsilon_title": "Epsilon Decay",
        "qtable_title":  "Q-Table Heatmap",
        "glass_title":   "Glass-Box \u2014 Step Trace",
        "summary_title": "Results Summary",
    }


def render():
    tx = _tx()
    st.title(tx["title"])
    st.caption(tx["subtitle"])

    _tab_main, _tab_handbook = st.tabs(["\U0001f52c Interactive Lab",
                                         "\U0001f4d8 Hands-On Guide EN"])
    with _tab_handbook:
        _render_handbook()
    with _tab_main:

        try:
            import rlvr_py
        except ImportError:
            st.error(tx["engine_missing"])
            return

        st.sidebar.header(tx["sidebar_title"])
        episodes      = st.sidebar.slider(tx["episodes"],      50, 1000, 300, 50)
        lr            = st.sidebar.slider(tx["lr"],            0.0001, 0.01, 0.001, 0.0001, format="%.4f")
        epsilon_start = st.sidebar.slider(tx["epsilon_start"], 0.1, 1.0, 1.0, 0.05)
        epsilon_end   = st.sidebar.slider(tx["epsilon_end"],   0.01, 0.2, 0.05, 0.01)
        epsilon_decay = st.sidebar.slider(tx["epsilon_decay"], 0.9, 0.999, 0.995, 0.001, format="%.3f")
        batch_size    = st.sidebar.select_slider(tx["batch_size"], [16, 32, 64, 128], value=32)
        target_update = st.sidebar.slider(tx["target_update"], 1, 20, 10, 1)
        buffer_size   = st.sidebar.select_slider(tx["buffer_size"], [500, 1000, 2000, 5000], value=1000)
        hidden_units  = st.sidebar.select_slider(tx["hidden_units"], [16, 32, 64, 128], value=64)
        seed          = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

        run = st.button(tx["run_btn"], type="primary")

        if run:
            with st.spinner("Running Rust DRL engine (tch/libtorch)..."):
                raw = rlvr_py.run_ch16(
                    int(episodes), float(lr),
                    float(epsilon_start), float(epsilon_end), float(epsilon_decay),
                    int(batch_size), int(target_update), int(buffer_size),
                    int(hidden_units), int(seed)
                )
            st.session_state["ch16_raw"] = raw

        if "ch16_raw" not in st.session_state:
            st.info("Configure settings and click **\u25b6 Run All Four Algorithms**.")
            return

        results = st.session_state["ch16_raw"]
        import pandas as pd
        import altair as alt
        import numpy as np

        COLORS = {
            "DQN":         "#0082F0",
            "Double DQN":  "#0FC373",
            "Dueling DQN": "#FF8C0A",
            "PPO":         "#8B5CF6",
        }

        # ── KPI row ──────────────────────────────────────────────────────────
        cols = st.columns(4)
        for i, res in enumerate(results):
            cols[i].metric(
                res["algorithm"],
                f"Avg Return: {res['total_reward']:.3f}",
                f"Avg Loss: {res['avg_loss']:.5f}",
            )

        # ── Episode Returns ───────────────────────────────────────────────────
        st.subheader(tx["reward_title"])
        rows = []
        for res in results:
            algo = res["algorithm"]
            for ep in res["episodes"]:
                rows.append({"Episode": ep["episode"], "Return": ep["total_reward"], "Algorithm": algo})
        df = pd.DataFrame(rows)
        df["Return_MA"] = df.groupby("Algorithm")["Return"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        chart = alt.Chart(df).mark_line(opacity=0.9).encode(
            x="Episode:Q",
            y=alt.Y("Return_MA:Q", title="Return (MA-20)"),
            color=alt.Color("Algorithm:N",
                scale=alt.Scale(domain=list(COLORS.keys()), range=list(COLORS.values()))),
            tooltip=["Episode", "Algorithm", alt.Tooltip("Return_MA:Q", format=".3f")]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

        # ── Loss curves ───────────────────────────────────────────────────────
        st.subheader(tx["loss_title"])
        loss_rows = []
        for res in results:
            if res["algorithm"] == "PPO":
                continue  # PPO loss scale differs
            for ep in res["episodes"]:
                if ep["avg_loss"] > 0:
                    loss_rows.append({"Episode": ep["episode"], "Loss": ep["avg_loss"],
                                      "Algorithm": res["algorithm"]})
        if loss_rows:
            df_loss = pd.DataFrame(loss_rows)
            df_loss["Loss_MA"] = df_loss.groupby("Algorithm")["Loss"].transform(
                lambda x: x.rolling(20, min_periods=1).mean()
            )
            chart_loss = alt.Chart(df_loss).mark_line(opacity=0.9).encode(
                x="Episode:Q",
                y=alt.Y("Loss_MA:Q", scale=alt.Scale(type="log"), title="MSE Loss (log, MA-20)"),
                color=alt.Color("Algorithm:N",
                    scale=alt.Scale(domain=list(COLORS.keys()), range=list(COLORS.values()))),
            ).properties(height=250)
            st.altair_chart(chart_loss, use_container_width=True)

        # ── Epsilon decay ─────────────────────────────────────────────────────
        st.subheader(tx["epsilon_title"])
        eps_rows = []
        for res in results:
            if res["algorithm"] == "PPO":
                continue
            for ep in res["episodes"]:
                eps_rows.append({"Episode": ep["episode"], "Epsilon": ep["epsilon"],
                                 "Algorithm": res["algorithm"]})
        if eps_rows:
            df_eps = pd.DataFrame(eps_rows)
            chart_eps = alt.Chart(df_eps).mark_line(opacity=0.9).encode(
                x="Episode:Q",
                y=alt.Y("Epsilon:Q", title="ε (exploration rate)"),
                color=alt.Color("Algorithm:N",
                    scale=alt.Scale(domain=list(COLORS.keys()), range=list(COLORS.values()))),
            ).properties(height=200)
            st.altair_chart(chart_eps, use_container_width=True)

        # ── Q-Table Heatmaps ──────────────────────────────────────────────────
        st.subheader(tx["qtable_title"])
        import plotly.graph_objects as go
        action_labels = ["Dispatch+1", "Dispatch-1", "Dispatch+2", "Dispatch-2"]
        state_labels  = [f"S{i}" for i in range(8)]

        cols_q = st.columns(2)
        for idx, res in enumerate(results[:2]):
            q = np.array(res["final_q_table"])
            fig = go.Figure(data=go.Heatmap(
                z=q, x=action_labels, y=state_labels,
                colorscale="Viridis", colorbar=dict(title="Q-value"),
            ))
            fig.update_layout(
                title=f"{res['algorithm']} — Q-Table",
                height=350, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font=dict(color="#e8eaf6"),
            )
            cols_q[idx].plotly_chart(fig, use_container_width=True)

        cols_q2 = st.columns(2)
        for idx, res in enumerate(results[2:]):
            q = np.array(res["final_q_table"])
            fig = go.Figure(data=go.Heatmap(
                z=q, x=action_labels, y=state_labels,
                colorscale="Plasma", colorbar=dict(title="Q/Logit"),
            ))
            fig.update_layout(
                title=f"{res['algorithm']} — Q-Table / Policy Logits",
                height=350, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font=dict(color="#e8eaf6"),
            )
            cols_q2[idx].plotly_chart(fig, use_container_width=True)

        # ── Glass-Box ─────────────────────────────────────────────────────────
        st.subheader(tx["glass_title"])
        algo_sel = st.selectbox("Algorithm", [r["algorithm"] for r in results])
        ep_sel   = st.slider("Episode", 0, episodes - 1, 0)
        for res in results:
            if res["algorithm"] == algo_sel:
                ep_data = res["episodes"][ep_sel] if ep_sel < len(res["episodes"]) else res["episodes"][-1]
                steps_data = ep_data.get("steps", [])
                if steps_data:
                    df_steps = pd.DataFrame(steps_data)
                    df_steps = df_steps[["step", "state", "action", "reward", "q_value", "loss", "epsilon"]]
                    df_steps.columns = ["Step", "State", "Action", "Reward", "Q-value", "Loss", "ε"]
                    df_steps["Action"] = df_steps["Action"].map(
                        {0: "Dispatch+1", 1: "Dispatch-1", 2: "Dispatch+2", 3: "Dispatch-2"}
                    )
                    for col in ["Reward", "Q-value", "Loss"]:
                        df_steps[col] = df_steps[col].round(4)
                    df_steps["ε"] = df_steps["ε"].round(3)
                    st.dataframe(df_steps, use_container_width=True)
                break

        # ── Summary ───────────────────────────────────────────────────────────
        st.subheader(tx["summary_title"])
        summary = [{"Algorithm": r["algorithm"],
                    "Avg Return": round(r["total_reward"], 4),
                    "Avg Loss":   round(r["avg_loss"], 6)}
                   for r in results]
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
