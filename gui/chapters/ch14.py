import streamlit as st
import os as _os
import json

def _render_handbook():
    _col1, _col2 = st.columns([8, 1])
    with _col1:
        st.subheader("Hands-On Guide \u2014 Chapter 14 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch14_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("\U0001f4be Save", data=_html,
                           file_name="handson_ch14_en.html", mime="text/html")
    st.iframe(_html, height=4000)


def _tx(lang):
    T = {
        "EN": {
            "title": "Chapter 14 \u2014 Foundational MARL Algorithms",
            "subtitle": "IQL \u00b7 VDN \u00b7 MAPG \u00b7 MADDPG \u00b7 5\u00d75 Grid World \u00b7 Warsaw ASP",
            "engine_missing": "\u274c Rust engine not found. Run: maturin develop --release",
            "sidebar_title": "Ch14 Settings",
            "episodes": "Episodes",
            "alpha": "\u03b1 (learning rate)",
            "gamma": "\u03b3 (discount)",
            "epsilon": "\u03b5 (exploration)",
            "beta": "\u03b2 (entropy bonus MAPG)",
            "tau": "\u03c4 (soft update MADDPG)",
            "seed": "Random seed",
            "run_btn": "\u25b6 Run All Four Algorithms",
            "returns_title": "Episode Returns",
            "coop_title": "Cooperation Rate",
            "td_title": "TD Error / Policy Gradient",
            "qtable_title": "Q-Table Heatmap",
            "glass_title": "Glass-Box \u2014 Step Trace",
            "summary_title": "Results Summary",
        }
    }
    return T.get(lang, T["EN"])


def render():
    lang = "EN"
    tx = _tx(lang)

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
        episodes = st.sidebar.slider(tx["episodes"], 50, 1000, 200, 50)
        alpha    = st.sidebar.slider(tx["alpha"],   0.01, 0.5, 0.1, 0.01)
        gamma    = st.sidebar.slider(tx["gamma"],   0.5, 0.999, 0.95, 0.005, format="%.3f")
        epsilon  = st.sidebar.slider(tx["epsilon"], 0.0, 1.0, 0.3, 0.05)
        beta     = st.sidebar.slider(tx["beta"],    0.0, 0.1, 0.01, 0.005, format="%.3f")
        tau      = st.sidebar.slider(tx["tau"],     0.001, 0.1, 0.01, 0.001, format="%.3f")
        seed     = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

        run = st.button(tx["run_btn"], type="primary")

        if run:
            with st.spinner("Running Rust MARL engine..."):
                raw = rlvr_py.run_ch14(
                    int(episodes), float(alpha), float(gamma),
                    float(epsilon), float(beta), float(tau), int(seed)
                )
            st.session_state["ch14_raw"] = raw

        if "ch14_raw" not in st.session_state:
            st.info("Configure settings and click **\u25b6 Run All Four Algorithms**.")
            return

        results = st.session_state["ch14_raw"]
        algo_colors = {
            "IQL":    "#0082F0",
            "VDN":    "#0FC373",
            "MAPG":   "#FF8C0A",
            "MADDPG": "#8B5CF6",
        }

        # ── KPI row ──────────────────────────────────────────────────────────
        cols = st.columns(4)
        for i, res in enumerate(results):
            algo = res["algorithm"]
            cols[i].metric(
                algo,
                f"Avg Return: {res['total_reward']:.2f}",
                f"Coop: {res['avg_cooperation']*100:.1f}%",
            )

        # ── Episode Returns ───────────────────────────────────────────────────
        st.subheader(tx["returns_title"])
        import altair as alt
        import pandas as pd

        rows = []
        for res in results:
            algo = res["algorithm"]
            for ep_data in res["episodes"]:
                rows.append({
                    "Episode": ep_data["episode"],
                    "Return":  ep_data["total_reward"],
                    "Algorithm": algo,
                })
        df = pd.DataFrame(rows)
        # Moving average
        df["Return_MA"] = df.groupby("Algorithm")["Return"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        chart = alt.Chart(df).mark_line(opacity=0.9).encode(
            x=alt.X("Episode:Q"),
            y=alt.Y("Return_MA:Q", title="Return (MA-20)"),
            color=alt.Color("Algorithm:N",
                scale=alt.Scale(
                    domain=list(algo_colors.keys()),
                    range=list(algo_colors.values())
                )),
            tooltip=["Episode", "Algorithm", alt.Tooltip("Return_MA:Q", format=".2f")]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

        # ── Cooperation Rate ──────────────────────────────────────────────────
        st.subheader(tx["coop_title"])
        rows_c = []
        for res in results:
            algo = res["algorithm"]
            for ep_data in res["episodes"]:
                rows_c.append({
                    "Episode": ep_data["episode"],
                    "Cooperation": ep_data["cooperation_rate"] * 100,
                    "Algorithm": algo,
                })
        df_c = pd.DataFrame(rows_c)
        df_c["Coop_MA"] = df_c.groupby("Algorithm")["Cooperation"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        chart_c = alt.Chart(df_c).mark_line(opacity=0.9).encode(
            x="Episode:Q",
            y=alt.Y("Coop_MA:Q", title="Cooperation Rate % (MA-20)"),
            color=alt.Color("Algorithm:N",
                scale=alt.Scale(
                    domain=list(algo_colors.keys()),
                    range=list(algo_colors.values())
                )),
        ).properties(height=250)
        st.altair_chart(chart_c, use_container_width=True)

        # ── TD Error ─────────────────────────────────────────────────────────
        st.subheader(tx["td_title"])
        rows_td = []
        for res in results:
            algo = res["algorithm"]
            for ep_data in res["episodes"]:
                rows_td.append({
                    "Episode": ep_data["episode"],
                    "TD_Error": ep_data["avg_td_error"],
                    "Algorithm": algo,
                })
        df_td = pd.DataFrame(rows_td)
        df_td["TD_MA"] = df_td.groupby("Algorithm")["TD_Error"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        chart_td = alt.Chart(df_td).mark_line(opacity=0.9).encode(
            x="Episode:Q",
            y=alt.Y("TD_MA:Q", title="Avg TD Error (MA-20)"),
            color=alt.Color("Algorithm:N",
                scale=alt.Scale(
                    domain=list(algo_colors.keys()),
                    range=list(algo_colors.values())
                )),
        ).properties(height=250)
        st.altair_chart(chart_td, use_container_width=True)

        # ── Q-Table Heatmaps ──────────────────────────────────────────────────
        st.subheader(tx["qtable_title"])
        import numpy as np
        import plotly.graph_objects as go

        action_labels = ["Up", "Down", "Left", "Right"]
        state_labels  = [f"({r},{c})" for r in range(5) for c in range(5)]

        cols_q = st.columns(2)
        for idx, res in enumerate(results[:2]):
            algo = res["algorithm"]
            q = np.array(res["final_q_tables"][0])  # Agent 0
            fig = go.Figure(data=go.Heatmap(
                z=q,
                x=action_labels,
                y=state_labels,
                colorscale="Viridis",
                colorbar=dict(title="Q-value"),
            ))
            fig.update_layout(
                title=f"{algo} — Agent 0 Q-Table",
                height=400,
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                font=dict(color="#e8eaf6"),
            )
            cols_q[idx].plotly_chart(fig, use_container_width=True)

        cols_q2 = st.columns(2)
        for idx, res in enumerate(results[2:]):
            algo = res["algorithm"]
            q = np.array(res["final_q_tables"][0])
            fig = go.Figure(data=go.Heatmap(
                z=q,
                x=action_labels,
                y=state_labels,
                colorscale="Plasma",
                colorbar=dict(title="Logit/Q"),
            ))
            fig.update_layout(
                title=f"{algo} — Agent 0 Policy Logits",
                height=400,
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
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
                    df_steps = df_steps[["step", "state", "action", "reward", "q_value", "td_error"]]
                    df_steps.columns = ["Step", "State", "Action", "Reward", "Q-value", "TD Error"]
                    df_steps["Action"] = df_steps["Action"].map(
                        {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
                    )
                    df_steps["Reward"]   = df_steps["Reward"].round(3)
                    df_steps["Q-value"]  = df_steps["Q-value"].round(4)
                    df_steps["TD Error"] = df_steps["TD Error"].round(4)
                    st.dataframe(df_steps, use_container_width=True)
                break

        # ── Summary ───────────────────────────────────────────────────────────
        st.subheader(tx["summary_title"])
        summary_rows = []
        for res in results:
            summary_rows.append({
                "Algorithm":      res["algorithm"],
                "Avg Return":     round(res["total_reward"], 3),
                "Avg Cooperation": f"{res['avg_cooperation']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
