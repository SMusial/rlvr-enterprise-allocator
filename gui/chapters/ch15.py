import streamlit as st
import os as _os


def _render_handbook():
    _col1, _col2 = st.columns([8, 1])
    with _col1:
        st.subheader("Hands-On Guide \u2014 Chapter 15 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch15_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("\U0001f4be Save", data=_html,
                           file_name="handson_ch15_en.html", mime="text/html")
    st.iframe(_html, height=4000)


def _tx():
    return {
        "title":    "Chapter 15 \u2014 Deep Learning Foundations",
        "subtitle": "FNN \u00b7 Activations \u00b7 Optimizers \u00b7 Regularization \u00b7 Warsaw ASP V*(s) Approximation",
        "engine_missing": "\u274c Rust engine not found. Run: cd rlvr-py && maturin develop --release",
        "sidebar_title": "Ch15 Settings",
        "epochs":   "Epochs",
        "lr":       "\u03b1 (learning rate)",
        "activation": "Activation function",
        "hidden":   "Hidden units per layer",
        "n_layers": "Hidden layers",
        "l2":       "\u03bb (L2 regularization)",
        "dropout":  "Dropout rate",
        "seed":     "Random seed",
        "run_btn":  "\u25b6 Run All Four Configurations",
        "loss_title":  "Training Loss Curve",
        "pred_title":  "V*(s) Predictions vs True Values",
        "arch_title":  "Network Architecture",
        "act_title":   "Activation Function Comparison",
        "glass_title": "Glass-Box \u2014 Epoch Trace",
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
        epochs     = st.sidebar.slider(tx["epochs"],   50, 2000, 500, 50)
        lr         = st.sidebar.slider(tx["lr"],       0.0001, 0.1, 0.01, 0.0001, format="%.4f")
        activation = st.sidebar.selectbox(tx["activation"],
                        ["relu", "leaky_relu", "elu", "swish", "tanh", "sigmoid"], index=0)
        hidden     = st.sidebar.slider(tx["hidden"],   4, 64, 16, 4)
        n_layers   = st.sidebar.slider(tx["n_layers"], 1, 4, 2, 1)
        l2_lambda  = st.sidebar.slider(tx["l2"],       0.0, 0.1, 0.001, 0.001, format="%.3f")
        dropout    = st.sidebar.slider(tx["dropout"],  0.0, 0.5, 0.1, 0.05)
        seed       = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

        run = st.button(tx["run_btn"], type="primary")

        if run:
            with st.spinner("Running Rust FNN engine..."):
                raw = rlvr_py.run_ch15(
                    int(epochs), float(lr), activation,
                    float(l2_lambda), float(dropout),
                    int(hidden), int(n_layers), int(seed)
                )
            st.session_state["ch15_raw"] = raw

        if "ch15_raw" not in st.session_state:
            st.info("Configure settings and click **\u25b6 Run All Four Configurations**.")
            return

        results = st.session_state["ch15_raw"]
        import pandas as pd
        import altair as alt
        import numpy as np

        COLORS = {
            results[0]["algorithm"]: "#9ca3af",
            results[1]["algorithm"]: "#0082F0",
            results[2]["algorithm"]: "#0FC373",
            results[3]["algorithm"]: "#FF8C0A",
        }

        # ── KPI row ──────────────────────────────────────────────────────────
        cols = st.columns(4)
        for i, res in enumerate(results):
            cols[i].metric(
                res["algorithm"][:25],
                f"Loss: {res['final_loss']:.5f}",
            )

        # ── Loss curves ───────────────────────────────────────────────────────
        st.subheader(tx["loss_title"])
        rows = []
        for res in results:
            for ep in res["epochs"]:
                rows.append({"Epoch": ep["epoch"], "Loss": ep["loss"],
                             "Algorithm": res["algorithm"]})
        df = pd.DataFrame(rows)
        chart = alt.Chart(df).mark_line(opacity=0.9).encode(
            x="Epoch:Q",
            y=alt.Y("Loss:Q", scale=alt.Scale(type="log"), title="MSE Loss (log scale)"),
            color=alt.Color("Algorithm:N",
                scale=alt.Scale(domain=list(COLORS.keys()), range=list(COLORS.values()))),
            tooltip=["Epoch", "Algorithm", alt.Tooltip("Loss:Q", format=".6f")]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

        # ── V*(s) predictions ─────────────────────────────────────────────────
        st.subheader(tx["pred_title"])
        state_names = ["S0 (Best)", "S1", "S2", "S3", "S4", "S5", "S6", "S7 (Worst)"]
        true_vals = results[0]["true_values"]

        pred_rows = []
        for res in results:
            for i, (pred, true) in enumerate(zip(res["final_predictions"], true_vals)):
                pred_rows.append({
                    "State": state_names[i],
                    "Value": pred,
                    "Algorithm": res["algorithm"],
                    "Type": "Predicted"
                })
        for i, tv in enumerate(true_vals):
            pred_rows.append({
                "State": state_names[i],
                "Value": tv,
                "Algorithm": "True V*(s) [Ch02]",
                "Type": "True"
            })

        df_pred = pd.DataFrame(pred_rows)
        chart_pred = alt.Chart(df_pred).mark_line(point=True).encode(
            x=alt.X("State:N", sort=state_names),
            y=alt.Y("Value:Q", title="V*(s)"),
            color="Algorithm:N",
            strokeDash=alt.condition(
                alt.datum.Type == "True",
                alt.value([8, 4]),
                alt.value([1, 0])
            ),
            tooltip=["State", "Algorithm", alt.Tooltip("Value:Q", format=".3f")]
        ).properties(height=300)
        st.altair_chart(chart_pred, use_container_width=True)

        # ── Architecture display ──────────────────────────────────────────────
        st.subheader(tx["arch_title"])
        dims = results[1]["layer_dims"]
        arch_str = " → ".join([f"**{d}**" for d in dims])
        st.markdown(f"Network: Input({dims[0]}) → " +
                    " → ".join([f"Dense({d}, {activation})" for d in dims[1:-1]]) +
                    f" → Output({dims[-1]})")
        total_params = sum(dims[i]*dims[i+1] + dims[i+1] for i in range(len(dims)-1))
        col1, col2, col3 = st.columns(3)
        col1.metric("Total parameters", total_params)
        col2.metric("Hidden layers", n_layers)
        col3.metric("Hidden units", hidden)

        # ── Activation comparison ─────────────────────────────────────────────
        st.subheader(tx["act_title"])
        x_vals = [i * 0.1 - 3.0 for i in range(61)]
        act_rows = []
        for act_name in ["relu", "leaky_relu", "elu", "swish", "tanh", "sigmoid"]:
            for x in x_vals:
                if act_name == "relu":       y = max(0.0, x)
                elif act_name == "leaky_relu": y = x if x > 0 else 0.01*x
                elif act_name == "elu":      y = x if x >= 0 else (np.exp(x) - 1)
                elif act_name == "swish":    y = x / (1 + np.exp(-x))
                elif act_name == "tanh":     y = np.tanh(x)
                else:                        y = 1 / (1 + np.exp(-x))
                act_rows.append({"x": x, "f(x)": y, "Activation": act_name})
        df_act = pd.DataFrame(act_rows)
        chart_act = alt.Chart(df_act).mark_line().encode(
            x="x:Q",
            y=alt.Y("f(x):Q", scale=alt.Scale(domain=[-2, 2])),
            color="Activation:N",
            tooltip=["Activation", alt.Tooltip("x:Q", format=".1f"),
                     alt.Tooltip("f(x):Q", format=".3f")]
        ).properties(height=250)
        st.altair_chart(chart_act, use_container_width=True)

        # ── Glass-Box ─────────────────────────────────────────────────────────
        st.subheader(tx["glass_title"])
        algo_sel = st.selectbox("Algorithm", [r["algorithm"] for r in results])
        for res in results:
            if res["algorithm"] == algo_sel:
                ep_data = [{"Epoch": e["epoch"],
                            "Loss": round(e["loss"], 6),
                            "Val Loss": round(e["val_loss"], 6),
                            "Pred S0": round(e["predictions"][0], 3),
                            "Pred S7": round(e["predictions"][-1], 3)}
                           for e in res["epochs"]]
                st.dataframe(pd.DataFrame(ep_data), use_container_width=True)
                break

        # ── Summary ───────────────────────────────────────────────────────────
        st.subheader(tx["summary_title"])
        summary = [{"Algorithm": r["algorithm"],
                    "Activation": r["activation"],
                    "Optimizer": r["optimizer"],
                    "L2 λ": r["l2_lambda"],
                    "Dropout": r["dropout_rate"],
                    "Final Loss": round(r["final_loss"], 6)}
                   for r in results]
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
