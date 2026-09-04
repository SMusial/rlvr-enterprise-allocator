import streamlit as st
import os as _os


def _render_handbook():
    _col1, _col2 = st.columns([8, 1])
    with _col1:
        st.subheader("Hands-On Guide \u2014 Chapter 17 (EN)")
    _base = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_base, "..", "..", "docs", "handson_ch17_en.html")
    with open(_path, encoding="utf-8") as _f:
        _html = _f.read()
    with _col2:
        st.download_button("\U0001f4be Save", data=_html,
                           file_name="handson_ch17_en.html", mime="text/html")
    st.components.v1.html(_html, height=4000, scrolling=True)


def _tx():
    return {
        "title":    "Chapter 17 \u2014 Model Explainability and Interpretability",
        "subtitle": "Gradient FI \u00b7 Saliency Maps \u00b7 LIME \u00b7 SHAP \u00b7 Warsaw ASP FNN \u00b7 tch/libtorch",
        "engine_missing": "\u274c Rust engine not found. Run: cd rlvr-py && maturin develop --release",
        "sidebar_title":  "Ch17 Settings",
        "epochs":         "Training epochs (FNN)",
        "lr":             "\u03b1 (learning rate)",
        "hidden":         "Hidden units",
        "lime_samples":   "LIME samples",
        "shap_samples":   "SHAP samples",
        "seed":           "Random seed",
        "run_btn":        "\u25b6 Run Explainability Analysis",
        "global_title":   "Global Feature Importance (SHAP)",
        "local_title":    "Local Explanations per State",
        "shap_title":     "SHAP Waterfall",
        "lime_title":     "LIME Local Approximation",
        "saliency_title": "Saliency Map",
        "compare_title":  "Method Comparison",
        "glass_title":    "Glass-Box \u2014 State Detail",
        "summary_title":  "Results Summary",
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
        epochs       = st.sidebar.slider(tx["epochs"],       100, 2000, 500, 100)
        lr           = st.sidebar.slider(tx["lr"],           0.0001, 0.05, 0.01, 0.0001, format="%.4f")
        hidden       = st.sidebar.select_slider(tx["hidden"], [8, 16, 32, 64], value=16)
        lime_samples = st.sidebar.slider(tx["lime_samples"], 50, 500, 200, 50)
        shap_samples = st.sidebar.slider(tx["shap_samples"], 50, 500, 200, 50)
        seed         = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

        run = st.button(tx["run_btn"], type="primary")

        if run:
            with st.spinner("Running Rust explainability engine..."):
                raw = rlvr_py.run_ch17(
                    int(epochs), float(lr), int(hidden),
                    int(lime_samples), int(shap_samples), int(seed)
                )
            st.session_state["ch17_raw"] = raw

        if "ch17_raw" not in st.session_state:
            st.info("Configure settings and click **\u25b6 Run Explainability Analysis**.")
            return

        res = st.session_state["ch17_raw"]
        import pandas as pd
        import altair as alt
        import numpy as np
        import plotly.graph_objects as go

        feature_names = res["feature_names"]
        state_names   = [f"S{i}" for i in range(8)]
        COLORS = ["#0082F0", "#0FC373", "#FF8C0A", "#8B5CF6"]

        # ── Model quality ─────────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        col1.metric("Model R²", f"{res['model_r2']:.4f}")
        col2.metric("Method", res["method"][:30])

        # ── Global Feature Importance ─────────────────────────────────────────
        st.subheader(tx["global_title"])
        gi = res["global_importance"]
        df_gi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": gi,
        }).sort_values("Importance", ascending=False)

        chart_gi = alt.Chart(df_gi).mark_bar().encode(
            x=alt.X("Importance:Q", title="Mean |SHAP value|"),
            y=alt.Y("Feature:N", sort="-x"),
            color=alt.Color("Feature:N",
                scale=alt.Scale(domain=feature_names, range=COLORS)),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")]
        ).properties(height=200)
        st.altair_chart(chart_gi, use_container_width=True)

        # ── Method Comparison Heatmap ─────────────────────────────────────────
        st.subheader(tx["compare_title"])
        methods = ["Gradient FI", "Saliency", "LIME", "SHAP"]
        exps = res["state_explanations"]

        rows = []
        for exp in exps:
            for j, feat in enumerate(feature_names):
                rows.append({"State": f"S{exp['state_idx']}", "Feature": feat,
                             "Gradient FI": round(exp["gradient_fi"][j], 4),
                             "Saliency":    round(exp["saliency"][j], 4),
                             "LIME":        round(exp["lime_coefs"][j], 4),
                             "SHAP":        round(exp["shap_values"][j], 4)})
        df_comp = pd.DataFrame(rows)

        method_sel = st.selectbox("Method", methods)
        pivot = df_comp.pivot(index="State", columns="Feature", values=method_sel)
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title=method_sel),
        ))
        fig_heat.update_layout(
            title=f"{method_sel} — Feature Attribution per State",
            height=350, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#e8eaf6"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── SHAP Waterfall ────────────────────────────────────────────────────
        st.subheader(tx["shap_title"])
        state_sel = st.selectbox("State", state_names, key="shap_state")
        s_idx = int(state_sel[1:])
        exp = exps[s_idx]

        shap_vals = exp["shap_values"]
        base = exp["shap_base"]
        cumsum = [base]
        for v in shap_vals:
            cumsum.append(cumsum[-1] + v)

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(
            x=feature_names,
            y=shap_vals,
            marker_color=["#0FC373" if v >= 0 else "#FF4B4B" for v in shap_vals],
            text=[f"{v:+.4f}" for v in shap_vals],
            textposition="outside",
        ))
        fig_wf.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
        fig_wf.update_layout(
            title=f"SHAP Waterfall — {state_sel} (base={base:.3f}, pred={exp['prediction']:.3f})",
            height=300, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#e8eaf6"), yaxis_title="SHAP value",
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        # ── LIME ─────────────────────────────────────────────────────────────
        st.subheader(tx["lime_title"])
        state_lime = st.selectbox("State", state_names, key="lime_state")
        s_lime = int(state_lime[1:])
        exp_lime = exps[s_lime]

        fig_lime = go.Figure()
        fig_lime.add_trace(go.Bar(
            x=feature_names,
            y=exp_lime["lime_coefs"],
            marker_color=["#0082F0" if v >= 0 else "#FF8C0A" for v in exp_lime["lime_coefs"]],
            text=[f"{v:+.4f}" for v in exp_lime["lime_coefs"]],
            textposition="outside",
        ))
        fig_lime.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
        fig_lime.update_layout(
            title=f"LIME — {state_lime} (R²={exp_lime['lime_r2']:.3f}, intercept={exp_lime['lime_intercept']:.3f})",
            height=300, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#e8eaf6"), yaxis_title="LIME coefficient",
        )
        st.plotly_chart(fig_lime, use_container_width=True)

        # ── Saliency ──────────────────────────────────────────────────────────
        st.subheader(tx["saliency_title"])
        sal_rows = []
        for exp in exps:
            for j, feat in enumerate(feature_names):
                sal_rows.append({"State": f"S{exp['state_idx']}", "Feature": feat,
                                 "Saliency": exp["saliency"][j]})
        df_sal = pd.DataFrame(sal_rows)
        chart_sal = alt.Chart(df_sal).mark_rect().encode(
            x=alt.X("Feature:N"),
            y=alt.Y("State:N"),
            color=alt.Color("Saliency:Q", scale=alt.Scale(scheme="oranges")),
            tooltip=["State", "Feature", alt.Tooltip("Saliency:Q", format=".4f")]
        ).properties(height=250, title="Saliency |∂f/∂x| per State and Feature")
        st.altair_chart(chart_sal, use_container_width=True)

        # ── Glass-Box ─────────────────────────────────────────────────────────
        st.subheader(tx["glass_title"])
        state_gb = st.selectbox("State", state_names, key="gb_state")
        s_gb = int(state_gb[1:])
        exp_gb = exps[s_gb]

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", f"{exp_gb['prediction']:.3f}")
        col2.metric("True V*(s)", f"{exp_gb['true_value']:.1f}")
        col3.metric("LIME R²", f"{exp_gb['lime_r2']:.3f}")

        df_detail = pd.DataFrame({
            "Feature":     feature_names,
            "Value":       [round(v, 3) for v in exp_gb["features"]],
            "Gradient FI": [round(v, 4) for v in exp_gb["gradient_fi"]],
            "Saliency":    [round(v, 4) for v in exp_gb["saliency"]],
            "LIME coef":   [round(v, 4) for v in exp_gb["lime_coefs"]],
            "SHAP":        [round(v, 4) for v in exp_gb["shap_values"]],
        })
        st.dataframe(df_detail, use_container_width=True)

        # ── Summary ───────────────────────────────────────────────────────────
        st.subheader(tx["summary_title"])
        summary_rows = []
        for exp in exps:
            summary_rows.append({
                "State":      f"S{exp['state_idx']}",
                "Prediction": round(exp["prediction"], 3),
                "True V*(s)": exp["true_value"],
                "Top Feature (SHAP)": feature_names[
                    int(np.argmax([abs(v) for v in exp["shap_values"]]))
                ],
                "LIME R²":    round(exp["lime_r2"], 3),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
