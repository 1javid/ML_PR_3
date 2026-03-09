import os
import pickle
from typing import Any, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

matplotlib.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.linewidth": 0.8,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "lines.linewidth": 2,
    "font.family": "sans-serif",
})

BMW_BLUE   = "#1C69D4"
BMW_DARK   = "#0d1117"
BMW_CARD   = "#161b22"
BMW_BORDER = "#30363d"
BMW_TEXT   = "#c9d1d9"
BMW_MUTED  = "#8b949e"
BMW_GREEN  = "#3fb950"
BMW_RED    = "#f85149"
BMW_YELLOW = "#d29922"


# ── CSS injection ─────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(
        f"""
<style>
/* ── global ── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {BMW_DARK};
    color: {BMW_TEXT};
    font-family: "Inter", "Segoe UI", sans-serif;
}}
[data-testid="stSidebar"] {{
    background-color: #0d1117;
    border-right: 1px solid {BMW_BORDER};
}}
[data-testid="stSidebar"] * {{
    color: {BMW_TEXT} !important;
}}

/* ── header banner ── */
.bmw-header {{
    background: linear-gradient(135deg, #0d1117 0%, #0a1628 50%, #112244 100%);
    border-bottom: 2px solid {BMW_BLUE};
    padding: 2rem 2.5rem 1.5rem;
    margin: -1rem -1rem 1.5rem -1rem;
}}
.bmw-header h1 {{
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.5px;
}}
.bmw-header .sub {{
    color: {BMW_MUTED};
    font-size: 0.9rem;
}}
.bmw-badge {{
    display: inline-block;
    background: {BMW_BLUE};
    color: #fff;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 6px;
    vertical-align: middle;
}}

/* ── metric cards ── */
.card-row {{
    display: flex;
    gap: 12px;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}}
.metric-card {{
    background: {BMW_CARD};
    border: 1px solid {BMW_BORDER};
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    flex: 1;
    min-width: 150px;
    text-align: center;
}}
.metric-card .label {{
    font-size: 0.75rem;
    color: {BMW_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}}
.metric-card .value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
}}
.metric-card .delta-pos {{
    font-size: 0.8rem;
    color: {BMW_GREEN};
    margin-top: 3px;
}}
.metric-card .delta-neg {{
    font-size: 0.8rem;
    color: {BMW_RED};
    margin-top: 3px;
}}
.metric-card .delta-zero {{
    font-size: 0.8rem;
    color: {BMW_MUTED};
    margin-top: 3px;
}}
.metric-card.highlight {{
    border-color: {BMW_BLUE};
    background: linear-gradient(135deg, {BMW_CARD}, #0a1628);
}}

/* ── info / example banner ── */
.info-banner {{
    background: #0a1628;
    border-left: 4px solid {BMW_BLUE};
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: {BMW_TEXT};
}}
.success-banner {{
    background: #0d1f0d;
    border-left: 4px solid {BMW_GREEN};
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: {BMW_TEXT};
}}
.warn-banner {{
    background: #1f1800;
    border-left: 4px solid {BMW_YELLOW};
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: {BMW_TEXT};
}}

/* ── section header ── */
.section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #ffffff;
    border-bottom: 1px solid {BMW_BORDER};
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}}

/* ── feature pill ── */
.pill {{
    display: inline-block;
    background: #21262d;
    border: 1px solid {BMW_BORDER};
    color: {BMW_BLUE};
    border-radius: 100px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}}

/* ── divider ── */
.divider {{
    border: none;
    border-top: 1px solid {BMW_BORDER};
    margin: 1.5rem 0;
}}

/* ── comparison bar ── */
.cmp-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}}
.cmp-label {{
    min-width: 110px;
    font-size: 0.82rem;
    color: {BMW_MUTED};
}}
.cmp-bar-bg {{
    flex: 1;
    background: #21262d;
    border-radius: 4px;
    height: 18px;
    overflow: hidden;
}}
.cmp-bar-fill {{
    height: 18px;
    border-radius: 4px;
    transition: width 0.3s;
}}
.cmp-val {{
    min-width: 60px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #fff;
    text-align: right;
}}

/* ── Streamlit overrides ── */
div[data-testid="metric-container"] {{
    background: {BMW_CARD};
    border: 1px solid {BMW_BORDER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
}}
div[data-testid="metric-container"] label {{
    color: {BMW_MUTED} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
div[data-testid="stTabs"] [data-baseweb="tab"] {{
    color: {BMW_MUTED};
    font-size: 0.9rem;
    font-weight: 500;
}}
div[data-testid="stTabs"] [aria-selected="true"] {{
    color: #ffffff !important;
    border-bottom-color: {BMW_BLUE} !important;
}}
.stDataFrame {{
    border: 1px solid {BMW_BORDER};
    border-radius: 8px;
    overflow: hidden;
}}
button[kind="primary"], .stButton > button {{
    background: {BMW_BLUE} !important;
    border: none !important;
    color: #fff !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}}
button[kind="primary"]:hover, .stButton > button:hover {{
    background: #1558b0 !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


# ── Data & artifact loading ───────────────────────────────────────────────────

@st.cache_data
def load_data(path: str = "bmw_global_sales_dataset.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_artifacts(models_dir: str = "models") -> Dict[str, Any]:
    keys = [
        "scaler", "ridge_model", "lasso_model",
        "ols_best_model", "best_features", "feature_columns", "plot_data",
    ]
    artifacts: Dict[str, Any] = {k: None for k in keys}
    if not os.path.isdir(models_dir):
        return artifacts
    for key in keys:
        p = os.path.join(models_dir, f"{key}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                artifacts[key] = pickle.load(f)
    return artifacts


# ── Preprocessing & prediction ────────────────────────────────────────────────

def encode_and_scale_row(row: pd.Series, scaler) -> Tuple[np.ndarray, pd.Index]:
    scaler_cols = pd.Index(scaler.feature_names_in_)

    # row.to_frame().T upcasts all columns to object dtype, so
    # select_dtypes would treat year/price_usd/… as categorical and dummify
    # them into year_2021, price_usd_82234, etc. — all zeroed by reindex.
    # Fix: inspect the Series values directly to split cat vs numeric.
    cat_cols = [c for c in row.index if isinstance(row[c], str)]
    num_cols = [c for c in row.index if c not in cat_cols]

    # Build one-row DataFrames with correct dtypes
    df_numeric = pd.DataFrame({c: [float(row[c])] for c in num_cols})
    df_cat     = pd.DataFrame({c: [row[c]]       for c in cat_cols})

    # get_dummies on string columns; drop_first=False so the baseline
    # category (e.g. country_Australia) gets a 1 — reindex will then
    # correctly zero-out the non-baseline dummies in scaler_cols.
    df_dummies = (
        pd.get_dummies(df_cat, drop_first=False)
        if not df_cat.empty else pd.DataFrame(index=df_numeric.index)
    )
    df_encoded = pd.concat([df_numeric, df_dummies], axis=1)

    X_row = df_encoded.reindex(columns=scaler_cols, fill_value=0)
    return scaler.transform(X_row)[0], scaler_cols


def predict_for_row(
    row: pd.Series,
    artifacts: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    scaler = artifacts.get("scaler")
    ridge_model = artifacts.get("ridge_model")
    lasso_model = artifacts.get("lasso_model")
    ols_best_model = artifacts.get("ols_best_model")

    if scaler is None or not hasattr(scaler, "feature_names_in_"):
        return None, None, None

    x_scaled, scaler_cols = encode_and_scale_row(row, scaler)
    x_df = pd.DataFrame([x_scaled], columns=scaler_cols)

    ridge_pred = float(ridge_model.predict(x_df)[0]) if ridge_model is not None else None
    lasso_pred = float(lasso_model.predict(x_df)[0]) if lasso_model is not None else None

    if ols_best_model is not None:
        exog_names = list(ols_best_model.model.exog_names)
        feat_names = [n for n in exog_names if n != "const"]
        X_pred = x_df.reindex(columns=feat_names, fill_value=0)
        if "const" in exog_names:
            X_pred = sm.add_constant(X_pred, has_constant="add")
        X_pred = X_pred[exog_names]
        raw = float(ols_best_model.predict(X_pred)[0])
        ols_pred = raw if (np.isfinite(raw) and -1e6 < raw < 1e6) else None
    else:
        ols_pred = None

    return ridge_pred, lasso_pred, ols_pred


def _delta_html(pred: float, actual: float) -> str:
    diff = pred - actual
    pct = 100 * abs(diff) / (actual + 1e-9)
    if diff > 0:
        return f'<div class="delta-pos">▲ {diff:+,.0f} ({pct:.1f}% over)</div>'
    elif diff < 0:
        return f'<div class="delta-neg">▼ {diff:+,.0f} ({pct:.1f}% under)</div>'
    else:
        return f'<div class="delta-zero">= exact</div>'


# ── Sidebar ───────────────────────────────────────────────────────────────────

def show_sidebar(df: pd.DataFrame, artifacts: Dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown(
            f'<div style="text-align:center;padding:1rem 0 0.5rem">'
            f'<span style="font-size:2.2rem">🚗</span>'
            f'<div style="font-size:1rem;font-weight:700;color:#fff;margin-top:4px">BMW ML Dashboard</div>'
            f'<div style="font-size:0.72rem;color:{BMW_MUTED};margin-top:2px">Group 4 · Assignment 3</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<hr style="border-color:{BMW_BORDER};margin:0.75rem 0">', unsafe_allow_html=True)

        if not df.empty:
            st.markdown("**Dataset**")
            c1, c2 = st.columns(2)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Features", df.shape[1] - 1)
            st.markdown(f'<hr style="border-color:{BMW_BORDER};margin:0.75rem 0">', unsafe_allow_html=True)

        st.markdown("**Models loaded**")
        model_map = {
            "scaler": ("⚙️", "Scaler"),
            "ridge_model": ("📐", "Ridge (RidgeCV)"),
            "lasso_model": ("✂️", "Lasso (LassoCV)"),
            "ols_best_model": ("📊", "OLS (min-AIC)"),
        }
        for key, (icon, label) in model_map.items():
            loaded = artifacts.get(key) is not None
            dot = f'<span style="color:{"#3fb950" if loaded else "#f85149"}">{"●" if loaded else "○"}</span>'
            st.markdown(
                f'{dot} {icon} <span style="font-size:0.85rem">{label}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<hr style="border-color:{BMW_BORDER};margin:0.75rem 0">', unsafe_allow_html=True)

        plot_data = artifacts.get("plot_data")
        if isinstance(plot_data, dict):
            ridge_opt = plot_data.get("ridge_opt_alpha")
            lasso_opt = plot_data.get("lasso_opt_alpha")
            if ridge_opt:
                st.markdown(f"**Ridge λ\*** `{ridge_opt:.4f}`")
            if lasso_opt:
                st.markdown(f"**Lasso λ\*** `{lasso_opt:.4f}`")
            num_predictors = np.array(plot_data.get("num_predictors", []))
            aic_list = np.array(plot_data.get("aic_list", []))
            if num_predictors.size and aic_list.size:
                best_n = num_predictors[np.argmin(aic_list)]
                st.markdown(f"**OLS features** `{best_n}` (min-AIC)")

        st.markdown(f'<hr style="border-color:{BMW_BORDER};margin:0.75rem 0">', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.72rem;color:{BMW_MUTED};text-align:center">'
            f'Target: <code>units_sold</code> (15 – 1 242)<br>'
            f'2015 – 2024 · 10 countries · 10 models'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Tab 1: Data Overview ──────────────────────────────────────────────────────

def _styled_bar_fig(x_vals, y_vals, title: str, color: str = BMW_BLUE) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.barh(x_vals[::-1], y_vals[::-1], color=color, alpha=0.85, height=0.6)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in y_vals[::-1]],
                 padding=4, color=BMW_MUTED, fontsize=7.5)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#fff", pad=8)
    ax.set_xlabel("Units Sold", fontsize=8)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.8)
    return fig


def show_data_overview_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.markdown('<div class="warn-banner">⚠️ Data file not found. Place <code>bmw_global_sales_dataset.csv</code> in the project root.</div>', unsafe_allow_html=True)
        return

    # KPI row
    total_sold = df["units_sold"].sum()
    avg_price  = df["price_usd"].mean()
    avg_mkt    = df["marketing_spend_usd"].mean()
    top_market = df.groupby("country")["units_sold"].sum().idxmax()

    st.markdown(
        f"""
<div class="card-row">
  <div class="metric-card highlight">
    <div class="label">Total Units Sold</div>
    <div class="value">{total_sold:,.0f}</div>
  </div>
  <div class="metric-card">
    <div class="label">Avg. Price (USD)</div>
    <div class="value">${avg_price:,.0f}</div>
  </div>
  <div class="metric-card">
    <div class="label">Avg. Marketing Spend</div>
    <div class="value">${avg_mkt:,.0f}</div>
  </div>
  <div class="metric-card">
    <div class="label">Top Market</div>
    <div class="value">{top_market}</div>
  </div>
  <div class="metric-card">
    <div class="label">Records</div>
    <div class="value">{df.shape[0]:,}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Snapshot</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=260)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sales Breakdown</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        cg = df.groupby("country")["units_sold"].sum().sort_values(ascending=False)
        fig = _styled_bar_fig(cg.index.tolist(), cg.values.tolist(), "Units Sold by Country", BMW_BLUE)
        st.pyplot(fig); plt.close(fig)

    with col2:
        mg = df.groupby("model")["units_sold"].sum().sort_values(ascending=False)
        fig = _styled_bar_fig(mg.index.tolist(), mg.values.tolist(), "Units Sold by Model", "#58a6ff")
        st.pyplot(fig); plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:
        eg = df.groupby("engine_type")["units_sold"].sum().sort_values(ascending=False)
        fig = _styled_bar_fig(eg.index.tolist(), eg.values.tolist(), "Units Sold by Engine Type", "#3fb950")
        st.pyplot(fig); plt.close(fig)

    with col4:
        sg = df.groupby("segment")["units_sold"].sum().sort_values(ascending=False)
        fig = _styled_bar_fig(sg.index.tolist(), sg.values.tolist(), "Units Sold by Segment", "#d29922")
        st.pyplot(fig); plt.close(fig)

    # Sales trend
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Annual Sales Trend (All Markets)</div>', unsafe_allow_html=True)
    yearly = df.groupby("year")["units_sold"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(yearly["year"], yearly["units_sold"], alpha=0.18, color=BMW_BLUE)
    ax.plot(yearly["year"], yearly["units_sold"], marker="o", color=BMW_BLUE, linewidth=2.5, markersize=7)
    for x, y in zip(yearly["year"], yearly["units_sold"]):
        ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=BMW_MUTED)
    ax.set_xlabel("Year"); ax.set_ylabel("Units Sold")
    ax.set_title("Total Units Sold per Year", fontweight="bold", color="#fff")
    ax.set_xticks(yearly["year"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ── Tab 2: Model Curves ───────────────────────────────────────────────────────

def show_model_curves_tab(plot_data: Optional[Dict[str, Any]]) -> None:

    # § Resampling
    st.markdown('<div class="section-title">§2 — Resampling: Polynomial Regression MSE vs Degree</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-banner">Feature: <code>marketing_spend_usd → units_sold</code>. '
        'Degrees 1–9 evaluated with 5-Fold CV (mean of 15 splits), 10-Fold CV, LOOCV, and Bootstrap.</div>',
        unsafe_allow_html=True,
    )

    img_poly = "images/resampling_poly.png"
    img_k5   = "images/resampling_poly_k5_repeats.png"
    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(img_poly):
            st.image(img_poly, caption="5-Fold CV · 10-Fold CV · LOOCV · Bootstrap", use_container_width=True)
        else:
            st.markdown(f'<div class="warn-banner">Image <code>{img_poly}</code> not found — run the notebook.</div>', unsafe_allow_html=True)
    with col_b:
        if os.path.exists(img_k5):
            st.image(img_k5, caption="Repeated 5-Fold CV — 15 different splits", use_container_width=True)
        else:
            st.markdown(f'<div class="warn-banner">Image <code>{img_k5}</code> not found — run the notebook.</div>', unsafe_allow_html=True)

    if not isinstance(plot_data, dict):
        st.markdown(
            '<div class="warn-banner">The remaining charts require <code>models/plot_data.pkl</code>. '
            'Run the notebook\'s <strong>§4 Save Models</strong> cell, then refresh.</div>',
            unsafe_allow_html=True,
        )
        return

    # § Forward Stepwise
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">§3.1 — Forward Stepwise Selection: AIC · BIC · Adjusted R²</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-banner">Greedy forward selection on the fully encoded &amp; scaled dataset. '
        'The feature that minimises AIC is added at each step. '
        'The saved OLS model corresponds to the <strong>global min-AIC</strong> point (red dashed line).</div>',
        unsafe_allow_html=True,
    )

    num_predictors = np.array(plot_data.get("num_predictors", []))
    aic_list  = np.array(plot_data.get("aic_list", []))
    bic_list  = np.array(plot_data.get("bic_list", []))
    adj_r2_list = np.array(plot_data.get("adj_r2_list", []))
    cp_list   = np.array(plot_data.get("cp_list", []))

    if num_predictors.size and aic_list.size and bic_list.size and adj_r2_list.size:
        best_aic_n   = num_predictors[np.argmin(aic_list)]
        best_bic_n   = num_predictors[np.argmin(bic_list)]
        best_adjr2_n = num_predictors[np.argmax(adj_r2_list)]

        n_panels = 4 if cp_list.size else 3
        fig, axs = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.2))

        palette = [BMW_BLUE, "#3fb950", "#f85149", "#d29922"]
        titles  = ["AIC", "BIC", "Adj R²", "Mallows' Cp"]
        data    = [aic_list, bic_list, adj_r2_list]
        best_ns = [best_aic_n, best_bic_n, best_adjr2_n]
        vert_colors = ["red", "red", "dodgerblue"]
        vert_labels = [f"Min AIC @ {best_aic_n}", f"Min BIC @ {best_bic_n}", f"Max Adj R² @ {best_adjr2_n}"]

        if cp_list.size:
            best_cp_n = num_predictors[np.argmin(cp_list)]
            data.append(cp_list)
            best_ns.append(best_cp_n)
            vert_colors.append("red")
            vert_labels.append(f"Min Cp @ {best_cp_n}")

        for i, (ax, d, bn, vc, vl, title, color) in enumerate(
            zip(axs, data, best_ns, vert_colors, vert_labels, titles, palette)
        ):
            ax.plot(num_predictors, d, marker="o", color=color, markersize=5)
            ax.axvline(bn, color=vc, linestyle="--", linewidth=1.5, label=vl)
            if title == "Mallows' Cp":
                ax.plot(num_predictors, num_predictors + 1, linestyle=":", color="#8b949e",
                        linewidth=1.2, label="Cp = p (ideal)")
            ax.set_xlabel("Predictors", fontsize=9)
            ax.set_title(f"{title} vs Predictors", fontsize=10, fontweight="bold", color="#fff")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.5)
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout(pad=1.0)
        st.pyplot(fig); plt.close(fig)

        summary_parts = [
            f"Min AIC at **{best_aic_n}** predictors",
            f"Min BIC at **{best_bic_n}** predictors",
            f"Max Adj R² at **{best_adjr2_n}** predictors",
        ]
        if cp_list.size:
            summary_parts.append(f"Min Mallows' Cp at **{best_cp_n}** predictors")
        st.markdown(
            '<div class="success-banner">📌 ' + " · ".join(summary_parts) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="warn-banner">Forward selection data missing from <code>plot_data.pkl</code>.</div>', unsafe_allow_html=True)

    # § Ridge & Lasso
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">§3.2 — Shrinkage Methods: Ridge & Lasso CV MSE vs λ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-banner">5-Fold CV MSE computed over a log-spaced grid of 100 λ values (0.1 → 10 000). '
        'Optimal λ chosen by <code>RidgeCV</code> (5-fold) and <code>LassoCV</code> (10-fold).</div>',
        unsafe_allow_html=True,
    )

    alphas    = np.array(plot_data.get("alphas", []))
    ridge_mses = np.array(plot_data.get("ridge_mses", []))
    lasso_mses = np.array(plot_data.get("lasso_mses", []))
    ridge_opt  = plot_data.get("ridge_opt_alpha")
    lasso_opt  = plot_data.get("lasso_opt_alpha")

    if alphas.size and ridge_mses.size and lasso_mses.size:
        log_alphas = np.log10(alphas)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(log_alphas, ridge_mses, color=BMW_BLUE, label="Ridge 5-Fold CV MSE", linewidth=2)
        ax.plot(log_alphas, lasso_mses, color="#d29922", label="Lasso 5-Fold CV MSE", linewidth=2)
        if ridge_opt is not None:
            ax.axvline(np.log10(ridge_opt), color=BMW_BLUE, linestyle="--", alpha=0.8,
                       label=f"Opt Ridge λ = {ridge_opt:.2f}")
        if lasso_opt is not None:
            ax.axvline(np.log10(lasso_opt), color="#d29922", linestyle="--", alpha=0.8,
                       label=f"Opt Lasso λ = {lasso_opt:.2f}")
        ax.fill_between(log_alphas, ridge_mses, alpha=0.07, color=BMW_BLUE)
        ax.fill_between(log_alphas, lasso_mses, alpha=0.07, color="#d29922")
        ax.set_xlabel("log₁₀(λ)"); ax.set_ylabel("5-Fold CV MSE")
        ax.set_title("CV MSE vs λ — Ridge and Lasso", fontweight="bold", color="#fff")
        ax.legend(); ax.grid(True, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        c1, c2 = st.columns(2)
        if ridge_opt:
            c1.markdown(
                f'<div class="metric-card"><div class="label">Optimal Ridge λ</div>'
                f'<div class="value">{ridge_opt:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        if lasso_opt:
            c2.markdown(
                f'<div class="metric-card"><div class="label">Optimal Lasso λ</div>'
                f'<div class="value">{lasso_opt:.4f}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="warn-banner">Ridge/Lasso data missing from <code>plot_data.pkl</code>.</div>', unsafe_allow_html=True)

    # § Bias-Variance
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">§3.2b — Bias-Variance Tradeoff: Ridge & Lasso</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-banner">Bootstrap decomposition (50 resamples, 30% held-out test set). '
        'As λ → ∞ coefficients shrink to zero: bias rises, variance falls.</div>',
        unsafe_allow_html=True,
    )

    bv_data = plot_data.get("bv_data")
    if isinstance(bv_data, dict) and alphas.size:
        log_alphas = np.log10(alphas)
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
        for ax, prefix, opt, title in [
            (axs[0], "ridge", ridge_opt, "Ridge"),
            (axs[1], "lasso", lasso_opt, "Lasso"),
        ]:
            bias2 = np.array(bv_data.get(f"{prefix}_bias2", []))
            var   = np.array(bv_data.get(f"{prefix}_var", []))
            if bias2.size:
                ax.plot(log_alphas, bias2, label="Bias²", color="#f85149", linewidth=2)
                ax.plot(log_alphas, var,   label="Variance", color="#3fb950", linewidth=2)
                if opt is not None:
                    ax.axvline(np.log10(opt), color="#d29922", linestyle=":", linewidth=1.8,
                               label=f"Opt λ = {opt:.2f}")
                ax.set_xlabel("log₁₀(λ)"); ax.set_ylabel("MSE component")
                ax.set_title(f"{title}: Bias² vs Variance", fontweight="bold", color="#fff")
                ax.legend(); ax.grid(True, alpha=0.5)
                for spine in ax.spines.values():
                    spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
    else:
        st.markdown(
            '<div class="info-banner">Bias-variance data not in <code>plot_data.pkl</code>. '
            'Re-run the notebook (§3.2b cell) to generate it.</div>',
            unsafe_allow_html=True,
        )

    # § PCR vs PLS
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">§3.3 — Dimensionality Reduction: PCR vs PLS — CV MSE vs Components</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-banner">10-Fold CV MSE as a function of the number of components (1 → min(p, 20)). '
        'PCR = PCA on scaled features + LinearRegression. PLS = PLSRegression.</div>',
        unsafe_allow_html=True,
    )

    components = np.array(plot_data.get("components", []))
    pca_mses   = np.array(plot_data.get("pca_mses", []))
    pls_mses   = np.array(plot_data.get("pls_mses", []))

    if components.size and pca_mses.size and pls_mses.size:
        best_pca_n = components[np.argmin(pca_mses)]
        best_pls_n = components[np.argmin(pls_mses)]

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(components, pca_mses, marker="o", color=BMW_BLUE,   label="PCR (PCA + LinReg)", markersize=5)
        ax.plot(components, pls_mses, marker="s", color="#d29922",  label="PLS", markersize=5)
        ax.axvline(best_pca_n, color=BMW_BLUE,  linestyle="--", alpha=0.8, label=f"Min PCR MSE @ {best_pca_n} comp")
        ax.axvline(best_pls_n, color="#d29922", linestyle="--", alpha=0.8, label=f"Min PLS MSE @ {best_pls_n} comp")
        ax.set_xlabel("Number of Components"); ax.set_ylabel("10-Fold CV MSE")
        ax.set_title("MSE vs Number of Components — PCR vs PLS", fontweight="bold", color="#fff")
        ax.legend(); ax.grid(True, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        st.markdown(
            f'<div class="success-banner">📌 Min PCR MSE at <strong>{best_pca_n}</strong> components · '
            f'Min PLS MSE at <strong>{best_pls_n}</strong> components</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="warn-banner">PCR/PLS data missing from <code>plot_data.pkl</code>.</div>', unsafe_allow_html=True)


# ── Tab 3: Predictions ────────────────────────────────────────────────────────

def _render_prediction_cards(actual: float, ridge: Optional[float], lasso: Optional[float], ols: Optional[float]) -> None:
    """Render a styled 4-column card row for actual vs predicted."""
    def _card(label: str, val: Optional[float], ref: Optional[float], color: str, emoji: str) -> str:
        if val is None:
            return f'<div class="metric-card"><div class="label">{emoji} {label}</div><div class="value" style="font-size:1rem;color:{BMW_MUTED}">N/A</div></div>'
        diff = val - ref if ref is not None else 0
        pct  = 100 * abs(diff) / (ref + 1e-9)
        if diff > 0:
            delta_html = f'<div class="delta-pos">▲ +{diff:,.0f} ({pct:.1f}% over)</div>'
        elif diff < 0:
            delta_html = f'<div class="delta-neg">▼ {diff:,.0f} ({pct:.1f}% under)</div>'
        else:
            delta_html = f'<div class="delta-zero">= exact</div>'
        return (
            f'<div class="metric-card" style="border-color:{color}">'
            f'<div class="label">{emoji} {label}</div>'
            f'<div class="value" style="color:{color}">{val:,.0f}</div>'
            f'{delta_html}'
            f'</div>'
        )

    actual_card = (
        f'<div class="metric-card highlight">'
        f'<div class="label">✅ Actual</div>'
        f'<div class="value">{actual:,.0f}</div>'
        f'<div class="delta-zero">ground truth</div>'
        f'</div>'
    )
    ridge_card = _card("Ridge (RidgeCV)", ridge, actual, BMW_BLUE, "📐")
    lasso_card = _card("Lasso (LassoCV)", lasso, actual, "#d29922", "✂️")
    ols_card   = _card("OLS (min-AIC)", ols, actual, "#3fb950", "📊")

    st.markdown(
        f'<div class="card-row">{actual_card}{ridge_card}{lasso_card}{ols_card}</div>',
        unsafe_allow_html=True,
    )


def _render_comparison_bars(actual: float, ridge: Optional[float], lasso: Optional[float], ols: Optional[float]) -> None:
    """Visual comparison bar chart."""
    entries = [("Actual", actual, "#58a6ff")]
    if ridge is not None:
        entries.append(("Ridge", ridge, BMW_BLUE))
    if lasso is not None:
        entries.append(("Lasso", lasso, "#d29922"))
    if ols is not None:
        entries.append(("OLS", ols, "#3fb950"))

    fig, ax = plt.subplots(figsize=(8, 2.6))
    labels = [e[0] for e in entries]
    vals   = [e[1] for e in entries]
    colors = [e[2] for e in entries]
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.55, alpha=0.9)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in vals[::-1]], padding=5, fontsize=9, color=BMW_TEXT)
    ax.set_xlabel("Units Sold")
    ax.set_title("Prediction Comparison", fontweight="bold", color="#fff", pad=8)
    ax.axvline(actual, color="#58a6ff", linestyle=":", linewidth=1.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


def show_predictions_tab(df: pd.DataFrame, artifacts: Dict[str, Any]) -> None:
    if df.empty:
        st.markdown('<div class="warn-banner">⚠️ No data loaded.</div>', unsafe_allow_html=True)
        return

    ols_model = artifacts.get("ols_best_model")
    if ols_model is not None:
        ols_feats = [n for n in ols_model.model.exog_names if n != "const"]
        feat_pills = " ".join(f'<span class="pill">{f}</span>' for f in ols_feats)
        st.markdown(
            f'<div class="info-banner">📊 OLS uses <strong>{len(ols_feats)}</strong> features selected by '
            f'min-AIC forward stepwise:<br><br>{feat_pills}</div>',
            unsafe_allow_html=True,
        )

    # ── Section A: Pick from dataset ────────────────────────────────────────
    st.markdown('<div class="section-title">A · Pick a Dataset Row</div>', unsafe_allow_html=True)
    choice_mode = st.radio("Row selection", ("Random row", "By index"), horizontal=True)

    if choice_mode == "Random row":
        if st.button("🎲 Sample new random row"):
            st.session_state["sample_idx"] = int(np.random.randint(0, len(df)))
        idx = st.session_state.get("sample_idx", 0)
    else:
        idx = st.slider("Row index", min_value=0, max_value=len(df) - 1, value=0)
        st.session_state["sample_idx"] = idx

    row = df.iloc[idx]
    st.markdown(f'<div class="info-banner">📍 Row <strong>{idx}</strong> selected</div>', unsafe_allow_html=True)
    st.dataframe(row.to_frame().T, use_container_width=True)

    actual = float(row["units_sold"])
    ridge_pred, lasso_pred, ols_pred = predict_for_row(row, artifacts)

    if ridge_pred is None and lasso_pred is None and ols_pred is None:
        st.markdown(
            '<div class="warn-banner">⚠️ Models not loaded. Run the notebook <strong>§4 Save Models</strong> cell '
            'and ensure all <code>.pkl</code> files exist in <code>models/</code>.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predicted vs Actual — Units Sold</div>', unsafe_allow_html=True)
    _render_prediction_cards(actual, ridge_pred, lasso_pred, ols_pred)
    _render_comparison_bars(actual, ridge_pred, lasso_pred, ols_pred)

    # ── Section B: Custom Input ──────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">B · Custom Prediction Input</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-banner">🔧 Build a custom observation and run all three models on it in real time.</div>',
        unsafe_allow_html=True,
    )

    with st.form("custom_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            year_in  = st.selectbox("Year",  sorted(df["year"].unique()), index=len(df["year"].unique()) - 1)
            month_in = st.selectbox("Month", list(range(1, 13)), index=0)
            country_in = st.selectbox("Country", sorted(df["country"].unique()))
        with c2:
            model_in   = st.selectbox("Model",   sorted(df["model"].unique()))
            segment_in = st.selectbox("Segment", sorted(df["segment"].unique()))
            engine_in  = st.selectbox("Engine Type", sorted(df["engine_type"].unique()))
        with c3:
            price_in  = st.slider("Price (USD)",  int(df.price_usd.min()), int(df.price_usd.max()),
                                  int(df.price_usd.median()), step=500)
            mkt_in    = st.slider("Marketing Spend (USD)", int(df.marketing_spend_usd.min()),
                                  int(df.marketing_spend_usd.max()),
                                  int(df.marketing_spend_usd.median()), step=1000)
            deal_in   = st.slider("Dealership Count", int(df.dealership_count.min()),
                                  int(df.dealership_count.max()),
                                  int(df.dealership_count.median()), step=1)

        c4, c5, c6 = st.columns(3)
        with c4:
            fuel_in = st.number_input("Fuel Price (USD)", value=float(df.fuel_price_usd.median()), format="%.2f")
        with c5:
            gdp_in  = st.number_input("GDP Growth (%)",   value=float(df.gdp_growth_percent.median()), format="%.2f")
        with c6:
            ir_in   = st.number_input("Interest Rate (%)", value=float(df.interest_rate_percent.median()), format="%.2f")

        comp_in = st.slider("Competition Index", int(df.competition_index.min()),
                             int(df.competition_index.max()),
                             int(df.competition_index.median()), step=1)

        submitted = st.form_submit_button("⚡ Run Prediction", use_container_width=True)

    if submitted:
        custom_row = pd.Series({
            "year": year_in, "month": month_in, "country": country_in,
            "model": model_in, "segment": segment_in, "engine_type": engine_in,
            "price_usd": price_in, "marketing_spend_usd": mkt_in,
            "dealership_count": deal_in, "fuel_price_usd": fuel_in,
            "gdp_growth_percent": gdp_in, "interest_rate_percent": ir_in,
            "competition_index": comp_in,
        })
        r_p, l_p, o_p = predict_for_row(custom_row, artifacts)

        st.markdown('<div class="section-title">Custom Prediction Result</div>', unsafe_allow_html=True)
        def _single_card(label: str, val: Optional[float], color: str, emoji: str) -> str:
            if val is None:
                return f'<div class="metric-card"><div class="label">{emoji} {label}</div><div class="value" style="font-size:1rem;color:{BMW_MUTED}">N/A</div></div>'
            return (
                f'<div class="metric-card" style="border-color:{color}">'
                f'<div class="label">{emoji} {label}</div>'
                f'<div class="value" style="color:{color}">{val:,.0f}</div>'
                f'<div class="delta-zero">estimated units sold</div></div>'
            )
        cards = (
            _single_card("Ridge (RidgeCV)", r_p, BMW_BLUE, "📐")
            + _single_card("Lasso (LassoCV)", l_p, "#d29922", "✂️")
            + _single_card("OLS (min-AIC)", o_p, "#3fb950", "📊")
        )
        st.markdown(f'<div class="card-row">{cards}</div>', unsafe_allow_html=True)

        if any(v is not None for v in [r_p, l_p, o_p]):
            _render_comparison_bars(
                float(np.nanmean([v for v in [r_p, l_p, o_p] if v is not None])),
                r_p, l_p, o_p,
            )

    # ── Section C: Example Results ───────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">C · Example Results Showcase</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-banner">📋 Pre-selected rows that illustrate model behaviour across different '
        'scenarios — high-volume markets, niche models, different engine types.</div>',
        unsafe_allow_html=True,
    )

    scaler = artifacts.get("scaler")
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        example_indices = []
        try:
            # Representative examples: highest & lowest units_sold, mid-range
            example_indices = [
                int(df["units_sold"].idxmax()),
                int(df["units_sold"].idxmin()),
                int(df[df["engine_type"] == "Electric"]["units_sold"].idxmax())
                    if "Electric" in df["engine_type"].values else 0,
                int(df[df["country"] == "China"]["units_sold"].idxmax())
                    if "China" in df["country"].values else 0,
                int(df[df["segment"] == "SUV"]["units_sold"].idxmax())
                    if "SUV" in df["segment"].values else 0,
            ]
            example_indices = list(dict.fromkeys(example_indices))[:5]
        except Exception:
            example_indices = list(range(min(5, len(df))))

        rows_data = []
        for ei in example_indices:
            erow = df.iloc[ei]
            r, l, o = predict_for_row(erow, artifacts)
            rows_data.append({
                "idx": ei,
                "Country": erow["country"],
                "Model": erow["model"],
                "Engine": erow["engine_type"],
                "Year": int(erow["year"]),
                "Actual": int(erow["units_sold"]),
                "Ridge": int(round(r)) if r is not None else None,
                "Lasso": int(round(l)) if l is not None else None,
                "OLS":   int(round(o)) if o is not None else None,
            })

        if rows_data:
            ex_df = pd.DataFrame(rows_data).set_index("idx")
            ex_df.index.name = "Row"
            st.dataframe(ex_df, use_container_width=True)

            # Visual bar chart of example predictions
            labels   = [f"R{r['idx']}: {r['Country']}/{r['Model']}" for r in rows_data]
            actuals  = [r["Actual"]   for r in rows_data]
            ridges   = [r["Ridge"]  if r["Ridge"]  is not None else 0 for r in rows_data]
            lassos   = [r["Lasso"]  if r["Lasso"]  is not None else 0 for r in rows_data]
            olss     = [r["OLS"]    if r["OLS"]    is not None else 0 for r in rows_data]

            x = np.arange(len(labels))
            w = 0.2
            fig, ax = plt.subplots(figsize=(13, 4.5))
            ax.bar(x - 1.5*w, actuals, w, label="Actual",  color="#58a6ff", alpha=0.9)
            ax.bar(x - 0.5*w, ridges,  w, label="Ridge",   color=BMW_BLUE,  alpha=0.9)
            ax.bar(x + 0.5*w, lassos,  w, label="Lasso",   color="#d29922", alpha=0.9)
            ax.bar(x + 1.5*w, olss,    w, label="OLS",     color="#3fb950", alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8.5)
            ax.set_ylabel("Units Sold")
            ax.set_title("Example Predictions — Actual vs Ridge vs Lasso vs OLS",
                         fontweight="bold", color="#fff")
            ax.legend(loc="upper right")
            ax.grid(True, axis="y", alpha=0.4); ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
    else:
        st.markdown(
            '<div class="warn-banner">⚠️ Load models first to see example results.</div>',
            unsafe_allow_html=True,
        )


# ── Tab 4: Data Explorer ──────────────────────────────────────────────────────

def show_explorer_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.markdown('<div class="warn-banner">⚠️ No data loaded.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="section-title">Filter the Dataset</div>', unsafe_allow_html=True)

    with st.expander("🔽 Filter controls", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        years        = sorted(df["year"].unique())
        months       = sorted(df["month"].unique())
        countries    = sorted(df["country"].unique())
        models_list  = sorted(df["model"].unique())
        segments     = sorted(df["segment"].unique())
        engines      = sorted(df["engine_type"].unique())

        year_sel    = col1.multiselect("Year",    years,       default=years)
        month_sel   = col2.multiselect("Month",   months,      default=months)
        country_sel = col3.multiselect("Country", countries,   default=countries)
        model_sel   = col4.multiselect("Model",   models_list, default=models_list)

        col5, col6, col7 = st.columns(3)
        seg_sel     = col5.multiselect("Segment",     segments, default=segments)
        eng_sel     = col6.multiselect("Engine Type", engines,  default=engines)
        price_range = col7.slider(
            "Price range (USD)",
            int(df.price_usd.min()), int(df.price_usd.max()),
            (int(df.price_usd.min()), int(df.price_usd.max())),
            step=500,
        )

    mask = (
        df["year"].isin(year_sel)
        & df["month"].isin(month_sel)
        & df["country"].isin(country_sel)
        & df["model"].isin(model_sel)
        & df["segment"].isin(seg_sel)
        & df["engine_type"].isin(eng_sel)
        & df["price_usd"].between(*price_range)
    )
    df_filt = df[mask]

    match_pct = 100 * df_filt.shape[0] / df.shape[0]
    badge_color = BMW_GREEN if match_pct > 50 else BMW_YELLOW if match_pct > 20 else BMW_RED
    st.markdown(
        f'<div class="info-banner">Showing <strong style="color:{badge_color}">{df_filt.shape[0]:,}</strong> '
        f'rows ({match_pct:.1f}%) out of {df.shape[0]:,} total</div>',
        unsafe_allow_html=True,
    )

    if df_filt.empty:
        st.markdown('<div class="warn-banner">⚠️ No rows match the current filters.</div>', unsafe_allow_html=True)
        return

    st.dataframe(df_filt, use_container_width=True, height=280)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Units Sold over Time</div>', unsafe_allow_html=True)
        df_time = (
            df_filt.groupby(["year", "month"])["units_sold"]
            .sum().reset_index().sort_values(["year", "month"])
        )
        df_time["ym"] = (
            df_time["year"].astype(str) + "-" + df_time["month"].astype(str).str.zfill(2)
        )
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.fill_between(range(len(df_time)), df_time["units_sold"], alpha=0.18, color=BMW_BLUE)
        ax.plot(range(len(df_time)), df_time["units_sold"], color=BMW_BLUE, linewidth=1.8)
        step = max(1, len(df_time) // 8)
        ax.set_xticks(range(0, len(df_time), step))
        ax.set_xticklabels(df_time["ym"].iloc[::step], rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("Units Sold")
        ax.set_title("Monthly Units Sold", fontweight="bold", color="#fff")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col_b:
        st.markdown('<div class="section-title">Average Units Sold by Country</div>', unsafe_allow_html=True)
        df_country = (
            df_filt.groupby("country")["units_sold"].mean()
            .sort_values(ascending=False).reset_index()
        )
        fig = _styled_bar_fig(
            df_country["country"].tolist(),
            df_country["units_sold"].tolist(),
            "Avg Units Sold by Country",
            BMW_BLUE,
        )
        st.pyplot(fig); plt.close(fig)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Summary Statistics (Filtered)</div>', unsafe_allow_html=True)
    num_cols = [
        "price_usd", "marketing_spend_usd", "dealership_count",
        "fuel_price_usd", "gdp_growth_percent", "interest_rate_percent",
        "competition_index", "units_sold",
    ]
    st.dataframe(df_filt[[c for c in num_cols if c in df_filt.columns]].describe(), use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="BMW Global Sales — ML Dashboard",
        layout="wide",
        page_icon="🚗",
        initial_sidebar_state="expanded",
    )

    inject_css()

    st.markdown(
        """
<div class="bmw-header">
  <h1>🚗 BMW Global Sales — ML Dashboard</h1>
  <div class="sub">
    <span class="bmw-badge">Ridge</span>
    <span class="bmw-badge">Lasso</span>
    <span class="bmw-badge">OLS</span>
    <span class="bmw-badge">PCR / PLS</span>
    Predict <strong>units_sold</strong> · Explore resampling, model selection & shrinkage methods
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    df        = load_data()
    artifacts = load_artifacts()

    show_sidebar(df, artifacts)

    missing = [
        k for k, v in artifacts.items()
        if v is None and k not in ("plot_data", "best_features", "feature_columns")
    ]
    if missing:
        st.markdown(
            f'<div class="warn-banner">⚠️ Missing model artifacts: '
            f'<code>{"</code>, <code>".join(missing)}</code>. '
            f'Run the notebook <strong>§4 Save Models</strong> cell, then refresh.</div>',
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Data Overview",
        "📈  Model Curves",
        "🔮  Predictions",
        "🔍  Data Explorer",
    ])

    with tab1:
        show_data_overview_tab(df)
    with tab2:
        show_model_curves_tab(artifacts.get("plot_data"))
    with tab3:
        show_predictions_tab(df, artifacts)
    with tab4:
        show_explorer_tab(df)


if __name__ == "__main__":
    main()
