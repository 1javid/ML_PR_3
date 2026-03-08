"""
Streamlit app for BMW sales visualization and model predictions.
Run the notebook and save models first (models/*.pkl), then: streamlit run app.py
"""
import os
import pickle
import warnings
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="BMW Sales – ML Assignment 3", layout="wide", initial_sidebar_state="expanded")

# Custom theme and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #0f0f12 0%, #1a1a22 50%, #12121a 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #e8e8ec !important;
        letter-spacing: -0.02em;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.04);
        padding: 6px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0066b3 0%, #004d87 100%) !important;
        color: white !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        color: #00d4aa;
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    
    .metric-card h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #888;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 0.25rem 0;
    }
    
    .pred-box {
        background: linear-gradient(145deg, rgba(0,102,179,0.15) 0%, rgba(0,212,170,0.08) 100%);
        border: 1px solid rgba(0,212,170,0.25);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .current-row-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0 0 1rem 0;
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        color: #b0b0b8;
    }
    
    .current-row-card strong { color: #e8e8ec; }
    
    .explorer-panel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #b0b0b8;
        font-size: 0.9rem;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #16161d 0%, #0f0f12 100%);
    }
    
    .stSidebar .stButton > button {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        border-radius: 10px;
        background: linear-gradient(135deg, #0066b3 0%, #004d87 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .stSidebar .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,102,179,0.4);
    }
    
    p, .stCaption, span {
        font-family: 'Outfit', sans-serif;
        color: #b0b0b8 !important;
    }
    
    .stInfo, .stWarning {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "bmw_global_sales_dataset.csv"
MODELS_DIR = "models"


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def load_models():
    """Load pickle files if present. Returns dict with keys or None for missing."""
    out = {}
    for name, filename in [
        ("scaler", "scaler.pkl"),
        ("ridge", "ridge_model.pkl"),
        ("lasso", "lasso_model.pkl"),
        ("ols_model", "ols_best_model.pkl"),
        ("best_features", "best_features.pkl"),
        ("feature_columns", "feature_columns.pkl"),
        ("plot_data", "plot_data.pkl"),
    ]:
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                out[name] = pickle.load(f)
        else:
            out[name] = None
    return out


# Load data and models
df = load_data()
models = load_models()
models_available = models.get("ridge") is not None and models.get("scaler") is not None

if df is None:
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()

st.title("BMW Global Sales")
st.caption("Visualization & model predictions · ML Assignment 3")

tab1, tab2, tab3, tab4 = st.tabs(["Data overview", "Model curves", "Predictions", "Data explorer"])

with tab1:
    st.markdown('<p class="section-header">Dataset snapshot</p>', unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True, height=320)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Target", "units_sold")
    st.markdown('<p class="section-header">Units sold by country</p>', unsafe_allow_html=True)
    by_country = df.groupby("country")["units_sold"].sum().sort_values(ascending=False)
    st.bar_chart(pd.DataFrame(by_country), height=280)
    st.markdown('<p class="section-header">Units sold by model</p>', unsafe_allow_html=True)
    by_model = df.groupby("model")["units_sold"].sum().sort_values(ascending=False)
    st.bar_chart(pd.DataFrame(by_model), height=280)

with tab2:
    st.markdown('<p class="section-header">Ridge & Lasso · CV MSE vs λ</p>', unsafe_allow_html=True)
    plot_data = models.get("plot_data")
    if plot_data is None:
        st.info("Run the notebook and save models (including plot_data.pkl) to see these charts.")
    else:
        log_alphas = np.log10(plot_data["alphas"])
        chart_df = pd.DataFrame({
            "log10(lambda)": log_alphas,
            "Ridge CV MSE": plot_data["ridge_mses"],
            "Lasso CV MSE": plot_data["lasso_mses"],
        })
        st.line_chart(chart_df.set_index("log10(lambda)"), height=300)
        st.caption(f"Optimal Ridge λ = {plot_data['ridge_opt_alpha']:.4f}  ·  Lasso λ = {plot_data['lasso_opt_alpha']:.4f}")
        st.markdown('<p class="section-header">Forward selection · AIC, BIC, Adj R²</p>', unsafe_allow_html=True)
        n_pred = plot_data["num_predictors"]
        sel_df = pd.DataFrame({
            "Number of predictors": n_pred,
            "AIC": plot_data["aic_list"],
            "BIC": plot_data["bic_list"],
            "Adj R²": plot_data["adj_r2_list"],
        })
        st.line_chart(sel_df.set_index("Number of predictors"), height=300)
        st.markdown('<p class="section-header">PCR vs PLS · MSE vs components</p>', unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            "Components": plot_data["components"],
            "PCR (PCA + LinReg) MSE": plot_data["pca_mses"],
            "PLS MSE": plot_data["pls_mses"],
        })
        st.line_chart(comp_df.set_index("Components"), height=300)

with tab3:
    st.markdown('<p class="section-header">Predict units sold</p>', unsafe_allow_html=True)
    if not models_available:
        st.info("Train and save models in the notebook first (run the 'Save models for Streamlit app' cell).")
    else:
        df_encoded = pd.get_dummies(df, drop_first=True)
        saved_cols = models.get("feature_columns")
        feature_cols = saved_cols if saved_cols is not None else [c for c in df_encoded.columns if c != "units_sold"]
        if "pred_row_index" not in st.session_state:
            st.session_state.pred_row_index = 0

        # Row selector: buttons in main tab so you see the effect immediately below
        st.caption("Choose which record to use as input. Predictions update as soon as you change it.")
        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
        with btn_col1:
            if st.button("Use random row", key="rand_row"):
                st.session_state.pred_row_index = int(np.random.randint(0, len(df_encoded)))
        with btn_col2:
            if st.button("Reset to first row", key="reset_row"):
                st.session_state.pred_row_index = 0

        idx = min(st.session_state.pred_row_index, len(df_encoded) - 1)
        sample = df_encoded.iloc[idx : idx + 1].copy()
        # Show the actual record so you see what "current row" is (this changes when you click the buttons)
        row_record = df.iloc[idx]
        st.markdown(
            f'<div class="current-row-card">'
            f'<strong>Current row (index {idx})</strong> · '
            f'Year {int(row_record["year"])} · {row_record["country"]} · {row_record["model"]} · '
            f'{row_record["segment"]} · {row_record["engine_type"]} · '
            f'Price ${int(row_record["price_usd"]):,} · <strong>Actual units_sold: {int(row_record["units_sold"])}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

        X_input = sample.reindex(columns=feature_cols, fill_value=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            X_scaled = models["scaler"].transform(X_input)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

        preds = {}
        ridge = models.get("ridge")
        lasso = models.get("lasso")
        ols_model = models.get("ols_model")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if ridge is not None:
                preds["Ridge"] = float(ridge.predict(X_scaled)[0])
            if lasso is not None:
                preds["Lasso"] = float(lasso.predict(X_scaled)[0])

        if ols_model is not None:
            try:
                param_names = list(ols_model.params.index)
                exog = pd.DataFrame(index=X_scaled_df.index)
                for col in param_names:
                    if col == "const":
                        exog["const"] = 1.0
                    else:
                        exog[col] = X_scaled_df[col] if col in X_scaled_df.columns else 0.0
                exog = exog[param_names]
                preds["OLS (forward selection)"] = float(ols_model.predict(exog)[0])
            except Exception as e:
                st.warning(f"OLS prediction failed: {e}")

        st.markdown('<p class="section-header">Model predictions for this row</p>', unsafe_allow_html=True)
        for name, value in preds.items():
            st.markdown(f'<div class="pred-box">{name}: <span style="color:#00d4aa">{value:.1f}</span> units</div>', unsafe_allow_html=True)
        with st.expander("Encoded input features (for this row)"):
            st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True, height=280)

with tab4:
    # Task 1: Compare by pair — select two dimensions, see how the view changes
    st.markdown('<p class="section-header">Task 1: Compare by pair</p>', unsafe_allow_html=True)
    st.caption("Pick two dimensions; the chart and table below update as soon as you change either.")
    pair_options = ["country", "model", "segment", "engine_type", "year"]
    p1, p2 = st.columns(2)
    with p1:
        dim1 = st.selectbox("First dimension", pair_options, key="dim1")
    with p2:
        dim2 = st.selectbox("Second dimension", [x for x in pair_options if x != dim1], key="dim2")

    agg_by_pair = df.groupby([dim1, dim2])["units_sold"].agg(["sum", "mean", "count"]).reset_index()
    agg_by_pair.columns = [dim1, dim2, "Total units", "Mean units", "Count"]
    # Chart: total units by first dimension (updates when you change the pair)
    chart_series = agg_by_pair.groupby(dim1)["Total units"].sum()
    st.bar_chart(pd.DataFrame({"Total units": chart_series}), height=300)
    st.caption("Breakdown by both dimensions:")
    st.dataframe(agg_by_pair, use_container_width=True, height=280)

    # Task 2: Twist parameters — filters with immediate feedback
    st.markdown('<p class="section-header">Task 2: Twist parameters</p>', unsafe_allow_html=True)
    st.caption("Change any filter; the table and summary metrics update immediately below.")
    years = list(sorted(df["year"].unique()))
    countries = list(sorted(df["country"].unique()))
    car_models = list(sorted(df["model"].unique()))
    segments = list(sorted(df["segment"].unique()))
    engine_types = list(sorted(df["engine_type"].unique()))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        year_sel = st.selectbox("Year", ["All"] + years, key="fy")
    with c2:
        country_sel = st.selectbox("Country", ["All"] + countries, key="fc")
    with c3:
        model_sel = st.selectbox("Model", ["All"] + car_models, key="fm")
    with c4:
        segment_sel = st.selectbox("Segment", ["All"] + segments, key="fs")

    subset = df.copy()
    if year_sel != "All":
        subset = subset[subset["year"] == year_sel]
    if country_sel != "All":
        subset = subset[subset["country"] == country_sel]
    if model_sel != "All":
        subset = subset[subset["model"] == model_sel]
    if segment_sel != "All":
        subset = subset[subset["segment"] == segment_sel]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Rows (filtered)", len(subset))
    with m2:
        st.metric("Mean units sold", f"{subset['units_sold'].mean():.1f}" if len(subset) else "—")
    with m3:
        st.metric("Total units sold", f"{subset['units_sold'].sum():,.0f}" if len(subset) else "—")
    st.dataframe(subset, use_container_width=True, height=320)
