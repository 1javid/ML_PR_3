import os
import pickle
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st


# ── Data & artifact loading ───────────────────────────────────────────────────

@st.cache_data
def load_data(path: str = "bmw_global_sales_dataset.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Data file `{path}` not found. Make sure it is in the project root.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_artifacts(models_dir: str = "models") -> Dict[str, Any]:
    """Load every pickle saved by the notebook's §4 Save Models cell."""
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

def encode_and_scale_row(
    row: pd.Series,
    feature_columns: pd.Index,
    scaler,
) -> np.ndarray:
    """
    Reproduce notebook preprocessing on a single row:
      pd.get_dummies(drop_first=True) → align to training columns → StandardScaler.
    """
    df_single = pd.get_dummies(row.to_frame().T, drop_first=True)
    X_row = df_single.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(X_row)[0]


def predict_for_row(
    row: pd.Series,
    artifacts: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    scaler = artifacts.get("scaler")
    ridge_model = artifacts.get("ridge_model")
    lasso_model = artifacts.get("lasso_model")
    ols_best_model = artifacts.get("ols_best_model")
    best_features = artifacts.get("best_features")
    feature_columns = artifacts.get("feature_columns")

    if scaler is None or feature_columns is None:
        return None, None, None

    x_scaled = encode_and_scale_row(row, pd.Index(feature_columns), scaler)
    x_df = pd.DataFrame([x_scaled], columns=feature_columns)

    ridge_pred = float(ridge_model.predict(x_df)[0]) if ridge_model is not None else None
    lasso_pred = float(lasso_model.predict(x_df)[0]) if lasso_model is not None else None

    if ols_best_model is not None:
        # Derive the exact feature/constant layout from the model itself so that
        # prediction is always consistent regardless of what best_features.pkl holds.
        exog_names = list(ols_best_model.model.exog_names)
        feat_names = [n for n in exog_names if n != "const"]
        X_pred = x_df.reindex(columns=feat_names, fill_value=0)
        if "const" in exog_names:
            X_pred = sm.add_constant(X_pred, has_constant="add")
        X_pred = X_pred[exog_names]  # enforce training column order
        ols_pred = float(ols_best_model.predict(X_pred)[0])
    else:
        ols_pred = None

    return ridge_pred, lasso_pred, ols_pred


# ── Tab 1: Data overview ──────────────────────────────────────────────────────

def show_data_overview_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No data loaded.")
        return

    st.subheader("Dataset snapshot")
    st.write(
        f"**{df.shape[0]} rows × {df.shape[1]} columns** — "
        "target variable: `units_sold`"
    )
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("---")
    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total units sold by country")
        cg = (
            df.groupby("country")["units_sold"]
            .sum().reset_index()
            .sort_values("units_sold", ascending=False)
        )
        st.bar_chart(cg, x="country", y="units_sold")

    with col2:
        st.subheader("Total units sold by model")
        mg = (
            df.groupby("model")["units_sold"]
            .sum().reset_index()
            .sort_values("units_sold", ascending=False)
        )
        st.bar_chart(mg, x="model", y="units_sold")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Units sold by engine type")
        eg = (
            df.groupby("engine_type")["units_sold"]
            .sum().reset_index()
            .sort_values("units_sold", ascending=False)
        )
        st.bar_chart(eg, x="engine_type", y="units_sold")

    with col4:
        st.subheader("Units sold by segment")
        sg = (
            df.groupby("segment")["units_sold"]
            .sum().reset_index()
            .sort_values("units_sold", ascending=False)
        )
        st.bar_chart(sg, x="segment", y="units_sold")


# ── Tab 2: Model curves ───────────────────────────────────────────────────────

def show_model_curves_tab(plot_data: Optional[Dict[str, Any]]) -> None:

    # §2 — Resampling ─────────────────────────────────────────────────────────
    st.subheader("§2 — Resampling: Polynomial Regression MSE vs Degree")
    st.caption(
        "Feature: `marketing_spend_usd` → `units_sold`. "
        "Degrees 1–9 evaluated with 5-Fold CV (mean of 15 splits), "
        "10-Fold CV, LOOCV, and Bootstrap."
    )

    img_poly = "images/resampling_poly.png"
    img_k5 = "images/resampling_poly_k5_repeats.png"
    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(img_poly):
            st.image(img_poly, caption="5-Fold CV · 10-Fold CV · LOOCV · Bootstrap", use_container_width=True)
        else:
            st.info(f"`{img_poly}` not found — run the notebook to generate it.")
    with col_b:
        if os.path.exists(img_k5):
            st.image(img_k5, caption="Repeated 5-Fold CV — 15 different splits", use_container_width=True)
        else:
            st.info(f"`{img_k5}` not found — run the notebook to generate it.")

    if not isinstance(plot_data, dict):
        st.markdown("---")
        st.info(
            "The remaining charts require `models/plot_data.pkl`. "
            "Run the notebook's **§4 Save Models** cell, then refresh."
        )
        return

    # §3.1 — Forward stepwise selection ──────────────────────────────────────
    st.markdown("---")
    st.subheader("§3.1 — Forward Stepwise Selection: AIC, BIC, Adjusted R²")
    st.caption(
        "Greedy forward selection on the fully encoded & scaled dataset. "
        "At each step the feature that minimises AIC is added. "
        "The saved OLS model corresponds to the **global min-AIC** point (red dashed line)."
    )

    num_predictors = np.array(plot_data.get("num_predictors", []))
    aic_list = np.array(plot_data.get("aic_list", []))
    bic_list = np.array(plot_data.get("bic_list", []))
    adj_r2_list = np.array(plot_data.get("adj_r2_list", []))

    if num_predictors.size and aic_list.size and bic_list.size and adj_r2_list.size:
        best_aic_n = num_predictors[np.argmin(aic_list)]
        best_bic_n = num_predictors[np.argmin(bic_list)]
        best_adjr2_n = num_predictors[np.argmax(adj_r2_list)]

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].plot(num_predictors, aic_list, marker="o")
        axs[0].axvline(best_aic_n, color="red", linestyle="--", label=f"Min AIC @ {best_aic_n}")
        axs[0].set_xlabel("Number of Predictors")
        axs[0].set_title("AIC vs Predictors")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(num_predictors, bic_list, marker="o", color="g")
        axs[1].axvline(best_bic_n, color="red", linestyle="--", label=f"Min BIC @ {best_bic_n}")
        axs[1].set_xlabel("Number of Predictors")
        axs[1].set_title("BIC vs Predictors")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(num_predictors, adj_r2_list, marker="o", color="r")
        axs[2].axvline(best_adjr2_n, color="blue", linestyle="--", label=f"Max Adj R² @ {best_adjr2_n}")
        axs[2].set_xlabel("Number of Predictors")
        axs[2].set_title("Adjusted R² vs Predictors")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            f"Min AIC at **{best_aic_n}** predictors · "
            f"Min BIC at **{best_bic_n}** predictors · "
            f"Max Adj R² at **{best_adjr2_n}** predictors"
        )
    else:
        st.warning("Forward selection data missing from `plot_data.pkl`.")

    # §3.2 — Ridge & Lasso ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("§3.2 — Shrinkage Methods: Ridge & Lasso CV MSE vs λ")
    st.caption(
        "5-Fold CV MSE computed over a log-spaced grid of 100 λ values (0.1 → 10 000). "
        "Optimal λ chosen by `RidgeCV` (5-fold) and `LassoCV` (10-fold)."
    )

    alphas = np.array(plot_data.get("alphas", []))
    ridge_mses = np.array(plot_data.get("ridge_mses", []))
    lasso_mses = np.array(plot_data.get("lasso_mses", []))
    ridge_opt = plot_data.get("ridge_opt_alpha")
    lasso_opt = plot_data.get("lasso_opt_alpha")

    if alphas.size and ridge_mses.size and lasso_mses.size:
        log_alphas = np.log10(alphas)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(log_alphas, ridge_mses, label="Ridge 5-Fold CV MSE")
        ax.plot(log_alphas, lasso_mses, label="Lasso 5-Fold CV MSE")
        if ridge_opt is not None:
            ax.axvline(
                np.log10(ridge_opt), color="blue", linestyle="--",
                label=f"Opt Ridge λ = {ridge_opt:.2f}",
            )
        if lasso_opt is not None:
            ax.axvline(
                np.log10(lasso_opt), color="orange", linestyle="--",
                label=f"Opt Lasso λ = {lasso_opt:.2f}",
            )
        ax.set_xlabel("log₁₀(λ)")
        ax.set_ylabel("5-Fold CV MSE")
        ax.set_title("CV MSE vs λ for Ridge and Lasso")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if ridge_opt is not None and lasso_opt is not None:
            st.caption(
                f"Optimal Ridge λ: **{ridge_opt:.4f}** · "
                f"Optimal Lasso λ: **{lasso_opt:.4f}**"
            )
    else:
        st.warning("Ridge/Lasso data missing from `plot_data.pkl`.")

    # §3.3 — PCR vs PLS ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("§3.3 — Dimensionality Reduction: PCR vs PLS — CV MSE vs Components")
    st.caption(
        "10-Fold CV MSE as a function of the number of components (1 → min(p, 20)). "
        "PCR = PCA on scaled features + LinearRegression. PLS = PLSRegression."
    )

    components = np.array(plot_data.get("components", []))
    pca_mses = np.array(plot_data.get("pca_mses", []))
    pls_mses = np.array(plot_data.get("pls_mses", []))

    if components.size and pca_mses.size and pls_mses.size:
        best_pca_n = components[np.argmin(pca_mses)]
        best_pls_n = components[np.argmin(pls_mses)]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(components, pca_mses, marker="o", label="PCR (PCA + LinReg)")
        ax.plot(components, pls_mses, marker="s", label="PLS")
        ax.axvline(
            best_pca_n, color="blue", linestyle="--",
            label=f"Min PCR MSE @ {best_pca_n} comp",
        )
        ax.axvline(
            best_pls_n, color="orange", linestyle="--",
            label=f"Min PLS MSE @ {best_pls_n} comp",
        )
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("10-Fold CV MSE")
        ax.set_title("MSE vs Number of Components (PCR vs PLS)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            f"Min PCR MSE at **{best_pca_n}** components · "
            f"Min PLS MSE at **{best_pls_n}** components"
        )
    else:
        st.warning("PCR/PLS data missing from `plot_data.pkl`.")


# ── Tab 3: Predictions ────────────────────────────────────────────────────────

def show_predictions_tab(df: pd.DataFrame, artifacts: Dict[str, Any]) -> None:
    if df.empty:
        st.warning("No data loaded.")
        return

    ols_model = artifacts.get("ols_best_model")
    if ols_model is not None:
        ols_feats = [n for n in ols_model.model.exog_names if n != "const"]
        st.info(
            f"OLS uses **{len(ols_feats)}** features selected by min-AIC forward stepwise: "
            f"`{'`, `'.join(ols_feats)}`"
        )

    st.subheader("Pick an observation")
    choice_mode = st.radio(
        "Row selection",
        ("Random row", "By index"),
        horizontal=True,
    )

    if choice_mode == "Random row":
        if st.button("Sample new random row"):
            st.session_state["sample_idx"] = int(np.random.randint(0, len(df)))
        idx = st.session_state.get("sample_idx", 0)
    else:
        idx = st.slider("Row index", min_value=0, max_value=len(df) - 1, value=0)
        st.session_state["sample_idx"] = idx

    row = df.iloc[idx]
    st.markdown(f"**Selected index:** `{idx}`")
    st.dataframe(row.to_frame().T, use_container_width=True)

    actual = float(row["units_sold"])
    ridge_pred, lasso_pred, ols_pred = predict_for_row(row, artifacts)

    if ridge_pred is None and lasso_pred is None and ols_pred is None:
        st.info(
            "Models not loaded. Run the notebook **§4 Save Models** cell and "
            "ensure all `.pkl` files exist in `models/`."
        )
        return

    st.markdown("---")
    st.subheader("Predicted vs actual units sold")
    cols = st.columns(4)
    cols[0].metric("Actual", f"{actual:,.0f}")
    if ridge_pred is not None:
        cols[1].metric("Ridge (RidgeCV)", f"{ridge_pred:,.0f}", f"{ridge_pred - actual:+,.0f}")
    if lasso_pred is not None:
        cols[2].metric("Lasso (LassoCV)", f"{lasso_pred:,.0f}", f"{lasso_pred - actual:+,.0f}")
    if ols_pred is not None:
        cols[3].metric("OLS (min-AIC)", f"{ols_pred:,.0f}", f"{ols_pred - actual:+,.0f}")


# ── Tab 4: Data explorer ──────────────────────────────────────────────────────

def show_explorer_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No data loaded.")
        return

    st.subheader("Filter the dataset")
    col1, col2, col3, col4 = st.columns(4)

    years = sorted(df["year"].unique())
    countries = sorted(df["country"].unique())
    models_list = sorted(df["model"].unique())
    segments = sorted(df["segment"].unique())

    year_sel = col1.multiselect("Year", years, default=years)
    country_sel = col2.multiselect("Country", countries, default=countries)
    model_sel = col3.multiselect("Model", models_list, default=models_list)
    segment_sel = col4.multiselect("Segment", segments, default=segments)

    mask = (
        df["year"].isin(year_sel)
        & df["country"].isin(country_sel)
        & df["model"].isin(model_sel)
        & df["segment"].isin(segment_sel)
    )
    df_filt = df[mask]

    st.write(f"**{df_filt.shape[0]}** rows match (out of {df.shape[0]} total)")

    if df_filt.empty:
        st.warning("No rows match the current filters.")
        return

    st.dataframe(df_filt, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Units sold over time")
        df_time = (
            df_filt.groupby(["year", "month"])["units_sold"]
            .sum().reset_index()
            .sort_values(["year", "month"])
        )
        df_time["year_month"] = (
            df_time["year"].astype(str) + "-"
            + df_time["month"].astype(str).str.zfill(2)
        )
        st.line_chart(df_time.set_index("year_month")["units_sold"])

    with col_b:
        st.subheader("Average units sold by country")
        df_country = (
            df_filt.groupby("country")["units_sold"]
            .mean().reset_index()
            .sort_values("units_sold", ascending=False)
        )
        st.bar_chart(df_country, x="country", y="units_sold")

    st.markdown("---")
    st.subheader("Summary statistics (filtered)")
    num_cols = [
        "price_usd", "marketing_spend_usd", "dealership_count",
        "fuel_price_usd", "gdp_growth_percent", "interest_rate_percent",
        "competition_index", "units_sold",
    ]
    st.dataframe(df_filt[num_cols].describe(), use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="BMW Global Sales — ML Dashboard",
        layout="wide",
        page_icon="🚗",
    )

    st.title("BMW Global Sales — Interactive ML Dashboard")
    st.caption(
        "Predict BMW units sold using Ridge, Lasso, and OLS (forward stepwise). "
        "Explore resampling curves, model selection criteria, and live predictions."
    )

    df = load_data()
    artifacts = load_artifacts()

    missing = [k for k, v in artifacts.items() if v is None and k != "plot_data"]
    if missing:
        st.warning(
            f"Some model artifacts are missing: `{'`, `'.join(missing)}`. "
            "Run the notebook **§4 Save Models** cell, then refresh."
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data overview", "📈 Model curves", "🔮 Predictions", "🔍 Data explorer"]
    )

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
