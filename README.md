# BMW Global Sales — ML Assignment 3

## Title

**BMW Global Sales: Regression, Model Selection & Interactive Dashboard**

## Description

This project uses the BMW global sales dataset to predict **units sold** from features such as year, month, country, model, segment, engine type, price, marketing spend, dealership count, and macroeconomic variables. It covers:

- Resampling methods (5-Fold CV, 10-Fold CV, LOOCV, Bootstrap) for polynomial regression (degrees 1–9)
- Model selection via forward stepwise OLS (AIC / BIC / Adjusted R²), with the optimal model chosen at the global min-AIC point
- Shrinkage methods (Ridge and Lasso) with cross-validated λ tuning
- Dimensionality reduction (PCR and PLS) and comparison of CV MSE vs number of components

Training and analysis are done in a Jupyter notebook; an interactive Streamlit app provides visualizations, live predictions, and a custom prediction builder.

---

## Project structure

```
ML_PR_3/
├── README.md
├── requirements.txt
├── bmw_global_sales_dataset.csv        # Input data (target: units_sold)
├── ML_Group_4_Assignment_3.ipynb       # Training, analysis, and saving models
├── app.py                              # Streamlit dashboard
├── images/                             # Plots saved by the notebook
│   ├── resampling_poly.png             # Resampling MSE curves (5-fold, 10-fold, LOOCV, bootstrap)
│   ├── resampling_poly_k5_repeats.png  # Repeated 5-fold CV variability
│   ├── subset_selection.png            # AIC / BIC / Adj R² vs number of predictors
│   ├── shrinkage.png                   # Ridge & Lasso CV MSE vs λ
│   ├── bias_variance.png               # Bias-variance tradeoff (Ridge & Lasso)
│   └── pca_pls.png                     # PCR vs PLS MSE vs components
└── models/                             # Created by notebook §4 (pickle artifacts)
    ├── scaler.pkl
    ├── ridge_model.pkl
    ├── lasso_model.pkl
    ├── ols_best_model.pkl
    └── plot_data.pkl
```

---

## Setup

1. Clone or download the project and go to the project folder:

   ```bash
   cd ML_PR_3
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Linux/macOS
   # .venv\Scripts\activate           # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the notebook

1. Start Jupyter (from the project root with the venv activated):

   ```bash
   jupyter notebook
   ```

2. Open `ML_Group_4_Assignment_3.ipynb`.

3. Run all cells in order (e.g. **Run → Run All Cells**). This will:
   - Load and encode the data
   - Run polynomial resampling (degrees 1–9), forward stepwise selection, Ridge/Lasso CV, and PCR/PLS
   - Save plots to the `images/` directory
   - Create the `models/` directory and save all pickle files (scaler, Ridge, Lasso, OLS, plot data)

4. Ensure the **§4 Save Models** cell runs successfully so the Streamlit app can load everything.

---

## Running the interactive app

1. From the project root with the same virtual environment activated:

   ```bash
   streamlit run app.py
   ```

2. Open the URL shown in the terminal (e.g. `http://localhost:8501`).

3. Use the tabs:
   - **📊 Data Overview** — KPI summary cards (total units sold, avg price, avg marketing spend, top market), dataset snapshot, descriptive statistics, bar charts of units sold by country, model, engine type, and segment, and an annual sales trend chart.
   - **📈 Model Curves** — Four sections matching the notebook: resampling MSE vs degree (images), forward selection AIC/BIC/Adj R²/Mallows' Cp with optimal-predictor markers, Ridge/Lasso CV MSE vs log₁₀(λ) with optimal-λ markers and metric cards, bias-variance tradeoff decomposition, and PCR vs PLS MSE vs components.
   - **🔮 Predictions** — Three sections: (A) pick any dataset row by index or at random and compare Ridge, Lasso, and OLS predictions against the actual value with delta badges; (B) build a fully custom observation using dropdowns and sliders and run all three models live; (C) example results showcase with pre-selected representative rows displayed in a table and grouped bar chart.
   - **🔍 Data Explorer** — Filter by year, month, country, model, segment, engine type, and price range; view the filtered table, units-sold time series, average units sold by country, and summary statistics for the filtered subset.

---

## Models covered

| Model / method | Description |
|---|---|
| **Polynomial regression (LinearRegression)** | MSE compared across degrees 1–9 using 5-Fold CV (mean of 15 splits), 10-Fold CV, LOOCV, and Bootstrap. Feature: `marketing_spend_usd`. |
| **Forward stepwise OLS** | Greedy selection by AIC; the model at the **global min-AIC** step is saved. AIC, BIC, Adj R², and Mallows' Cp are plotted vs number of predictors. |
| **Ridge regression** | L2 regularization; optimal λ chosen by `RidgeCV` over a log-spaced grid of 100 values (5-fold CV). |
| **Lasso regression** | L1 regularization; optimal λ chosen by `LassoCV` (10-fold CV). |
| **PCR (Principal Component Regression)** | PCA on scaled features + LinearRegression; 10-fold CV MSE vs number of components (1 to min(p, 20)). |
| **PLS (Partial Least Squares)** | `PLSRegression`; 10-fold CV MSE vs number of components. |

Target variable: **units_sold**. Categorical features are one-hot encoded (string columns only, `drop_first=True` equivalent via `reindex`) and all features are standardized (`StandardScaler`) for Ridge, Lasso, and OLS.

---

## Best results

- **Ridge / Lasso:** Optimal λ is chosen by cross-validation and marked with a vertical dashed line on the CV MSE vs λ plot in the **Model Curves** tab. The exact values are also shown as metric cards below the chart and in the sidebar.
- **Forward stepwise OLS:** The step with the **globally lowest AIC** is selected; its OLS model is saved to `models/ols_best_model.pkl` and used for predictions in the app. The selected feature count is shown in the sidebar.
- **PCR vs PLS:** The number of components that minimises 10-fold CV MSE is highlighted on the plot. Both methods are compared side-by-side.

Exact values (optimal λ, minimum MSE, best number of components/predictors) appear in the notebook outputs and in the Streamlit app once models are loaded.

---

## Requirements

- Python 3.9+
- See `requirements.txt` (pandas, numpy, scikit-learn, statsmodels, matplotlib, streamlit, notebook).
