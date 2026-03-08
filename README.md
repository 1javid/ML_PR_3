# BMW Global Sales — ML Assignment 3

## Title

**BMW Global Sales: Regression, Model Selection & Interactive Dashboard**

## Description

This project uses the BMW global sales dataset to predict **units sold** from features such as year, country, model, segment, engine type, price, marketing spend, dealership count, and macroeconomic variables. It covers:

- Resampling methods (K-fold CV, LOOCV, bootstrap) for polynomial regression
- Model selection via forward stepwise OLS (AIC/BIC)
- Shrinkage methods (Ridge and Lasso) with cross-validated tuning
- Dimensionality reduction (PCR and PLS) and comparison of MSE vs number of components

Training and analysis are done in a Jupyter notebook; an interactive Streamlit app provides visualizations and live predictions from the saved models.

---

## Project structure

```
ML_PR_3/
├── README.md
├── requirements.txt
├── bmw_global_sales_dataset.csv   # Input data (target: units_sold)
├── ML_Group_4_Assignment_3.ipynb   # Training, analysis, and saving models
├── app.py                         # Streamlit dashboard
└── models/                        # Created by notebook (pickle artifacts)
    ├── scaler.pkl
    ├── ridge_model.pkl
    ├── lasso_model.pkl
    ├── ols_best_model.pkl
    ├── best_features.pkl
    ├── feature_columns.pkl
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
   # .venv\Scripts\activate          # Windows
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
   - Run polynomial resampling (degrees 1–5), forward selection, Ridge/Lasso CV, and PCA/PLS
   - Create the `models/` directory and save all pickle files (scaler, Ridge, Lasso, OLS, feature lists, plot data)

4. Ensure the last section **“Save models for Streamlit app”** runs successfully so the app can load the models.

---

## Running the interactive app

1. From the project root with the same virtual environment activated:

   ```bash
   streamlit run app.py
   ```

2. Open the URL shown in the terminal (e.g. `http://localhost:8501`).

3. Use the tabs:
   - **Data overview** — Table and bar charts (units by country/model).
   - **Model curves** — Ridge/Lasso MSE vs λ, forward-selection (AIC/BIC/Adj R²), PCR vs PLS (requires `plot_data.pkl`).
   - **Predictions** — Choose a row (random/first), see the record and Ridge/Lasso/OLS predictions.
   - **Data explorer** — Compare by pair (two dimensions) and twist filters (year, country, model, segment) to see updated table and metrics.

---

## Models covered

| Model / method | Description |
|---------------|-------------|
| **Polynomial regression (LinearRegression)** | MSE compared across degrees 1–5 using 5-fold CV, 10-fold CV, LOOCV, and bootstrap. |
| **Forward stepwise OLS** | Stepwise selection by AIC; final OLS model and selected features saved for the app. |
| **Ridge regression** | L2 regularization; optimal α chosen by `RidgeCV` over a log-spaced grid (5-fold CV). |
| **Lasso regression** | L1 regularization; optimal α chosen by `LassoCV` (10-fold CV). |
| **PCR (Principal Component Regression)** | PCA on scaled features + LinearRegression; 10-fold CV MSE vs number of components (1 to max). |
| **PLS (Partial Least Squares)** | `PLSRegression`; 10-fold CV MSE vs number of components. |

Target variable: **units_sold**. Features are one-hot encoded (categorical) and standardized for Ridge/Lasso/OLS.

---

## Best result(s)

- **Ridge / Lasso:** Best regularization strength is chosen by cross-validation (optimal λ printed in the notebook and shown in the app’s “Model curves” tab).
- **Forward selection OLS:** The model with **lowest AIC** over the stepwise sequence is kept; its coefficients and selected features are saved and used for predictions in the app.
- **PCR vs PLS:** The **number of components** that minimizes 10-fold CV MSE is read from the notebook’s PCA/PLS plot (and from `plot_data.pkl` in the app).

Exact values (e.g. optimal α, minimum MSE, best number of components) appear in the notebook outputs and in the Streamlit app after you run the notebook and load the saved models.

---

## Requirements

- Python 3.9+
- See `requirements.txt` (pandas, numpy, scikit-learn, statsmodels, matplotlib, streamlit, notebook).
