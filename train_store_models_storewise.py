from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from walmart_utils import DATA_FILE, MODELS_DIR, clean_sales_data, ensure_models_dir, load_sales_data, safe_filename

warnings.filterwarnings("ignore")

ensure_models_dir()

df = load_sales_data(DATA_FILE)

print("Data loaded successfully")
print("Total rows:", len(df))
print("Stores found:", df["Branch"].nunique())

# =====================================================
# 3. TRAIN MODELS PER STORE
# =====================================================
results_summary = []

stores = sorted(df["Branch"].unique())

for idx, store in enumerate(stores, 1):
    print("\n======================================")
    print(f"({idx}/{len(stores)}) Training model for Category: {store}")
    print("======================================")

    store_df = df[df["Branch"] == store].copy()

    weekly = (
        store_df
        .set_index("date")
        .resample("W")["sales"]
        .sum()
        .to_frame("weekly_sales")
    )

    if len(weekly) < 12:
        print("Not enough data — skipping")
        continue

    weekly["lag_1"] = weekly["weekly_sales"].shift(1)
    weekly["lag_2"] = weekly["weekly_sales"].shift(2)
    weekly["lag_4"] = weekly["weekly_sales"].shift(4)
    weekly["ma_4"] = weekly["weekly_sales"].shift(1).rolling(4).mean()

    weekly["weekofyear"] = weekly.index.isocalendar().week.astype(int)
    weekly["month"] = weekly.index.month
    weekly["quarter"] = weekly.index.quarter
    weekly["year"] = weekly.index.year

    weekly = weekly.dropna()

    feature_cols = [
        "lag_1", "lag_2", "lag_4",
        "ma_4", "weekofyear", "month",
        "quarter", "year"
    ]

    X = weekly[feature_cols]
    y = weekly["weekly_sales"]

    split_idx = int(len(weekly) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "Linear Reg": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=42))
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=0.1, random_state=42, max_iter=10000))
        ]),
        "Elastic Net": Pipeline([
            ("scaler", StandardScaler()),
            ("elasticnet", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000))
        ]),
        "Polynomial": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression())
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boost": GradientBoostingRegressor(
            random_state=42
        )
    }

    metrics = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = np.mean(np.abs((y_test - preds) / y_test.replace(0, np.nan))) * 100
        r2 = r2_score(y_test, preds)
        n = len(y_test)
        p = X_test.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan

        metrics[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2,
            "Adjusted_R2": adjusted_r2
        }
        trained_models[name] = model

        print(
            f"{name:18s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | MAPE: {mape:6.2f}%"
            f" | R2: {r2:6.3f} | Adj R2: {adjusted_r2:6.3f}"
        )

    best_model_name = max(metrics, key=lambda m: metrics[m]["R2"])
    best_model = trained_models[best_model_name]

    model_bundle = {
        "model": best_model,
        "feature_cols": feature_cols,
        "store": store
    }

    safe_store = store.replace(" ", "_").replace("&", "and")
    model_path = f"{MODELS_DIR}/store_{safe_store}_model.joblib"
    joblib.dump(model_bundle, model_path)

    print(f"Best model: {best_model_name}")
    print(f"Model saved to: {model_path}")

    results_summary.append({
        "Category": store,
        "Best_Model": best_model_name,
        "R2": metrics[best_model_name]["R2"],
        "Adjusted_R2": metrics[best_model_name]["Adjusted_R2"]
    })

# =====================================================
# 4. SAVE TRAINING SUMMARY
# =====================================================
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv("training_summary_storewise.csv", index=False)

print("\n======================================")
print("TRAINING COMPLETE")
print("Store-wise models saved in /models/")
print("Summary saved to training_summary_storewise.csv")
print("======================================")
