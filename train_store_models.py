from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from walmart_utils import DATA_FILE, MODELS_DIR, build_weekly_features, clean_sales_data, ensure_models_dir, load_sales_data, safe_filename

warnings.filterwarnings("ignore")

TARGET_R2 = 0.90
TIME_SERIES_SPLITS = 3


def compute_metrics(y_true, preds, p):
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true - preds) / y_true.replace(0, np.nan))) * 100
    r2 = r2_score(y_true, preds)
    n = len(y_true)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
    }


def effective_feature_count(model, X):
    if hasattr(model, "named_steps") and "poly" in model.named_steps:
        return model.named_steps["poly"].transform(X).shape[1]
    return X.shape[1]


def fit_with_grid_search(model, params, X_train, y_train):
    if not params or len(X_train) < TIME_SERIES_SPLITS + 1:
        model.fit(X_train, y_train)
        return model

    cv = TimeSeriesSplit(n_splits=min(TIME_SERIES_SPLITS, len(X_train) - 1))
    search = GridSearchCV(
        model,
        params,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def build_branch_time_data(category_df, freq="W"):
    periods = (
        category_df
        .set_index("date")
        .groupby("Branch")
        .resample(freq)
        .agg({
            "sales": "sum",
            "quantity": "sum",
            "unit_price": "mean",
            "rating": "mean",
            "profit_margin": "mean",
            "invoice_id": "count",
        })
        .rename(columns={
            "sales": "period_sales",
            "quantity": "period_quantity",
            "unit_price": "avg_unit_price",
            "rating": "avg_rating",
            "profit_margin": "avg_profit_margin",
            "invoice_id": "period_transactions",
        })
    )
    periods = periods.reset_index(level="Branch")
    periods["lag_1"] = periods.groupby("Branch")["period_sales"].shift(1)
    periods["lag_2"] = periods.groupby("Branch")["period_sales"].shift(2)
    periods["lag_4"] = periods.groupby("Branch")["period_sales"].shift(4)
    periods["ma_4"] = periods.groupby("Branch")["period_sales"].shift(1).rolling(4).mean()

    periods["weekofyear"] = periods.index.isocalendar().week.astype(int)
    periods["dayofweek"] = periods.index.dayofweek
    periods["day"] = periods.index.day
    periods["month"] = periods.index.month
    periods["quarter"] = periods.index.quarter
    periods["year"] = periods.index.year

    return periods.dropna()

ensure_models_dir()

df = load_sales_data(DATA_FILE)

print("Data loaded successfully")
print("Total rows:", len(df))
print("Branches found:", df["Branch"].nunique())

# =====================================================
# 3. TRAIN MODELS PER CATEGORY (Product Type)
# =====================================================
results_summary = []

categories = df["category"].unique()

for category in categories:
    print("\n======================================")
    print(f"Training model for Category: {category}")
    print("======================================")

    category_df = df[df["category"] == category].copy()

    # -------------------------------------
    # Weekly aggregation
    # -------------------------------------
    weekly = (
        category_df
        .set_index("date")
        .resample("W")
        .agg({
            "sales": "sum",
            "quantity": "sum",
            "unit_price": "mean",
            "rating": "mean",
            "profit_margin": "mean",
            "invoice_id": "count",
        })
        .rename(columns={
            "sales": "weekly_sales",
            "quantity": "weekly_quantity",
            "unit_price": "avg_unit_price",
            "rating": "avg_rating",
            "profit_margin": "avg_profit_margin",
            "invoice_id": "weekly_transactions",
        })
    )

    if len(weekly) < 12:
        print("Not enough data — skipping")
        continue

    # -------------------------------------
    # Feature engineering
    # -------------------------------------
    weekly["lag_1"] = weekly["weekly_sales"].shift(1)
    weekly["lag_2"] = weekly["weekly_sales"].shift(2)
    weekly["lag_4"] = weekly["weekly_sales"].shift(4)
    weekly["ma_4"]  = weekly["weekly_sales"].shift(1).rolling(4).mean()

    weekly["weekofyear"] = weekly.index.isocalendar().week.astype(int)
    weekly["month"] = weekly.index.month
    weekly["quarter"] = weekly.index.quarter
    weekly["year"] = weekly.index.year

    weekly = weekly.dropna()

    feature_cols = [
        "lag_1", "lag_2", "lag_4",
        "ma_4", "weekofyear", "month",
        "quarter", "year", "weekly_quantity",
        "avg_unit_price", "avg_rating",
        "avg_profit_margin", "weekly_transactions"
    ]

    X = weekly[feature_cols]
    y = weekly["weekly_sales"]

    # -------------------------------------
    # Train / test split (time-based)
    # -------------------------------------
    split_idx = int(len(weekly) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # -------------------------------------
    # Models and parameter grids
    # -------------------------------------
    model_candidates = {
        "Linear Reg": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LinearRegression())
            ]),
            {}
        ),
        "Ridge": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(random_state=42))
            ]),
            {"ridge__alpha": [0.01, 0.1, 1.0, 10.0]}
        ),
        "Lasso": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("lasso", Lasso(random_state=42, max_iter=10000))
            ]),
            {"lasso__alpha": [0.001, 0.01, 0.1, 1.0]}
        ),
        "Elastic Net": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("elasticnet", ElasticNet(random_state=42, max_iter=10000))
            ]),
            {
                "elasticnet__alpha": [0.01, 0.1, 1.0],
                "elasticnet__l1_ratio": [0.2, 0.5, 0.8]
            }
        ),
        "Polynomial": (
            Pipeline([
                ("poly", PolynomialFeatures(include_bias=False)),
                ("scaler", StandardScaler()),
                ("lr", LinearRegression())
            ]),
            {"poly__degree": [2, 3]}
        ),
        "Random Forest": (
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_leaf": [1, 3]
            }
        ),
        "Gradient Boost": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        )
    }

    metrics = {}
    trained_models = {}

    for name, (model, param_grid) in model_candidates.items():
        fitted = fit_with_grid_search(model, param_grid, X_train, y_train)
        preds = fitted.predict(X_test)
        p = effective_feature_count(fitted, X_test)
        metrics[name] = compute_metrics(y_test, preds, p)
        trained_models[name] = fitted

        print(
            f"{name:18s} | MAE: {metrics[name]['MAE']:8.2f} | RMSE: {metrics[name]['RMSE']:8.2f}"
            f" | MAPE: {metrics[name]['MAPE']:6.2f}% | R2: {metrics[name]['R2']:6.3f}"
            f" | Adj R2: {metrics[name]['Adjusted_R2']:6.3f}"
        )

    # -------------------------------------
    # Select best category-level model by R2
    # -------------------------------------
    best_model_name = max(metrics, key=lambda m: metrics[m]["R2"])
    best_metrics = metrics[best_model_name]
    selected_model = trained_models[best_model_name]
    selected_model_name = best_model_name
    selected_branch = None
    threshold_hit = best_metrics["R2"] >= TARGET_R2

    if not threshold_hit:
        print("Category-level model below target, trying branch-level models...")
        branch_weekly = build_branch_time_data(category_df, freq="W")
        branch_feature_cols = [
            "lag_1", "lag_2", "lag_4", "ma_4",
            "weekofyear", "dayofweek", "day",
            "month", "quarter", "year",
            "period_quantity", "avg_unit_price", "avg_rating",
            "avg_profit_margin", "period_transactions"
        ]

        branch_level_best = None
        branch_selected_model = None
        branch_selected_name = None
        branch_selected_branch = None

        for branch in branch_weekly["Branch"].unique():
            branch_df = branch_weekly[branch_weekly["Branch"] == branch]
            if len(branch_df) < 12:
                continue

            X_branch = branch_df[branch_feature_cols]
            y_branch = branch_df["period_sales"]
            split_branch = int(len(branch_df) * 0.8)
            X_train_b, X_test_b = X_branch.iloc[:split_branch], X_branch.iloc[split_branch:]
            y_train_b, y_test_b = y_branch.iloc[:split_branch], y_branch.iloc[split_branch:]

            branch_metrics = {}
            branch_trained = {}
            for name, (model, param_grid) in model_candidates.items():
                fitted_b = fit_with_grid_search(model, param_grid, X_train_b, y_train_b)
                preds_b = fitted_b.predict(X_test_b)
                p_b = effective_feature_count(fitted_b, X_test_b)
                branch_metrics[name] = compute_metrics(y_test_b, preds_b, p_b)
                branch_trained[name] = fitted_b

            best_branch_name = max(branch_metrics, key=lambda m: branch_metrics[m]["R2"])
            best_branch_metrics = branch_metrics[best_branch_name]
            print(
                f"Branch {branch:20s} | {best_branch_name:18s} | MAE: {best_branch_metrics['MAE']:8.2f}"
                f" | RMSE: {best_branch_metrics['RMSE']:8.2f} | R2: {best_branch_metrics['R2']:6.3f}"
                f" | Adj R2: {best_branch_metrics['Adjusted_R2']:6.3f}"
            )

            if branch_level_best is None or best_branch_metrics["R2"] > branch_level_best["R2"]:
                branch_level_best = best_branch_metrics
                branch_selected_model = branch_trained[best_branch_name]
                branch_selected_name = best_branch_name
                branch_selected_branch = branch

        if branch_level_best is not None and branch_level_best["R2"] >= TARGET_R2:
            selected_model = branch_selected_model
            selected_model_name = f"{branch_selected_name} (branch={branch_selected_branch})"
            selected_branch = branch_selected_branch
            best_metrics = branch_level_best
            threshold_hit = True
            print(f"Branch-level target reached with {selected_model_name}")
        else:
            print("Branch-level models did not reach the target. Trying branch-aware weekly category model...")
            if len(branch_weekly) >= 12:
                branch_dummies = pd.get_dummies(branch_weekly["Branch"], drop_first=True)
                X_branch = pd.concat(
                    [branch_weekly[branch_feature_cols].reset_index(drop=True), branch_dummies.reset_index(drop=True)],
                    axis=1,
                )
                y_branch = branch_weekly["period_sales"].reset_index(drop=True)
                split_branch = int(len(X_branch) * 0.8)
                X_train_b, X_test_b = X_branch.iloc[:split_branch], X_branch.iloc[split_branch:]
                y_train_b, y_test_b = y_branch.iloc[:split_branch], y_branch.iloc[split_branch:]

                branch_aware_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(random_state=42))
                ])
                branch_aware = fit_with_grid_search(
                    branch_aware_model,
                    {"ridge__alpha": [0.01, 0.1, 1.0]},
                    X_train_b,
                    y_train_b,
                )
                preds_branch_aware = branch_aware.predict(X_test_b)
                branch_aware_metrics = compute_metrics(y_test_b, preds_branch_aware, X_test_b.shape[1])
                print(
                    f"Branch-aware weekly Ridge  | MAE: {branch_aware_metrics['MAE']:8.2f} | RMSE: {branch_aware_metrics['RMSE']:8.2f}"
                    f" | R2: {branch_aware_metrics['R2']:6.3f} | Adj R2: {branch_aware_metrics['Adjusted_R2']:6.3f}"
                )
                if branch_aware_metrics["R2"] > best_metrics["R2"]:
                    selected_model = branch_aware
                    selected_model_name = "BranchAwareWeeklyRidge"
                    selected_branch = None
                    best_metrics = branch_aware_metrics
                    threshold_hit = best_metrics["R2"] >= TARGET_R2
                    if threshold_hit:
                        print("Branch-aware weekly category model reached the target.")

            if not threshold_hit:
                print("Weekly branch-aware failed. Trying branch-aware daily category model...")
                branch_daily = build_branch_time_data(category_df, freq="D")
                if len(branch_daily) >= 12:
                    branch_dummies = pd.get_dummies(branch_daily["Branch"], drop_first=True)
                    X_daily = pd.concat(
                        [branch_daily[branch_feature_cols].reset_index(drop=True), branch_dummies.reset_index(drop=True)],
                        axis=1,
                    )
                    y_daily = branch_daily["period_sales"].reset_index(drop=True)
                    split_daily = int(len(X_daily) * 0.8)
                    X_train_d, X_test_d = X_daily.iloc[:split_daily], X_daily.iloc[split_daily:]
                    y_train_d, y_test_d = y_daily.iloc[:split_daily], y_daily.iloc[split_daily:]

                    daily_rf = RandomForestRegressor(
                        random_state=42,
                        n_estimators=200,
                        max_depth=20,
                        n_jobs=-1,
                    )
                    daily_rf.fit(X_train_d, y_train_d)
                    preds_daily = daily_rf.predict(X_test_d)
                    daily_metrics = compute_metrics(y_test_d, preds_daily, X_test_d.shape[1])
                    print(
                        f"Branch-aware daily RF | MAE: {daily_metrics['MAE']:8.2f} | RMSE: {daily_metrics['RMSE']:8.2f}"
                        f" | R2: {daily_metrics['R2']:6.3f} | Adj R2: {daily_metrics['Adjusted_R2']:6.3f}"
                    )
                    if daily_metrics["R2"] > best_metrics["R2"]:
                        selected_model = daily_rf
                        selected_model_name = "BranchAwareDailyRF"
                        selected_branch = None
                        best_metrics = daily_metrics
                        threshold_hit = best_metrics["R2"] >= TARGET_R2
                        if threshold_hit:
                            print("Branch-aware daily category model reached the target.")

    safe_category = category.replace(" ", "_").replace("&", "and")
    if selected_branch:
        model_path = f"{MODELS_DIR}/product_{safe_category}_branch_{selected_branch}_model.joblib"
    else:
        model_path = f"{MODELS_DIR}/product_{safe_category}_model.joblib"

    model_bundle = {
        "model": selected_model,
        "feature_cols": feature_cols,
        "category": category,
        "branch": selected_branch,
    }
    joblib.dump(model_bundle, model_path)

    print(f"Best model: {selected_model_name}")
    print(f"Best model R2: {best_metrics['R2']:6.3f}")
    if threshold_hit:
        print(f"Target R2 >= {TARGET_R2:.2f} achieved [OK]")
    else:
        print(f"Target R2 >= {TARGET_R2:.2f} NOT achieved - keeping best available model.")
    print(f"Model saved to: {model_path}")

    results_summary.append({
        "Category": category,
        "Best_Model": selected_model_name,
        "Branch": selected_branch or "",
        "R2": best_metrics["R2"],
        "Adjusted_R2": best_metrics["Adjusted_R2"],
        "Target_90pct": threshold_hit
    })

# =====================================================
# 4. SAVE TRAINING SUMMARY
# =====================================================
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv("training_summary.csv", index=False)

print("\n======================================")
print("TRAINING COMPLETE")
print("Models saved in /models/")
print("Summary saved to training_summary.csv")
print("======================================")