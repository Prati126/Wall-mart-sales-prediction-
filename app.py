import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
from datetime import datetime
from pathlib import Path
from typing import Any

st.set_page_config(page_title="Walmart Store + Product Forecasts", page_icon="🛒", layout="wide")

# Comprehensive Custom CSS to implement Material You / 2026 aesthetics
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

/* Main Body & Layout */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #0B0F19 !important;
    color: #F8FAFC !important;
}

[data-testid="stHeader"] {
    background-color: rgba(11, 15, 25, 0.8) !important;
    backdrop-filter: blur(12px) !important;
}

header {
    display: none !important;
}
#MainMenu {
    display: none !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0B0F19;
}
::-webkit-scrollbar-thumb {
    background: #1E293B;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3B82F6;
}

/* Container Paddings */
div.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1300px !important;
}

/* Custom Card System */
.m3-card {
    background: linear-gradient(135deg, rgba(23, 28, 40, 0.7), rgba(15, 23, 42, 0.4));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 20px;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease, box-shadow 0.3s ease;
    backdrop-filter: blur(12px);
}
.m3-card:hover {
    transform: translateY(-4px);
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4), 0 0 15px rgba(59, 130, 246, 0.1);
}

.m3-card-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}

.m3-card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.2;
}

.m3-card-footer {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 12px;
    font-size: 0.875rem;
    font-weight: 500;
}

/* Delta Indicators */
.delta-positive {
    color: #10B981;
    background: rgba(16, 185, 129, 0.12);
    padding: 2px 8px;
    border-radius: 100px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.delta-negative {
    color: #EF4444;
    background: rgba(239, 68, 68, 0.12);
    padding: 2px 8px;
    border-radius: 100px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.delta-neutral {
    color: #94A3B8;
    background: rgba(148, 163, 184, 0.12);
    padding: 2px 8px;
    border-radius: 100px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

/* Icon Wrap */
.m3-icon-box {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(59, 130, 246, 0.12);
    color: #3B82F6;
    margin-bottom: 16px;
}
.m3-icon-box.purple {
    background: rgba(139, 92, 246, 0.12);
    color: #8B5CF6;
}
.m3-icon-box.emerald {
    background: rgba(16, 185, 129, 0.12);
    color: #10B981;
}
.m3-icon-box.coral {
    background: rgba(239, 68, 68, 0.12);
    color: #EF4444;
}

/* Streamlit Tabs Customization */
div[data-testid="stTabs"] {
    background-color: transparent !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-bottom: 8px !important;
}
div[data-testid="stTabs"] button {
    border-radius: 14px !important;
    padding: 10px 20px !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    background-color: rgba(23, 28, 40, 0.4) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stTabs"] button:hover {
    color: #F8FAFC !important;
    background-color: rgba(59, 130, 246, 0.08) !important;
    border-color: rgba(59, 130, 246, 0.2) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #FFFFFF !important;
    background-color: #3B82F6 !important;
    border-color: #3B82F6 !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
}

/* Selectbox Customization */
div[data-testid="stSelectbox"] > div {
    background-color: #171C28 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    color: #F8FAFC !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: transparent !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] input {
    caret-color: transparent !important;
    pointer-events: none !important;
    user-select: none !important;
}

/* Slider Customization */
div[data-testid="stSlider"] {
    padding-top: 10px !important;
}

/* Button Overrides */
.stButton>button {
    background-color: #3B82F6 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 8px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: auto !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(59, 130, 246, 0.3) !important;
    background-color: #2563EB !important;
}
.stButton>button:active {
    transform: translateY(0px) !important;
}

/* Dataframe skinning */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    background-color: #171C28 !important;
}

/* Alert styling */
.m3-alert {
    padding: 16px 20px;
    border-radius: 20px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    line-height: 1.5;
}
.m3-alert.info {
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.15);
    color: #93C5FD;
}
.m3-alert.warning {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.15);
    color: #FDE047;
}
.m3-alert .material-symbols-rounded {
    font-size: 28px;
    flex-shrink: 0;
}

/* Custom Dashboard Header */
.dashboard-header {
    background: linear-gradient(90deg, #171C28, #0B0F19);
    border-radius: 24px;
    padding: 24px 32px;
    margin-bottom: 28px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.dashboard-header-left {
    display: flex;
    align-items: center;
    gap: 20px;
}
.dashboard-logo {
    width: 52px;
    height: 52px;
    border-radius: 16px;
    background: #3B82F6;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 26px;
    font-weight: 800;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3);
}
.dashboard-titles h1 {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.02em;
}
.dashboard-titles p {
    font-size: 0.9rem !important;
    color: #94A3B8 !important;
    margin: 4px 0 0 0 !important;
}

.m3-subheading {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

script_path = Path(__file__).resolve()
data_path = script_path.parent / "data.csv"
models_dir = script_path.parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Branch"] = df["Branch"].astype(str).str.strip()
    df["City"] = df["City"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
    df["unit_price"] = pd.to_numeric(
        df["unit_price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0.0)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["sales"] = df["unit_price"] * df["quantity"]
    return df.dropna(subset=["date", "Branch", "category", "sales"])


@st.cache_data
def load_sales_data() -> pd.DataFrame:
    if not data_path.exists():
        st.error("data.csv not found in Walmart directory.")
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    return clean_sales_data(df)


@st.cache_resource(max_entries=10)
def load_product_models() -> list[dict[str, Any]]:
    model_files = sorted(models_dir.glob("product_*_model.joblib"))
    bundles = []

    if not model_files:
        st.warning("No product-level model files found in the Walmart models folder.")

    for model_file in model_files:
        try:
            bundle = joblib.load(model_file)
            category = bundle.get("category") if isinstance(bundle, dict) else None
            if not category:
                category = (
                    model_file.name
                    .replace("product_", "")
                    .replace("_model.joblib", "")
                    .replace("_", " ")
                )
            model = bundle.get("model") if isinstance(bundle, dict) else bundle
            feature_cols = bundle.get("feature_cols") if isinstance(bundle, dict) else ["lag_1", "lag_2", "lag_4", "ma_4", "weekofyear", "year"]
            bundles.append({
                "category": category,
                "model": model,
                "feature_cols": feature_cols,
                "path": str(model_file),
            })
        except Exception as e:
            st.warning(f"Unable to load product model {model_file.name}: {e}")
    return bundles


@st.cache_resource(max_entries=10)
def load_store_models() -> dict[str, dict[str, Any]]:
    model_files = sorted(models_dir.glob("store_*_model.joblib"))
    store_models = {}

    if not model_files:
        st.warning("No store-level model files found in the Walmart models folder.")

    for model_file in model_files:
        try:
            bundle = joblib.load(model_file)
            branch = model_file.name.split("_")[1]
            model = bundle.get("model") if isinstance(bundle, dict) else bundle
            feature_cols = bundle.get("feature_cols") if isinstance(bundle, dict) else ["lag_1", "lag_2", "lag_4", "ma_4", "weekofyear", "year"]
            store_models[branch] = {
                "model": model,
                "feature_cols": feature_cols,
                "path": str(model_file),
            }
        except Exception as e:
            st.warning(f"Unable to load store model {model_file.name}: {e}")
    return store_models


@st.cache_data
def load_training_summary() -> pd.DataFrame:
    summary_path = script_path.parent / "training_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame()


@st.cache_data
def load_training_summary_storewise() -> pd.DataFrame:
    summary_path = script_path.parent / "training_summary_storewise.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame()


def is_holiday_week() -> bool:
    today = datetime.now()
    month = today.month
    day = today.day
    holidays = [
        (1, 1),
        (7, 4),
        (11, 27),
        (12, 25),
        (12, 26),
    ]
    for holiday_month, holiday_day in holidays:
        if month == holiday_month and abs(day - holiday_day) <= 3:
            return True
    if month == 11 and 22 <= day <= 28:
        return True
    return False


def get_weekly_sales(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.set_index("date")
        .resample("W")
        .agg(
            weekly_sales=("sales", "sum"),
            weekly_quantity=("quantity", "sum"),
            avg_unit_price=("unit_price", "mean"),
            avg_rating=("rating", "mean"),
            avg_profit_margin=("profit_margin", "mean"),
            weekly_transactions=("invoice_id", "count"),
        )
    )
    weekly["week_end"] = weekly.index
    weekly["weekofyear"] = weekly.index.isocalendar().week.astype(int)
    weekly["month"] = weekly.index.month
    weekly["quarter"] = weekly.index.quarter
    weekly["year"] = weekly.index.year
    return weekly


def build_weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    features = weekly.copy()
    features["lag_1"] = features["weekly_sales"].shift(1)
    features["lag_2"] = features["weekly_sales"].shift(2)
    features["lag_4"] = features["weekly_sales"].shift(4)
    features["ma_4"] = features["weekly_sales"].shift(1).rolling(4).mean()
    return features


def simple_forecast_fallback(weekly: pd.DataFrame) -> float | None:
    if weekly.empty or "weekly_sales" not in weekly:
        return None
    last_sales = weekly["weekly_sales"].dropna()
    if last_sales.empty:
        return None
    if len(last_sales) >= 3:
        return float(last_sales.tail(3).mean())
    return float(last_sales.iloc[-1])


def predict_next_week(model, feature_cols: list[str], weekly: pd.DataFrame) -> float:
    x_last = weekly[feature_cols].iloc[[-1]]
    return max(0.0, float(model.predict(x_last)[0]))


def predict_multiple_weeks(model, feature_cols: list[str], weekly: pd.DataFrame, steps: int = 3) -> list[float]:
    df_temp = weekly.copy()
    predictions = []
    
    for step in range(steps):
        if step > 0:
            last_idx = df_temp.index[-1]
            next_idx = last_idx + pd.Timedelta(days=7)
            
            new_row = df_temp.iloc[[-1]].copy()
            new_row.index = [next_idx]
            
            new_row["weekly_sales"] = predictions[-1]
            df_temp = pd.concat([df_temp, new_row])
            
            df_temp["lag_1"] = df_temp["weekly_sales"].shift(1)
            df_temp["lag_2"] = df_temp["weekly_sales"].shift(2)
            df_temp["lag_4"] = df_temp["weekly_sales"].shift(4)
            df_temp["ma_4"] = df_temp["weekly_sales"].shift(1).rolling(4).mean()
            
            df_temp["weekofyear"] = df_temp.index.isocalendar().week.astype(int)
            df_temp["month"] = df_temp.index.month
            df_temp["quarter"] = df_temp.index.quarter
            df_temp["year"] = df_temp.index.year
            
        x_last = df_temp[feature_cols].iloc[[-1]]
        pred = max(0.0, float(model.predict(x_last)[0]))
        predictions.append(pred)
        
    return predictions


def fallback_multiple_weeks(weekly: pd.DataFrame, steps: int = 3) -> list[float]:
    val = simple_forecast_fallback(weekly)
    if val is None:
        return [0.0] * steps
    return [val] * steps


def format_currency(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"${value:,.2f}"


def branch_to_store_name(branch: str, city: str) -> str:
    try:
        store_number = int(branch.replace("M", ""))
        return f"Store {store_number} - {city}"
    except Exception:
        return f"{branch} - {city}"


def format_r2(val: Any) -> str:
    try:
        f_val = float(val)
        if np.isnan(f_val):
            return "-"
        return f"{f_val:.4f}"
    except Exception:
        return "-"


def render_m3_card(icon: str, title: str, value: str, footer_text: str, delta_type: str = "neutral", color_class: str = ""):
    icon_color_style = f" {color_class}" if color_class else ""
    delta_class = f"delta-{delta_type}"
    
    delta_icon = ""
    if delta_type == "positive":
        delta_icon = '<span class="material-symbols-rounded" style="font-size: 14px;">trending_up</span>'
    elif delta_type == "negative":
        delta_icon = '<span class="material-symbols-rounded" style="font-size: 14px;">trending_down</span>'
    elif delta_type == "neutral":
        delta_icon = '<span class="material-symbols-rounded" style="font-size: 14px;">trending_flat</span>'

    st.markdown(f"""
    <div class="m3-card">
        <div class="m3-icon-box{icon_color_style}">
            <span class="material-symbols-rounded">{icon}</span>
        </div>
        <div class="m3-card-title">{title}</div>
        <div class="m3-card-value">{value}</div>
        <div class="m3-card-footer">
            <span class="{delta_class}">{delta_icon} {footer_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- Load data and models ---
df = load_sales_data()
product_models = load_product_models()
product_model_map = {bundle["category"]: bundle for bundle in product_models}
store_models = load_store_models()

if df.empty:
    st.error("Unable to load the Walmart sales data. Please check data.csv.")
    st.stop()

# --- Extract Store Metadata ---
store_meta = (
    df[["Branch", "City"]]
    .drop_duplicates()
    .assign(store_number=lambda d: d["Branch"].str.replace("M", "").astype(int))
    .sort_values("store_number")
    .reset_index(drop=True)
)
store_meta["display"] = store_meta.apply(lambda row: branch_to_store_name(row["Branch"], row["City"]), axis=1)

# --- Top Navigation Workspace Control ---
col_logo, col_select, col_space = st.columns([1.5, 3.5, 5])
with col_logo:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; height: 100%; margin-top:14px;">
        <div class="dashboard-logo" style="width: 42px; height: 42px; font-size:22px; border-radius:12px;">🛒</div>
        <div style="font-weight: 800; font-size: 1.1rem; color: #FFFFFF; letter-spacing: -0.01em;">Walmart</div>
    </div>
    """, unsafe_allow_html=True)
with col_select:
    selected_store_display = st.selectbox(
        "Workspace Store Selection",
        options=store_meta["display"].tolist(),
        index=0,
        label_visibility="collapsed"
    )
selected_store_branch = store_meta.loc[store_meta["display"] == selected_store_display, "Branch"].iloc[0]
selected_store_city = store_meta.loc[store_meta["display"] == selected_store_display, "City"].iloc[0]
selected_store_number = int(selected_store_branch.replace("M", ""))

# --- Setup Store specific data ---
store_df = df[df["Branch"] == selected_store_branch].copy()
store_sales_total = store_df["sales"].sum()
product_groups = (
    store_df.groupby("category", observed=False)["sales"].sum().reset_index().rename(columns={"sales": "total_sales"})
)
product_groups["model_available"] = product_groups["category"].isin(product_model_map)
product_groups["total_sales_formatted"] = product_groups["total_sales"].map(format_currency)

store_model_bundle = store_models.get(selected_store_branch)
store_weekly = get_weekly_sales(store_df) if not store_df.empty else pd.DataFrame()
store_forecasts = []
store_forecast_note = ""
store_dates = []

# --- Compute Forecasts ---
if store_model_bundle is not None and not store_weekly.empty:
    feature_weekly = build_weekly_features(store_weekly)
    available_features = feature_weekly.dropna()
    if not available_features.empty:
        try:
            store_forecasts = predict_multiple_weeks(store_model_bundle["model"], store_model_bundle["feature_cols"], available_features, steps=3)
            store_forecast_note = "Store forecast based on trained model."
        except Exception as e:
            store_forecasts = fallback_multiple_weeks(store_weekly, steps=3)
            store_forecast_note = f"Model forecasting error, using baseline history: {e}"
    else:
        store_forecasts = fallback_multiple_weeks(store_weekly, steps=3)
        store_forecast_note = "Insufficient history for model features. Defaulted to recent historical average."
else:
    store_forecasts = fallback_multiple_weeks(store_weekly, steps=3) if not store_weekly.empty else []
    store_forecast_note = "No store model file found. Defaulted to recent historical average." if store_forecasts else "No forecast available."

store_forecast = store_forecasts[0] if len(store_forecasts) > 0 else None
store_forecast_delta = 0.0
if store_forecast is not None and not store_weekly.empty:
    last_store_week = store_weekly["weekly_sales"].iloc[-1]
    store_forecast_delta = ((store_forecast - last_store_week) / last_store_week * 100) if last_store_week != 0 else 0

if not store_weekly.empty:
    last_week_end = store_weekly.index[-1]
    store_dates = [
        (last_week_end + pd.Timedelta(days=7)).date().isoformat(),
        (last_week_end + pd.Timedelta(days=14)).date().isoformat(),
        (last_week_end + pd.Timedelta(days=21)).date().isoformat()
    ]
else:
    store_dates = ["Week 1", "Week 2", "Week 3"]

# Initialize session state for showing extra weeks
if "show_extra_weeks" not in st.session_state:
    st.session_state.show_extra_weeks = False

# --- Prepare Category Forecast Details ---
product_rows = []
for _, row in product_groups.sort_values("total_sales", ascending=False).iterrows():
    category = row["category"]
    total_sales = float(row["total_sales"])
    bundle = product_model_map.get(category)
    forecasts = []
    previous_week_date = "-"
    previous_week_sales = None
    current_week_date = "-"
    current_week_sales = None
    next_week_date = "-"
    next_week_plus_1_date = "-"
    next_week_plus_2_date = "-"

    category_df = store_df[store_df["category"] == category].copy()
    category_weekly = get_weekly_sales(category_df)
    
    if len(category_weekly) >= 1:
        current_week_sales = float(category_weekly["weekly_sales"].iloc[-1])
        current_week_date = category_weekly.index[-1].date().isoformat()
        next_week_date = (category_weekly.index[-1] + pd.Timedelta(days=7)).date().isoformat()
        next_week_plus_1_date = (category_weekly.index[-1] + pd.Timedelta(days=14)).date().isoformat()
        next_week_plus_2_date = (category_weekly.index[-1] + pd.Timedelta(days=21)).date().isoformat()
        if len(category_weekly) >= 2:
            previous_week_sales = float(category_weekly["weekly_sales"].iloc[-2])
            previous_week_date = category_weekly.index[-2].date().isoformat()
        else:
            previous_week_date = (category_weekly.index[-1] - pd.Timedelta(days=7)).date().isoformat()
            previous_week_sales = current_week_sales
    elif not category_df.empty:
        last_date = category_df["date"].max()
        if pd.notna(last_date):
            current_week_date = last_date.date().isoformat()
            current_week_sales = float(category_df["sales"].sum())
            next_week_date = (last_date + pd.Timedelta(days=7)).date().isoformat()
            next_week_plus_1_date = (last_date + pd.Timedelta(days=14)).date().isoformat()
            next_week_plus_2_date = (last_date + pd.Timedelta(days=21)).date().isoformat()
            previous_week_date = (last_date - pd.Timedelta(days=7)).date().isoformat()
            previous_week_sales = current_week_sales

    feature_weekly = build_weekly_features(category_weekly)
    available_features = feature_weekly.dropna()
    if bundle is not None and not available_features.empty:
        try:
            forecasts = predict_multiple_weeks(bundle["model"], bundle["feature_cols"], available_features, steps=3)
        except Exception:
            forecasts = fallback_multiple_weeks(category_weekly, steps=3)
    else:
        forecasts = fallback_multiple_weeks(category_weekly, steps=3)

    forecast = forecasts[0] if len(forecasts) > 0 else None
    forecast_plus_1 = forecasts[1] if len(forecasts) > 1 else None
    forecast_plus_2 = forecasts[2] if len(forecasts) > 2 else None

    current_week_delta = None
    if current_week_sales is not None and previous_week_sales is not None:
        if previous_week_sales != 0:
            current_week_delta = ((current_week_sales - previous_week_sales) / previous_week_sales * 100)
        else:
            current_week_delta = 0.0

    forecast_delta = None
    if forecast is not None and current_week_sales is not None:
        if current_week_sales != 0:
            forecast_delta = ((forecast - current_week_sales) / current_week_sales * 100)
        else:
            forecast_delta = 0.0

    forecast_plus_1_delta = None
    if forecast_plus_1 is not None and forecast is not None:
        if forecast != 0:
            forecast_plus_1_delta = ((forecast_plus_1 - forecast) / forecast * 100)
        else:
            forecast_plus_1_delta = 0.0

    forecast_plus_2_delta = None
    if forecast_plus_2 is not None and forecast_plus_1 is not None:
        if forecast_plus_1 != 0:
            forecast_plus_2_delta = ((forecast_plus_2 - forecast_plus_1) / forecast_plus_1 * 100)
        else:
            forecast_plus_2_delta = 0.0

    product_rows.append({
        "Category": category,
        "Total Sales": total_sales,
        "Total Sales Display": format_currency(total_sales),
        "Previous Week Date": previous_week_date,
        "Previous Week Sales": format_currency(previous_week_sales) if previous_week_sales is not None else "-",
        "Current Week Date": current_week_date,
        "Current Week Sales": format_currency(current_week_sales) if current_week_sales is not None else "-",
        "Current Week % Change": f"{current_week_delta:.2f}%" if current_week_delta is not None else "-",
        "Next 1st Week Date": next_week_date,
        "Next 1st Week Forecast": format_currency(forecast) if forecast is not None else "-",
        "Next 1st Week % Change": f"{forecast_delta:.2f}%" if forecast_delta is not None else "-",
        "Next 2nd Week Date": next_week_plus_1_date,
        "Next 2nd Week Forecast": format_currency(forecast_plus_1) if forecast_plus_1 is not None else "-",
        "Next 2nd Week % Change": f"{forecast_plus_1_delta:.2f}%" if forecast_plus_1_delta is not None else "-",
        "Next 3rd Week Date": next_week_plus_2_date,
        "Next 3rd Week Forecast": format_currency(forecast_plus_2) if forecast_plus_2 is not None else "-",
        "Next 3rd Week % Change": f"{forecast_plus_2_delta:.2f}%" if forecast_plus_2_delta is not None else "-",
    })

product_report = pd.DataFrame(product_rows)

# --- Define Page Layout Workspace Tabs ---
tab_overview, tab_forecasts, tab_models = st.tabs([
    "📊 Store Overview", 
    "🔮 Forecasting Terminal", 
    "🧠 Model Registry & Insights"
])

# ==========================================
# TAB 1: OVERVIEW DASHBOARD
# ==========================================
with tab_overview:
    # Evolved Header
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="dashboard-header-left">
            <div class="dashboard-logo">W</div>
            <div class="dashboard-titles">
                <h1>{selected_store_display} Workspace</h1>
                <p>Overview of historical performance, operations metrics, and baseline trends.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Holiday warning banner
    is_holiday = is_holiday_week()
    if is_holiday:
        st.markdown("""
        <div class="m3-alert warning">
            <span class="material-symbols-rounded">campaign</span>
            <div><strong>Active Holiday Period:</strong> Special weights and multipliers are active for this week's forecasts to account for historical spikes in customer shopping patterns.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="m3-alert info">
            <span class="material-symbols-rounded">info</span>
            <div><strong>Standard Baseline Period:</strong> Demands are currently evaluated against normal seasonality profiles. No major anomalies are active.</div>
        </div>
        """, unsafe_allow_html=True)

    # Calculate growth rate
    if len(store_weekly) >= 2:
        last_sales = store_weekly["weekly_sales"].iloc[-1]
        prev_sales = store_weekly["weekly_sales"].iloc[-2]
        growth = ((last_sales - prev_sales) / prev_sales) * 100
        growth_val = f"{growth:+.2f}%"
        growth_type = "positive" if growth >= 0 else "negative"
        growth_footer = "vs. previous week"
    else:
        growth_val = "N/A"
        growth_type = "neutral"
        growth_footer = "Insufficient history"

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        render_m3_card(
            icon="store", 
            title="Store Location", 
            value=f"Store {selected_store_number}", 
            footer_text=selected_store_city, 
            delta_type="neutral"
        )
    with col_m2:
        render_m3_card(
            icon="payments", 
            title="Total Historical Sales", 
            value=format_currency(store_sales_total), 
            footer_text="Total aggregated sales", 
            delta_type="positive",
            color_class="emerald"
        )
    with col_m3:
        num_categories = len(product_groups)
        num_modeled = int(product_groups["model_available"].sum())
        pct_modeled = (num_modeled / num_categories * 100) if num_categories > 0 else 0
        render_m3_card(
            icon="widgets", 
            title="Product Categories", 
            value=f"{num_modeled}/{num_categories}", 
            footer_text=f"{pct_modeled:.0f}% categories modeled", 
            delta_type="neutral" if pct_modeled < 100 else "positive",
            color_class="purple"
        )
    with col_m4:
        render_m3_card(
            icon="monitoring", 
            title="Sales Momentum", 
            value=growth_val, 
            footer_text=growth_footer, 
            delta_type=growth_type,
            color_class="coral" if growth_type == "negative" else "emerald"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Store sales historical summary visualization
    col_hist_chart, col_cat_list = st.columns([7, 3])
    
    with col_hist_chart:
        st.markdown('<div class="m3-subheading"><span class="material-symbols-rounded">timeline</span> 24-Week Sales History</div>', unsafe_allow_html=True)
        if not store_weekly.empty:
            hist_plot_df = store_weekly.tail(24).reset_index()
            hist_plot_df.rename(columns={"date": "Week Ending", "weekly_sales": "Sales"}, inplace=True)
            
            line_chart = alt.Chart(hist_plot_df).mark_area(
                line={'color':'#3B82F6', 'size':3},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='rgba(59, 130, 246, 0.25)', offset=0),
                           alt.GradientStop(color='rgba(59, 130, 246, 0)', offset=1)],
                    x1=1, y1=1, x2=1, y2=0
                ),
                interpolate='monotone'
            ).encode(
                x=alt.X('Week Ending:T', axis=alt.Axis(format='%b %d, %Y', title='Week Ending', grid=False)),
                y=alt.Y('Sales:Q', axis=alt.Axis(title='Weekly Sales ($)', format='$,.0f')),
                tooltip=[alt.Tooltip('Week Ending:T', format='%B %d, %Y'), alt.Tooltip('Sales:Q', format='$,.2f')]
            ).properties(
                height=320
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                labelFont='Plus Jakarta Sans',
                titleFont='Plus Jakarta Sans',
                labelColor='#94A3B8',
                titleColor='#94A3B8',
                gridColor='rgba(255, 255, 255, 0.05)'
            )
            st.altair_chart(line_chart, use_container_width=True)
        else:
            st.info("No historical trends available.")

    with col_cat_list:
        st.markdown('<div class="m3-subheading"><span class="material-symbols-rounded">pie_chart</span> Categories in Store</div>', unsafe_allow_html=True)
        st.dataframe(
            product_groups[["category", "total_sales_formatted"]]
            .rename(columns={"category": "Category", "total_sales_formatted": "Total Sales"}),
            use_container_width=True,
            height=320
        )


# ==========================================
# TAB 2: FORECASTING TERMINAL
# ==========================================
with tab_forecasts:
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="dashboard-header-left">
            <div class="dashboard-logo" style="background: #8B5CF6; box-shadow: 0 4px 14px rgba(139, 92, 246, 0.3);">F</div>
            <div class="dashboard-titles">
                <h1>{selected_store_display} Forecasts</h1>
                <p>Machine learning-derived demand estimates. Configure forecast horizon and inspect distributions.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Forecast Horizon Selection
    col_horizon, col_hor_space = st.columns([3, 7])
    with col_horizon:
        horizon_weeks = st.select_slider(
            "Select Forecast Horizon Range",
            options=[1, 2, 3],
            value=3 if st.session_state.show_extra_weeks else 1,
            format_func=lambda x: f"Next {x} Week{'s' if x > 1 else ''}"
        )
        st.session_state.show_extra_weeks = (horizon_weeks > 1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Store Forecast Metric Cards
    if store_forecast is not None:
        st.markdown('<div class="m3-subheading"><span class="material-symbols-rounded">online_prediction</span> Store-Level Predictions</div>', unsafe_allow_html=True)
        f_cols = st.columns(horizon_weeks)
        
        for idx in range(horizon_weeks):
            with f_cols[idx]:
                if idx == 0:
                    delta_pct = store_forecast_delta
                    delta_label = "vs. last hist week"
                else:
                    prev_wk = store_forecasts[idx - 1]
                    curr_wk = store_forecasts[idx]
                    delta_pct = ((curr_wk - prev_wk) / prev_wk * 100) if prev_wk != 0 else 0
                    delta_label = f"vs. Week {idx}"
                
                delta_type = "positive" if delta_pct >= 0 else "negative"
                
                render_m3_card(
                    icon="calendar_today",
                    title=f"Week {idx + 1} ({store_dates[idx]})",
                    value=format_currency(store_forecasts[idx]),
                    footer_text=f"{delta_pct:+.2f}% {delta_label}",
                    delta_type=delta_type,
                    color_class="purple" if idx == 0 else ""
                )
        if store_forecast_note:
            st.markdown(f"""
            <div style="font-size: 0.85rem; color:#94A3B8; margin-top:-8px; display:flex; align-items:center; gap:6px;">
                <span class="material-symbols-rounded" style="font-size:16px;">info</span> {store_forecast_note}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No store forecasts generated.")

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0;' />", unsafe_allow_html=True)

    # Combined Line Chart (History + Forecast)
    if not store_weekly.empty and len(store_forecasts) > 0:
        st.markdown('<div class="m3-subheading"><span class="material-symbols-rounded">trending_up</span> Historical Trend vs Forecast Path</div>', unsafe_allow_html=True)
        
        # 1. Historical Data (Last 12 weeks)
        hist_df = store_weekly.tail(12).reset_index()
        hist_df = hist_df[["date", "weekly_sales"]].copy()
        hist_df.rename(columns={"date": "Week Ending", "weekly_sales": "Sales"}, inplace=True)
        hist_df["Type"] = "Historical"
        
        # 2. Forecasted Data (connect smoothly to last hist week)
        forecast_rows = []
        last_hist_date = hist_df["Week Ending"].iloc[-1]
        last_hist_sales = hist_df["Sales"].iloc[-1]
        forecast_rows.append({
            "Week Ending": last_hist_date,
            "Sales": last_hist_sales,
            "Type": "Forecasted"
        })
        
        for idx in range(horizon_weeks):
            forecast_rows.append({
                "Week Ending": pd.to_datetime(store_dates[idx]),
                "Sales": store_forecasts[idx],
                "Type": "Forecasted"
            })
        fore_df = pd.DataFrame(forecast_rows)
        
        # Combine
        combined_df = pd.concat([hist_df, fore_df], ignore_index=True)
        
        # Build Altair line chart
        combo_chart = alt.Chart(combined_df).mark_line(point=True, interpolate='monotone', size=3.5).encode(
            x=alt.X('Week Ending:T', axis=alt.Axis(format='%b %d, %Y', title='Week Ending', grid=False)),
            y=alt.Y('Sales:Q', axis=alt.Axis(title='Weekly Sales ($)', format='$,.0f')),
            color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Forecasted'], range=['#3B82F6', '#8B5CF6']), legend=alt.Legend(title="Data Type")),
            strokeDash=alt.StrokeDash('Type:N', scale=alt.Scale(domain=['Historical', 'Forecasted'], range=[[], [4, 4]]), legend=None),
            tooltip=[
                alt.Tooltip('Week Ending:T', format='%B %d, %Y'), 
                alt.Tooltip('Sales:Q', format='$,.2f'),
                'Type'
            ]
        ).properties(
            height=320
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            labelFont='Plus Jakarta Sans',
            titleFont='Plus Jakarta Sans',
            labelColor='#94A3B8',
            titleColor='#94A3B8',
            gridColor='rgba(255, 255, 255, 0.05)'
        ).configure_legend(
            labelFont='Plus Jakarta Sans',
            titleFont='Plus Jakarta Sans',
            labelColor='#94A3B8',
            titleColor='#94A3B8'
        )
        
        st.altair_chart(combo_chart, use_container_width=True)

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0;' />", unsafe_allow_html=True)

    # Detailed table view
    st.markdown('<div class="m3-subheading"><span class="material-symbols-rounded">table_chart</span> Product Category Predictions</div>', unsafe_allow_html=True)
    
    # Filter columns to display based on horizon slider
    if horizon_weeks == 3:
        display_cols = [
            "Category",
            "Total Sales Display",
            "Previous Week Date",
            "Previous Week Sales",
            "Current Week Date",
            "Current Week Sales",
            "Current Week % Change",
            "Next 1st Week Date",
            "Next 1st Week Forecast",
            "Next 1st Week % Change",
            "Next 2nd Week Date",
            "Next 2nd Week Forecast",
            "Next 2nd Week % Change",
            "Next 3rd Week Date",
            "Next 3rd Week Forecast",
            "Next 3rd Week % Change",
        ]
    elif horizon_weeks == 2:
        display_cols = [
            "Category",
            "Total Sales Display",
            "Previous Week Date",
            "Previous Week Sales",
            "Current Week Date",
            "Current Week Sales",
            "Current Week % Change",
            "Next 1st Week Date",
            "Next 1st Week Forecast",
            "Next 1st Week % Change",
            "Next 2nd Week Date",
            "Next 2nd Week Forecast",
            "Next 2nd Week % Change",
        ]
    else:
        display_cols = [
            "Category",
            "Total Sales Display",
            "Previous Week Date",
            "Previous Week Sales",
            "Current Week Date",
            "Current Week Sales",
            "Current Week % Change",
            "Next 1st Week Date",
            "Next 1st Week Forecast",
            "Next 1st Week % Change",
        ]

    if product_report.empty:
        st.warning("No category predictions available.")
    else:
        st.dataframe(
            product_report[display_cols].rename(columns={"Total Sales Display": "Total Sales"}),
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="m3-subheading"><span class="material-symbols-rounded">donut_large</span> Forecasted Category Distributions</div>', unsafe_allow_html=True)
        
        chart_data = product_report.copy()
        chart_data["Next 1st Week Forecast Numeric"] = chart_data["Next 1st Week Forecast"].replace("-", np.nan).replace(r"[$,]", "", regex=True).astype(float)
        
        tab_names = [f"Week 1 ({store_dates[0]})"]
        if horizon_weeks >= 2:
            tab_names.append(f"Week 2 ({store_dates[1]})")
            chart_data["Next 2nd Week Forecast Numeric"] = chart_data["Next 2nd Week Forecast"].replace("-", np.nan).replace(r"[$,]", "", regex=True).astype(float)
        if horizon_weeks >= 3:
            tab_names.append(f"Week 3 ({store_dates[2]})")
            chart_data["Next 3rd Week Forecast Numeric"] = chart_data["Next 3rd Week Forecast"].replace("-", np.nan).replace(r"[$,]", "", regex=True).astype(float)
            
        forecast_tabs = st.tabs(tab_names)
        
        with forecast_tabs[0]:
            chart_data_clean = chart_data.dropna(subset=["Next 1st Week Forecast Numeric"])
            if not chart_data_clean.empty:
                fig1 = alt.Chart(chart_data_clean).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta(field="Next 1st Week Forecast Numeric", type="quantitative"),
                    color=alt.Color(field="Category", type="nominal", scale=alt.Scale(scheme='category20'), legend=alt.Legend(title="Category")),
                    tooltip=["Category", alt.Tooltip("Next 1st Week Forecast Numeric", title="Sales Forecast", format="$,.2f")]
                ).properties(
                    height=280
                ).configure_view(
                    strokeWidth=0
                )
                st.altair_chart(fig1, use_container_width=True)
                
        if horizon_weeks >= 2:
            with forecast_tabs[1]:
                chart_data_clean2 = chart_data.dropna(subset=["Next 2nd Week Forecast Numeric"])
                if not chart_data_clean2.empty:
                    fig2 = alt.Chart(chart_data_clean2).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta(field="Next 2nd Week Forecast Numeric", type="quantitative"),
                        color=alt.Color(field="Category", type="nominal", scale=alt.Scale(scheme='category20'), legend=alt.Legend(title="Category")),
                        tooltip=["Category", alt.Tooltip("Next 2nd Week Forecast Numeric", title="Sales Forecast", format="$,.2f")]
                    ).properties(
                        height=280
                    ).configure_view(
                        strokeWidth=0
                    )
                    st.altair_chart(fig2, use_container_width=True)
                    
        if horizon_weeks >= 3:
            with forecast_tabs[2]:
                chart_data_clean3 = chart_data.dropna(subset=["Next 3rd Week Forecast Numeric"])
                if not chart_data_clean3.empty:
                    fig3 = alt.Chart(chart_data_clean3).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta(field="Next 3rd Week Forecast Numeric", type="quantitative"),
                        color=alt.Color(field="Category", type="nominal", scale=alt.Scale(scheme='category20'), legend=alt.Legend(title="Category")),
                        tooltip=["Category", alt.Tooltip("Next 3rd Week Forecast Numeric", title="Sales Forecast", format="$,.2f")]
                    ).properties(
                        height=280
                    ).configure_view(
                        strokeWidth=0
                    )
                    st.altair_chart(fig3, use_container_width=True)


# ==========================================
# TAB 3: MODEL REGISTRY & INSIGHTS
# ==========================================
with tab_models:
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="dashboard-header-left">
            <div class="dashboard-logo" style="background: #10B981; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);">M</div>
            <div class="dashboard-titles">
                <h1>Model Registry & Performance</h1>
                <p>Comprehensive register of trained predictive architectures, fit validation, and safety state.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Selected Store's Model Status Card
    st.subheader(f"🧠 Selected Store Model Info: {selected_store_display}")
    
    storewise_df = load_training_summary_storewise()
    store_model_info = None
    if not storewise_df.empty:
        store_match = storewise_df[storewise_df["Category"] == selected_store_branch]
        if not store_match.empty:
            store_model_info = store_match.iloc[0]
            
    if store_model_info is not None:
        best_model_name = store_model_info.get("Best_Model", "N/A")
        r2_score = store_model_info.get("R2", np.nan)
        adj_r2_score = store_model_info.get("Adjusted_R2", np.nan)
        
        # Determine status
        try:
            r2_val = float(r2_score)
            if np.isnan(r2_val):
                status_text = "No Metrics Available"
                status_badge = "neutral"
            elif r2_val > 0.7:
                status_text = "Excellent Fit"
                status_badge = "positive"
            elif r2_val >= 0.3:
                status_text = "Moderate Fit"
                status_badge = "positive"
            elif r2_val >= 0.0:
                status_text = "Low Fit (Active)"
                status_badge = "neutral"
            else:
                status_text = "Underperforming (Fallback Active)"
                status_badge = "negative"
        except Exception:
            status_text = "Unknown"
            status_badge = "neutral"
            
        col_sm1, col_sm2, col_sm3 = st.columns(3)
        with col_sm1:
            render_m3_card(
                icon="psychology",
                title="Best Trained Model Type",
                value=best_model_name,
                footer_text="Determined by cross-validation",
                delta_type="neutral"
            )
        with col_sm2:
            r2_str = format_r2(r2_score)
            render_m3_card(
                icon="speed",
                title="Model R² Score",
                value=r2_str,
                footer_text="Coefficient of determination",
                delta_type="positive" if (not np.isnan(r2_val) and r2_val >= 0) else "negative",
                color_class="emerald" if (not np.isnan(r2_val) and r2_val >= 0) else "coral"
            )
        with col_sm3:
            render_m3_card(
                icon="fact_check",
                title="Model Quality Status",
                value=status_text,
                footer_text="Safety constraints active",
                delta_type=status_badge,
                color_class="purple" if status_badge == "positive" else ("coral" if status_badge == "negative" else "")
            )
            
        if not np.isnan(r2_val) and r2_val < 0:
            st.markdown("""
            <div class="m3-alert warning" style="margin-top: 10px;">
                <span class="material-symbols-rounded">gavel</span>
                <div><strong>Automatic Fallback System Engaged:</strong> The store-level machine learning model scored below the acceptable threshold (R² < 0) on evaluation datasets due to high variance or localized trend shifts. A robust 3-week average historical sales fallback predictor is currently managing forecast pipelines.</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No store-specific model metrics loaded for this store.")
        
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0;' />", unsafe_allow_html=True)
    
    # Category Model Performance Summaries
    st.subheader("📚 Category-Level Global Models")
    cat_summary_df = load_training_summary()
    if not cat_summary_df.empty:
        formatted_cat_df = cat_summary_df.copy()
        formatted_cat_df["R2"] = formatted_cat_df["R2"].map(format_r2)
        formatted_cat_df["Adjusted_R2"] = formatted_cat_df["Adjusted_R2"].map(format_r2)
        formatted_cat_df = formatted_cat_df.rename(columns={
            "Category": "Product Category",
            "Best_Model": "Best Model Architecture",
            "R2": "R² Score",
            "Adjusted_R2": "Adjusted R²",
            "Target_90pct": "Met 90% Target"
        })
        if "Branch" in formatted_cat_df.columns:
            formatted_cat_df = formatted_cat_df.drop(columns=["Branch"])
            
        st.dataframe(formatted_cat_df, use_container_width=True)
    else:
        st.warning("Category model training summary file not found.")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Storewide summary overview (all stores)
    st.subheader("🏬 Store-Level Model Benchmarks (All Branches)")
    if not storewise_df.empty:
        formatted_storewise_df = storewise_df.copy()
        formatted_storewise_df["R2"] = formatted_storewise_df["R2"].map(format_r2)
        formatted_storewise_df["Adjusted_R2"] = formatted_storewise_df["Adjusted_R2"].map(format_r2)
        
        formatted_storewise_df = formatted_storewise_df.rename(columns={
            "Category": "Branch Code",
            "Best_Model": "Trained Model",
            "R2": "R² Score",
            "Adjusted_R2": "Adjusted R²"
        })
        st.dataframe(formatted_storewise_df, use_container_width=True)
    else:
        st.warning("Storewise training summary file not found.")
