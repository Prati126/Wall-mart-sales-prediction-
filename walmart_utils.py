from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

WALMART_ROOT = Path(__file__).resolve().parent
DATA_FILE = WALMART_ROOT / "data.csv"
MODELS_DIR = WALMART_ROOT / "models"


def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Branch"] = df["Branch"].astype(str).str.strip()
    if "City" in df.columns:
        df["City"] = df["City"].astype(str).str.strip()
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip()

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
    df["unit_price"] = pd.to_numeric(
        df["unit_price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["sales"] = df["unit_price"] * df["quantity"]
    return df.dropna(subset=["date", "sales", "Branch"])


def load_sales_data(data_path: Union[Path, str] = DATA_FILE) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing Walmart sales data file: {data_path}")
    df = pd.read_csv(data_path)
    return clean_sales_data(df)


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
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
    weekly["lag_1"] = weekly["weekly_sales"].shift(1)
    weekly["lag_2"] = weekly["weekly_sales"].shift(2)
    weekly["lag_4"] = weekly["weekly_sales"].shift(4)
    weekly["ma_4"] = weekly["weekly_sales"].shift(1).rolling(4).mean()
    weekly["weekofyear"] = weekly.index.isocalendar().week.astype(int)
    weekly["month"] = weekly.index.month
    weekly["quarter"] = weekly.index.quarter
    weekly["year"] = weekly.index.year
    return weekly.dropna()


def safe_filename(value: str) -> str:
    return str(value).replace(" ", "_").replace("&", "and")
