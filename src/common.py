import os, json
import pandas as pd
from typing import Tuple

def load_expense_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "total_expense" not in df.columns:
        raise ValueError("CSV must have columns: date,total_expense")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date")
    return df

def to_daily_series(df: pd.DataFrame) -> pd.Series:
    ts = df.set_index("date")["total_expense"].sort_index()
    full_idx = pd.date_range(ts.index.min(), ts.index.max(), freq="D")
    # forward-fill missing days; then fill any leading NaN with 0
    ts = ts.reindex(full_idx).ffill().fillna(0.0)
    ts.index.name = "date"
    return ts

def month_bounds(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthEnd(1))  # inclusive last day of month
    return start, end

def save_metadata(path: str, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_metadata(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
