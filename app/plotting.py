import io
import matplotlib.pyplot as plt
import pandas as pd

# Do not set explicit colors or styles per your constraints.

def plot_history(ts: pd.Series, days: int = 365) -> bytes:
    tail = ts.iloc[-days:] if len(ts) > days else ts
    fig = plt.figure(figsize=(10, 4))
    plt.plot(tail.index, tail.values)
    plt.title(f"History (last {len(tail)} days)")
    plt.xlabel("Date"); plt.ylabel("Total Expense")
    buf = io.BytesIO(); plt.tight_layout(); fig.savefig(buf, format="png")
    plt.close(fig); buf.seek(0); return buf.getvalue()

def plot_forecast(ts: pd.Series, forecast_df: pd.DataFrame, days: int = 365) -> bytes:
    tail = ts.iloc[-days:] if len(ts) > days else ts
    fig = plt.figure(figsize=(10, 4))
    plt.plot(tail.index, tail.values, label="History")
    plt.plot(forecast_df["date"], forecast_df["value"], label="Forecast")
    plt.legend(); plt.title(f"Forecast next {len(forecast_df)} days")
    plt.xlabel("Date"); plt.ylabel("Total Expense")
    buf = io.BytesIO(); plt.tight_layout(); fig.savefig(buf, format="png")
    plt.close(fig); buf.seek(0); return buf.getvalue()
