import pandas as pd
import matplotlib.pyplot as plt
import joblib, os
from prophet import Prophet

def plot_model_comparison(csv_path="./data/expenses.csv", plots_dir="./plots"):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # ---------------------
    # RandomForest Daily Predictions (last 30 days)
    # ---------------------
    rf_model = joblib.load("./models/expense_rf.pkl")
    recent_df = df.iloc[-30:].copy()

    feature_cols = [
        "month", "year", "day_of_week", "holiday_flag",
        "total_meal_count","average_meal_cost","overhead_cost",
        "previous_day_expense","rolling_7day_avg_expense","rolling_30day_trend"
    ]
    X_recent = recent_df[feature_cols]
    rf_pred = rf_model.predict(X_recent)

    # ---------------------
    # Prophet Predictions (same range)
    # ---------------------
    prophet_model = joblib.load("./models/expense_prophet.pkl")
    prophet_df = df.rename(columns={"date": "ds", "total_expense": "y"})[["ds", "y"]]
    forecast = prophet_model.predict(prophet_df[["ds"]])
    prophet_pred = forecast.tail(30)["yhat"].values

    # ---------------------
    # Plot comparison
    # ---------------------
    plt.figure(figsize=(12, 6))
    plt.plot(recent_df["date"], recent_df["total_expense"], label="Actual", linewidth=2)
    plt.plot(recent_df["date"], rf_pred, label="RandomForest", linestyle="--", color="orange")
    plt.plot(recent_df["date"], prophet_pred, label="Prophet", linestyle="--", color="green")
    plt.title("Model Comparison - Last 30 Days")
    plt.xlabel("Date"); plt.ylabel("Expense (BDT)"); plt.legend(); plt.grid(True)

    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "model_comparison.png")
    plt.tight_layout(); plt.savefig(path)
    plt.close()

    print(f"✅ Saved model comparison plot to: {path}")

if __name__ == "__main__":
    plot_model_comparison()
