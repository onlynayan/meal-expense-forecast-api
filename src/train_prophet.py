import os, json, argparse, joblib
import numpy as np, pandas as pd
from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero = y_true != 0
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100


def train_prophet(csv_path, model_path, plots_dir="./plots", metadata_path="./models/metadata_prophet.json"):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Prophet expects columns: ds (date), y (target)
    prophet_df = df.rename(columns={"date": "ds", "total_expense": "y"})[["ds", "y"]]

    # Split last 30 days for validation
    train_df = prophet_df.iloc[:-30]
    test_df = prophet_df.iloc[-30:]

    # Initialize and fit model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train_df)

    # Predict next 30 days
    forecast = model.predict(test_df[["ds"]])
    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(metadata_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot forecast vs actual
    plt.figure(figsize=(12,5))
    plt.plot(test_df["ds"], y_true, label="Actual", linewidth=2)
    plt.plot(test_df["ds"], y_pred, label="Prophet Predicted", linestyle="--")
    plt.title("Prophet Model - Actual vs Predicted (Last 30 Days)")
    plt.xlabel("Date"); plt.ylabel("Expense"); plt.legend()
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "train_prophet_results.png"))
    plt.close()

    print("✅ Prophet model trained successfully!")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./data/expenses.csv", help="Path to dataset CSV")
    parser.add_argument("--model", type=str, default="./models/expense_prophet.pkl", help="Path to save Prophet model")
    args = parser.parse_args()

    train_prophet(args.csv, args.model)
