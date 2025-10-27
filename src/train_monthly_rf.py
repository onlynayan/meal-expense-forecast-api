import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from datetime import datetime


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_mask = y_true != 0
    return np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100


def train_monthly_model(csv_path, model_path, plots_dir="./plots", metadata_path="./models/metadata_rf_monthly.json"):
    # Load dataset
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # -----------------------------------------------------------------
    # 1. Aggregate data monthly
    # -----------------------------------------------------------------
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    monthly_df = df.groupby("year_month").agg({
        "total_expense": "sum",
        "total_meal_count": "sum",
        "average_meal_cost": "mean",
        "overhead_cost": "mean",
        "holiday_flag": "sum",
        "rolling_7day_avg_expense": "mean",
        "rolling_30day_trend": "mean"
    }).reset_index()

    # Add separate year/month numeric columns
    monthly_df["year"] = monthly_df["year_month"].apply(lambda x: int(x.split("-")[0]))
    monthly_df["month"] = monthly_df["year_month"].apply(lambda x: int(x.split("-")[1]))

    feature_cols = [
        "year", "month", "total_meal_count", "average_meal_cost",
        "overhead_cost", "holiday_flag", "rolling_7day_avg_expense", "rolling_30day_trend"
    ]
    target_col = "total_expense"

    # -----------------------------------------------------------------
    # 2. Train/Test Split
    # -----------------------------------------------------------------
    train_df = monthly_df.iloc[:-2]  # all but last 2 months
    test_df = monthly_df.iloc[-2:]   # last 2 months to test

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # -----------------------------------------------------------------
    # 3. Train RandomForest Model
    # -----------------------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -----------------------------------------------------------------
    # 4. Metrics
    # -----------------------------------------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "train_months": len(train_df),
        "test_months": len(test_df),
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # -----------------------------------------------------------------
    # 5. Save Model + Metadata
    # -----------------------------------------------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    joblib.dump(model, model_path)
    with open(metadata_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # -----------------------------------------------------------------
    # 6. Plot actual vs predicted (last 6 months)
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_df["year_month"], monthly_df["total_expense"], label="Actual", linewidth=2)
    plt.plot(test_df["year_month"], y_pred, label="Predicted (Test)", linestyle="--", marker="o")
    plt.title("Monthly Expense Forecast (RandomForest)")
    plt.xlabel("Year-Month")
    plt.ylabel("Total Expense")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "train_monthly_results.png"))
    plt.close()

    print("✅ Monthly model trained successfully!")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./data/expenses.csv", help="Path to dataset CSV")
    parser.add_argument("--model", type=str, default="./models/expense_rf_monthly.pkl", help="Path to save model")
    args = parser.parse_args()

    train_monthly_model(args.csv, args.model)
