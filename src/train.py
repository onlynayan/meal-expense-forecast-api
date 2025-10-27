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

# ---------------------------------------------------------------------
# Helper: Mean Absolute Percentage Error
# ---------------------------------------------------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_mask = y_true != 0
    return np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100


# ---------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------
def train_model(csv_path, model_path, plots_dir="./plots", metadata_path="./models/metadata_rf.json"):
    # Load dataset
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Feature selection
    feature_cols = [
        "month", "year", "day_of_week", "holiday_flag",
        "total_meal_count", "average_meal_cost", "overhead_cost",
        "previous_day_expense", "rolling_7day_avg_expense", "rolling_30day_trend"
    ]
    target_col = "total_expense"

    # Fill any missing values
    df[feature_cols] = df[feature_cols].fillna(0)
    df[target_col] = df[target_col].fillna(0)

    # Prepare train/test split (last 30 days as test)
    train_df = df.iloc[:-30]
    test_df = df.iloc[-30:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Train RandomForest
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # -----------------------------------------------------------------
    # Save model and metadata
    # -----------------------------------------------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    joblib.dump(model, model_path)
    with open(metadata_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # -----------------------------------------------------------------
    # Plot actual vs predicted
    # -----------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["date"], y_test, label="Actual", linewidth=2)
    plt.plot(test_df["date"], y_pred, label="Predicted", linestyle="--")
    plt.title("RandomForest - Actual vs Predicted Daily Expense")
    plt.xlabel("Date")
    plt.ylabel("Expense")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "train_results.png"))
    plt.close()

    print("✅ Training completed successfully!")
    print(json.dumps(metrics, indent=4))


# ---------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./data/expenses.csv", help="Path to expense CSV")
    parser.add_argument("--model", type=str, default="./models/expense_rf.pkl", help="Path to save model")
    args = parser.parse_args()

    train_model(args.csv, args.model)
