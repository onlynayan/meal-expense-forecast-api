from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import joblib, json, os
from datetime import datetime
from calendar import monthrange
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------
load_dotenv()
DATA_CSV = os.getenv("DATA_CSV", "./data/expenses.csv")

# Models
DAILY_MODEL_PATH = "./models/expense_rf.pkl"
MONTHLY_MODEL_PATH = "./models/expense_rf_monthly.pkl"
METRICS_DAILY = "./models/metadata_rf.json"
METRICS_MONTHLY = "./models/metadata_rf_monthly.json"

app = FastAPI(title="Meal Expense Forecast API (RF + Monthly Model)")

# ---------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------
daily_model = joblib.load(DAILY_MODEL_PATH) if os.path.exists(DAILY_MODEL_PATH) else None
monthly_model = joblib.load(MONTHLY_MODEL_PATH) if os.path.exists(MONTHLY_MODEL_PATH) else None

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def prepare_features_for_dates(dates: pd.Series, ref_df: pd.DataFrame):
    df = pd.DataFrame({"date": dates})
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["holiday_flag"] = 0
    for col in [
        "total_meal_count","average_meal_cost","overhead_cost",
        "previous_day_expense","rolling_7day_avg_expense","rolling_30day_trend",
    ]:
        df[col] = ref_df[col].mean()
    return df[
        ["month","year","day_of_week","holiday_flag",
         "total_meal_count","average_meal_cost","overhead_cost",
         "previous_day_expense","rolling_7day_avg_expense","rolling_30day_trend"]
    ]

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Meal Expense Forecast API (RandomForest + Monthly Model) is running!"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "daily_model_loaded": daily_model is not None,
        "monthly_model_loaded": monthly_model is not None,
        "data_csv": DATA_CSV,
    }


# ---------------------------------------------------------------------
# Daily + Monthly prediction combined
# ---------------------------------------------------------------------
@app.get("/predict/month")
def predict_month(year: int = Query(..., ge=2000), month: int = Query(..., ge=1, le=12)):
    if daily_model is None:
        return JSONResponse(status_code=500, content={"error": "Model not found."})
    if not os.path.exists(DATA_CSV):
        return JSONResponse(status_code=500, content={"error": "Dataset not found."})

    ref_df = pd.read_csv(DATA_CSV)
    ref_df["date"] = pd.to_datetime(ref_df["date"])

    # Get correct number of days for that month
    num_days = monthrange(year, month)[1]
    pred_dates = pd.date_range(f"{year}-{month:02d}-01", periods=num_days, freq="D")

    # Prepare features for each day in the month
    features = prepare_features_for_dates(pred_dates, ref_df)

    # Predict daily expenses using RandomForest
    preds = daily_model.predict(features)

    # Combine into a dataframe
    daily_results = pd.DataFrame({"date": pred_dates, "predicted_expense": preds})

    # Aggregate to monthly total
    total_pred = daily_results["predicted_expense"].sum()

    # Return both daily list + total monthly
    return {
        "year": year,
        "month": month,
        "total_predicted_expense": round(float(total_pred), 2),
        "daily_predictions": [
            {"date": d.strftime("%Y-%m-%d"), "predicted_expense": round(float(v), 2)}
            for d, v in zip(daily_results["date"], daily_results["predicted_expense"])
        ],
    }



# ---------------------------------------------------------------------
# Visualization route
# ---------------------------------------------------------------------
@app.get("/plots/forecast")
def plot_forecast(year: int = Query(...), month: int = Query(...)):
    ref_df = pd.read_csv(DATA_CSV)
    ref_df["date"] = pd.to_datetime(ref_df["date"])

    num_days = monthrange(year, month)[1]
    pred_dates = pd.date_range(f"{year}-{month:02d}-01", periods=num_days, freq="D")
    features = prepare_features_for_dates(pred_dates, ref_df)
    preds = daily_model.predict(features)

    plt.figure(figsize=(10, 5))
    plt.plot(pred_dates, preds, label="Predicted", color="orange", linewidth=2)
    plt.title(f"Forecasted Daily Expenses - {year}-{month:02d}")
    plt.xlabel("Date")
    plt.ylabel("Expense (BDT)")
    plt.grid(True)
    plt.legend()

    os.makedirs("./plots", exist_ok=True)
    file_path = f"./plots/forecast_{year}_{month}.png"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    return FileResponse(file_path, media_type="image/png")


# ---------------------------------------------------------------------
# Metrics route (compare both)
# ---------------------------------------------------------------------
@app.get("/metrics")
def metrics():
    results = {}
    if os.path.exists(METRICS_DAILY):
        with open(METRICS_DAILY) as f:
            results["daily_model"] = json.load(f)
    if os.path.exists(METRICS_MONTHLY):
        with open(METRICS_MONTHLY) as f:
            results["monthly_model"] = json.load(f)
    return results


@app.get("/compare")
def compare_models():
    path = "./models/model_comparison.json"
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Comparison file not found."})
    with open(path, "r") as f:
        data = json.load(f)
    return data



@app.get("/plots/compare")
def get_model_comparison_plot():
    path = "./plots/model_comparison.png"
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Comparison plot not found."})
    return FileResponse(path, media_type="image/png")
