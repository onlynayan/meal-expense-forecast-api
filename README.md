# 🚀 Meal Expense Forecast API (Live Demo)

### 🔗 Live Railway API Endpoints
| Endpoint | Description | URL |
|---------|-------------|-----|
| 🌐 **Base URL** | Root of the API | https://web-production-e2163.up.railway.app |
| 🩺 **Health Check** | API health status | https://web-production-e2163.up.railway.app/health |
| 📈 **Monthly Prediction** | Predict expenses for a given month | https://web-production-e2163.up.railway.app/predict/month?year=2025&month=10 |
| 🧮 **Model Metrics** | Compare model performance | https://web-production-e2163.up.railway.app/metrics |
| 🧠 **Model Retrain (POST)** | Retrain the model dynamically | https://web-production-e2163.up.railway.app/retrain |
| 📊 **Forecast Plot** | Visualize future expense forecast trends | https://web-production-e2163.up.railway.app/plots/forecast |
| 📉 **Historical Trend Plot** | View historical expense data visualization | https://web-production-e2163.up.railway.app/plots/history |

---

### 🌐 Frontend UI (NestJS + React, Deployed on Vercel)
| Feature | Description | URL |
|--------|-------------|-----|
| 🍽️ **Meal Expense Dashboard** | Visual interface built on top of this API | https://meal-app-delta-tan.vercel.app/predictions |

> This UI interactively displays forecast results, monthly expense breakdowns, and helps plan next month's meal budget using data served from this API.

---

## 📘 Project Overview
**Meal Expense Forecast API** is a Machine Learning–driven system built using **FastAPI** that predicts **daily and monthly meal expenses** from historical company data.

It’s optimized to help with:
- Budget forecasting 📊  
- Trend visualization 📈  
- Real-time predictions via REST API ⚙️  

---

## 🧠 Tech Stack
| Layer | Technology |
|--------|-------------|
| Backend | FastAPI (Python 3.11) |
| ML Models | RandomForestRegressor, Prophet |
| Data | Pandas, NumPy |
| Visualization | Matplotlib / Plotly |
| Deployment | Railway.app |
| Environment | .env for configuration |

---

## ⚙️ Local Setup
```bash
# Clone the repo
git clone https://github.com/<your-username>/meal-expense-forecast-api.git
cd meal-expense-forecast-api

# Create virtual environment
python -m venv .venv
.\.venv\Scriptsctivate  # For Windows

# Install dependencies
pip install -r requirements.txt

# Train model
python -m src.train --csv ./data/expenses.csv --model ./models/expense_rf.pkl

# Run FastAPI server
uvicorn app.main:app --reload --port 8000

# Open docs
http://127.0.0.1:8000/docs
```

---

## 🧮 Model Performance Summary
| Model | MAE | RMSE | R² | MAPE |
|--------|------|------|------|------|
| **RandomForest (Daily)** | 159.22 | 251.56 | 0.990 | 2.83% |
| **RandomForest (Monthly)** | 5659.49 | 5660.82 | 0.523 | 3.04% |
| **Prophet** | 1387.70 | 2006.88 | 0.379 | 29.38% |

🏆 **Best Model:** RandomForest (Daily)

---

## 🧩 Example API Usage (Postman)

### ✅ Health Check
`GET` → https://web-production-e2163.up.railway.app/health

### 📈 Monthly Prediction
`GET` → https://web-production-e2163.up.railway.app/predict/month?year=2025&month=10

### 🧠 Retrain
`POST` → https://web-production-e2163.up.railway.app/retrain

Body (JSON):
```json
{
  "csv_path": "./data/expenses.csv"
}
```

---

## 🌍 Deployment Steps (Railway)
1. Push this repo to GitHub.
2. Create new project on [Railway](https://railway.app/).
3. Connect to your GitHub repo.
4. Add Environment Variables:
   ```env
   DATA_CSV=./data/expenses.csv
   MODEL_PATH=./models/expense_rf.pkl
   TIMEZONE=Asia/Dhaka
   ```
5. Set Start Command:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
6. Deploy 🚀

---

## 👨‍💻 Author
**Developed by:** *Phoenix (Nayan)*  
🔗 GitHub: [https://github.com/onlynayan](https://github.com/onlynayan)  
💡 “Smarter budgeting starts with intelligent forecasting.”
