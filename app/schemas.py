from pydantic import BaseModel, Field
from typing import List

class DayForecast(BaseModel):
    date: str
    value: float
    source: str = Field(description="'actual' or 'forecast'")

class MonthForecastResponse(BaseModel):
    year: int
    month: int
    days: List[DayForecast]
    total: float
    note: str

class NDForecastResponse(BaseModel):
    start: str
    days: int
    forecast: List[DayForecast]

class MetricsResponse(BaseModel):
    MAE: float
    RMSE: float
    R2: float
    MAPE: float
