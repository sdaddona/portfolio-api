# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from portfolio_core import run_full_analysis

app = FastAPI(
    title="Portfolio Analytics API",
    description="Calcolo performance portafoglio + allocazioni ETF",
    version="0.1.0",
)

class AnalyzeRequest(BaseModel):
    lots_text: str
    benchmark: Optional[str] = "VT"
    headless: Optional[bool] = True  # per lo scraper

class AnalyzeResponse(BaseModel):
    status: str
    benchmark: str
    positions: list
    performance: dict
    allocations: dict

@app.get("/health")
def health():
    return {"status": "alive"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    result = run_full_analysis(
        lots_text=payload.lots_text,
        benchmark=payload.benchmark,
        headless=payload.headless
    )
    return result

