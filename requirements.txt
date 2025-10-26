# server.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from portfolio_core import run_full_analysis

app = FastAPI(title="Portfolio Analyzer API")

class LotsRequest(BaseModel):
    lots_text: str
    bench: str = "VT"

@app.post("/analyze")
def analyze(req: LotsRequest):
    """
    Esegue l'analisi del portafoglio con i lotti passati dal frontend.
    """
    result = run_full_analysis(req.lots_text, req.bench)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}
