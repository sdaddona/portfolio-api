# server.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Portfolio API",
    description="API di test: il backend è vivo su Render",
    version="0.0.1",
)

# --- MODELLI -------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    note: str

# --- ENDPOINTS ----------------------------------------------

@app.get("/health", response_model=HealthResponse)
def healthcheck():
    """
    Semplice endpoint per verificare che l'app giri su Render.
    Nessuna analisi, nessun Playwright.
    """
    return HealthResponse(
        status="ok",
        note="Render is running 👍"
    )

@app.get("/")
def root():
    # homepage JSON minimale
    return {
        "message": "Portfolio API online",
        "next": {
            "/health": "stato servizio",
            "/analyze (coming soon)": "analisi portafoglio"
        }
    }

