# Portfolio Analytics API (Render deploy)

## 1. Requisiti locali
Creare virtualenv (solo per sviluppo locale):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
uvicorn server:app --reload

