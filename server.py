import os
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file, make_response

from portfolio_core import run_full_analysis

# ------------------------------------------------------------------------------
# Setup Flask app + logging
# ------------------------------------------------------------------------------

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("portfolio-api")


# ------------------------------------------------------------------------------
# CORS CONFIG
# ------------------------------------------------------------------------------
# Inizio semplice: permettiamo tutte le origini.
# Quando pubblichi su WordPress, puoi sostituire "*" con "https://www.daddona.it"
ALLOWED_ORIGINS = "*"


def add_cors_headers(resp):
    """Aggiunge gli header CORS standard alla risposta Flask."""
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGINS
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.after_request
def after_request(response):
    """
    Questo viene chiamato su OGNI risposta "normale" (GET/POST riuscita).
    Aggiungiamo qui gli header CORS in modo automatico.
    """
    return add_cors_headers(response)


# ------------------------------------------------------------------------------
# HEALTHCHECK /
# ------------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def root():
    """
    Endpoint di salute. Lo lasciamo compatibile con la tua versione attuale
    ma arricchito con qualche info in più per debugging.
    """
    payload = {
        "status": "ok",
        "message": "portfolio API online",
        "time_utc": datetime.utcnow().isoformat() + "Z"
    }
    resp = jsonify(payload)
    return add_cors_headers(resp)


# ------------------------------------------------------------------------------
# /analyze
# ------------------------------------------------------------------------------

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    """
    Richiesta attesa dal front-end WordPress:

    POST /analyze
    Content-Type: application/json
    {
        "lots_text": "VT 2024-01-02 10 100\nVT 2024-03-10 5 95",
        "bench": "VT"
    }

    Risposta:
    {
        "ok": true,
        "summary_lines": [...],
        "risk": {...},
        "pme": {...},
        "plot_path": "/tmp/outputs/crescita_cumulata.png",
        ...
    }

    CORS:
    Gestiamo anche OPTIONS per il preflight del browser.
    """

    # 1. Risposta CORS preflight dal browser (prima della POST vera)
    if request.method == "OPTIONS":
        resp = make_response("", 204)
        return add_cors_headers(resp)

    try:
        # 2. Parse input JSON
        data = request.get_json(silent=True)
        if data is None:
            resp = jsonify({"ok": False, "error": "Body JSON mancante o non valido"})
            return add_cors_headers(resp), 400

        lots_text = data.get("lots_text", "")
        bench = data.get("bench", "")

        if not lots_text.strip():
            resp = jsonify({"ok": False, "error": "lots_text mancante"})
            return add_cors_headers(resp), 400
        if not bench.strip():
            resp = jsonify({"ok": False, "error": "bench mancante"})
            return add_cors_headers(resp), 400

        # 3. Esegui analisi completa (tuo core business)
        result = run_full_analysis(lots_text, bench)

        # 4. Risposta finale
        out = {
            "ok": True,
            **result
        }
        resp = jsonify(out)
        return add_cors_headers(resp), 200

    except Exception as e:
        logger.exception("Errore in /analyze")
        resp = jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()
        })
        return add_cors_headers(resp), 500


# ------------------------------------------------------------------------------
# /plot
# ------------------------------------------------------------------------------

@app.route("/plot", methods=["GET", "OPTIONS"])
def get_plot():
    """
    Ritorna il grafico PNG dell'ultima analisi effettuata.

    Come lo useremo nel front-end:
    <img
      src="https://portfolio-api-docker.onrender.com/plot"
      style="max-width:100%;border:1px solid #d7e1de;border-radius:8px;"
    />

    Nota di architettura:
    - Per ora serviamo un file fisso /tmp/outputs/crescita_cumulata.png
      (cioè l'ultima analisi fatta sul container).
    - In futuro, per multi-utente, potremo salvare un file per analisi con un ID
      e fare GET /plot?id=123.
    """

    # Preflight CORS
    if request.method == "OPTIONS":
        resp = make_response("", 204)
        return add_cors_headers(resp)

    try:
        plot_path = "/tmp/outputs/crescita_cumulata.png"
        if not os.path.exists(plot_path):
            resp = jsonify({
                "ok": False,
                "error": "Plot non disponibile. Esegui prima /analyze."
            })
            return add_cors_headers(resp), 404

        # send_file ritorna una Response Flask con contenuto binario PNG
        resp = send_file(plot_path, mimetype="image/png", as_attachment=False)
        return add_cors_headers(resp), 200

    except Exception as e:
        logger.exception("Errore in /plot")
        resp = jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()
        })
        return add_cors_headers(resp), 500


# ------------------------------------------------------------------------------
# RUN LOCALE (non usato da Render perché lì parte gunicorn)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # uso locale/manuale
    port = int(os.environ.get("PORT", "5000"))
    # debug=True solo per sviluppo locale; su Render gira gunicorn senza debug
    app.run(host="0.0.0.0", port=port, debug=True)
