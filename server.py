# server.py
# -*- coding: utf-8 -*-

import os
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file, make_response

from werkzeug.exceptions import RequestEntityTooLarge, BadRequest
from portfolio_core import run_full_analysis

# ------------------------------------------------------------------------------
# Config base
# ------------------------------------------------------------------------------
app = Flask(__name__)

# Limite opzionale body (MB) per evitare 413 non gestiti; 5MB di default.
MAX_BODY_MB = float(os.environ.get("MAX_BODY_MB", "5"))
app.config["MAX_CONTENT_LENGTH"] = int(MAX_BODY_MB * 1024 * 1024)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("portfolio-api")

# ------------------------------------------------------------------------------
# CORS Helpers
# ------------------------------------------------------------------------------
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_HEADERS = os.environ.get(
    "ALLOWED_HEADERS",
    "Content-Type, Authorization, X-Requested-With, Accept, Origin"
)
ALLOWED_METHODS = os.environ.get("ALLOWED_METHODS", "GET,POST,OPTIONS")

def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGINS
    resp.headers["Access-Control-Allow-Methods"] = ALLOWED_METHODS
    resp.headers["Access-Control-Allow-Headers"] = ALLOWED_HEADERS
    # Se usi credenziali cross-site, abilita anche:
    # resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp

@app.after_request
def after_request(response):
    return add_cors_headers(response)

# ------------------------------------------------------------------------------
# Error Handlers (per avere JSON SEMPRE e LOG chiari)
# ------------------------------------------------------------------------------
@app.errorhandler(RequestEntityTooLarge)
def handle_413(e):
    logger.warning(f"413 Payload troppo grande: size > {MAX_BODY_MB} MB")
    resp = jsonify({"ok": False, "error": f"Payload troppo grande (> {MAX_BODY_MB} MB)"})
    return add_cors_headers(resp), 413

@app.errorhandler(BadRequest)
def handle_400(e):
    logger.warning(f"400 Bad Request: {e}")
    resp = jsonify({"ok": False, "error": "Bad Request: JSON mancante o non valido"})
    return add_cors_headers(resp), 400

@app.errorhandler(Exception)
def handle_500(e):
    logger.exception("500 Internal Server Error")
    resp = jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()})
    return add_cors_headers(resp), 500

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET", "OPTIONS"])
def root():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response("", 204))
    payload = {
        "status": "ok",
        "message": "portfolio API online",
        "time_utc": datetime.utcnow().isoformat() + "Z"
    }
    resp = jsonify(payload)
    return add_cors_headers(resp)

@app.route("/debug/ping", methods=["GET", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response("", 204))
    info = {
        "ok": True,
        "method": request.method,
        "ua": request.headers.get("User-Agent", ""),
        "content_type": request.headers.get("Content-Type", ""),
        "length": int(request.headers.get("Content-Length", "0") or "0"),
        "time_utc": datetime.utcnow().isoformat() + "Z",
    }
    return add_cors_headers(jsonify(info)), 200

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        # Preflight CORS
        return add_cors_headers(make_response("", 204))

    # --- Log ingresso richiesta
    ua = request.headers.get("User-Agent", "")
    ct = request.headers.get("Content-Type", "")
    clen = int(request.headers.get("Content-Length", "0") or "0")
    logger.info(f"POST /analyze | UA={ua!r} | CT={ct!r} | LEN={clen}")

    # --- Parse JSON robusto
    data = None
    try:
        # Accettiamo application/json; se non c'è, tentiamo comunque un parse safe
        if ct and "application/json" in ct.lower():
            data = request.get_json(silent=False)
        else:
            data = request.get_json(silent=True)
    except BadRequest:
        # scatta l'handler 400 se rilanciamo
        raise
    except Exception as ex:
        logger.warning(f"JSON parse fallito: {ex}")
        data = None

    if not isinstance(data, dict):
        msg = "Body JSON mancante o non valido"
        logger.warning(f"/analyze -> {msg}")
        return add_cors_headers(jsonify({"ok": False, "error": msg})), 400

    # --- Estraggo parametri
    lots_text    = data.get("lots_text", "")
    bench        = data.get("bench", "")
    use_adjclose = bool(data.get("use_adjclose", False))
    rf_source    = str(data.get("rf_source", "fred_1y"))
    # Nota: rf passato come percentuale decimale (es: 0.04 = 4%)
    rf           = float(data.get("rf", 0.0))

    if not lots_text.strip():
        return add_cors_headers(jsonify({"ok": False, "error": "lots_text mancante"})), 400
    if not bench.strip():
        return add_cors_headers(jsonify({"ok": False, "error": "bench mancante"})), 400

    logger.info(
        f"Analyze called | bench={bench} | use_adjclose={use_adjclose} | "
        f"rf_source={rf_source} | rf={rf:.4f} | chars(lots)={len(lots_text)}"
    )

    try:
        result = run_full_analysis(
            lots_text=lots_text,
            bench=bench,
            use_adjclose=use_adjclose,
            rf_source=rf_source,
            rf=rf,
        )
        resp = jsonify({"ok": True, **result})
        return add_cors_headers(resp), 200

    except Exception as e:
        logger.error("Errore in /analyze", exc_info=True)
        resp = jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()})
        return add_cors_headers(resp), 500

@app.route("/plot", methods=["GET", "OPTIONS"])
def get_plot():
    if request.method == "OPTIONS":
        return add_cors_headers(make_response("", 204))
    try:
        plot_path = "/tmp/outputs/crescita_cumulata.png"
        if not os.path.exists(plot_path):
            return add_cors_headers(jsonify({"ok": False, "error": "Plot non disponibile. Esegui prima /analyze."})), 404
        resp = send_file(plot_path, mimetype="image/png", as_attachment=False)
        return add_cors_headers(resp), 200
    except Exception as e:
        logger.exception("Errore in /plot")
        resp = jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()})
        return add_cors_headers(resp), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # Avvio di sviluppo; in produzione usa gunicorn
    app.run(host="0.0.0.0", port=port, debug=True)
