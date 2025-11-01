import os
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file, make_response

from portfolio_core import run_full_analysis

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("portfolio-api")

# CORS (puoi restringere a "https://www.daddona.it" quando sei pronto)
ALLOWED_ORIGINS = "*"

def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGINS
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.after_request
def after_request(response):
    return add_cors_headers(response)

@app.route("/", methods=["GET"])
def root():
    payload = {
        "status": "ok",
        "message": "portfolio API online",
        "time_utc": datetime.utcnow().isoformat() + "Z"
    }
    resp = jsonify(payload)
    return add_cors_headers(resp)

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        resp = make_response("", 204)
        return add_cors_headers(resp)

    try:
        data = request.get_json(silent=True)
        if data is None:
            resp = jsonify({"ok": False, "error": "Body JSON mancante o non valido"})
            return add_cors_headers(resp), 400

        lots_text   = (data.get("lots_text") or "").strip()
        bench       = (data.get("bench") or "").strip()
        # —— nuovi parametri per replicare il locale ——
        use_adjclose = bool(data.get("use_adjclose", False))   # locale: False
        rf_source    = (data.get("rf_source") or "fred_1y").strip().lower()
        rf_fixed     = float(data.get("rf_fixed", 0.04))

        if not lots_text:
            resp = jsonify({"ok": False, "error": "lots_text mancante"})
            return add_cors_headers(resp), 400
        if not bench:
            resp = jsonify({"ok": False, "error": "bench mancante"})
            return add_cors_headers(resp), 400

        logger.info(
            "Analyze called | bench=%s | use_adjclose=%s | rf_source=%s | rf_fixed=%s | chars(lots)=%d",
            bench, use_adjclose, rf_source, rf_fixed, len(lots_text)
        )

        result = run_full_analysis(
            lots_text=lots_text,
            bench=bench,
            use_adjclose=use_adjclose,
            rf_source=rf_source,
            rf_fixed=rf_fixed,
        )

        out = {"ok": True, **result}
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

@app.route("/plot", methods=["GET", "OPTIONS"])
def get_plot():
    if request.method == "OPTIONS":
        resp = make_response("", 204)
        return add_cors_headers(resp)

    try:
        plot_path = "/tmp/outputs/crescita_cumulata.png"
        if not os.path.exists(plot_path):
            resp = jsonify({"ok": False, "error": "Plot non disponibile. Esegui prima /analyze."})
            return add_cors_headers(resp), 404

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
