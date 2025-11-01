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
# In produzione puoi mettere: ALLOWED_ORIGINS = {"https://www.daddona.it", "https://daddona.it"}
ALLOWED_ORIGINS = {"*"}  # per ora aperto

ALLOWED_HEADERS = "Content-Type, Accept, X-Requested-With"
ALLOWED_METHODS = "GET,POST,OPTIONS"


def _pick_origin(req_origin: str | None) -> str:
    """Se vuoi restringere, restituisce l'origin solo se ammesso, altrimenti 'null'."""
    if not req_origin:
        return "*"
    if "*" in ALLOWED_ORIGINS:
        return "*"
    return req_origin if req_origin in ALLOWED_ORIGINS else "null"


def add_cors_headers(resp):
    """Aggiunge header CORS standard alla risposta Flask."""
    origin = _pick_origin(request.headers.get("Origin"))
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Methods"] = ALLOWED_METHODS
    resp.headers["Access-Control-Allow-Headers"] = ALLOWED_HEADERS
    # niente credenziali (cookies) -> non necessario per questa API
    return resp


@app.after_request
def after_request(response):
    """
    Questo viene chiamato su OGNI risposta "normale".
    Aggiungiamo qui gli header CORS in modo automatico.
    """
    return add_cors_headers(response)

# ------------------------------------------------------------------------------
# HEALTHCHECK /
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    payload = {
        "status": "ok",
        "message": "portfolio API online",
        "time_utc": datetime.utcnow().isoformat() + "Z",
    }
    resp = jsonify(payload)
    return add_cors_headers(resp)

# ------------------------------------------------------------------------------
# /analyze
# ------------------------------------------------------------------------------
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    """
    POST /analyze
    Body JSON:
      {
        "lots_text": "ACWI 2012-12-21 200 47.81\n...",
        "bench": "VT",
        "use_adjclose": false   # opzionale
      }
    """
    # Preflight CORS
    if request.method == "OPTIONS":
        resp = make_response("", 204)
        return add_cors_headers(resp)

    try:
        data = request.get_json(silent=True)
        if data is None:
            resp = jsonify({"ok": False, "error": "Body JSON mancante o non valido"})
            return add_cors_headers(resp), 400

        lots_text = (data.get("lots_text") or "").strip()
        bench = (data.get("bench") or "").strip()
        use_adjclose = bool(data.get("use_adjclose", False))

        if not lots_text:
            resp = jsonify({"ok": False, "error": "lots_text mancante"})
            return add_cors_headers(resp), 400
        if not bench:
            resp = jsonify({"ok": False, "error": "bench mancante"})
            return add_cors_headers(resp), 400

        logger.info("Analyze called | bench=%s | use_adjclose=%s | chars(lots)=%d",
                    bench, use_adjclose, len(lots_text))

        # Supporto opzionale a use_adjclose: se la tua funzione non lo accetta, fallback.
        try:
            result = run_full_analysis(lots_text, bench, use_adjclose=use_adjclose)
        except TypeError:
            # vecchia firma (lots_text, bench)
            result = run_full_analysis(lots_text, bench)

        out = {"ok": True, **result}
        resp = jsonify(out)
        return add_cors_headers(resp), 200

    except Exception as e:
        logger.exception("Errore in /analyze")
        resp = jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        })
        return add_cors_headers(resp), 500

# ------------------------------------------------------------------------------
# /plot
# ------------------------------------------------------------------------------
@app.route("/plot", methods=["GET", "OPTIONS"])
def get_plot():
    """
    Ritorna il PNG dell’ultima analisi.
    <img src="https://portfolio-api-docker.onrender.com/plot" />
    """
    # Preflight CORS
    if request.method == "OPTIONS":
        resp = make_response("", 204)
        return add_cors_headers(resp)

    try:
        plot_path = "/tmp/outputs/crescita_cumulata.png"
        if not os.path.exists(plot_path):
            resp = jsonify({"ok": False, "error": "Plot non disponibile. Esegui prima /analyze."})
            return add_cors_headers(resp), 404

        resp = make_response(send_file(plot_path, mimetype="image/png", as_attachment=False))
        # evita cache aggressive del browser/CDN
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return add_cors_headers(resp), 200

    except Exception as e:
        logger.exception("Errore in /plot")
        resp = jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()})
        return add_cors_headers(resp), 500

# ------------------------------------------------------------------------------
# RUN LOCALE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
