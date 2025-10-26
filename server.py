import os
from flask import Flask, request, jsonify
from portfolio_core import run_full_analysis

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "portfolio API online"})

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expects JSON like:
    {
        "lots_text": "ACWI 2024-01-02 10 100\nVT 2024-01-03 5 90\n...",
        "bench": "VT"
    }
    Returns summary data.
    """
    data = request.get_json(silent=True) or {}
    lots_text = data.get("lots_text", "")
    bench = data.get("bench", "VT")

    try:
        result = run_full_analysis(lots_text, bench)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # local debug
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
