import os
from portfolio_analysis_web import analyze_portfolio_from_text

def run_full_analysis(lots_text: str, bench: str):
    """
    Glue che chiama la logica finanziaria e restituisce
    un dizionario JSON-safe per server.py.

    lots_text: contenuto del file lotti (ticker, data, qty, prezzo)
    bench: ticker benchmark (es 'VT')
    """

    outdir = "/tmp/outputs"
    os.makedirs(outdir, exist_ok=True)

    analysis = analyze_portfolio_from_text(
        lots_text=lots_text,
        bench=bench,
        outdir=outdir,
    )

    # analysis contiene:
    # {
    #   "summary_lines": [...],
    #   "plot_path": "...png",
    #   "alloc": {...},
    #   "risk": {...},
    #   ...
    # }

    return analysis
