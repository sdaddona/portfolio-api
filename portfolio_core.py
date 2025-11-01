import os
from portfolio_analysis_web import analyze_portfolio_from_text

def run_full_analysis(
    lots_text: str,
    bench: str,
    use_adjclose: bool = False,
    rf_source: str = "fred_1y",
    rf: float = 0.04,
):
    """
    Glue che chiama la logica finanziaria e restituisce
    un dizionario JSON-safe per server.py.
    """
    outdir = "/tmp/outputs"
    os.makedirs(outdir, exist_ok=True)

    analysis = analyze_portfolio_from_text(
        lots_text=lots_text,
        bench=bench,
        outdir=outdir,
        use_adjclose=use_adjclose,
        rf_source=rf_source,
        rf=rf,
    )
    return analysis
