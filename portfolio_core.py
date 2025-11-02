# portfolio_core.py
import os
from portfolio_analysis_web import analyze_portfolio_from_text

def run_full_analysis(
    lots_text: str,
    bench: str,
    use_adjclose: bool = False,
    rf_source: str = "fred_1y",
    rf: float = 0.0,
    start_buffer_days: int = 7,
):
    """
    Pass-through verso analyze_portfolio_from_text con stessa firma
    della versione locale (senza allocazioni).
    """
    outdir = "/tmp/outputs"
    os.makedirs(outdir, exist_ok=True)

    analysis = analyze_portfolio_from_text(
        lots_text=lots_text,
        bench=bench,
        outdir=outdir,
        start_buffer_days=start_buffer_days,
        use_adjclose=use_adjclose,
        rf_source=rf_source,
        rf=rf,
    )
    return analysis
