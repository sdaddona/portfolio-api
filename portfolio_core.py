# ============================== portfolio_core.py =============================
# -*- coding: utf-8 -*-
import os
from portfolio_analysis_web import analyze_portfolio_from_text

def run_full_analysis(
    lots_text: str,
    bench: str,
    use_adjclose: bool = False,
    rf_source: str = "fred_1y",
    rf: float = 0.0,
):
    """
    Glue per server.py: invoca la logica e ritorna un dict JSON-safe.
    """
    outdir = "/tmp/outputs"
    os.makedirs(outdir, exist_ok=True)

    analysis = analyze_portfolio_from_text(
        lots_text=lots_text,
        bench=bench,
        outdir=outdir,
        start_buffer_days=7,
        use_adjclose=use_adjclose,
        rf_source=rf_source,
        rf=rf,
    )
    return analysis
