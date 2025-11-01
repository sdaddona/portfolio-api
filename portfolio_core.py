import os
from portfolio_analysis_web import analyze_portfolio_from_text

def run_full_analysis(
    lots_text: str,
    bench: str,
    *,
    use_adjclose: bool = False,     # come il tuo locale
    rf_source: str = "fred_1y",     # come il tuo locale
    rf_fixed: float = 0.04          # usato solo se rf_source == "fixed"
):
    """
    Bridge tra server e logica finanziaria.

    lots_text: contenuto della textarea (ticker, data, qty, prezzo)
    bench: ticker benchmark (es. 'VT')
    use_adjclose: False = usa Close e sostituisce il prezzo del file nel trade day
    rf_source: 'fred_1y' | 'irx_13w' | 'fixed'
    rf_fixed: valore annuo (solo se rf_source='fixed')
    """
    outdir = "/tmp/outputs"
    os.makedirs(outdir, exist_ok=True)

    analysis = analyze_portfolio_from_text(
        lots_text=lots_text,
        bench=bench,
        outdir=outdir,
        use_adjclose=use_adjclose,
        rf_source=rf_source,
        rf=rf_fixed
    )
    return analysis
