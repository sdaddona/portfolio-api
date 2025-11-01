# portfolio_core.py
import os
from portfolio_analysis_web import analyze_portfolio_from_text

def run_full_analysis(lots_text: str,
                      bench: str,
                      use_adjclose: bool = False) -> dict:
    """
    Entry point chiamato da server.py.

    Parametri
    ---------
    lots_text : str
        Righe con i lotti: TICKER DATA QUANTITA PREZZO
    bench : str
        Ticker del benchmark (es. 'VT')
    use_adjclose : bool
        Se True: ignora il prezzo del file e usa gli AdjClose di mercato.
        Se False: usa AdjClose ma sovrascrive il prezzo nel giorno del trade
                  con il prezzo del file (replica la logica locale).
    """
    outdir = "/tmp/outputs"
    os.makedirs(outdir, exist_ok=True)

    analysis = analyze_portfolio_from_text(
        lots_text=lots_text,
        bench=bench,
        outdir=outdir,
        use_adjclose=use_adjclose,
    )

    return analysis
