# /app/portfolio_core.py

import os
from typing import Any, Dict

from portfolio_analysis_web import analyze_portfolio_from_text


def run_full_analysis(
    lots_text: str,
    bench: str,
    *,
    use_adjclose: bool = False,
    rf_source: str = "fred_1y",
    rf: float = 0.0,
    start_buffer_days: int = 7,
    data_source: str = "auto",
) -> Dict[str, Any]:
    """
    Colla di integrazione tra server.py e la logica di analisi.

    Parametri
    ---------
    lots_text : str
        Contenuto del file lotti (ticker, data, qty, prezzo).
    bench : str
        Ticker benchmark (es. "VT").
    use_adjclose : bool
        Se True ignora i prezzi del file lotti e usa gli AdjClose per tutti.
    rf_source : {"fred_1y","irx_13w","fixed"}
        Sorgente del risk-free per Sharpe/Var.
    rf : float
        Valore annuo del risk-free (solo se rf_source="fixed").
    start_buffer_days : int
        Giorni di buffer prima della prima data trade per iniziare i download.
    data_source : {"auto","eod","yahoo"}
        Sorgente prezzi preferita. "auto" prova EOD (se c'è API key) poi Yahoo.

    Ritorna
    -------
    dict (JSON-safe) con chiavi:
      - summary_lines: list[str]
      - plot_path: str
      - (eventuali) risk, pme, alloc, ecc.
    """
    outdir = "/tmp/outputs"
    os.makedirs(outdir, exist_ok=True)

    eod_api_key = os.environ.get("EOD_API_KEY", "")

    # Kwargs "completi" per la versione più recente della funzione web
    full_kwargs = dict(
        lots_text=lots_text,
        bench=bench,
        outdir=outdir,
        use_adjclose=use_adjclose,
        rf_source=rf_source,
        rf=rf,
        start_buffer_days=start_buffer_days,
        data_source=data_source,
        eod_api_key=eod_api_key,
    )

    # 1) Prova chiamata "completa"
    try:
        return analyze_portfolio_from_text(**full_kwargs)  # type: ignore[arg-type]
    except TypeError:
        # 2) Fallback: togli i parametri più "nuovi" per compatibilità retro
        fallback_kwargs = dict(
            lots_text=lots_text,
            bench=bench,
            outdir=outdir,
            use_adjclose=use_adjclose,
            rf_source=rf_source,
            rf=rf,
        )
        try:
            return analyze_portfolio_from_text(**fallback_kwargs)  # type: ignore[arg-type]
        except TypeError:
            # 3) Fallback minimale: vecchissima firma (lots_text, bench, outdir)
            minimal_kwargs = dict(
                lots_text=lots_text,
                bench=bench,
                outdir=outdir,
            )
            return analyze_portfolio_from_text(**minimal_kwargs)  # type: ignore[arg-type]
