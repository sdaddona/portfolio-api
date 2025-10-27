# portfolio_analysis_web.py
#
# Versione robusta per deploy su Render (Docker).
# - Scarico prezzi via yfinance con retry e lookback esteso
# - Fallback benchmark se Yahoo non risponde subito
# - Genero output JSON strutturato + grafico PNG
#
# Assunzioni:
# - lots_text è tipo:
#     VT 2024-01-02 10 100
#     VT 2024-03-10 5 95
#   cioè: <ticker> <YYYY-MM-DD> <qty> <price_paid>
#
# - bench è un ticker Yahoo (es. "VT")
#
# Output principale:
# {
#   "ok": true,
#   "summary_lines": [...],
#   "pme": {...},
#   "risk": {...},
#   "alloc": {...},
#   "plot_path": "/tmp/outputs/crescita_cumulata.png"
# }
#
# NOTE:
# - Alcune metriche (PME, VaR, Sharpe) qui sono implementate in modo
#   ragionevole/simplificato per avere qualcosa di funzionante e stabile.
#   Se la tua versione locale calcola cose più raffinate (tipo IRR esatto),
#   puoi reincollarle 1:1 all'interno dei placeholder segnati sotto.


import os
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Utility: parsing input lotti
# ----------------------------------------------------------------------

def parse_lots_text(lots_text: str):
    """
    Converte il testo dei lotti in un DataFrame con colonne:
    ['ticker', 'date', 'qty', 'price']
    """
    rows = []
    for raw_line in lots_text.strip().splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = raw_line.split()
        # atteso: TICKER DATE QTY PRICE
        # es: VT 2024-01-02 10 100
        if len(parts) != 4:
            # se non matcha, salto la riga
            continue
        ticker, date_str, qty_str, price_str = parts
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            qty = float(qty_str.replace(",", "."))
            prc = float(price_str.replace(",", "."))
        except Exception:
            continue
        rows.append({
            "ticker": ticker.upper(),
            "date": dt,
            "qty": qty,
            "price": prc,
        })
    if not rows:
        return pd.DataFrame(columns=["ticker","date","qty","price"])
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------
# 2. Download prezzi robusto
# ----------------------------------------------------------------------

def try_symbol(ticker: str,
               min_history_days: int = 365,
               max_lookback_years: int = 10):
    """
    Scarica storico daily da Yahoo Finance per `ticker`.
    Fa più tentativi di lookback, restituisce DataFrame index datetime
    con colonna 'Close', oppure None se proprio non trova prezzi.

    - Primo tentativo: ~1 anno di dati
    - Secondo tentativo: fino a ~10 anni
    """
    end = datetime.utcnow().date()
    start_candidates = [
        end - timedelta(days=min_history_days),
        end - timedelta(days=365 * max_lookback_years),
    ]

    for start in start_candidates:
        try:
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=True,
            )
        except Exception:
            df = None

        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            out = df[["Close"]].copy()
            out = out.dropna()
            if not out.empty:
                # Normalizza in modo che l'indice sia datetime.date coerente
                out.index = pd.to_datetime(out.index).tz_localize(None)
                out = out.sort_index()
                return out

    # Se nessun tentativo ha prodotto dati validi:
    return None


# ----------------------------------------------------------------------
# 3. Costruzione della curva di portafoglio nel tempo
# ----------------------------------------------------------------------

def build_portfolio_nav(lots_df: pd.DataFrame,
                        price_cache: dict):
    """
    Dato l'elenco dei lotti e un cache prezzi per ciascun ticker,
    calcola la curva di valore del portafoglio (NAV) giorno per giorno.

    price_cache: dict { ticker: DataFrame( index=datetime, Close=float ) }

    Ritorna:
    - nav_df: DataFrame con colonne ['portfolio_value'] indicizzato per data (datetime)
    """

    if lots_df.empty:
        # se non hai lotti, ritorna vuoto
        return pd.DataFrame(columns=["portfolio_value"])

    # 1. costruiamo la serie delle posizioni nel tempo
    #    per semplicità: assumiamo buy&hold (solo acquisti), nessuna vendita
    #    quantity cumulata per ticker nel tempo
    nav_by_date = {}

    # timeline globale = unione di tutte le date prezzo di tutti i ticker
    all_dates = set()
    for tkr in lots_df["ticker"].unique():
        px = price_cache.get(tkr)
        if px is not None and not px.empty:
            for d in px.index:
                all_dates.add(d)

    if not all_dates:
        return pd.DataFrame(columns=["portfolio_value"])

    all_dates = sorted(all_dates)

    # per velocità, prepariamo le cumulative quantities per ticker
    # idea: per ogni ticker, al giorno D, qty held = somma qty di tutte le operazioni con date <= D
    ticker_lots = {
        tkr: lots_df[lots_df["ticker"] == tkr].sort_values("date")
        for tkr in lots_df["ticker"].unique()
    }

    # calcolo NAV giornaliero
    for d in all_dates:
        total_val = 0.0
        for tkr, tdf in ticker_lots.items():
            # qty cumulata fino a quel giorno
            qty_held = tdf.loc[tdf["date"] <= d.date(), "qty"].sum()
            if qty_held <= 0:
                continue
            px = price_cache.get(tkr)
            if px is None or px.empty:
                continue
            # prezzo close del giorno d (se manca quel giorno, prendo ultimo disponibile <= d)
            # forward-fill all'interno px
            prc = px.loc[px.index <= d, "Close"]
            if prc.empty:
                continue
            last_price = prc.iloc[-1]
            total_val += qty_held * last_price

        nav_by_date[d] = total_val

    nav_df = (
        pd.Series(nav_by_date)
        .sort_index()
        .to_frame(name="portfolio_value")
    )
    return nav_df


# ----------------------------------------------------------------------
# 4. Normalizza due curve per confronto (base 100)
# ----------------------------------------------------------------------

def base_100(series: pd.Series):
    """
    Porta la serie (tipicamente NAV o Close benchmark) a base 100
    sul primo valore non-NaN.
    """
    s = series.dropna().astype(float)
    if s.empty:
        return s
    first = s.iloc[0]
    if first == 0:
        return s * 0.0
    return s / first * 100.0


# ----------------------------------------------------------------------
# 5. Rischio, Sharpe, VaR semplici
# ----------------------------------------------------------------------

def compute_risk_metrics(nav_series: pd.Series,
                         bench_series: pd.Series):
    """
    Calcola alcune metriche di rischio base.
    nav_series e bench_series sono serie base 100 (float).
    """

    out = {
        "volatility_ann_pct": None,
        "sharpe_12m": None,
        "var_1d_pct": None,
        "tracking_error_ann_pct": None,
    }

    # rendimenti giornalieri %
    rets = nav_series.pct_change().dropna()
    if len(rets) > 2:
        # volatilità ann (std giornaliera * sqrt(252))
        vol_ann = float(rets.std() * math.sqrt(252))  # es. 0.15 = 15%
        out["volatility_ann_pct"] = vol_ann * 100.0

        # VaR 1-day 95% (quantile 5%)
        var_1d = float(np.percentile(rets, 5))
        out["var_1d_pct"] = var_1d * 100.0

    # Sharpe 12m (molto semplificato):
    # - calcoliamo rendimento medio giornaliero *252 / vol_ann
    # - risk-free: fallback 4% annuo (0.04)
    if len(rets) > 10:
        avg_daily = float(rets.mean())
        rf_ann = 0.04  # fallback default
        rf_daily = (1.0 + rf_ann) ** (1.0/252.0) - 1.0
        excess_daily = avg_daily - rf_daily
        vol_daily = float(rets.std())
        if vol_daily > 0:
            sharpe = (excess_daily * 252.0) / (vol_daily * math.sqrt(252.0))
            out["sharpe_12m"] = sharpe

    # Tracking error: std dei residui vs benchmark
    if bench_series is not None and not bench_series.dropna().empty:
        bench_rets = bench_series.pct_change().dropna()
        # allinea gli indici
        comb = pd.concat([rets, bench_rets], axis=1).dropna()
        comb.columns = ["p", "b"]
        if len(comb) > 2:
            diff = comb["p"] - comb["b"]
            te_ann = float(diff.std() * math.sqrt(252))
            out["tracking_error_ann_pct"] = te_ann * 100.0

    return out


# ----------------------------------------------------------------------
# 6. PME / IRR placeholder
# ----------------------------------------------------------------------

def compute_pme_metrics(nav_series: pd.Series,
                        bench_series: pd.Series):
    """
    Placeholder PME/IRR.
    Nel tuo codice originale probabilmente calcoli IRR del portafoglio,
    PME (Public Market Equivalent), ecc.
    Qui ritorno qualche stub coerente per non rompere il JSON.

    Sostituisci liberamente con la tua funzione reale.
    """
    out = {
        "irr_portfolio": None,
        "irr_benchmark": None,
        "pme_ratio": None,
    }

    # Se hai già funzioni tipo calc_irr(...) nel tuo codice originale,
    # puoi chiamarle qui usando i flussi di cassa reali.

    return out


# ----------------------------------------------------------------------
# 7. Allocazioni (settori / paesi) placeholder
# ----------------------------------------------------------------------

def compute_allocations(lots_df: pd.DataFrame):
    """
    Qui potresti calcolare allocazioni per settore / paese se hai
    già un mapping ticker -> breakdown.
    Per ora restituisco strutture vuote ma consistenti.
    """
    return {
        "portfolio_sectors": [],
        "portfolio_countries": [],
    }


# ----------------------------------------------------------------------
# 8. Funzione principale chiamata dal server
# ----------------------------------------------------------------------

def analyze_portfolio_from_text(lots_text: str,
                                bench: str,
                                outdir: str):
    """
    Entry point usata da portfolio_core.run_full_analysis()

    - parse lotti
    - scarica prezzi di ogni ticker
    - scarica benchmark con retry robusto
    - costruisce curve base100
    - calcola metriche rischio/performance
    - salva grafico PNG in outdir
    - ritorna dict pronto per il JSON
    """

    # -------------------------------------------------
    # parse input
    lots_df = parse_lots_text(lots_text)
