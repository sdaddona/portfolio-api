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
                                outdir: str = "/tmp/outputs",
                                start_buffer_days: int = 7):
    """
    Passi:
    - parse lotti
    - scarica prezzi robusti per TUTTI i ticker + benchmark
    - costruisce NAV portafoglio, TWR base 100
    - costruisce benchmark base 100 (con fallback)
    - calcola Sharpe, VaR, PME, IRR
    - salva grafico
    - ritorna dict API-friendly
    """

    os.makedirs(outdir, exist_ok=True)

    # 1. parse input lotti
    lots = read_lots_from_text(lots_text)
    first_tx_date = lots["data"].min()

    tickers_portafoglio = sorted(set(lots["ticker"].tolist()))
    all_tickers = sorted(set(tickers_portafoglio + [bench]))

    # 2. scarica prezzi robusti
    price_series = {}
    for sym in all_tickers:
        s = robust_fetch_prices_for_symbol(sym, first_tx_date, start_buffer_days)
        if s is not None and not s.empty:
            # s è una Series con index datetime e valori Close
            price_series[sym] = s.astype(float)
        time.sleep(0.05)

    # retry esplicito sul benchmark se mancante
    if bench not in price_series or price_series[bench].empty:
        s_retry = robust_fetch_prices_for_symbol(
            bench,
            first_tx_date - timedelta(days=365),
            start_buffer_days,
        )
        if s_retry is not None and not s_retry.empty:
            price_series[bench] = s_retry.astype(float)

    # se è ancora mancante il benchmark, marchiamo fallback
    benchmark_is_fallback = False
    if bench not in price_series or price_series[bench].empty:
        benchmark_is_fallback = True

    # controllo duro: se non ho nemmeno un ticker con dati -> errore esplicito leggibile
    if len(price_series) == 0:
        raise RuntimeError(
            f"Nessun dato prezzi scaricato da Yahoo Finance. Tickers richiesti: {all_tickers}"
        )

    # 3. costruisci dataframe prezzi
    #    (solo serie effettivamente presenti, e tutte castate a float)
    px_list = [ser for ser in price_series.values() if ser is not None and not ser.empty]
    if len(px_list) == 0:
        raise RuntimeError(
            f"Nessuna serie valida dopo il download. Tickers richiesti: {all_tickers}"
        )

    px = pd.concat(px_list, axis=1).sort_index().ffill()
    px = to_naive(px).asfreq("B").ffill()

    # filtra date dalla prima operazione
    px = px.loc[px.index >= (first_tx_date - timedelta(days=start_buffer_days))]

    # 4. se il benchmark era mancante, fallback al portafoglio stesso
    if benchmark_is_fallback:
        # NOTE: non abbiamo ancora calcolato port_val, quindi per ora segnamo
        # che useremo portafoglio come bench dopo aver calcolato port_val.
        bench_fallback_flag = True
    else:
        bench_fallback_flag = False

    # 5. ricostruisci shares giornaliere
    present = sorted(set(px.columns) & set(tickers_portafoglio))
    shares = pd.DataFrame(0.0, index=px.index, columns=present)

    trades = []
    for _, r in lots.iterrows():
        sym = r["ticker"]
        d = r["data"]
        qty = float(r["quantità"])
        px_file = float(r["prezzo"])
        # allinea la data di acquisto alla prima data di mercato >= d
        # se la data d non esiste nell'indice px, mappiamo alla successiva disponibile
        pos = px.index.searchsorted(d, side="left")
        if pos >= len(px.index):
            # nessuna data di mercato >= d -> ignoriamo questa trade perché fuori range
            continue
        d_eff = px.index[pos]

        trades.append((sym, d, d_eff, qty, px_file))

        if sym in shares.columns:
            # aggiungiamo la quantità a partire da d_eff in avanti
            shares.iloc[pos:, shares.columns.get_loc(sym)] += qty

    # 6. valore portafoglio giornaliero
    port_val = (shares * px[present]).sum(axis=1)

    # tieni solo da quando il portafoglio ha valore > 0
    first_mv_mask = port_val > 0
    if not first_mv_mask.any():
        raise RuntimeError("Portafoglio senza valore positivo (controlla i lotti).")

    first_mv_date = first_mv_mask.idxmax()
    port_val = port_val.loc[first_mv_date:].dropna()

    # 7. benchmark value series
    if bench_fallback_flag:
        bench_val_raw = port_val.copy()
        bench_name = "PORTFOLIO"
    else:
        if bench not in px.columns:
            # se siamo qui bench non è fallback ma non è in px? edge case
            # => usiamo fallback comunque
            bench_val_raw = port_val.copy()
            bench_name = "PORTFOLIO"
        else:
            bench_val_raw = px[bench].dropna()
            bench_name = bench

    # allinea su intersezione date
    idx_common = port_val.index.intersection(bench_val_raw.index)
    port_val = port_val.loc[idx_common]
    bench_val = bench_val_raw.loc[idx_common]

    # controllo duro: se dopo l'allineamento non resta nulla -> errore chiaro
    if len(port_val) == 0 or len(bench_val) == 0:
        raise RuntimeError("Serie portafoglio/benchmark vuote dopo allineamento date.")

    # 8. cash flow giornaliero (positivo = contribuzione)
    cf = pd.Series(0.0, index=port_val.index)
    for sym, d0, d_eff, qty, px_file in trades:
        if d_eff in cf.index:
            cf.loc[d_eff] += qty * px_file

    # 9. TWR portafoglio base 100
    dates = port_val.index
    twr_chunks = []
    for i in range(1, len(dates)):
        t, tm = dates[i], dates[i - 1]
        mv_t, mv_tm = float(port_val.loc[t]), float(port_val.loc[tm])
        cf_t = float(cf.loc[t])
        if mv_tm > 0:
            twr_chunks.append((mv_t - cf_t) / mv_tm - 1)
        else:
            twr_chunks.append(0.0)

    twr_ret = pd.Series(twr_chunks, index=dates[1:])
    port_idx = (1 + twr_ret).cumprod() * 100.0
    port_idx = pd.concat([pd.Series([100.0], index=dates[:1]), port_idx])

    # 10. benchmark base 100
    bench_ret = bench_val.pct_change()
    bench_idx = (1 + bench_ret.iloc[1:]).cumprod() * 100.0
    bench_idx = pd.concat([pd.Series([100.0], index=bench_val.index[:1]), bench_idx])

    # 11. risk-free daily
    rf_daily, rf_meta = build_rf_daily_series_web(twr_ret.index)

    # 12. PME (replica cash flow sul benchmark)
    bench_pme_val = []
    units = 0.0
    for t in port_val.index:
        px_b = float(bench_val.loc[t])
        invest_today = float(cf.loc[t])
        if px_b != 0:
            units += invest_today / px_b
        bench_pme_val.append(units * px_b)
    bench_pme_val = pd.Series(bench_pme_val, index=port_val.index)

    # 13. rischio 12m / Sharpe / VaR
    lb = TRADING_DAYS

    port_r_12m = (twr_ret.iloc[-lb:] if len(twr_ret) >= lb else twr_ret.dropna()).copy()
    rf_12m = rf_daily.reindex(port_r_12m.index).ffill().fillna(0.0)

    vol_port_12m = np.nan if port_r_12m.empty else float(port_r_12m.std(ddof=1) * np.sqrt(TRADING_DAYS))

    sharpe_port_12m = np.nan
    if (not port_r_12m.empty) and (port_r_12m.std(ddof=1) > 0):
        ex = port_r_12m - rf_12m
        sharpe_port_12m = float(np.sqrt(TRADING_DAYS) * ex.mean() / ex.std(ddof=1))

    z_95 = 1.65
    sigma_1d = float(port_r_12m.std(ddof=1)) if not port_r_12m.empty else np.nan
    var95_pct = np.nan if np.isnan(sigma_1d) else z_95 * sigma_1d
    current_value_port = float(port_val.iloc[-1])
    var95_usd = np.nan if np.isnan(var95_pct) else var95_pct * current_value_port

    bench_r_12m = (bench_ret.iloc[-lb:] if len(bench_ret) >= lb else bench_ret.dropna()).copy()
    rf_12m_b = rf_daily.reindex(bench_r_12m.index).ffill().fillna(0.0)

    vol_bench_12m = np.nan if bench_r_12m.empty else float(bench_r_12m.std(ddof=1) * np.sqrt(TRADING_DAYS))

    sharpe_bench_12m = np.nan
    if (not bench_r_12m.empty) and (bench_r_12m.std(ddof=1) > 0):
        exb = bench_r_12m - rf_12m_b
        sharpe_bench_12m = float(np.sqrt(TRADING_DAYS) * exb.mean() / exb.std(ddof=1))

    sigma_1d_b = float(bench_r_12m.std(ddof=1)) if not bench_r_12m.empty else np.nan
    var95_bench_pct = np.nan if np.isnan(sigma_1d_b) else z_95 * sigma_1d_b
    current_value_bench_pme = float(bench_pme_val.iloc[-1])
    var95_bench_usd = np.nan if np.isnan(var95_bench_pct) else var95_bench_pct * current_value_bench_pme

    # 14. summary IRR / PME
    contrib = cf.clip(upper=0) * -1.0   # soldi entrati (acquisti)
    withdrw = cf.clip(lower=0)          # soldi usciti (vendite)
    gross_contrib = float(contrib.sum())
    gross_withdrw = float(withdrw.sum())
    net_invested = float(cf.sum())

    current_value_port_val = current_value_port
    current_value_pme = float(bench_pme_val.iloc[-1])

    r_net_port = (current_value_port_val / net_invested - 1) if net_invested > 0 else np.nan
    r_net_bench = (current_value_pme / net_invested - 1) if net_invested > 0 else np.nan
    r_net_excess = (
        r_net_port - r_net_bench
        if (np.isfinite(r_net_port) and np.isfinite(r_net_bench))
        else np.nan
    )

    cf_list = [
        (t, -float(cf.loc[t]))
        for t in port_val.index
        if abs(float(cf.loc[t])) != 0.0
    ]
    cf_port = sorted(
        cf_list + [(port_val.index[-1], current_value_port_val)],
        key=lambda x: x[0]
    )
    cf_bench = sorted(
        cf_list + [(port_val.index[-1], current_value_pme)],
        key=lambda x: x[0]
    )

    irr_port = xirr(cf_port)
    irr_bench = xirr(cf_bench)
    irr_excess = (
        irr_port - irr_bench
        if (np.isfinite(irr_port) and np.isfinite(irr_bench))
        else np.nan
    )

    # 15. grafico cumulato
    plot_path = os.path.join(outdir, "crescita_cumulata.png")
    plt.figure(figsize=(10, 6))
    plt.plot(port_idx, label="Portafoglio (TWR base 100)")
    plt.plot(bench_idx, label=f"Benchmark {bench_name} (base 100)")
    plt.title(f"Andamento storico (base 100)\n{rf_meta}")
    plt.ylabel("Indice (base 100)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    # 16. allocazioni (stub)
    sectors_p, countries_p = aggregate_allocations_portfolio(
        shares_df=shares[present],
        prices_df=px,
        tickers=present,
        bench=bench_name,
    )

    bench_sectors, bench_countries = get_etf_allocations(bench_name)
    if not bench_sectors.empty:
        bench_sectors = bench_sectors.rename(columns={"weight":"weight_portfolio"}).copy()
    else:
        bench_sectors = pd.DataFrame(columns=["label","weight_portfolio"])
    if not bench_countries.empty:
        bench_countries = bench_countries.rename(columns={"weight":"weight_portfolio"}).copy()
    else:
        bench_countries = pd.DataFrame(columns=["label","weight_portfolio"])

    # 17. summary_lines finale
    summary_lines = []
    summary_lines.append("=== SUMMARY (Benchmark PME-consistent) ===")
    rows_out = [
        ("Start Date", str(port_val.index[0].date())),
        ("End Date", str(port_val.index[-1].date())),
        ("Return (Net Invested) – Portfolio", fmt_pct(r_net_port)),
        ("Return (Net Invested) – Bench (PME)", fmt_pct(r_net_bench)),
        ("Excess Return (Net)", fmt_pct(r_net_excess)),
        ("IRR – Portfolio", fmt_pct(irr_port) if np.isfinite(irr_port) else "n/a"),
        ("IRR – Bench (PME)", fmt_pct(irr_bench) if np.isfinite(irr_bench) else "n/a"),
        ("Excess IRR", fmt_pct(irr_excess) if np.isfinite(irr_excess) else "n/a"),
        ("Contributions (Gross)", money(gross_contrib)),
        ("Withdrawals (Gross)", money(gross_withdrw)),
        ("Net Invested", money(net_invested)),
        ("Current Value – Portfolio", money(current_value_port_val)),
        ("Current Value – Bench (PME)", money(current_value_pme)),
        ("Volatility 12m (ann.) – Portfolio", fmt_pct(vol_port_12m)),
        ("Volatility 12m (ann.) – Benchmark", fmt_pct(vol_bench_12m)),
        (f"Sharpe 12m – Portfolio [{rf_meta}]", f"{sharpe_port_12m:.2f}" if np.isfinite(sharpe_port_12m) else "n/a"),
        (f"Sharpe 12m – Benchmark [{rf_meta}]", f"{sharpe_bench_12m:.2f}" if np.isfinite(sharpe_bench_12m) else "n/a"),
        ("1D VaR(95%) – Portfolio (pct)", fmt_pct(var95_pct)),
        ("1D VaR(95%) – Portfolio (USD)", money(var95_usd) if np.isfinite(var95_usd) else "n/a"),
        ("1D VaR(95%) – Benchmark (pct)", fmt_pct(var95_bench_pct)),
        ("1D VaR(95%) – Benchmark (USD)", money(var95_bench_usd) if np.isfinite(var95_bench_usd) else "n/a"),
    ]
    for k, v in rows_out:
        summary_lines.append(f"{k.ljust(45)} {v}")

    # 18. salva summary.txt
    sum_path = os.path.join(outdir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # 19. ritorno finale
    out = {
        "summary_lines": summary_lines,
        "plot_path": plot_path,
        "summary_path": sum_path,
        "risk": {
            "vol_port_12m": vol_port_12m,
            "vol_bench_12m": vol_bench_12m,
            "sharpe_port_12m": sharpe_port_12m,
            "sharpe_bench_12m": sharpe_bench_12m,
            "var95_pct": var95_pct,
            "var95_usd": var95_usd,
            "var95_bench_pct": var95_bench_pct,
            "var95_bench_usd": var95_bench_usd,
        },
        "pme": {
            "r_net_port": r_net_port,
            "r_net_bench": r_net_bench,
            "r_net_excess": r_net_excess,
            "irr_port": irr_port,
            "irr_bench": irr_bench,
            "irr_excess": irr_excess,
            "current_value_port": current_value_port_val,
            "current_value_pme": current_value_pme,
            "net_invested": net_invested,
        },
        "alloc": {
            "portfolio_sectors": sectors_p.to_dict(orient="records"),
            "portfolio_countries": countries_p.to_dict(orient="records"),
            "bench_sectors": bench_sectors.to_dict(orient="records"),
            "bench_countries": bench_countries.to_dict(orient="records"),
        },
    }

    return out
