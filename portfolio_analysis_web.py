import os
import re
import time
import math
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt

TRADING_DAYS = 252


###############################################################################
# Utils base
###############################################################################

def to_naive(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)) and isinstance(obj.index, pd.DatetimeIndex):
        try:
            obj.index = obj.index.tz_localize(None)
        except Exception:
            pass
    return obj

def to_date(s: str) -> pd.Timestamp:
    s = str(s).strip()
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    if pd.isna(dt):
        raise ValueError(f"Data non riconosciuta: {s}")
    try:
        return dt.tz_localize(None)
    except Exception:
        return dt

def money(x: float) -> str:
    return f"{x:,.2f} USD"

def fmt_pct(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return "n/a"
    return f"{x:.2%}"

def xnpv(r, cf):
    t0 = cf[0][0]
    return sum(a / (1 + r) ** ((t - t0).days / 365.25) for t, a in cf)

def xirr(cf):
    has_pos = any(a > 0 for _, a in cf)
    has_neg = any(a < 0 for _, a in cf)
    if not (has_pos and has_neg):
        return np.nan
    lo, hi = -0.999, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        val = xnpv(mid, cf)
        if abs(val) < 1e-8:
            return mid
        if xnpv(lo, cf) * val <= 0:
            hi = mid
        else:
            lo = mid
    return mid


###############################################################################
# Lettura lotti dalla textarea
###############################################################################

def read_lots_from_text(txt: str) -> pd.DataFrame:
    """
    Formato righe:
      TICKER DATA QUANTITA PREZZO
      VT 2024-01-02 10 100
      VT 2024-03-10 5 95
    """
    lines = []
    for ln in txt.splitlines():
        ln_strip = ln.strip()
        if not ln_strip or ln_strip.startswith("#"):
            continue
        lines.append(ln_strip)
    if not lines:
        raise ValueError("Nessuna riga valida nei lotti.")

    rows = []
    for ln in lines:
        parts = re.split(r"[,\t;]+|\s+", ln.strip())
        if len(parts) < 4:
            raise ValueError(f"Riga lotti invalida: {ln}")
        ticker = parts[0].upper()
        data = to_date(parts[1])
        qty = float(str(parts[2]).replace(",", "."))
        px = float(str(parts[3]).replace(",", "."))
        rows.append((ticker, data, qty, px))

    df = pd.DataFrame(rows, columns=["ticker","data","quantità","prezzo"])
    df = df.sort_values("data").reset_index(drop=True)
    return df


###############################################################################
# Risk-free da FRED API (con fallback fisso)
###############################################################################

def fetch_fred_series(series_id: str,
                      start_date: pd.Timestamp,
                      end_date: pd.Timestamp,
                      api_key: str | None) -> pd.Series:
    """
    DGS1 = 1-Year Treasury Yield (% annuo). Ritorna Series con index=Data, valori=float(%).
    Se fallisce -> Series vuota.
    """
    if api_key is None or api_key.strip() == "":
        return pd.Series(dtype=float)

    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date.strftime("%Y-%m-%d"),
        "observation_end": end_date.strftime("%Y-%m-%d"),
    }

    try:
        r = requests.get(base_url, params=params, timeout=10)
        if r.status_code != 200:
            return pd.Series(dtype=float)
        data = r.json().get("observations", [])
    except Exception:
        return pd.Series(dtype=float)

    dates = []
    vals = []
    for obs in data:
        dstr = obs.get("date")
        vstr = obs.get("value")
        if vstr in (None, ".", ""):
            continue
        try:
            d = pd.to_datetime(dstr).tz_localize(None)
            v = float(vstr)
        except Exception:
            continue
        dates.append(d)
        vals.append(v)

    if not dates:
        return pd.Series(dtype=float)

    ser = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()
    return ser

def build_rf_daily_series_web(index: pd.DatetimeIndex,
                              fallback_annual_rf: float = 0.04) -> tuple[pd.Series, str]:
    """
    Restituisce:
    - rf_daily: Serie del tasso risk-free giornaliero (frazione tipo 0.0001)
    - rf_meta: stringa esplicativa
    """
    api_key = os.getenv("FRED_API_KEY", "").strip()
    start_fetch = index[0] - pd.Timedelta(days=10)
    end_fetch   = index[-1] + pd.Timedelta(days=3)

    dgs1 = fetch_fred_series("DGS1", start_fetch, end_fetch, api_key)
    if not dgs1.empty:
        dgs1 = dgs1.reindex(index).ffill()
        rf_annual = dgs1 / 100.0
        rf_daily = rf_annual / TRADING_DAYS
        rf_meta = "RF: FRED DGS1 (1Y Treasury, ann.)"
        return rf_daily, rf_meta

    rf_daily = pd.Series(fallback_annual_rf / TRADING_DAYS, index=index)
    rf_meta = f"Fixed RF {fallback_annual_rf:.2%} (ann., fallback)"
    return rf_daily, rf_meta


###############################################################################
# Allocazioni ETF (stub, al momento vuoto su Render)
###############################################################################

def get_etf_allocations(ticker: str):
    return (
        pd.DataFrame(columns=["label","weight"]),
        pd.DataFrame(columns=["label","weight"]),
    )

def aggregate_allocations_portfolio(shares_df: pd.DataFrame,
                                    prices_df: pd.DataFrame,
                                    tickers: list[str],
                                    bench: str):
    end_date = shares_df.index[-1]
    mv = shares_df.loc[end_date] * prices_df.loc[end_date, shares_df.columns]
    w_etf = (mv / mv.sum()).fillna(0.0)
    w_etf = w_etf[w_etf > 0]

    sector_rows, country_rows = [], []

    for etf, w in w_etf.items():
        s_df, c_df = get_etf_allocations(etf)
        if not s_df.empty:
            tmp = s_df.copy()
            tmp["weight_portfolio"] = w * tmp["weight"]
            tmp["etf"] = etf
            sector_rows.append(tmp[["etf", "label", "weight_portfolio"]])
        if not c_df.empty:
            tmpc = c_df.copy()
            tmpc["weight_portfolio"] = w * tmpc["weight"]
            tmpc["etf"] = etf
            country_rows.append(tmpc[["etf", "label", "weight_portfolio"]])
        time.sleep(0.05)

    if sector_rows:
        sectors = (
            pd.concat(sector_rows)
              .groupby("label", as_index=False)["weight_portfolio"].sum()
              .sort_values("weight_portfolio", ascending=False)
              .reset_index(drop=True)
        )
    else:
        sectors = pd.DataFrame(columns=["label","weight_portfolio"])

    if country_rows:
        countries = (
            pd.concat(country_rows)
              .groupby("label", as_index=False)["weight_portfolio"].sum()
              .sort_values("weight_portfolio", ascending=False)
              .reset_index(drop=True)
        )
    else:
        countries = pd.DataFrame(columns=["label","weight_portfolio"])

    def norm(df):
        if df.empty:
            return df
        tot = df["weight_portfolio"].sum()
        if tot > 0:
            df = df.copy()
            df["weight_portfolio"] = df["weight_portfolio"] / tot
        return df

    return norm(sectors), norm(countries)


###############################################################################
# Download prezzi da Alpha Vantage
###############################################################################

def download_price_history_av(ticker: str,
                              start: pd.Timestamp,
                              end: pd.Timestamp) -> pd.Series | None:
    """
    Chiede a Alpha Vantage TIME_SERIES_DAILY_ADJUSTED.
    Ritorna una Series 'Close' (in realtà 'adjusted close') indicizzata per data,
    oppure None se fallisce / vuota.
    """
    api_key = os.getenv("ALPHAVANTAGE_KEY", "").strip()
    if api_key == "":
        return None  # se qualcuno lancia senza chiave in locale

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",   # prendiamo storico lungo
        "apikey": api_key,
    }

    try:
        r = requests.get(url, params=params, timeout=10)
    except Exception:
        return None

    if r.status_code != 200:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    # struttura attesa: data["Time Series (Daily)"] = { "2025-10-24": {...}, ... }
    ts = None
    for k in data.keys():
        if "Time Series" in k:
            ts = data[k]
            break
    if ts is None or not isinstance(ts, dict) or len(ts) == 0:
        # può essere rate limit ("Thank you for using Alpha Vantage...") oppure ticker sconosciuto
        return None

    # Costruiamo dataframe da ts
    # ts[date_str]["5. adjusted close"] tipicamente
    dates = []
    closes = []
    for dstr, row in ts.items():
        # dstr tipo "2025-10-24"
        try:
            d = pd.to_datetime(dstr).tz_localize(None)
        except Exception:
            continue
        # filtra per range richiesto
        if d < start - timedelta(days=7):
            # teniamo comunque un po' di buffer in dietro
            pass
        if d > (end + timedelta(days=2)):
            continue

        # preferiamo adjusted close
        # Alpha Vantage usa chiavi tipo "5. adjusted close"
        adj_close = None
        if isinstance(row, dict):
            # prova varie chiavi possibili
            for keycand in [
                "5. adjusted close",
                "adjusted close",
                "4. close",
                "close"
            ]:
                if keycand in row:
                    try:
                        adj_close = float(row[keycand])
                        break
                    except Exception:
                        pass
        if adj_close is None:
            continue

        dates.append(d)
        closes.append(adj_close)

    if not dates:
        return None

    s = pd.Series(closes, index=pd.DatetimeIndex(dates)).sort_index().ffill()
    s = to_naive(s)
    return s


def robust_fetch_prices_for_symbol(sym: str,
                                   first_tx_date: pd.Timestamp,
                                   start_buffer_days: int = 7) -> pd.Series | None:
    """
    Tenta di ottenere la serie prezzi per un ticker usando Alpha Vantage.
    Prova:
    - sym
    - sym.US
    - sym.L
    - sym.MI
    Restituisce una Series con index datetime e valori float.
    """
    candidates = [sym, f"{sym}.US", f"{sym}.L", f"{sym}.MI"]
    # dedup mantenendo ordine
    candidates = list(dict.fromkeys(candidates))

    starts = [
        first_tx_date - timedelta(days=start_buffer_days + 365),
        first_tx_date - timedelta(days=start_buffer_days + 365*10),
    ]
    end = pd.Timestamp(datetime.today().date() + timedelta(days=2))

    for ticker_candidate in candidates:
        for st in starts:
            s = download_price_history_av(ticker_candidate, st, end)
            # rate limiting Alpha Vantage: facciamo una piccola pausa
            time.sleep(1.2)
            if s is not None and not s.empty:
                # rinominiamo al ticker logico (quello dell'utente)
                return s.rename(sym)

    return None


###############################################################################
# Funzione principale di analisi
###############################################################################

def analyze_portfolio_from_text(lots_text: str,
                                bench: str,
                                outdir: str = "/tmp/outputs",
                                start_buffer_days: int = 7):
    """
    Flusso:
    - parse lotti
    - scarica prezzi via Alpha Vantage
    - calcola NAV portafoglio, benchmark, TWR, Sharpe, VaR, PME, IRR
    - genera grafico e summary
    - ritorna dict serializzabile
    """

    os.makedirs(outdir, exist_ok=True)

    # 1. parse input lotti
    lots = read_lots_from_text(lots_text)
    first_tx_date = lots["data"].min()

    tickers_portafoglio = sorted(set(lots["ticker"].tolist()))
    all_tickers = sorted(set(tickers_portafoglio + [bench]))

    # 2. scarica prezzi con robust_fetch_prices_for_symbol (Alpha Vantage based)
    price_series = {}
    for sym in all_tickers:
        s = robust_fetch_prices_for_symbol(sym, first_tx_date, start_buffer_days)
        if s is not None and not s.empty:
            price_series[sym] = s.astype(float)

    # prova una volta in più il benchmark se mancante
    if bench not in price_series or price_series[bench].empty:
        s_retry = robust_fetch_prices_for_symbol(
            bench,
            first_tx_date - timedelta(days=365),
            start_buffer_days,
        )
        if s_retry is not None and not s_retry.empty:
            price_series[bench] = s_retry.astype(float)

    benchmark_is_fallback = False
    if bench not in price_series or price_series[bench].empty:
        benchmark_is_fallback = True

    # sanity check sui dati scaricati
    if len(price_series) == 0:
        raise RuntimeError(
            f"Nessun dato prezzi disponibile per i ticker richiesti: {all_tickers}. "
            f"Potrebbe essere rate limit Alpha Vantage o ticker sconosciuto."
        )

    px_list = [
        ser for ser in price_series.values()
        if ser is not None and not ser.empty
    ]
    if len(px_list) == 0:
        raise RuntimeError(
            f"Nessuna serie valida dopo il download. "
            f"Tickers richiesti: {all_tickers}"
        )

    px = pd.concat(px_list, axis=1).sort_index().ffill()
    px = to_naive(px).asfreq("B").ffill()

    px = px.loc[px.index >= (first_tx_date - timedelta(days=start_buffer_days))]

    # 3. ricostruisci le shares giornaliere
    present = sorted(set(px.columns) & set(tickers_portafoglio))
    shares = pd.DataFrame(0.0, index=px.index, columns=present)

    trades = []
    for _, r in lots.iterrows():
        sym = r["ticker"]
        d = r["data"]
        qty = float(r["quantità"])
        px_file = float(r["prezzo"])

        pos = px.index.searchsorted(d, side="left")
        if pos >= len(px.index):
            continue
        d_eff = px.index[pos]

        trades.append((sym, d, d_eff, qty, px_file))

        if sym in shares.columns:
            shares.iloc[pos:, shares.columns.get_loc(sym)] += qty

    port_val = (shares * px[present]).sum(axis=1)

    first_mv_mask = port_val > 0
    if not first_mv_mask.any():
        raise RuntimeError("Portafoglio senza valore positivo (controlla i lotti).")

    first_mv_date = first_mv_mask.idxmax()
    port_val = port_val.loc[first_mv_date:].dropna()

    # 4. costruiamo la serie benchmark
    if benchmark_is_fallback:
        bench_val_raw = port_val.copy()
        bench_name = "PORTFOLIO"
    else:
        if bench not in px.columns:
            bench_val_raw = port_val.copy()
            bench_name = "PORTFOLIO"
        else:
            bench_val_raw = px[bench].dropna()
            bench_name = bench

    idx_common = port_val.index.intersection(bench_val_raw.index)
    port_val = port_val.loc[idx_common]
    bench_val = bench_val_raw.loc[idx_common]

    if len(port_val) == 0 or len(bench_val) == 0:
        raise RuntimeError("Serie portafoglio/benchmark vuote dopo allineamento date.")

    # 5. cash flow giornaliero (positivo = contributo / acquisto)
    cf = pd.Series(0.0, index=port_val.index)
    for sym, d0, d_eff, qty, px_file in trades:
        if d_eff in cf.index:
            cf.loc[d_eff] += qty * px_file

    # 6. TWR del portafoglio base 100
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

    bench_ret = bench_val.pct_change()
    bench_idx = (1 + bench_ret.iloc[1:]).cumprod() * 100.0
    bench_idx = pd.concat([pd.Series([100.0], index=bench_val.index[:1]), bench_idx])

    # 7. risk-free daily
    rf_daily, rf_meta = build_rf_daily_series_web(twr_ret.index)

    # 8. PME (replica i cash flow sul benchmark)
    bench_pme_val = []
    units = 0.0
    for t in port_val.index:
        px_b = float(bench_val.loc[t])
        invest_today = float(cf.loc[t])
        if px_b != 0:
            units += invest_today / px_b
        bench_pme_val.append(units * px_b)
    bench_pme_val = pd.Series(bench_pme_val, index=port_val.index)

    # 9. rischio 12m / Sharpe / VaR
    lb = TRADING_DAYS

    port_r_12m = (twr_ret.iloc[-lb:] if len(twr_ret) >= lb else twr_ret.dropna()).copy()
    rf_12m = rf_daily.reindex(port_r_12m.index).ffill().fillna(0.0)

    vol_port_12m = np.nan if port_r_12m.empty else float(
        port_r_12m.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    sharpe_port_12m = np.nan
    if (not port_r_12m.empty) and (port_r_12m.std(ddof=1) > 0):
        ex = port_r_12m - rf_12m
        sharpe_port_12m = float(
            np.sqrt(TRADING_DAYS) * ex.mean() / ex.std(ddof=1)
        )

    z_95 = 1.65
    sigma_1d = float(port_r_12m.std(ddof=1)) if not port_r_12m.empty else np.nan
    var95_pct = np.nan if np.isnan(sigma_1d) else z_95 * sigma_1d
    current_value_port = float(port_val.iloc[-1])
    var95_usd = np.nan if np.isnan(var95_pct) else var95_pct * current_value_port

    bench_r_12m = (bench_ret.iloc[-lb:] if len(bench_ret) >= lb else bench_ret.dropna()).copy()
    rf_12m_b = rf_daily.reindex(bench_r_12m.index).ffill().fillna(0.0)

    vol_bench_12m = np.nan if bench_r_12m.empty else float(
        bench_r_12m.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    sharpe_bench_12m = np.nan
    if (not bench_r_12m.empty) and (bench_r_12m.std(ddof=1) > 0):
        exb = bench_r_12m - rf_12m_b
        sharpe_bench_12m = float(
            np.sqrt(TRADING_DAYS) * exb.mean() / exb.std(ddof=1)
        )

    sigma_1d_b = float(bench_r_12m.std(ddof=1)) if not bench_r_12m.empty else np.nan
    var95_bench_pct = np.nan if np.isnan(sigma_1d_b) else z_95 * sigma_1d_b
    current_value_bench_pme = float(bench_pme_val.iloc[-1])
    var95_bench_usd = np.nan if np.isnan(var95_bench_pct) else var95_bench_pct * current_value_bench_pme

    # 10. summary IRR / PME
    contrib = cf.clip(upper=0) * -1.0  # soldi entrati (acquisti)
    withdrw = cf.clip(lower=0)        # soldi usciti (vendite positive)
    gross_contrib = float(contrib.sum())
    gross_withdrw = float(withdrw.sum())
    net_invested = float(cf.sum())

    current_value_port_val = float(port_val.iloc[-1])
    current_value_pme = float(bench_pme_val.iloc[-1])

    r_net_port = (
        (current_value_port_val / net_invested - 1)
        if net_invested > 0 else np.nan
    )
    r_net_bench = (
        (current_value_pme / net_invested - 1)
        if net_invested > 0 else np.nan
    )
    if (np.isfinite(r_net_port) and np.isfinite(r_net_bench)):
        r_net_excess = r_net_port - r_net_bench
    else:
        r_net_excess = np.nan

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
    if (np.isfinite(irr_port) and np.isfinite(irr_bench)):
        irr_excess = irr_port - irr_bench
    else:
        irr_excess = np.nan

    # 11. grafico cumulato
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

    # 12. allocazioni (stub)
    sectors_p, countries_p = aggregate_allocations_portfolio(
        shares_df=shares[present],
        prices_df=px,
        tickers=present,
        bench=bench_name,
    )

    bench_sectors, bench_countries = get_etf_allocations(bench_name)
    if not bench_sectors.empty:
        bench_sectors = bench_sectors.rename(
            columns={"weight":"weight_portfolio"}
        ).copy()
    else:
        bench_sectors = pd.DataFrame(columns=["label","weight_portfolio"])
    if not bench_countries.empty:
        bench_countries = bench_countries.rename(
            columns={"weight":"weight_portfolio"}
        ).copy()
    else:
        bench_countries = pd.DataFrame(columns=["label","weight_portfolio"])

    # 13. summary_lines finale
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
        (f"Sharpe 12m – Portfolio [{rf_meta}]",
         f"{sharpe_port_12m:.2f}" if np.isfinite(sharpe_port_12m) else "n/a"),
        (f"Sharpe 12m – Benchmark [{rf_meta}]",
         f"{sharpe_bench_12m:.2f}" if np.isfinite(sharpe_bench_12m) else "n/a"),
        ("1D VaR(95%) – Portfolio (pct)", fmt_pct(var95_pct)),
        ("1D VaR(95%) – Portfolio (USD)",
         money(var95_usd) if np.isfinite(var95_usd) else "n/a"),
        ("1D VaR(95%) – Benchmark (pct)", fmt_pct(var95_bench_pct)),
        ("1D VaR(95%) – Benchmark (USD)",
         money(var95_bench_usd) if np.isfinite(var95_bench_usd) else "n/a"),
    ]
    for k, v in rows_out:
        summary_lines.append(f"{k.ljust(45)} {v}")

    # 14. salva summary.txt
    sum_path = os.path.join(outdir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # 15. ritorno finale per l'API
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
