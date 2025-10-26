import os
import io
import re
import json
import time
import math
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
import yfinance as yf

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

def try_symbol(sym: str) -> str:
    """Prova suffissi comuni; ritorna il primo che ha storico valido."""
    for t in [sym, f"{sym}.US", f"{sym}.L", f"{sym}.MI"]:
        try:
            h = yf.Ticker(t).history(period="10d", auto_adjust=False)
            if isinstance(h, pd.DataFrame) and not h.empty:
                return t
        except Exception:
            pass
    return ""

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

def map_to_effective_date(d: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp | None:
    pos = idx.searchsorted(d, side="left")
    if pos >= len(idx):
        return None
    return idx[pos]

def money(x: float) -> str:
    return f"{x:,.2f} USD"

def fmt_pct(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return "n/a"
    return f"{x:.2%}"

###############################################################################
# Lettura lotti dalla textarea (invece che da file)
###############################################################################

def read_lots_from_text(txt: str) -> pd.DataFrame:
    """
    txt deve avere righe tipo:
    TICKER DATA QUANTITA PREZZO
    ACWI 2024-01-02 10 100
    VT 2024-01-05 5 90
    ecc.

    Ritorna DataFrame con colonne:
    ticker, data, quantità, prezzo
    """
    lines = []
    for ln in txt.splitlines():
        ln_strip = ln.strip()
        if not ln_strip or ln_strip.startswith("#"):
            continue
        lines.append(ln_strip)
    if not lines:
        raise ValueError("Nessuna riga valida nei lotti.")

    # tenta split intelligente
    rows = []
    for ln in lines:
        parts = re.split(r"[,\t;]+|\s+", ln.strip())
        if len(parts) < 4:
            raise ValueError(f"Riga lotti invalida: {ln}")
        ticker = parts[0]
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
    Scarica serie FRED (es. DGS1 = 1Y Treasury yield, annualizzato %) come pandas.Series.
    Se fallisce -> Series vuota.

    Usa l'endpoint FRED ufficiale che fornisce osservazioni giornaliere di
    rendimento dei Treasury USA a 1 anno, misurati come tasso percentuale annuo,
    pubblicato dalla Federal Reserve Bank of St. Louis. 
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
    Ritorna rf_daily (Series con rf giornaliero in frazione) e rf_meta (descrizione).
    1) prova FRED DGS1 (1-Year Treasury Yield, % annuo)
    2) se fallisce -> fallback fisso (es. 4% annuo)
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
# Allocazioni ETF (stub lato Render per ora)
###############################################################################

def get_etf_allocations(ticker: str):
    """
    Stub per Render: ritorna DataFrame vuoti.
    In locale (desktop) tu usi già etfdb_alloc con Playwright per
    riempire 'settori' (sector breakdown) e 'paesi' (country breakdown).
    Qui su Render non possiamo ancora farlo (niente Chromium free tier).

    Ritorna tuple (df_sectors, df_countries) con colonne ['label','weight'].
    'weight' in frazione (0..1).
    """
    return (
        pd.DataFrame(columns=["label","weight"]),
        pd.DataFrame(columns=["label","weight"]),
    )

def aggregate_allocations_portfolio(shares_df: pd.DataFrame,
                                    prices_df: pd.DataFrame,
                                    tickers: list[str],
                                    bench: str):
    """
    Calcola esposizione settoriale / geografica del portafoglio come
    somma pesata delle allocazioni degli ETF.

    Su Render questa userà get_etf_allocations() che per ora è stub -> vuoto.
    """
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

    # normalizza
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
# Core calcolo portafoglio
###############################################################################

def analyze_portfolio_from_text(lots_text: str,
                                bench: str,
                                outdir: str = "/tmp/outputs",
                                start_buffer_days: int = 7):
    """
    Fa tutto:
      - parse lotti da textarea
      - scarica prezzi con yfinance
      - calcola TWR base 100 vs benchmark base 100
      - calcola PME, IRR, Sharpe 12m, VaR(95%) 1d
      - calcola allocazioni (stub su Render)
      - salva grafico cumulato su outdir
      - ritorna dict pronto per JSON
    """

    os.makedirs(outdir, exist_ok=True)

    lots = read_lots_from_text(lots_text)
    first_tx_date = lots["data"].min()

    # tickers totali (portafoglio + benchmark)
    tickers = sorted(set(lots["ticker"].tolist()) | {bench})
    resolved = {}
    for s in tickers:
        rs = try_symbol(s)
        resolved[s] = rs

    if not resolved.get(bench):
        raise RuntimeError(f"Benchmark {bench} non risolto su Yahoo")

    # Prezzi: usiamo Adj Close semplicemente (auto_adjust=True)
    start = pd.Timestamp(first_tx_date.date() - timedelta(days=start_buffer_days))
    end = pd.Timestamp(datetime.today().date() + timedelta(days=2))
    series = {}
    for k, ysym in resolved.items():
        if not ysym:
            continue
        h = yf.Ticker(ysym).history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True  # usa sempre adjusted per web
        )
        if h is None or h.empty or "Close" not in h.columns:
            continue
        s = h["Close"].rename(k).sort_index().ffill()
        series[k] = to_naive(s)
        time.sleep(0.05)

    if not series:
        raise RuntimeError("Nessun prezzo scaricato.")

    px = pd.concat(series.values(), axis=1).sort_index().ffill()
    px = to_naive(px).asfreq("B").ffill()

    # ricostruiamo le shares giornaliere applicando le operazioni
    present = sorted(set(px.columns) & set(lots["ticker"].unique()))
    shares = pd.DataFrame(0.0, index=px.index, columns=present)

    trades = []
    for _, r in lots.iterrows():
        sym = r["ticker"]
        d = r["data"]
        qty = float(r["quantità"])
        px_file = float(r["prezzo"])  # lo manteniamo per flussi di cassa
        d_eff = map_to_effective_date(d, px.index)
        if d_eff is None:
            continue
        # con AdjClose, il prezzo effettivo di costo per il nav day lo mettiamo px_file per i cash flows
        trades.append((sym, d, d_eff, qty, px_file))

        if sym in shares.columns:
            pos = shares.index.searchsorted(d_eff, side="left")
            if pos < len(shares.index):
                shares.iloc[pos:, shares.columns.get_loc(sym)] += qty

    # valore portafoglio giornaliero
    port_val = (shares * px[present]).sum(axis=1)
    first_mv = port_val[port_val > 0].first_valid_index()
    port_val = port_val.loc[first_mv:].dropna()

    # benchmark value series
    bench_val = px[bench].loc[port_val.index[0]:].dropna()
    idx_common = port_val.index.intersection(bench_val.index)
    port_val = port_val.loc[idx_common]
    bench_val = bench_val.loc[idx_common]

    # flussi di cassa giornalieri (positivo = contributo / acquisto)
    cf = pd.Series(0.0, index=port_val.index)
    for sym, d0, d_eff, qty, px_file in trades:
        if d_eff in cf.index:
            cf.loc[d_eff] += qty * px_file

    # calcolo TWR (time-weighted return)
    twr_ret = []
    dates = port_val.index
    for i in range(1, len(dates)):
        t, tm = dates[i], dates[i - 1]
        mv_t, mv_tm = float(port_val.loc[t]), float(port_val.loc[tm])
        cf_t = float(cf.loc[t])
        if mv_tm > 0:
            twr_ret.append((mv_t - cf_t) / mv_tm - 1)
        else:
            twr_ret.append(0.0)
    twr_ret = pd.Series(twr_ret, index=dates[1:])
    port_idx = (1 + twr_ret).cumprod() * 100.0
    port_idx = pd.concat([pd.Series([100.0], index=dates[:1]), port_idx])

    # benchmark base 100
    bench_ret = bench_val.pct_change()
    bench_idx = (1 + bench_ret.iloc[1:]).cumprod() * 100.0
    bench_idx = pd.concat([pd.Series([100.0], index=bench_val.index[:1]), bench_idx])

    # RF daily (FRED con fallback)
    rf_daily, rf_meta = build_rf_daily_series_web(twr_ret.index)

    # PME (Replica flussi sul benchmark)
    bench_pme_val = []
    units = 0.0
    for t in port_val.index:
        px_b = float(bench_val.loc[t])
        invest_today = float(cf.loc[t])
        if px_b != 0:
            units += invest_today / px_b
        bench_pme_val.append(units * px_b)
    bench_pme_val = pd.Series(bench_pme_val, index=port_val.index)

    # Rischio 12m
    lb = TRADING_DAYS

    # --- Portafoglio ---
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

    # --- Benchmark (per confronto Sharpe/vol) ---
    bench_r_12m = (bench_ret.iloc[-lb:] if len(bench_ret) >= lb else bench_ret.dropna()).copy()
    rf_12m_b = rf_daily.reindex(bench_r_12m.index).ffill().fillna(0.0)

    vol_bench_12m = np.nan if bench_r_12m.empty else float(bench_r_12m.std(ddof=1) * np.sqrt(TRADING_DAYS))

    sharpe_bench_12m = np.nan
    if (not bench_r_12m.empty) and (bench_r_12m.std(ddof=1) > 0):
        exb = bench_r_12m - rf_12m_b
        sharpe_bench_12m = float(np.sqrt(TRADING_DAYS) * exb.mean() / exb.std(ddof=1))

    sigma_1d_b = float(bench_r_12m.std(ddof=1)) if not bench_r_12m.empty else np.nan
    var95_bench_pct = np.nan if np.isnan(sigma_1d_b) else z_95 * sigma_1d_b
    # Per coerenza col PME:
    current_value_bench_pme = float(bench_pme_val.iloc[-1])
    var95_bench_usd = np.nan if np.isnan(var95_bench_pct) else var95_bench_pct * current_value_bench_pme

    # Summary Net Invested / IRR ecc.
    contrib = cf.clip(upper=0) * -1.0   # soldi entrati (acquisti)
    withdrw = cf.clip(lower=0)          # soldi usciti (vendite positive)
    gross_contrib = float(contrib.sum())
    gross_withdrw = float(withdrw.sum())
    net_invested = float(cf.sum())

    current_value_port_val = current_value_port
    current_value_pme = float(bench_pme_val.iloc[-1])

    r_net_port = (current_value_port_val / net_invested - 1) if net_invested > 0 else np.nan
    r_net_bench = (current_value_pme / net_invested - 1) if net_invested > 0 else np.nan
    r_net_excess = r_net_port - r_net_bench if (np.isfinite(r_net_port) and np.isfinite(r_net_bench)) else np.nan

    cf_list = [(t, -float(cf.loc[t])) for t in port_val.index if abs(float(cf.loc[t])) != 0.0]
    cf_port = sorted(cf_list + [(port_val.index[-1], current_value_port_val)], key=lambda x: x[0])
    cf_bench = sorted(cf_list + [(port_val.index[-1], current_value_pme)], key=lambda x: x[0])

    irr_port = xirr(cf_port)
    irr_bench = xirr(cf_bench)
    irr_excess = (irr_port - irr_bench) if (np.isfinite(irr_port) and np.isfinite(irr_bench)) else np.nan

    # GRAFICO cumulato
    plot_path = os.path.join(outdir, "crescita_cumulata.png")
    plt.figure(figsize=(10, 6))
    plt.plot(port_idx, label="Portafoglio (TWR base 100)")
    plt.plot(bench_idx, label=f"Benchmark {bench} (base 100)")
    plt.title(f"Andamento storico (base 100)\n{rf_meta}")
    plt.ylabel("Indice (base 100)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    # Allocazioni (stub lato Render)
    sectors_p, countries_p = aggregate_allocations_portfolio(
        shares_df=shares[present],
        prices_df=px,
        tickers=present,
        bench=bench,
    )

    # also benchmark alloc stub
    bench_sectors, bench_countries = get_etf_allocations(bench)
    if not bench_sectors.empty:
        bench_sectors = bench_sectors.rename(columns={"weight":"weight_portfolio"}).copy()
    else:
        bench_sectors = pd.DataFrame(columns=["label","weight_portfolio"])
    if not bench_countries.empty:
        bench_countries = bench_countries.rename(columns={"weight":"weight_portfolio"}).copy()
    else:
        bench_countries = pd.DataFrame(columns=["label","weight_portfolio"])

    # SUMMARY lines
    summary_lines = []
    summary_lines.append("=== SUMMARY (Benchmark PME-consistent) ===")
    rows = [
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
    for k, v in rows:
        summary_lines.append(f"{k.ljust(45)} {v}")

    # Output strutturato per l’API
    out = {
        "summary_lines": summary_lines,
        "plot_path": plot_path,
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

    # per debug, scriviamo anche un file summary nel tmp
    sum_path = os.path.join(outdir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    out["summary_path"] = sum_path

    return out
