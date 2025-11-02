# portfolio_analysis_web.py
# -*- coding: utf-8 -*-

import os
import re
import time
import random
import warnings
from io import StringIO
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import yfinance as yf
from pandas_datareader import data as pdr  # FRED + STOOQ

warnings.filterwarnings("ignore", category=UserWarning)
TRADING_DAYS = 252

# --------------------------- Cache locale ----------------------------
CACHE_DIR = "/tmp/yf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(sym: str, src: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{src}_{sym}")
    return os.path.join(CACHE_DIR, f"{safe}.csv")

def _save_cache(sym: str, df: pd.DataFrame, src: str):
    try:
        if isinstance(df, (pd.DataFrame, pd.Series)) and not df.empty:
            if isinstance(df, pd.Series):
                df.to_frame("Close").to_csv(_cache_path(sym, src))
            else:
                df.to_csv(_cache_path(sym, src))
    except Exception:
        pass

def _load_cache(sym: str, src: str) -> pd.DataFrame | None:
    p = _cache_path(sym, src)
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            return None
    return None

def _sleep_jitter(base: float):
    time.sleep(base + random.uniform(0, 0.5))

# --------------------------- Util calcoli ----------------------------
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

# --------------------------- Lettura lotti ---------------------------
def read_lots_from_text(txt: str) -> pd.DataFrame:
    if txt is None:
        raise ValueError("lots_text mancante.")
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("Nessuna riga valida nei lotti.")
    sample = "\n".join(lines)

    def _try(buf, header=None):
        return pd.read_csv(buf, sep=r"[,\t;]+|\s{2,}", engine="python", header=header, comment="#")

    buf = StringIO(sample)
    try:
        df = _try(buf, header=None)
    except Exception:
        buf = StringIO(sample)
        df = pd.read_csv(buf, sep=r"\s+", engine="python", header=None, comment="#")

    if df.shape[1] < 4:
        raise ValueError("Attese 4 colonne: ticker, data, quantità, prezzo.")
    if df.shape[1] > 4:
        df = df.iloc[:, :4]

    first = " ".join(str(x).lower() for x in df.iloc[0].tolist())
    if any(k in first for k in ["ticker", "symbol", "simbolo", "data", "date", "quantita", "quantità", "qty", "prezzo", "price", "px"]):
        buf = StringIO(sample)
        df = _try(buf, header=0)
        if df.shape[1] > 4:
            df = df.iloc[:, :4]

    df.columns = [str(c).strip().lower() for c in df.columns]
    posmap = {0: "ticker", 1: "data", 2: "quantità", 3: "prezzo"}
    name_map = {}
    for i, c in enumerate(df.columns):
        if c in ("ticker", "symbol", "simbolo"):
            name_map[c] = "ticker"
        elif c in ("data", "date"):
            name_map[c] = "data"
        elif c in ("quantità", "quantita", "qty", "qta", "quantity"):
            name_map[c] = "quantità"
        elif c in ("prezzo", "price", "px"):
            name_map[c] = "prezzo"
        else:
            name_map[c] = posmap[i]
    df = df.rename(columns=name_map)[["ticker", "data", "quantità", "prezzo"]]

    def to_float(x):
        return float(str(x).strip().replace(" ", "").replace(",", "."))
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["quantità"] = df["quantità"].apply(to_float)
    df["prezzo"] = df["prezzo"].apply(to_float)
    df["data"] = df["data"].apply(to_date)
    return df.sort_values("data").reset_index(drop=True)

# --------------------------- Risk-free ---------------------------
def build_rf_daily_series(index: pd.DatetimeIndex, rf_source: str, rf_fixed: float):
    rf_source = (rf_source or "fred_1y").lower()

    if rf_source == "fixed":
        rf_daily = pd.Series(rf_fixed / TRADING_DAYS, index=index)
        rf_meta = f"Fixed RF {rf_fixed:.2%} (ann.)"
        return rf_daily, rf_meta

    if rf_source == "irx_13w":
        tkr = "^IRX"
        try:
            # ^IRX via Yahoo singolo, con cache
            df = _load_cache(tkr, "yahoo")
            if df is None or df.empty:
                h = yf.Ticker(tkr).history(period="10y", interval="1d", auto_adjust=False, threads=False)
                if h is not None and not h.empty:
                    df = h[["Close"]]
                    _save_cache(tkr, df, "yahoo")
            ser = df["Close"].rename("IRX").sort_index()
            ser = to_naive(ser).reindex(index).ffill()
            rf_annual = ser / 100.0
            rf_daily = rf_annual / TRADING_DAYS
            rf_meta = "RF: ^IRX (13W T-Bill, ann.)"
            return rf_daily, rf_meta
        except Exception:
            rf_daily = pd.Series(0.0, index=index)
            return rf_daily, "RF fallback 0.00% (IRX download failed)"

    # default: FRED DGS1
    try:
        fred = pdr.DataReader("DGS1", "fred", index[0]-pd.Timedelta(days=10), index[-1]+pd.Timedelta(days=3))
        ser = fred["DGS1"].rename("DGS1").sort_index()
        ser = ser.reindex(index).ffill()
        rf_annual = ser / 100.0
        rf_daily = rf_annual / TRADING_DAYS
        rf_meta = "RF: FRED DGS1 (1Y Treasury, ann.)"
        return rf_daily, rf_meta
    except Exception:
        rf_daily = pd.Series(0.0, index=index)
        return rf_daily, "RF fallback 0.00% (FRED DGS1 download failed)"

# --------------------------- Downloader: STOOQ → cache → Yahoo ---------------------------
def _stooq_try(sym: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Tenta Stooq con 'sym' e, se vuoto, con 'sym.US'.
    Stooq fornisce prezzi *adjusted* nel campo 'Close'. Ordiniamo asc.
    """
    for variant in (sym, f"{sym}.US"):
        try:
            df = pdr.DataReader(variant, "stooq", start, end)  # può tornare desc
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.sort_index()
                s = df["Close"].rename(sym).dropna()
                if not s.empty:
                    _save_cache(sym, s, "stooq")
                    return s
        except Exception:
            continue
    # cache Stooq
    cached = _load_cache(sym, "stooq")
    if cached is not None and not cached.empty and "Close" in cached.columns:
        return cached["Close"].rename(sym).sort_index()
    return pd.Series(dtype=float)

def _yahoo_try(sym: str, start: pd.Timestamp, end: pd.Timestamp, auto_adjust: bool) -> pd.Series:
    """Tentativo Yahoo finale (singolo), con backoff dolce e cache."""
    back = [1.5, 3.0, 6.0]
    for delay in back:
        try:
            h = yf.download(sym, start=start, end=end, interval="1d",
                            auto_adjust=auto_adjust, progress=False, threads=False)
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h.columns:
                s = h["Close"].rename(sym).sort_index()
                if not s.empty:
                    _save_cache(sym, s, "yahoo")
                    return s
        except Exception:
            pass
        _sleep_jitter(delay)
    cached = _load_cache(sym, "yahoo")
    if cached is not None and not cached.empty and "Close" in cached.columns:
        return cached["Close"].rename(sym).sort_index()
    return pd.Series(dtype=float)

def _download_prices(symbols: List[str], start: pd.Timestamp, end: pd.Timestamp, auto_adjust: bool) -> Dict[str, pd.Series]:
    """
    Strategia cloud-friendly:
      1) Stooq (sym, sym.US)
      2) Cache (Stooq / Yahoo)
      3) Yahoo (ultimissimo tentativo singolo)
    """
    out: Dict[str, pd.Series] = {}
    for sym in symbols:
        sym = sym.upper().strip()
        if not sym:
            continue
        # 1) Stooq
        s = _stooq_try(sym, start, end)
        if s is not None and not s.empty:
            out[sym] = s
            continue
        # 2) Yahoo (solo se Stooq non ha nulla)
        s2 = _yahoo_try(sym, start, end, auto_adjust=auto_adjust)
        if s2 is not None and not s2.empty:
            out[sym] = s2
    return out

# --------------------------- Analisi principale ---------------------------
def analyze_portfolio_from_text(
    lots_text: str,
    bench: str,
    outdir: str = "/tmp/outputs",
    use_adjclose: bool = False,
    rf_source: str = "fred_1y",
    rf: float = 0.0,
    start_buffer_days: int = 7,
):
    """
    Calcoli IDENTICI alla tua versione locale.
    Cambia solo l’approvvigionamento dati: Stooq→Cache→Yahoo, per eliminare i 429 su Render.
    """
    os.makedirs(outdir, exist_ok=True)

    # LOTTI
    lots = read_lots_from_text(lots_text)
    first_tx_date = lots["data"].min()
    bench = (bench or "").upper().strip()
    if not bench:
        raise ValueError("Benchmark mancante.")

    # LISTA TICKER
    tickers = sorted(set(lots["ticker"].tolist()) | {bench})

    # PREZZI
    start = pd.Timestamp(first_tx_date.date() - timedelta(days=start_buffer_days))
    end = pd.Timestamp(datetime.today().date() + timedelta(days=2))

    series_map = _download_prices(tickers, start, end, auto_adjust=True if use_adjclose else False)

    present_syms = [s for s in tickers if s in series_map and isinstance(series_map[s], pd.Series) and not series_map[s].empty]
    if not present_syms or bench not in present_syms:
        raise RuntimeError(f"Nessun prezzo disponibile per: {tickers}")

    px = pd.concat([series_map[s] for s in present_syms], axis=1)
    px.columns = present_syms
    px = px.sort_index().ffill()
    px = to_naive(px).asfreq("B").ffill()

    # COSTRUZIONE POSIZIONI
    calendar = px.index
    trades = []
    for _, r in lots.iterrows():
        sym = r["ticker"]
        d = r["data"]
        qty = float(r["quantità"])
        px_file = float(r["prezzo"])
        d_eff = map_to_effective_date(d, calendar)
        if d_eff is None:
            continue
        px_eff_trade = float(px.loc[d_eff, sym]) if use_adjclose else px_file
        trades.append((sym, d, d_eff, qty, px_file, px_eff_trade))

    px_eff = px.copy()
    if not use_adjclose:
        # override del prezzo nel giorno del trade con il prezzo del file (identico alla tua logica)
        for sym, d0, d_eff, qty, px_file, px_eff_trade in trades:
            if sym in px_eff.columns and d_eff in px_eff.index:
                px_eff.at[d_eff, sym] = px_file

    present = sorted(set(px_eff.columns) & set(lots["ticker"].unique()))
    shares = pd.DataFrame(0.0, index=px_eff.index, columns=present)
    for sym, d0, d_eff, qty, px_file, px_eff_trade in trades:
        if sym not in shares.columns:
            continue
        pos = shares.index.searchsorted(d_eff, side="left")
        if pos < len(shares.index):
            shares.iloc[pos:, shares.columns.get_loc(sym)] += qty

    # PORTAFOGLIO vs BENCH
    port_val = (shares * px_eff[present]).sum(axis=1)
    first_mv = port_val[port_val > 0].first_valid_index()
    port_val = port_val.loc[first_mv:].dropna()

    if bench not in px.columns:
        raise RuntimeError(f"Benchmark {bench} non presente nelle serie scaricate")
    bench_val = px[bench].loc[port_val.index[0]:].dropna()

    idx_common = port_val.index.intersection(bench_val.index)
    port_val = port_val.loc[idx_common]
    bench_val = bench_val.loc[idx_common]

    # CASH FLOWS
    cf = pd.Series(0.0, index=port_val.index)
    for sym, d0, d_eff, qty, px_file, px_eff_trade in trades:
        if d_eff in cf.index:
            cf.loc[d_eff] += qty * (px_eff_trade)

    # TWR base 100
    twr_ret = []
    dates = port_val.index
    for i in range(1, len(dates)):
        t, tm = dates[i], dates[i - 1]
        mv_t, mv_tm = float(port_val.loc[t]), float(port_val.loc[tm])
        cf_t = float(cf.loc[t])
        twr_ret.append((mv_t - cf_t) / mv_tm - 1 if mv_tm > 0 else 0.0)
    twr_ret = pd.Series(twr_ret, index=dates[1:])
    port_idx = (1 + twr_ret).cumprod() * 100.0
    port_idx = pd.concat([pd.Series([100.0], index=dates[:1]), port_idx])

    # Benchmark base 100
    bench_ret = bench_val.pct_change()
    bench_idx = (1 + bench_ret.iloc[1:]).cumprod() * 100.0
    bench_idx = pd.concat([pd.Series([100.0], index=bench_val.index[:1]), bench_idx])

    # RF daily
    rf_daily, rf_meta = build_rf_daily_series(twr_ret.index, rf_source, rf)

    # PME (replica flussi sul benchmark)
    bench_pme_val = []
    units = 0.0
    for t in port_val.index:
        px_b = float(bench_val.loc[t])
        invest_today = float(cf.loc[t])
        if px_b != 0:
            units += invest_today / px_b
        bench_pme_val.append(units * px_b)
    bench_pme_val = pd.Series(bench_pme_val, index=port_val.index)

    # RISK METRICS 12m
    lb = TRADING_DAYS

    # Portafoglio
    port_r_12m = (twr_ret.iloc[-lb:] if len(twr_ret) >= lb else twr_ret.dropna()).copy()
    rf_12m = rf_daily.reindex(port_r_12m.index).ffill().fillna(0.0)
    vol_port_12m = np.nan if port_r_12m.empty else float(port_r_12m.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe_port_12m = np.nan
    if not port_r_12m.empty and port_r_12m.std(ddof=1) > 0:
        ex = port_r_12m - rf_12m
        sharpe_port_12m = float(np.sqrt(TRADING_DAYS) * ex.mean() / ex.std(ddof=1))
    z_95 = 1.65
    sigma_1d = float(port_r_12m.std(ddof=1)) if not port_r_12m.empty else np.nan
    var95_pct = np.nan if np.isnan(sigma_1d) else z_95 * sigma_1d
    current_value_port = float(port_val.iloc[-1])
    var95_usd = np.nan if np.isnan(var95_pct) else var95_pct * current_value_port

    # Benchmark
    bench_r_12m = (bench_ret.iloc[-lb:] if len(bench_ret) >= lb else bench_ret.dropna()).copy()
    rf_12m_b = rf_daily.reindex(bench_r_12m.index).ffill().fillna(0.0)
    vol_bench_12m = np.nan if bench_r_12m.empty else float(bench_r_12m.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe_bench_12m = np.nan
    if not bench_r_12m.empty and bench_r_12m.std(ddof=1) > 0:
        exb = bench_r_12m - rf_12m_b
        sharpe_bench_12m = float(np.sqrt(TRADING_DAYS) * exb.mean() / exb.std(ddof=1))
    sigma_1d_b = float(bench_r_12m.std(ddof=1)) if not bench_r_12m.empty else np.nan
    current_value_bench_pme = float(bench_pme_val.iloc[-1])
    var95_bench_pct = np.nan if np.isnan(sigma_1d_b) else z_95 * sigma_1d_b
    var95_bench_usd = np.nan if np.isnan(var95_bench_pct) else var95_bench_pct * current_value_bench_pme

    # GRAFICO (compatibile col tuo /plot)
    os.makedirs(outdir, exist_ok=True)
    out_hist_main = os.path.join(outdir, "crescita_cumulata.png")
    out_hist_copy = os.path.join(outdir, "01_crescita_cumulata.png")
    plt.figure(figsize=(10, 6))
    plt.plot(port_idx, label="Portafoglio (TWR, base 100)")
    plt.plot(bench_idx, label=f"Benchmark {bench} (prezzo, base 100)")
    mode_label = "AdjClose (ignora prezzo file)" if use_adjclose else "Close (usa prezzo file al trade day)"
    plt.title(f"Andamento storico (base 100)\nMode: {mode_label} | {rf_meta}")
    plt.ylabel("Indice (base 100)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_hist_main, dpi=160)
    try:
        plt.savefig(out_hist_copy, dpi=160)
    except Exception:
        pass
    plt.close()

    # SUMMARY
    contrib = cf.clip(upper=0) * -1.0
    withdrw = cf.clip(lower=0)
    gross_contrib = float(contrib.sum())
    gross_withdrw = float(withdrw.sum())
    net_invested = float(cf.sum())
    current_value_port = float(port_val.iloc[-1])
    current_value_pme = float(bench_pme_val.iloc[-1])

    r_net_port = (current_value_port / net_invested - 1) if net_invested > 0 else np.nan
    r_net_bench = (current_value_pme / net_invested - 1) if net_invested > 0 else np.nan
    r_net_excess = r_net_port - r_net_bench if (np.isfinite(r_net_port) and np.isfinite(r_net_bench)) else np.nan

    cf_list = [(t, -float(cf.loc[t])) for t in port_val.index if abs(float(cf.loc[t])) != 0.0]
    cf_port = sorted(cf_list + [(port_val.index[-1], current_value_port)], key=lambda x: x[0])
    cf_bench = sorted(cf_list + [(port_val.index[-1], current_value_pme)], key=lambda x: x[0])
    irr_port = xirr(cf_port)
    irr_bench = xirr(cf_bench)
    irr_excess = (irr_port - irr_bench) if (np.isfinite(irr_port) and np.isfinite(irr_bench)) else np.nan

    def fmt_pct(x):
        return "n/a" if (x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x)))) else f"{x:.2%}"
    def money(x): return f"{x:,.2f} USD"

    summary_lines = []
    summary_lines.append("=== SUMMARY (Benchmark Equivalente, coerente) ===")
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
        ("Current Value – Portfolio", money(current_value_port)),
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

    out = {
        "summary_lines": summary_lines,
        "plot_path": out_hist_main,
        "summary_path": os.path.join(outdir, "02_summary.txt"),
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
            "current_value_port": current_value_port,
            "current_value_pme": current_value_pme,
            "net_invested": net_invested,
        },
        "alloc": {
            "portfolio_sectors": [],
            "portfolio_countries": [],
            "bench_sectors": [],
            "bench_countries": [],
        },
    }

    with open(out["summary_path"], "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return out
