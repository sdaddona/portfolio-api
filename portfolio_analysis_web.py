import os
import re
import time
import json
import unicodedata
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr  # FRED
import requests

TRADING_DAYS = 252


# ---------------- Utils ----------------

def to_naive(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)) and isinstance(obj.index, pd.DatetimeIndex):
        try:
            obj.index = obj.index.tz_localize(None)
        except Exception:
            pass
    return obj

def to_date(s: str) -> pd.Timestamp:
    s = str(s).strip()
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        raise ValueError(f"Data non riconosciuta: {s}")
    try:
        return dt.tz_localize(None)
    except Exception:
        return dt

def fmt_pct(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return "n/a"
    return f"{x:.2%}"

def money(x):
    return f"{x:,.2f} USD"

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


# ---------------- Normalizzazione & numeri robusti ----------------

_UNICODE_HYPHENS = ["\u2212", "\u2013", "\u2014", "\u2010"]  # − – — ‐

def _normalize_line(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for ch in _UNICODE_HYPHENS:
        s = s.replace(ch, "-")
    s = s.replace("\u00A0", " ").replace("\u202F", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return s.strip()

def _to_float_robust(val: str) -> float:
    raw = str(val) if val is not None else ""
    s = unicodedata.normalize("NFKC", raw)
    for ch in _UNICODE_HYPHENS:
        s = s.replace(ch, "-")
    s = s.replace("\u00A0", "").replace("\u202F", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s.count(".") > 1:
        head, tail = s.split(".", 1)
        tail = tail.replace(".", "")
        s = f"{head}.{tail}"
    if s in ("", ".", "-", "-.", ".-"):
        raise ValueError(f"Stringa numerica non valida: {repr(raw)}")
    try:
        return float(s)
    except Exception:
        raise ValueError(f"Impossibile convertire a float: {repr(raw)} -> '{s}'")


# ---------------- Input lots ----------------

def read_lots_from_text(txt: str) -> pd.DataFrame:
    if txt is None:
        raise ValueError("lots_text mancante.")
    norm_lines = []
    for ln in txt.splitlines():
        ln = _normalize_line(ln)
        if not ln or ln.startswith("#"):
            continue
        norm_lines.append(ln)

    if not norm_lines:
        raise ValueError("Nessuna riga valida nei lotti.")

    rows = []
    for ln in norm_lines:
        parts = re.split(r"[,\t;]+|\s+", ln.strip())
        parts = [p for p in parts if p is not None and str(p).strip() != ""]
        if len(parts) < 4:
            raise ValueError(f"Riga lotti invalida (colonne < 4): {repr(ln)}")

        ticker = str(parts[0]).strip().upper()
        data_str = parts[1]
        qty_str  = parts[2]
        px_str   = parts[3]

        try:
            data = to_date(data_str)
        except Exception:
            raise ValueError(f"Data non valida: {repr(data_str)} (riga: {repr(ln)})")

        try:
            qty = _to_float_robust(qty_str)
        except Exception as e:
            raise ValueError(f"Quantità non valida: {repr(qty_str)} (riga: {repr(ln)}) | {e}")

        try:
            px  = _to_float_robust(px_str)
        except Exception as e:
            raise ValueError(f"Prezzo non valido: {repr(px_str)} (riga: {repr(ln)}) | {e}")

        rows.append((ticker, data, qty, px))

    df = pd.DataFrame(rows, columns=["ticker", "data", "quantità", "prezzo"])
    return df.sort_values("data").reset_index(drop=True)


# ---------------- Risk-free ----------------

def build_rf_daily_series(index: pd.DatetimeIndex, rf_source="fred_1y", rf=0.04):
    rf_source = (rf_source or "fred_1y").lower()

    if rf_source == "fixed":
        rf_daily = pd.Series(rf / TRADING_DAYS, index=index)
        rf_meta = f"Fixed RF {rf:.2%} (ann.)"
        return rf_daily, rf_meta

    if rf_source == "irx_13w":
        try:
            h = yf.Ticker("^IRX").history(start=index[0]-pd.Timedelta(days=10),
                                          end=index[-1]+pd.Timedelta(days=3),
                                          interval="1d")
            ser = to_naive(h["Close"]).reindex(index).ffill()
            rf_annual = ser / 100.0
            rf_daily = rf_annual / TRADING_DAYS
            return rf_daily, "RF: ^IRX (13W T-Bill, ann.)"
        except Exception:
            pass

    try:
        fred = pdr.DataReader("DGS1", "fred",
                              index[0]-pd.Timedelta(days=10),
                              index[-1]+pd.Timedelta(days=3))
        ser = fred["DGS1"].rename("DGS1").sort_index()
        ser = ser.reindex(index).ffill()
        rf_annual = ser / 100.0
        rf_daily = rf_annual / TRADING_DAYS
        return rf_daily, "RF: FRED DGS1 (1Y Treasury, ann.)"
    except Exception:
        rf_daily = pd.Series(rf / TRADING_DAYS, index=index)
        return rf_daily, "RF fallback (FRED failed)"


# ---------------- Yahoo fetch (fallback diretto) ----------------

def _yahoo_chart_url(symbol: str, start_ts: int, end_ts: int) -> str:
    # interval=1d, includiamo adjclose
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}"
        f"?period1={start_ts}&period2={end_ts}"
        "&interval=1d&events=div%2Csplit&includeAdjustedClose=true"
    )

def _download_yahoo_direct(symbol: str, start: pd.Timestamp, end: pd.Timestamp, want_adjclose: bool) -> pd.Series | None:
    # Yahoo vuole epoch seconds (UTC); aggiungiamo buffer +/- qualche giorno
    start_utc = (start.tz_localize("UTC") - pd.Timedelta(days=3)).timestamp()
    end_utc   = (end.tz_localize("UTC") + pd.Timedelta(days=2)).timestamp()
    url = _yahoo_chart_url(symbol, int(start_utc), int(end_utc))

    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    chart = data.get("chart", {})
    results = chart.get("result", [])
    if not results:
        return None
    r0 = results[0]
    ts = r0.get("timestamp", [])
    if not ts:
        return None

    indicators = r0.get("indicators", {})
    adj = indicators.get("adjclose", [{}])
    qte = indicators.get("quote", [{}])

    series_vals = None
    if want_adjclose:
        series_vals = adj[0].get("adjclose", []) if adj else []
    else:
        series_vals = qte[0].get("close", []) if qte else []

    if not series_vals or len(series_vals) != len(ts):
        return None

    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None)
    s = pd.Series(series_vals, index=idx, dtype=float)
    s = s.dropna()
    return s if not s.empty else None

def _fetch_series_with_fallback(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, use_adjclose: bool) -> dict:
    """
    Per ogni simbolo prova:
      1) yfinance (come in locale)
      2) yahoo v8 chart diretto (con suffissi .US / .L / .MI)
    Ritorna: dict[symbol] = pandas.Series
    """
    out = {}
    for sym in symbols:
        got = None
        # 1) yfinance
        try:
            h = yf.Ticker(sym).history(start=start, end=end, interval="1d",
                                       auto_adjust=use_adjclose)
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h.columns:
                s = h["Close"].rename(sym).sort_index().ffill()
                got = to_naive(s)
        except Exception:
            pass

        # 2) fallback diretto (anche con suffissi)
        if got is None or got.empty:
            for cand in [sym, f"{sym}.US", f"{sym}.L", f"{sym}.MI"]:
                s2 = _download_yahoo_direct(cand, start, end, want_adjclose=use_adjclose)
                if s2 is not None and not s2.empty:
                    got = s2.rename(sym)
                    break

        if got is not None and not got.empty:
            out[sym] = got
        time.sleep(0.15)  # mini throttle
    return out


# ---------------- Core analysis ----------------

def analyze_portfolio_from_text(
    lots_text: str,
    bench: str,
    outdir: str = "/tmp/outputs",
    start_buffer_days: int = 7,
    use_adjclose: bool = False,
    rf_source: str = "fred_1y",
    rf: float = 0.04,
):
    os.makedirs(outdir, exist_ok=True)

    lots = read_lots_from_text(lots_text)
    first_tx_date = lots["data"].min()
    bench = bench.upper()

    tickers = sorted(set(lots["ticker"].tolist()) | {bench})

    start = pd.Timestamp(first_tx_date.date() - timedelta(days=start_buffer_days))
    end = pd.Timestamp(datetime.today().date() + timedelta(days=2))

    series = _fetch_series_with_fallback(tickers, start, end, use_adjclose=use_adjclose)
    if not series:
        raise RuntimeError(f"Nessun prezzo disponibile per: {tickers}")

    px = pd.concat(series.values(), axis=1).sort_index().ffill()
    px = to_naive(px).asfreq("B").ffill()

    # Calendario, trades e posizioni (shares)
    calendar = px.index
    trades = []
    for _, r in lots.iterrows():
        sym = r["ticker"]
        d = r["data"]
        qty = float(r["quantità"])
        px_file = float(r["prezzo"])
        pos = calendar.searchsorted(d, side="left")
        if pos >= len(calendar):
            continue
        d_eff = calendar[pos]
        px_eff_trade = float(px.loc[d_eff, sym]) if use_adjclose else px_file
        trades.append((sym, d, d_eff, qty, px_file, px_eff_trade))

    present = sorted(set(px.columns) & set(lots["ticker"].unique()))
    shares = pd.DataFrame(0.0, index=calendar, columns=present)
    for sym, d0, d_eff, qty, px_file, px_eff_trade in trades:
        if sym not in shares.columns:
            continue
        pos = shares.index.searchsorted(d_eff, side="left")
        if pos < len(shares.index):
            shares.iloc[pos:, shares.columns.get_loc(sym)] += qty

    port_val = (shares * px[present]).sum(axis=1)
    first_mv = port_val[port_val > 0].first_valid_index()
    port_val = port_val.loc[first_mv:].dropna()

    bench_val = px[bench].loc[port_val.index[0]:].dropna()
    idx_common = port_val.index.intersection(bench_val.index)
    port_val = port_val.loc[idx_common]
    bench_val = bench_val.loc[idx_common]

    # Cash flows (positivo = acquisti)
    cf = pd.Series(0.0, index=port_val.index)
    for sym, d0, d_eff, qty, px_file, px_eff_trade in trades:
        if d_eff in cf.index:
            cf.loc[d_eff] += qty * px_eff_trade

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

    bench_ret = bench_val.pct_change()
    bench_idx = (1 + bench_ret.iloc[1:]).cumprod() * 100.0
    bench_idx = pd.concat([pd.Series([100.0], index=bench_val.index[:1]), bench_idx])

    # RF
    rf_daily, rf_meta = build_rf_daily_series(twr_ret.index, rf_source, rf)

    # PME
    bench_pme_val = []
    units = 0.0
    for t in port_val.index:
        px_b = float(bench_val.loc[t])
        invest_today = float(cf.loc[t])
        if px_b != 0:
            units += invest_today / px_b
        bench_pme_val.append(units * px_b)
    bench_pme_val = pd.Series(bench_pme_val, index=port_val.index)

    # Rischio 12m + VaR
    lb = TRADING_DAYS
    port_r_12m = (twr_ret.iloc[-lb:] if len(twr_ret) >= lb else twr_ret)
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

    bench_r_12m = (bench_ret.iloc[-lb:] if len(bench_ret) >= lb else bench_ret)
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

    # Metriche “net invested”
    contrib = cf.clip(upper=0) * -1.0
    withdrw = cf.clip(lower=0)
    gross_contrib = float(contrib.sum())
    gross_withdrw = float(withdrw.sum())
    net_invested = float(cf.sum())
    current_value_pme = float(bench_pme_val.iloc[-1])

    r_net_port = (current_value_port / net_invested - 1) if net_invested > 0 else np.nan
    r_net_bench = (current_value_pme / net_invested - 1) if net_invested > 0 else np.nan
    r_net_excess = r_net_port - r_net_bench if (np.isfinite(r_net_port) and np.isfinite(r_net_bench)) else np.nan

    # IRR
    cf_list = [(t, -float(cf.loc[t])) for t in port_val.index if abs(float(cf.loc[t])) != 0.0]
    cf_port = sorted(cf_list + [(port_val.index[-1], current_value_port)], key=lambda x: x[0])
    cf_bench = sorted(cf_list + [(port_val.index[-1], current_value_pme)], key=lambda x: x[0])
    irr_port = xirr(cf_port)
    irr_bench = xirr(cf_bench)
    irr_excess = (irr_port - irr_bench) if (np.isfinite(irr_port) and np.isfinite(irr_bench)) else np.nan

    # Grafico
    os.makedirs(outdir, exist_ok=True)
    plot_path = os.path.join(outdir, "crescita_cumulata.png")
    plt.figure(figsize=(10, 6))
    plt.plot(port_idx, label="Portafoglio (TWR, base 100)")
    plt.plot(bench_idx, label=f"Benchmark {bench} (base 100)")
    mode_label = "AdjClose (ignora prezzo file)" if use_adjclose else "Close (usa prezzo file al trade day)"
    plt.title(f"Andamento storico (base 100)\nMode: {mode_label} | {rf_meta}")
    plt.ylabel("Indice (base 100)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Summary
    summary_lines = []
    summary_lines.append("=== SUMMARY (Benchmark PME-consistent) ===")
    for k, v in [
        ("Start Date", str(port_val.index[0].date())),
        ("End Date", str(port_val.index[-1].date())),
        ("Return (Net Invested) – Portfolio", fmt_pct(r_net_port)),
        ("Return (Net Invested) – Bench (PME)", fmt_pct(r_net_bench)),
        ("Excess Return (Net)", fmt_pct(r_net_excess)),
        ("IRR – Portfolio", fmt_pct(irr_port)),
        ("IRR – Bench (PME)", fmt_pct(irr_bench)),
        ("Excess IRR", fmt_pct(irr_excess)),
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
    ]:
        summary_lines.append(f"{k.ljust(45)} {v}")

    sum_path = os.path.join(outdir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Allocazioni (placeholder su Render)
    alloc = {
        "portfolio_sectors": [],
        "portfolio_countries": [],
        "bench_sectors": [],
        "bench_countries": [],
    }

    return {
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
            "current_value_port": current_value_port,
            "current_value_pme": current_value_pme,
            "net_invested": net_invested,
        },
        "alloc": alloc,
    }
