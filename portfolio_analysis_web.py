import os
import re
import time
import math
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRADING_DAYS = 252

_price_cache = {}  # cache prezzi per evitare richieste duplicate


###############################################################################
# Utility di base
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
# Lettura lotti (patch robusta con supporto TAB, NBSP e virgole)
###############################################################################

NBSP = u"\u00A0"
NARROW_NBSP = u"\u202F"
ZERO_WIDTH = u"\u200B"


def _clean_num(token: str) -> float:
    """
    Converte numeri 'sporchi' in float.
    Gestisce NBSP, virgola decimale e separatori migliaia.
    """
    if token is None:
        raise ValueError("numero mancante")
    s = str(token).strip()
    s = s.replace(NBSP, " ").replace(NARROW_NBSP, " ").replace(ZERO_WIDTH, "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9\-\+\,\.]", "", s)
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    if s in ("", "+", "-"):
        raise ValueError(f"numero vuoto: {token!r}")
    return float(s)


def read_lots_from_text(txt: str) -> pd.DataFrame:
    """
    Parser robusto:
    - accetta spazi, TAB, virgole o ';' come separatori
    - tollera caratteri invisibili/unicode
    - individua quantità e prezzo cercando i PRIMI due token 'numerici' dopo la data
    Formato logico per riga:  TICKER  DATA  QTY  PREZZO
    """
    if not isinstance(txt, str) or not txt.strip():
        raise ValueError("Nessuna riga valida nei lotti.")

    rows = []
    for raw in txt.splitlines():
        line = (raw or "").strip()
        if not line or line.startswith("#"):
            continue

        # normalizza unicode fastidiosi
        line = (line
                .replace(NBSP, " ")
                .replace(NARROW_NBSP, " ")
                .replace(ZERO_WIDTH, ""))

        # split permissivo
        parts = [p for p in re.split(r"[,\t;]+|\s+", line) if p != ""]
        if len(parts) < 4:
            line2 = re.sub(r"\s+", " ", line).strip()
            parts = [p for p in re.split(r"[,\t;]+|\s+", line2) if p != ""]

        if len(parts) < 3:
            raise ValueError(f"Riga lotti invalida: {raw!r}")

        ticker = parts[0].strip().upper()

        # data robusta
        try:
            dt = to_date(parts[1])
        except Exception:
            dt_try = pd.to_datetime(parts[1], errors="coerce", dayfirst=True, utc=False)
            if pd.isna(dt_try):
                raise ValueError(f"Data non riconosciuta: {parts[1]!r} (riga: {raw!r})")
            dt = pd.to_datetime(dt_try).tz_localize(None)

        # quantità e prezzo
        tail = parts[2:]
        numeric_tokens = []
        for tok in tail:
            if re.search(r"\d", tok):
                numeric_tokens.append(tok)
            if len(numeric_tokens) >= 2:
                break

        if len(numeric_tokens) < 2:
            raise ValueError(f"Quantità/Prezzo mancanti o illeggibili (riga: {raw!r})")

        qty = _clean_num(numeric_tokens[0])
        px = _clean_num(numeric_tokens[1])

        rows.append((ticker, dt, float(qty), float(px)))

    df = pd.DataFrame(rows, columns=["ticker", "data", "quantità", "prezzo"])
    df = df.sort_values("data").reset_index(drop=True)
    return df


###############################################################################
# Risk-free e prezzi
###############################################################################

def build_rf_daily_series_web(index: pd.DatetimeIndex, fallback_annual_rf: float = 0.04):
    rf_daily = pd.Series(fallback_annual_rf / TRADING_DAYS, index=index)
    rf_meta = f"Fixed RF {fallback_annual_rf:.2%} (ann., fallback)"
    return rf_daily, rf_meta


def _yahoo_chart_url(symbol: str, start_ts: int, end_ts: int) -> str:
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}"
        f"?period1={start_ts}&period2={end_ts}"
        "&interval=1d&events=div%2Csplit&includeAdjustedClose=true"
    )


def download_price_history_yahoo(ticker: str, start: pd.Timestamp, end: pd.Timestamp):
    start_sec = int((start.tz_localize("UTC") - pd.Timedelta(days=3)).timestamp())
    end_sec = int((end.tz_localize("UTC") + pd.Timedelta(days=1)).timestamp())
    cache_key = (ticker, str(start_sec), str(end_sec))
    if cache_key in _price_cache:
        return _price_cache[cache_key].copy()
    url = _yahoo_chart_url(ticker, start_sec, end_sec)
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()
    except Exception:
        return None
    if r.status_code != 200:
        return None
    result = data.get("chart", {}).get("result", [])
    if not result:
        return None
    res0 = result[0]
    ts = res0.get("timestamp", [])
    adj = res0.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
    if not ts or not adj:
        return None
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None)
    s = pd.Series(adj, index=idx, name=ticker, dtype=float).dropna()
    _price_cache[cache_key] = s.copy()
    return s if not s.empty else None


def robust_fetch_prices_for_symbol(sym: str, first_tx_date: pd.Timestamp, start_buffer_days: int = 7):
    start = first_tx_date - timedelta(days=start_buffer_days)
    end = pd.Timestamp(datetime.today().date())
    for ticker_try in [sym, f"{sym}.US", f"{sym}.L", f"{sym}.MI"]:
        s = download_price_history_yahoo(ticker_try, start, end)
        if s is not None and not s.empty:
            return s.rename(sym)
    return None


###############################################################################
# Allocazioni (stub) e analisi principale
###############################################################################

def get_etf_allocations(ticker: str):
    return (
        pd.DataFrame(columns=["label", "weight"]),
        pd.DataFrame(columns=["label", "weight"]),
    )


def aggregate_allocations_portfolio(shares_df, prices_df, tickers, bench):
    return (
        pd.DataFrame(columns=["label", "weight_portfolio"]),
        pd.DataFrame(columns=["label", "weight_portfolio"]),
    )


def analyze_portfolio_from_text(lots_text: str, bench: str, outdir: str = "/tmp/outputs", start_buffer_days: int = 7):
    os.makedirs(outdir, exist_ok=True)

    lots = read_lots_from_text(lots_text)
    first_tx_date = lots["data"].min()
    tickers_portafoglio = sorted(set(lots["ticker"].tolist()))
    all_tickers = sorted(set(tickers_portafoglio + [bench]))

    price_series = {}
    for sym in all_tickers:
        s = robust_fetch_prices_for_symbol(sym, first_tx_date, start_buffer_days)
        if s is not None and not s.empty:
            price_series[sym] = s.astype(float)
        time.sleep(0.1)

    if len(price_series) == 0:
        raise RuntimeError(f"Nessun dato prezzi disponibile per i ticker richiesti: {all_tickers}")

    px = pd.concat(price_series.values(), axis=1).sort_index().ffill()
    px = to_naive(px).asfreq("B").ffill()
    px = px.loc[px.index >= (first_tx_date - timedelta(days=start_buffer_days))]

    present = sorted(set(px.columns) & set(tickers_portafoglio))
    shares = pd.DataFrame(0.0, index=px.index, columns=present)
    trades = []
    for _, r in lots.iterrows():
        sym, d, qty, px_file = r["ticker"], r["data"], float(r["quantità"]), float(r["prezzo"])
        pos = px.index.searchsorted(d, side="left")
        if pos >= len(px.index):
            continue
        d_eff = px.index[pos]
        trades.append((sym, d, d_eff, qty, px_file))
        if sym in shares.columns:
            shares.iloc[pos:, shares.columns.get_loc(sym)] += qty

    port_val = (shares * px[present]).sum(axis=1)
    port_val = port_val.loc[port_val > 0]
    bench_val = px[bench] if bench in px.columns else port_val.copy()

    idx_common = port_val.index.intersection(bench_val.index)
    port_val, bench_val = port_val.loc[idx_common], bench_val.loc[idx_common]

    cf = pd.Series(0.0, index=port_val.index)
    for _, _, d_eff, qty, px_file in trades:
        if d_eff in cf.index:
            cf.loc[d_eff] += qty * px_file

    dates = port_val.index
    twr_chunks = []
    for i in range(1, len(dates)):
        mv_t, mv_tm = float(port_val.iloc[i]), float(port_val.iloc[i - 1])
        cf_t = float(cf.iloc[i])
        twr_chunks.append((mv_t - cf_t) / mv_tm - 1 if mv_tm > 0 else 0)
    twr_ret = pd.Series(twr_chunks, index=dates[1:])
    port_idx = pd.concat([pd.Series([100.0], index=dates[:1]), (1 + twr_ret).cumprod() * 100.0])

    bench_ret = bench_val.pct_change()
    bench_idx = pd.concat([pd.Series([100.0], index=bench_val.index[:1]), (1 + bench_ret.iloc[1:]).cumprod() * 100.0])

    rf_daily, rf_meta = build_rf_daily_series_web(twr_ret.index)
    current_value_port = float(port_val.iloc[-1])

    z_95 = 1.65
    sigma_1d = float(twr_ret.std(ddof=1))
    var95_pct = z_95 * sigma_1d
    var95_usd = var95_pct * current_value_port

    vol_port_12m = float(twr_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe_port_12m = float(np.sqrt(TRADING_DAYS) * (twr_ret.mean() - rf_daily.mean()) / twr_ret.std(ddof=1))

    summary_lines = [
        "=== SUMMARY ===",
        f"Start Date: {str(port_val.index[0].date())}",
        f"End Date: {str(port_val.index[-1].date())}",
        f"Volatility 12m: {fmt_pct(vol_port_12m)}",
        f"Sharpe: {sharpe_port_12m:.2f}",
        f"1D VaR(95%): {fmt_pct(var95_pct)} = {money(var95_usd)}",
    ]

    sum_path = os.path.join(outdir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    plot_path = os.path.join(outdir, "crescita_cumulata.png")
    plt.figure(figsize=(10, 6))
    plt.plot(port_idx, label="Portafoglio (base 100)")
    plt.plot(bench_idx, label=f"Benchmark {bench} (base 100)")
    plt.title(f"Andamento storico (base 100)\n{rf_meta}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    return {
        "summary_lines": summary_lines,
        "plot_path": plot_path,
        "summary_path": sum_path,
        "risk": {
            "vol_port_12m": vol_port_12m,
            "sharpe_port_12m": sharpe_port_12m,
            "var95_pct": var95_pct,
            "var95_usd": var95_usd,
        },
        "alloc": {"portfolio_sectors": [], "portfolio_countries": [], "bench_sectors": [], "bench_countries": []},
        "ok": True,
    }
