import os
import re
import time
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf
from pandas_datareader import data as pdr  # FRED

TRADING_DAYS = 252

# =============================================================================
# Utils
# =============================================================================

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

def map_to_effective_date(d: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp | None:
    pos = idx.searchsorted(d, side="left")
    if pos >= len(idx):
        return None
    return idx[pos]

# =============================================================================
# Parser lotti (robusto per tab, spazi multipli, NBSP, ecc.)
# =============================================================================
NBSP = u"\u00A0"
NARROW_NBSP = u"\u202F"
ZERO_WIDTH = u"\u200B"

def _clean_num(token: str) -> float:
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
    if not isinstance(txt, str) or not txt.strip():
        raise ValueError("Nessuna riga valida nei lotti.")

    rows = []
    for raw in txt.splitlines():
        line = (raw or "").strip()
        if not line or line.startswith("#"):
            continue
        line = (line
                .replace(NBSP, " ")
                .replace(NARROW_NBSP, " ")
                .replace(ZERO_WIDTH, ""))
        parts = [p for p in re.split(r"[,\t;]+|\s+", line) if p != ""]
        if len(parts) < 4:
            line2 = re.sub(r"\s+", " ", line).strip()
            parts = [p for p in re.split(r"[,\t;]+|\s+", line2) if p != ""]

        if len(parts) < 3:
            raise ValueError(f"Riga lotti invalida: {raw!r}")

        ticker = parts[0].strip().upper()

        # data
        try:
            dt = to_date(parts[1])
        except Exception:
            dt_try = pd.to_datetime(parts[1], errors="coerce", dayfirst=True, utc=False)
            if pd.isna(dt_try):
                raise ValueError(f"Data non riconosciuta: {parts[1]!r} (riga: {raw!r})")
            dt = pd.to_datetime(dt_try).tz_localize(None)

        # qty + prezzo = primi 2 token numerici
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
        px  = _clean_num(numeric_tokens[1])

        rows.append((ticker, dt, float(qty), float(px)))

    df = pd.DataFrame(rows, columns=["ticker", "data", "quantità", "prezzo"])
    df = df.sort_values("data").reset_index(drop=True)
    return df

# =============================================================================
# Risk-free (FRED DGS1) con fallback
# =============================================================================
def build_rf_daily_series_web(index: pd.DatetimeIndex,
                              fallback_annual_rf: float = 0.04) -> tuple[pd.Series, str]:
    if len(index) == 0:
        return pd.Series(dtype=float), "RF: n/a"

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
        rf_daily = pd.Series(fallback_annual_rf / TRADING_DAYS, index=index)
        return rf_daily, f"Fixed RF {fallback_annual_rf:.2%} (ann., fallback)"

# =============================================================================
# Allocazioni (stub su Render)
# =============================================================================
def get_etf_allocations(ticker: str):
    return (
        pd.DataFrame(columns=["label", "weight"]),
        pd.DataFrame(columns=["label", "weight"]),
    )

def aggregate_allocations_portfolio(shares_df: pd.DataFrame,
                                    prices_df: pd.DataFrame,
                                    tickers: list[str],
                                    bench: str):
    return (
        pd.DataFrame(columns=["label", "weight_portfolio"]),
        pd.DataFrame(columns=["label", "weight_portfolio"]),
    )

# =============================================================================
# Download prezzi con yfinance (identico alla logica locale)
# =============================================================================
def _dl_series_yf(symbol: str,
                  start: pd.Timestamp,
                  end: pd.Timestamp,
                  use_adjclose: bool) -> pd.Series | None:
    h = yf.Ticker(symbol).history(
        start=start, end=end, interval="1d",
        auto_adjust=True if use_adjclose else False
    )
    col = "Close"
    if h is None or h.empty or col not in h.columns:
        return None
    s = h[col].rename(symbol.split(".")[0]).sort_index().ffill()
    return to_naive(s)

def robust_fetch_prices_for_symbol(sym: str,
                                   first_tx_date: pd.Timestamp,
                                   start_buffer_days: int,
                                   use_adjclose: bool) -> pd.Series | None:
    start = pd.Timestamp(first_tx_date.date() - timedelta(days=start_buffer_days))
    end = pd.Timestamp(datetime.today().date() + timedelta(days=2))
    for candidate in [sym, f"{sym}.US", f"{sym}.L", f"{sym}.MI"]:
        try:
            ser = _dl_series_yf(candidate, start, end, use_adjclose=use_adjclose)
            if ser is not None and not ser.empty:
                return ser.rename(sym)
        except Exception:
            pass
        time.sleep(0.1)
    return None

# =============================================================================
# ANALISI principale (identica nelle formule allo script locale)
# =============================================================================
def analyze_portfolio_from_text(lots_text: str,
                                bench: str,
                                outdir: str = "/tmp/outputs",
                                start_buffer_days: int = 7,
                                use_adjclose: bool = False):
    """
    Default = use_adjclose=False:
      - usa 'Close' yfinance
      - sostituisce il prezzo del giorno-trade con il prezzo del file (px_file)
    Così i risultati replicano lo script locale.
    """
    os.makedirs(outdir, exist_ok=True)

    lots = read_lots_from_text(lots_text)
    first_tx_date = lots["data"].min()
    bench = bench.upper()

    tickers_port = sorted(set(lots["ticker"].tolist()))
    all_tickers = sorted(set(tickers_port + [bench]))

    # Prezzi da yfinance
    series = {}
    for sym in all_tickers:
        s = robust_fetch_prices_for_symbol(sym, first_tx_date, start_buffer_days, use_adjclose=use_adjclose)
        if s is not None and not s.empty:
            series[sym] = s.astype(float)
        time.sleep(0.1)
    if not series:
        raise RuntimeError(f"Nessun prezzo disponibile per: {all_tickers}")

    px = pd.concat(series.values(), axis=1).sort_index().ffill()
    px = to_naive(px).asfreq("B").ffill()
    px = px.loc[px.index >= (first_tx_date - timedelta(days=start_buffer_days))]

    # Ricostruzione posizioni e override prezzo trade-day
    present = sorted(set(px.columns) & set(tickers_port))
    shares = pd.DataFrame(0.0, index=px.index, columns=present)
    px_eff = px.copy()

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

        # identico al locale: se NON adjclose, forziamo il prezzo del file al trade day
        if not use_adjclose and (sym in px_eff.columns) and (d_eff in px_eff.index):
            try:
                px_eff.at[d_eff, sym] = float(px_file)
            except Exception:
                pass

    # Valore portafoglio
    port_val = (shares * px_eff[present]).sum(axis=1)
    first_mv = port_val[port_val > 0].first_valid_index()
    if first_mv is None:
        raise RuntimeError("Portafoglio senza valore positivo (controlla i lotti).")
    port_val = port_val.loc[first_mv:].dropna()

    # Benchmark sui prezzi reali (nessun override)
    if bench in px.columns:
        bench_val_raw = px[bench].dropna()
        bench_name = bench
    else:
        bench_val_raw = port_val.copy()
        bench_name = "PORTFOLIO"

    idx_common = port_val.index.intersection(bench_val_raw.index)
    port_val = port_val.loc[idx_common]
    bench_val = bench_val_raw.loc[idx_common]
    if len(port_val) == 0 or len(bench_val) == 0:
        raise RuntimeError("Serie portafoglio/benchmark vuote dopo allineamento date.")

    # Cash flows (come locale: usa px_file quando not use_adjclose)
    cf = pd.Series(0.0, index=port_val.index)
    for sym, d0, d_eff, qty, px_file in trades:
        if d_eff in cf.index:
            invest_today = px_file if not use_adjclose else float(px.loc[d_eff, sym])
            cf.loc[d_eff] += qty * invest_today

    # TWR base 100
    dates = port_val.index
    twr_ret = []
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

    # Risk-free
    rf_daily, rf_meta = build_rf_daily_series_web(twr_ret.index)

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

    # Rischio 12m / Sharpe / VaR
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

    # Summary / IRR
    contrib = cf.clip(upper=0) * -1.0
    withdrw = cf.clip(lower=0)
    gross_contrib = float(contrib.sum())
    gross_withdrw = float(withdrw.sum())
    net_invested = float(cf.sum())

    current_value_port_val = current_value_port
    current_value_pme = float(bench_pme_val.iloc[-1])
    r_net_port = (current_value_port_val / net_invested - 1) if net_invested > 0 else np.nan
    r_net_bench = (current_value_pme / net_invested - 1) if net_invested > 0 else np.nan
    r_net_excess = (r_net_port - r_net_bench) if (np.isfinite(r_net_port) and np.isfinite(r_net_bench)) else np.nan

    cf_list = [(t, -float(cf.loc[t])) for t in port_val.index if abs(float(cf.loc[t])) != 0.0]
    cf_port = sorted(cf_list + [(port_val.index[-1], current_value_port_val)], key=lambda x: x[0])
    cf_bench = sorted(cf_list + [(port_val.index[-1], current_value_pme)], key=lambda x: x[0])
    irr_port = xirr(cf_port)
    irr_bench = xirr(cf_bench)
    irr_excess = (irr_port - irr_bench) if (np.isfinite(irr_port) and np.isfinite(irr_bench)) else np.nan

    # Grafico
    mode_label = "AdjClose (ignora prezzo file)" if use_adjclose else "Close (usa prezzo file al trade day)"
    plot_path = os.path.join(outdir, "crescita_cumulata.png")
    plt.figure(figsize=(10, 6))
    plt.plot(port_idx, label="Portafoglio (TWR base 100)")
    plt.plot(bench_idx, label=f"Benchmark {bench_name} (base 100)")
    plt.title(f"Andamento storico (base 100)\nMode: {mode_label} | {rf_meta}")
    plt.ylabel("Indice (base 100)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    # Alloc (stub)
    sectors_p, countries_p = aggregate_allocations_portfolio(
        shares_df=shares[present], prices_df=px, tickers=present, bench=bench_name
    )
    bench_sectors, bench_countries = get_etf_allocations(bench_name)
    if not bench_sectors.empty:
        bench_sectors = bench_sectors.rename(columns={"weight": "weight_portfolio"}).copy()
    else:
        bench_sectors = pd.DataFrame(columns=["label","weight_portfolio"])
    if not bench_countries.empty:
        bench_countries = bench_countries.rename(columns={"weight": "weight_portfolio"}).copy()
    else:
        bench_countries = pd.DataFrame(columns=["label","weight_portfolio"])

    # Summary lines
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

    # Salva summary.txt
    sum_path = os.path.join(outdir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Output
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
