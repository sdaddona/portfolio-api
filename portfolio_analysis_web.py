# portfolio_analysis_web.py
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

TRADING_DAYS = 252

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Parser lotti robusto (tabs / spazi / virgola decimale)
# -----------------------------------------------------------------------------
def read_lots_from_text(txt: str) -> pd.DataFrame:
    """
    Formato atteso (tollerante a TAB, spazi multipli, virgole/punti come separatori,
    migliaia, spazi unificatori, ecc.):
      TICKER  DATA         QTA     PREZZO
      ACWI    2012-12-21   200     47.81
      ACWV    2017-05-04   270     77.9683
      EEM     2025-08-20   -150    49,71
    """
    def to_number_strict(x_raw: str) -> float:
        """
        Converte stringhe numeriche molto “sporche”:
        - rimuove spazi normali e non-breaking (U+00A0, U+202F)
        - gestisce migliaia e decimali con virgola o punto
        - accetta segni +/-
        Esempi validi: "1 234,56" -> 1234.56 ; "1,234.56" -> 1234.56 ; "77.9683" ; "-150"
        """
        if x_raw is None:
            raise ValueError("numero mancante")

        s = str(x_raw).strip()

        # normalizza spazi “strani”
        for ws in ("\u00A0", "\u202F"):
            s = s.replace(ws, "")
        s = s.replace(" ", "").replace("\t", "")

        # tieni solo cifre, virgole, punti e segni
        s = re.sub(r"[^0-9,.\-\+]", "", s)

        if s == "" or s in {"+", "-"}:
            raise ValueError(f"numero vuoto: '{x_raw}'")

        # se contiene sia virgola che punto, decide chi è il decimale guardando l’ultimo separatore
        if "," in s and "." in s:
            last_dot = s.rfind(".")
            last_com = s.rfind(",")
            if last_dot > last_com:
                # punto = decimale, togli le virgole come migliaia
                s = s.replace(",", "")
            else:
                # virgola = decimale, togli i punti come migliaia e sostituisci virgola con punto
                s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            # solo virgola presente: se più di una, consideriamo tutte migliaia tranne l’ultima
            if s.count(",") > 1:
                # es: "1,234,567,89" -> rimuovi tutte le virgole tranne l’ultima
                parts = s.split(",")
                s = "".join(parts[:-1]) + "." + parts[-1]
            else:
                s = s.replace(",", ".")
        else:
            # solo punto o niente: se più di un punto, rimuovi tutti tranne l’ultimo come migliaia
            if s.count(".") > 1:
                parts = s.split(".")
                s = "".join(parts[:-1]) + "." + parts[-1]

        try:
            return float(s)
        except Exception as ex:
            raise ValueError(f"numero non convertibile: '{x_raw}' -> '{s}'") from ex

    # --- pulizia righe ---
    lines = []
    for ln in txt.splitlines():
        ln_strip = (ln or "").strip()
        if not ln_strip or ln_strip.startswith("#"):
            continue
        lines.append(ln_strip)
    if not lines:
        raise ValueError("Nessuna riga valida nei lotti.")

    # --- parsing righe ---
    rows = []
    for ln in lines:
        # split molto tollerante: virgole, punto e virgola, TAB, spazi multipli
        parts = re.split(r"[,\t;]+|\s+", ln.strip())
        # filtra eventuali token vuoti residui
        parts = [p for p in parts if str(p).strip() != ""]
        if len(parts) < 4:
            raise ValueError(f"Riga lotti invalida (attese 4 colonne): {ln}")

        ticker = str(parts[0]).strip().upper()
        try:
            data = to_date(parts[1])
        except Exception:
            raise ValueError(f"Data non riconosciuta nella riga: {ln}")

        try:
            qty = to_number_strict(parts[2])   # consente anche quantità negative (vendite)
        except Exception as ex:
            raise ValueError(f"Quantità non valida nella riga: {ln} | {ex}")

        try:
            px = to_number_strict(parts[3])
        except Exception as ex:
            raise ValueError(f"Prezzo non valido nella riga: {ln} | {ex}")

        rows.append((ticker, data, qty, px))

    df = pd.DataFrame(rows, columns=["ticker", "data", "quantità", "prezzo"])
    df = df.sort_values("data").reset_index(drop=True)
    return df

# -----------------------------------------------------------------------------
# Risk-free (FRED se disponibile via requests? qui fallback fisso)
#   Su Render spesso le API esterne hanno limitazioni: usiamo un RF fisso
#   coerente con quanto avevamo già: 4% annuo di fallback.
# -----------------------------------------------------------------------------
def build_rf_daily_series_web(index: pd.DatetimeIndex,
                              fallback_annual_rf: float = 0.04) -> tuple[pd.Series, str]:
    rf_daily = pd.Series(fallback_annual_rf / TRADING_DAYS, index=index)
    rf_meta = f"Fixed RF {fallback_annual_rf:.2%} (ann., fallback)"
    return rf_daily, rf_meta

# -----------------------------------------------------------------------------
# Yahoo Chart API (raw)
#   Ritorna AdjClose giornalieri; headers per evitare blocchi
# -----------------------------------------------------------------------------
def _yahoo_chart_url(symbol: str, start_ts: int, end_ts: int) -> str:
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}"
        f"?period1={start_ts}&period2={end_ts}"
        "&interval=1d&events=div%2Csplit&includeAdjustedClose=true"
    )

def download_price_history_yahoo(ticker: str,
                                 start: pd.Timestamp,
                                 end: pd.Timestamp) -> pd.Series | None:
    start_utc = (start.tz_localize("UTC") - pd.Timedelta(days=3)).timestamp()
    end_utc   = (end.tz_localize("UTC") + pd.Timedelta(days=1)).timestamp()
    url = _yahoo_chart_url(ticker, int(start_utc), int(end_utc))

    try:
        r = requests.get(
            url,
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
                "Connection": "close",
            },
        )
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    chart = data.get("chart", {})
    results = chart.get("result", [])
    if not results:
        return None

    res0 = results[0]
    timestamps = res0.get("timestamp", [])
    indicators = res0.get("indicators", {})
    adj = indicators.get("adjclose", [{}])
    adjcl = adj[0].get("adjclose", []) if adj else []
    if not timestamps or not adjcl:
        return None

    idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
    s = pd.Series(adjcl, index=idx, name=ticker, dtype=float).dropna()
    if s.empty:
        return None
    return s

def robust_fetch_prices_for_symbol(sym: str,
                                   first_tx_date: pd.Timestamp,
                                   start_buffer_days: int = 7) -> pd.Series | None:
    start = first_tx_date - timedelta(days=start_buffer_days)
    end = pd.Timestamp(datetime.today().date())
    for t in [sym, f"{sym}.US", f"{sym}.L", f"{sym}.MI"]:
        s = download_price_history_yahoo(t, start, end)
        if s is not None and not s.empty:
            return s.rename(sym)
        time.sleep(0.15)
    return None

# -----------------------------------------------------------------------------
# Stub allocazioni (niente Playwright su Render free)
# -----------------------------------------------------------------------------
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
    # Nessun dato reale (vedi stub): ritorniamo DataFrame vuoti normalizzati
    sectors = pd.DataFrame(columns=["label","weight_portfolio"])
    countries = pd.DataFrame(columns=["label","weight_portfolio"])
    return sectors, countries

# -----------------------------------------------------------------------------
# Analisi principale (con modalità use_adjclose come nel tuo locale)
# -----------------------------------------------------------------------------
def analyze_portfolio_from_text(lots_text: str,
                                bench: str,
                                outdir: str = "/tmp/outputs",
                                start_buffer_days: int = 7,
                                use_adjclose: bool = False):
    os.makedirs(outdir, exist_ok=True)

    # 1) Lotti
    lots = read_lots_from_text(lots_text)
    first_tx_date = lots["data"].min()
    tickers_portafoglio = sorted(set(lots["ticker"].tolist()))
    all_tickers = sorted(set(tickers_portafoglio + [bench]))

    # 2) Prezzi (AdjClose Yahoo raw)
    series = {}
    for sym in all_tickers:
        s = robust_fetch_prices_for_symbol(sym, first_tx_date, start_buffer_days)
        if s is not None and not s.empty:
            series[sym] = s.astype(float)
        time.sleep(0.1)

    if not series:
        raise RuntimeError(f"Nessun prezzo disponibile per: {all_tickers}")

    px = pd.concat(series.values(), axis=1).sort_index().ffill()
    px = to_naive(px).asfreq("B").ffill()
    px = px.loc[px.index >= (first_tx_date - timedelta(days=start_buffer_days))]

    # 3) Costruzione posizioni + gestione prezzo trade day
    present = sorted(set(px.columns) & set(tickers_portafoglio))
    shares = pd.DataFrame(0.0, index=px.index, columns=present)

    trades = []
    for _, r in lots.iterrows():
        sym = r["ticker"]
        d = r["data"]
        qty = float(r["quantità"])
        px_file = float(r["prezzo"])
        d_eff = map_to_effective_date(d, px.index)
        if d_eff is None:
            continue
        px_eff_trade = float(px.loc[d_eff, sym]) if use_adjclose else px_file
        trades.append((sym, d, d_eff, qty, px_file, px_eff_trade))
        if sym in shares.columns:
            pos = shares.index.searchsorted(d_eff, side="left")
            if pos < len(shares.index):
                shares.iloc[pos:, shares.columns.get_loc(sym)] += qty

    # px_eff = px (AdjClose) ma se non use_adjclose sovrascriviamo il prezzo nel giorno trade
    px_eff = px.copy()
    if not use_adjclose:
        for sym, d0, d_eff, qty, px_file, px_eff_trade in trades:
            if sym in px_eff.columns and d_eff in px_eff.index:
                px_eff.at[d_eff, sym] = px_file

    # 4) Valori Portafoglio / Bench
    port_val = (shares * px_eff[present]).sum(axis=1)
    first_mv = port_val[port_val > 0].first_valid_index()
    if first_mv is None:
        raise RuntimeError("Portafoglio senza valore positivo (controlla i lotti).")
    port_val = port_val.loc[first_mv:].dropna()

    if bench in px.columns:
        bench_val = px[bench].loc[port_val.index[0]:].dropna()
        bench_name = bench
    else:
        bench_val = port_val.copy()
        bench_name = "PORTFOLIO"

    idx_common = port_val.index.intersection(bench_val.index)
    port_val = port_val.loc[idx_common]
    bench_val = bench_val.loc[idx_common]
    if len(port_val) == 0 or len(bench_val) == 0:
        raise RuntimeError("Serie portafoglio/benchmark vuote dopo allineamento date.")

    # 5) Cash flows (usiamo px_eff_trade per coerenza con la scelta Adj/Close)
    cf = pd.Series(0.0, index=port_val.index)
    for sym, d0, d_eff, qty, px_file, px_eff_trade in trades:
        if d_eff in cf.index:
            cf.loc[d_eff] += qty * px_eff_trade

    # 6) TWR base 100
    dates = port_val.index
    twr_ret = []
    for i in range(1, len(dates)):
        t, tm = dates[i], dates[i-1]
        mv_t, mv_tm = float(port_val.loc[t]), float(port_val.loc[tm])
        cf_t = float(cf.loc[t])
        twr_ret.append((mv_t - cf_t) / mv_tm - 1 if mv_tm > 0 else 0.0)
    twr_ret = pd.Series(twr_ret, index=dates[1:])
    port_idx = (1 + twr_ret).cumprod() * 100.0
    port_idx = pd.concat([pd.Series([100.0], index=dates[:1]), port_idx])

    bench_ret = bench_val.pct_change()
    bench_idx = (1 + bench_ret.iloc[1:]).cumprod() * 100.0
    bench_idx = pd.concat([pd.Series([100.0], index=bench_val.index[:1]), bench_idx])

    # 7) Risk-free
    rf_daily, rf_meta = build_rf_daily_series_web(twr_ret.index)

    # 8) PME
    bench_pme_val, units = [], 0.0
    for t in port_val.index:
        px_b = float(bench_val.loc[t])
        invest_today = float(cf.loc[t])
        if px_b != 0:
            units += invest_today / px_b
        bench_pme_val.append(units * px_b)
    bench_pme_val = pd.Series(bench_pme_val, index=port_val.index)

    # 9) Rischio 12m / Sharpe / VaR
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
    current_value_bench_pme = float(bench_pme_val.iloc[-1])
    var95_bench_pct = np.nan if np.isnan(sigma_1d_b) else z_95 * sigma_1d_b
    var95_bench_usd = np.nan if np.isnan(var95_bench_pct) else var95_bench_pct * current_value_bench_pme

    # 10) Grafico
    plot_path = os.path.join(outdir, "crescita_cumulata.png")
    mode_label = "AdjClose (ignora prezzo file)" if use_adjclose else "AdjClose con prezzo file al trade day"
    plt.figure(figsize=(10, 6))
    plt.plot(port_idx, label="Portafoglio (TWR, base 100)")
    plt.plot(bench_idx, label=f"Benchmark {bench_name} (base 100)")
    plt.title(f"Andamento storico (base 100)\nMode: {mode_label} | {rf_meta}")
    plt.ylabel("Indice (base 100)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    # 11) Allocazioni (stub)
    sectors_p, countries_p = aggregate_allocations_portfolio(
        shares_df=shares[present],
        prices_df=px_eff,
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

    # 12) Summary
    contrib = cf.clip(upper=0) * -1.0
    withdrw = cf.clip(lower=0)
    gross_contrib = float(contrib.sum())
    gross_withdrw = float(withdrw.sum())
    net_invested = float(cf.sum())

    r_net_port = (current_value_port / net_invested - 1) if net_invested > 0 else np.nan
    r_net_bench = (current_value_bench_pme / net_invested - 1) if net_invested > 0 else np.nan
    r_net_excess = (r_net_port - r_net_bench) if np.isfinite(r_net_port) and np.isfinite(r_net_bench) else np.nan

    cf_list = [(t, -float(cf.loc[t])) for t in port_val.index if abs(float(cf.loc[t])) != 0.0]
    cf_port = sorted(cf_list + [(port_val.index[-1], current_value_port)], key=lambda x: x[0])
    cf_bench = sorted(cf_list + [(port_val.index[-1], current_value_bench_pme)], key=lambda x: x[0])
    irr_port = xirr(cf_port)
    irr_bench = xirr(cf_bench)
    irr_excess = (irr_port - irr_bench) if np.isfinite(irr_port) and np.isfinite(irr_bench) else np.nan

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
        ("Current Value – Portfolio", money(current_value_port)),
        ("Current Value – Bench (PME)", money(current_value_bench_pme)),
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

    # 13) Salva summary.txt
    sum_path = os.path.join(outdir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # 14) Output API
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
            "current_value_port": current_value_port,
            "current_value_pme": current_value_bench_pme,
            "net_invested": float(cf.sum()),
        },
        "alloc": {
            "portfolio_sectors": sectors_p.to_dict(orient="records"),
            "portfolio_countries": countries_p.to_dict(orient="records"),
            "bench_sectors": bench_sectors.to_dict(orient="records"),
            "bench_countries": bench_countries.to_dict(orient="records"),
        },
    }
    return out
