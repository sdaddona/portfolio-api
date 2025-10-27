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
import yfinance as yf

TRADING_DAYS = 252


###############################################################################
# Utils base
###############################################################################

def to_naive(obj):
    """
    Rimuove timezone dagli indici datetime di Series/DataFrame (operando in place).
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)) and isinstance(obj.index, pd.DatetimeIndex):
        try:
            obj.index = obj.index.tz_localize(None)
        except Exception:
            pass
    return obj


def to_date(s: str) -> pd.Timestamp:
    """
    Prova a interpretare una stringa come data.
    Supporta "YYYY-MM-DD" oppure formati comuni tipo "DD/MM/YYYY".
    Restituisce un Timestamp naive (senza tz).
    """
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
    """
    Valore attuale netto di una serie di flussi cf = [(date, amount), ...]
    r è il tasso annuo.
    """
    t0 = cf[0][0]
    return sum(a / (1 + r) ** ((t - t0).days / 365.25) for t, a in cf)


def xirr(cf):
    """
    IRR dei flussi cf (date crescenti).
    Se non ci sono sia flussi positivi sia negativi → NaN.
    """
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
    txt deve avere righe tipo:
      TICKER DATA QUANTITA PREZZO
      VT 2024-01-02 10 100
      VT 2024-03-10 5 95

    Ritorna DataFrame con colonne:
    ticker (str), data (Timestamp), quantità (float), prezzo (float)
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
    Scarica serie FRED (es. DGS1: 1Y Treasury yield, % annuo) e
    restituisce una pandas.Series indicizzata per data.
    In caso di errore: Series vuota.
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
    Ritorna:
    - rf_daily: Series dell'rf giornaliero (frazzionale tipo 0.0001)
    - rf_meta: descrizione stringa usata nei summary

    1) prova DGS1 (1Y Treasury yield) da FRED
    2) se non disponibile → fallback costante, es. 4% annuo
    """
    api_key = os.getenv("FRED_API_KEY", "").strip()
    start_fetch = index[0] - pd.Timedelta(days=10)
    end_fetch   = index[-1] + pd.Timedelta(days=3)

    dgs1 = fetch_fred_series("DGS1", start_fetch, end_fetch, api_key)
    if not dgs1.empty:
        dgs1 = dgs1.reindex(index).ffill()
        rf_annual = dgs1 / 100.0  # es. 4.2 -> 0.042
        rf_daily = rf_annual / TRADING_DAYS
        rf_meta = "RF: FRED DGS1 (1Y Treasury, ann.)"
        return rf_daily, rf_meta

    rf_daily = pd.Series(fallback_annual_rf / TRADING_DAYS, index=index)
    rf_meta = f"Fixed RF {fallback_annual_rf:.2%} (ann., fallback)"
    return rf_daily, rf_meta


###############################################################################
# Allocazioni ETF (stub su Render)
###############################################################################

def get_etf_allocations(ticker: str):
    """
    Stub: su Render free non possiamo fare scraping headless serio.
    Ritorna due DataFrame vuoti (settori, paesi).
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
    Combina le allocazioni ETF pesandole per il peso di mercato nel portafoglio.
    Su Render → spesso vuoto, ed è ok.
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
# Download prezzi robusto con retry
###############################################################################

def download_price_history(ticker: str,
                           start: pd.Timestamp,
                           end: pd.Timestamp):
    """
    Scarica storico daily con yfinance (auto_adjust=True).
    Restituisce una Series Close indicizzata per data (tz naive), ffillata.
    Se vuoto -> None.
    """
    try:
        h = yf.Ticker(ticker).history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True
        )
    except Exception:
        return None

    if h is None or h.empty or "Close" not in h.columns:
        return None

    s = h["Close"].rename(ticker).sort_index().ffill()
    s = to_naive(s)
    return s


def robust_fetch_prices_for_symbol(sym: str,
                                   first_tx_date: pd.Timestamp,
                                   start_buffer_days: int = 7) -> pd.Series | None:
    """
    Tenta di ottenere la serie prezzi per un ticker, provando:
    - il ticker "così com'è"
    - suffissi comuni (.US, .L, .MI)
    - lookback 1y poi 10y
    Ritorna Series (index datetime, valori float) o None.
    """
    candidates = [sym, f"{sym}.US", f"{sym}.L", f"{sym}.MI"]
    candidates = list(dict.fromkeys(candidates))  # dedup mantenendo ordine

    starts = [
        first_tx_date - timedelta(days=start_buffer_days + 365),
        first_tx_date - timedelta(days=start_buffer_days + 365*10),
    ]
    end = pd.Timestamp(datetime.today().date() + timedelta(days=2))

    for ticker_candidate in candidates:
        for st in starts:
            s = download_price_history(ticker_candidate, st, end)
            if s is not None and not s.empty:
                # rinomina col ticker logico (sym) così rimane coerente nel portafoglio
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
    Passi:
    - parse lotti
    - scarica prezzi robusti per TUTTI i ticker + benchmark
    - costruisce NAV portafoglio, TWR base 100
    - costruisce benchmark base 100 (con fallback benchmark->PORTFOLIO)
    - calcola Sharpe, VaR, PME, IRR
    - salva grafico cumulato
    - ritorna dict pronto per l'API
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

    # controllo duro: se non ho nemmeno un ticker con dati
    if len(price_series) == 0:
        raise RuntimeError(
            f"Nessun dato prezzi scaricato da Yahoo Finance. "
            f"Tickers richiesti: {all_tickers}"
        )

    # 3. costruisci dataframe prezzi
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

    # filtra date dalla prima operazione
    px = px.loc[px.index >= (first_tx_date - timedelta(days=start_buffer_days))]

    # 4. Se il benchmark era mancante, lo sostituiremo col portafoglio più avanti
    if benchmark_is_fallback:
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

        # trova la prima data di mercato >= d
        pos = px.index.searchsorted(d, side="left")
        if pos >= len(px.index):
            # se non c'è nessuna data >= d (dopo tutti i filtri), saltiamo questa operazione
            continue
        d_eff = px.index[pos]

        trades.append((sym, d, d_eff, qty, px_file))

        if sym in shares.columns:
            # aggiungi quantità da d_eff in avanti
            shares.iloc[pos:, shares.columns.get_loc(sym)] += qty

    # 6. valore portafoglio giornaliero
    port_val = (shares * px[present]).sum(axis=1)

    # tieni solo da quando il NAV è >0
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
            # edge case: bench dichiarato non fallback ma non presente in px
            bench_val_raw = port_val.copy()
            bench_name = "PORTFOLIO"
        else:
            bench_val_raw = px[bench].dropna()
            bench_name = bench

    # allinea sulle stesse date
    idx_common = port_val.index.intersection(bench_val_raw.index)
    port_val = port_val.loc[idx_common]
    bench_val = bench_val_raw.loc[idx_common]

    if len(port_val) == 0 or len(bench_val) == 0:
        raise RuntimeError("Serie portafoglio/benchmark vuote dopo allineamento date.")

    # 8. cash flow giornaliero
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

    # 11. risk-free giornaliero
    rf_daily, rf_meta = build_rf_daily_series_web(twr_ret.index)

    # 12. PME (replico i cashflow sul benchmark)
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

    # 14. summary IRR / PME
    contrib = cf.clip(upper=0) * -1.0   # soldi entrati (acquisti)
    withdrw = cf.clip(lower=0)          # soldi usciti (vendite positive)
    gross_contrib = float(contrib.sum())
    gross_withdrw = float(withdrw.sum())
    net_invested = float(cf.sum())

    current_value_port_val = current_value_port
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

    # 18. salva summary.txt
    sum_path = os.path.join(outdir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # 19. ritorno finale per l'API
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
