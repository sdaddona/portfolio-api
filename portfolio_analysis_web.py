import os
import io
import re
import json
import time
import math
import requests
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt

TRADING_DAYS = 252

###############################################################################
# Caching in-memory per i prezzi (evita richieste duplicate a Yahoo)
###############################################################################

_price_cache = {}  # dict[(ticker, start_date_str, end_date_str)] = pandas.Series


###############################################################################
# Utils di base
###############################################################################

def to_naive(obj):
    # rimuove timezone dagli indici datetime di pandas
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
    # cf = list of (date, amount)
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


###############################################################################
# Lettura lotti dalla textarea (versione robusta)
###############################################################################

# --- parsing numeri "sporchi" (tab, NBSP, migliaia, virgola decimale, ecc.)
NBSP = u"\u00A0"

def _clean_num(token: str) -> float:
    """
    Converte stringhe 'sporche' in float.
    - Rimuove tutto tranne cifre, segno, virgola e punto
    - Gestisce separatore migliaia e virgola decimale
    - Accetta quantità negative
    """
    if token is None:
        raise ValueError("numero mancante")
    s = str(token).strip().replace(NBSP, " ")
    # tieni solo caratteri utili: cifre, segno, virgola, punto
    s = re.sub(r"[^0-9\-\+\,\.]", "", s)
    # se ci sono sia virgole che punti → considera la virgola separatore migliaia
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    if s in ("", "+", "-"):
        raise ValueError(f"numero vuoto: {token!r}")
    return float(s)

def read_lots_from_text(txt: str) -> pd.DataFrame:
    """
    Accetta righe con qualsiasi combinazione di spazi, tab, virgole o ';' come separatori.

    Formato:  TICKER  DATA(YYYY-MM-DD)  QUANTITA  PREZZO
    Esempi:
      ACWI 2012-12-21 200 47.81
      EEM  2012-12-21 235 43,2348
      ACWV 2025-09-23 -270 119.5
      DWM  2025-10-01 -190 66,43
    """
    if not isinstance(txt, str) or not txt.strip():
        raise ValueError("Nessuna riga valida nei lotti.")

    rows = []
    for raw in txt.splitlines():
        line = (raw or "").strip().replace(NBSP, " ")
        if not line or line.startswith("#"):
            continue

        # split permissivo: spazi, tab, virgole o ';'
        parts = re.split(r"[,\t;]+|\s+", line)
        if len(parts) < 4:
            # prova a comprimere spazi multipli e rifare split
            line2 = re.sub(r"\s+", " ", line)
            parts = re.split(r"[,\t;]+|\s+", line2)

        if len(parts) < 4:
            raise ValueError(f"Riga lotti invalida: {raw!r}")

        ticker = parts[0].strip().upper()

        # parsing data robusto (YYYY-MM-DD o day-first)
        try:
            dt = to_date(parts[1])
        except Exception:
            dt = pd.to_datetime(parts[1], errors="coerce", dayfirst=True)
            if pd.isna(dt):
                raise ValueError(f"Data non riconosciuta: {parts[1]!r} (riga: {raw!r})")
            dt = pd.to_datetime(dt).tz_localize(None)

        try:
            qty = _clean_num(parts[2])
        except Exception as e:
            raise ValueError(f"Quantità non valida: {parts[2]!r} (riga: {raw!r})") from e

        try:
            px = _clean_num(parts[3])
        except Exception as e:
            raise ValueError(f"Prezzo non valido: {parts[3]!r} (riga: {raw!r})") from e

        rows.append((ticker, dt, float(qty), float(px)))

    if not rows:
        raise ValueError("Nessuna riga valida nei lotti.")

    df = pd.DataFrame(rows, columns=["ticker", "data", "quantità", "prezzo"])
    df = df.sort_values("data").reset_index(drop=True)
    return df


###############################################################################
# Risk-free (prova FRED con fallback fisso)
###############################################################################

def fetch_fred_series(series_id: str,
                      start_date: pd.Timestamp,
                      end_date: pd.Timestamp,
                      api_key: str | None) -> pd.Series:
    """
    Scarica serie giornaliera dalla FRED API (es. DGS1 = 1Y Treasury yield)
    Ritorna Series con index datetime e valori float (% annuo).
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
    Ritorna:
      rf_daily = tasso risk-free giornaliero (frazione al giorno)
      rf_meta  = descrizione della sorgente
    1) prova FRED DGS1 (Treasury 1 anno)
    2) fallback costante (es. 4% annuo)
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
# Yahoo Finance raw fetch (senza yfinance)
###############################################################################

def _yahoo_chart_url(symbol: str,
                     start_ts: int,
                     end_ts: int) -> str:
    # base endpoint pubblico
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}"
        f"?period1={start_ts}&period2={end_ts}"
        "&interval=1d&events=div%2Csplit&includeAdjustedClose=true"
    )

def download_price_history_yahoo(ticker: str,
                                 start: pd.Timestamp,
                                 end: pd.Timestamp) -> pd.Series | None:
    """
    Scarica prezzi giornalieri Adjusted Close per ticker da Yahoo Finance.
    Ritorna pandas.Series indicizzata da datetime naive, name=ticker.
    Se fallisce o non trova dati -> None.
    """

    # Yahoo vuole epoch seconds (UTC)
    # start deve stare un po' prima perché se parte esattamente quel giorno
    # a volte non include la prima candela. Buffer di 3gg.
    start_utc = (start.tz_localize("UTC") - pd.Timedelta(days=3)).timestamp()
    end_utc   = (end.tz_localize("UTC") + pd.Timedelta(days=1)).timestamp()

    start_sec = int(start_utc)
    end_sec   = int(end_utc)

    # cache key
    cache_key = (ticker, str(start_sec), str(end_sec))
    if cache_key in _price_cache:
        return _price_cache[cache_key].copy()

    url = _yahoo_chart_url(ticker, start_sec, end_sec)

    try:
        r = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0"
        })
    except Exception:
        return None

    if r.status_code != 200:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    # Struttura attesa:
    # data["chart"]["result"][0]["timestamp"] -> [unix...]
    # data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"] -> [floats...]
    chart = data.get("chart", {})
    result_list = chart.get("result", [])
    if not result_list:
        return None

    result0 = result_list[0]
    timestamps = result0.get("timestamp", [])
    indicators = result0.get("indicators", {})
    adjclose_list = (
        indicators.get("adjclose", [{}])[0].get("adjclose", [])
    )

    if not timestamps or not adjclose_list:
        return None

    # costruiamo la series
    # timestamp sono epoch second -> convertiamo in datetime UTC -> naive
    idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)

    s = pd.Series(adjclose_list, index=idx, name=ticker, dtype=float)

    # togli eventuali NaN a fine serie
    s = s.dropna()
    if s.empty:
        return None

    # mettiamo in cache
    _price_cache[cache_key] = s.copy()

    return s


def robust_fetch_prices_for_symbol(sym: str,
                                   first_tx_date: pd.Timestamp,
                                   start_buffer_days: int = 7) -> pd.Series | None:
    """
    Prova a scaricare i prezzi per sym provando suffissi comuni,
    usando la fetch Yahoo custom.
    Ritorna Series (Adjusted Close) o None.
    """
    start = first_tx_date - timedelta(days=start_buffer_days)
    end = pd.Timestamp(datetime.today().date())

    # tentativi di ticker: sym, sym.US, sym.L, sym.MI
    candidates = [sym, f"{sym}.US", f"{sym}.L", f"{sym}.MI"]

    for ticker_try in candidates:
        ser = download_price_history_yahoo(ticker_try, start, end)
        if ser is not None and not ser.empty:
            # rinominiamo a sym "logico" (senza suffisso), così tutto il resto
            # del codice continua a vedere colonne con nomi semplici (VT, VOO, etc.)
            ser = ser.rename(sym)
            return ser

    return None


###############################################################################
# Allocazioni ETF (stub lato Render)
###############################################################################

def get_etf_allocations(ticker: str):
    """
    Stub (nessun scraping sul free tier).
    Ritorna tuple (df_sectors, df_countries) entrambe con colonne
    ['label','weight'] in frazione 0..1
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
    Somma pesata delle allocazioni degli ETF.
    Qui ritorniamo DataFrame vuoti perché su Render non possiamo ancora
    fare scraping con browser headless.
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
# ANALISI PORTAFOGLIO PRINCIPALE
###############################################################################

def analyze_portfolio_from_text(lots_text: str,
                                bench: str,
                                outdir: str = "/tmp/outputs",
                                start_buffer_days: int = 7):
    """
    Passi principali:
    1. parse lotti
    2. scarica prezzi (Yahoo raw API) per tickers e benchmark
    3. ricostruisce NAV portafoglio e TWR base 100
    4. confronta con benchmark base 100
    5. calcola Risk (vol, sharpe), VaR, PME, IRR, ecc.
    6. salva grafico e summary
    7. ritorna un dict serializzabile in JSON
    """

    os.makedirs(outdir, exist_ok=True)

    # 1. parse input lotti
    lots = read_lots_from_text(lots_text)
    first_tx_date = lots["data"].min()

    tickers_portafoglio = sorted(set(lots["ticker"].tolist()))
    all_tickers = sorted(set(tickers_portafoglio + [bench]))

    # 2. scarica prezzi per ogni ticker richiesto
    price_series = {}
    for sym in all_tickers:
        s = robust_fetch_prices_for_symbol(sym, first_tx_date, start_buffer_days)
        if s is not None and not s.empty:
            # s è una Series con index datetime e valori float
            price_series[sym] = s.astype(float)
        time.sleep(0.15)  # mini throttle per evitare di infastidire Yahoo

    # controllo: se non abbiamo NESSUN dato -> errore leggibile
    if len(price_series) == 0:
        raise RuntimeError(
            f"Nessun dato prezzi disponibile per i ticker richiesti: {all_tickers}."
        )

    # dataframe prezzi
    px_list = [ser for ser in price_series.values() if ser is not None and not ser.empty]
    if len(px_list) == 0:
        raise RuntimeError(
            f"Nessuna serie valida dopo il download. Tickers richiesti: {all_tickers}"
        )

    px = pd.concat(px_list, axis=1).sort_index().ffill()
    px = to_naive(px).asfreq("B").ffill()

    # tieni date da (prima operazione - buffer)
    px = px.loc[px.index >= (first_tx_date - timedelta(days=start_buffer_days))]

    # 3. ricostruisci posizione giornaliera (shares)
    present = sorted(set(px.columns) & set(tickers_portafoglio))
    shares = pd.DataFrame(0.0, index=px.index, columns=present)

    trades = []
    for _, r in lots.iterrows():
        sym = r["ticker"]
        d = r["data"]
        qty = float(r["quantità"])
        px_file = float(r["prezzo"])

        # mappiamo alla prima data di mercato >= d
        pos = px.index.searchsorted(d, side="left")
        if pos >= len(px.index):
            # fuori range
            continue
        d_eff = px.index[pos]

        trades.append((sym, d, d_eff, qty, px_file))

        if sym in shares.columns:
            shares.iloc[pos:, shares.columns.get_loc(sym)] += qty

    # 4. valore portafoglio nel tempo
    port_val = (shares * px[present]).sum(axis=1)

    first_mv_mask = port_val > 0
    if not first_mv_mask.any():
        raise RuntimeError("Portafoglio senza valore positivo (controlla i lotti).")

    first_mv_date = first_mv_mask.idxmax()
    port_val = port_val.loc[first_mv_date:].dropna()

    # benchmark
    if bench in px.columns:
        bench_val_raw = px[bench].dropna()
        bench_name = bench
    else:
        # fallback: se non ho il bench nei dati, uso il portafoglio stesso
        bench_val_raw = port_val.copy()
        bench_name = "PORTFOLIO"

    idx_common = port_val.index.intersection(bench_val_raw.index)
    port_val = port_val.loc[idx_common]
    bench_val = bench_val_raw.loc[idx_common]

    if len(port_val) == 0 or len(bench_val) == 0:
        raise RuntimeError("Serie portafoglio/benchmark vuote dopo allineamento date.")

    # 5. cash flow giornaliero (positivo = contribuzione/acquisto)
    cf = pd.Series(0.0, index=port_val.index)
    for _, d0, d_eff, qty, px_file in trades:
        if d_eff in cf.index:
            cf.loc[d_eff] += qty * px_file

    # 6. TWR portafoglio base 100
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

    # benchmark base 100
    bench_ret = bench_val.pct_change()
    bench_idx = (1 + bench_ret.iloc[1:]).cumprod() * 100.0
    bench_idx = pd.concat([pd.Series([100.0], index=bench_val.index[:1]), bench_idx])

    # 7. risk-free daily (FRED o fallback fisso)
    rf_daily, rf_meta = build_rf_daily_series_web(twr_ret.index)

    # 8. PME (replica dei flussi sul benchmark)
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

    # benchmark 12m
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

    # 10. summary IRR / PME
    contrib = cf.clip(upper=0) * -1.0   # soldi entrati (acquisti)
    withdrw = cf.clip(lower=0)          # soldi usciti (vendite positive)
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
        bench_sectors = bench_sectors.rename(columns={"weight": "weight_portfolio"}).copy()
    else:
        bench_sectors = pd.DataFrame(columns=["label","weight_portfolio"])
    if not bench_countries.empty:
        bench_countries = bench_countries.rename(columns={"weight": "weight_portfolio"}).copy()
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
        (f"Sharpe 12m – Portfolio [{rf_meta}]", f"{sharpe_port_12m:.2f}" if np.isfinite(sharpe_port_12m) else "n/a"),
        (f"Sharpe 12m – Benchmark [{rf_meta}]", f"{sharpe_bench_12m:.2f}" if np.isfinite(sharpe_bench_12m) else "n/a"),
        ("1D VaR(95%) – Portfolio (pct)", fmt_pct(var95_pct)),
        ("1D VaR(95%) – Portfolio (USD)", money(var95_usd) if np.isfinite(var95_usd) else "n/a"),
        ("1D VaR(95%) – Benchmark (pct)", fmt_pct(var95_bench_pct)),
        ("1D VaR(95%) – Benchmark (USD)", money(var95_bench_usd) if np.isfinite(var95_bench_usd) else "n/a"),
    ]
    for k, v in rows_out:
        summary_lines.append(f"{k.ljust(45)} {v}")

    # 14. salva summary.txt per eventuale download
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
