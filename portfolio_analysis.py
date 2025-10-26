#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portafoglio:
- storico TWR (base 100) vs Benchmark (base 100)
- SUMMARY (PME/IRR)
- Rischio (Sharpe 12m con RF FRED DGS1 o ^IRX o fixed, VaR 1d)
- Allocazioni SETTORI / PAESI tramite ETFdb (Playwright, sez. Charts)
  e confronto Portafoglio vs Benchmark (CSV + grafici differenze).

Esempi:
  python portfolio_full.py --lots lots.txt --bench VT --alloc --headless
  python portfolio_full.py --lots lots.txt --bench VT --alloc --no-headless
"""

import os
import re
import io
import json
import time
import argparse
import warnings
from io import StringIO
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr  # FRED
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=UserWarning)
TRADING_DAYS = 252

# =============================================================================
# ----------------------- UTIL -----------------------
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

def try_symbol(sym: str) -> str:
    """Risolvi ticker su Yahoo con suffissi comuni."""
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

# =============================================================================
# ----------------------- LETTURA LOTTI -----------------------
# =============================================================================

def read_lots(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    txt = open(path, "r", encoding="utf-8").read()
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("File lotti vuoto.")
    sample = "\n".join(lines)

    def _try(buf, header=None):
        # separatori: più spazi, tab, virgole, punto e virgola
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

    def to_float(x): return float(str(x).strip().replace(" ", "").replace(",", "."))
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["quantità"] = df["quantità"].apply(to_float)   # vendite possibili (<0)
    df["prezzo"] = df["prezzo"].apply(to_float)
    df["data"] = df["data"].apply(to_date)
    return df.sort_values("data").reset_index(drop=True)

# =============================================================================
# ----------------------- RISK-FREE -----------------------
# =============================================================================

def build_rf_daily_series(index: pd.DatetimeIndex, rf_source: str, rf_fixed: float):
    rf_source = (rf_source or "fred_1y").lower()

    if rf_source == "fixed":
        rf_daily = pd.Series(rf_fixed / TRADING_DAYS, index=index)
        rf_meta = f"Fixed RF {rf_fixed:.2%} (ann.)"
        return rf_daily, rf_meta

    if rf_source == "irx_13w":
        tkr = "^IRX"
        try:
            h = yf.Ticker(tkr).history(start=index[0]-pd.Timedelta(days=10),
                                       end=index[-1]+pd.Timedelta(days=3),
                                       interval="1d", auto_adjust=False)
            ser = h["Close"].rename("IRX").sort_index()
            ser = to_naive(ser)
            ser = ser.reindex(index).ffill()
            rf_annual = ser / 100.0
            rf_daily = rf_annual / TRADING_DAYS
            rf_meta = "RF: ^IRX (13W T-Bill, ann.)"
            return rf_daily, rf_meta
        except Exception:
            rf_daily = pd.Series(0.0, index=index)
            return rf_daily, "RF fallback 0.00% (IRX download failed)"

    # default: FRED DGS1 (1-Year Treasury)
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

# =============================================================================
# ----------------------- ETFDB SCRAPER (Allocazioni) -----------------------
# =============================================================================

def _clean_pct(txt: str) -> float | None:
    """
    '62.66%' -> 0.6266
    return frazione 0..1
    """
    if txt is None:
        return None
    t = txt.strip().replace(",", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", t)
    if not m:
        return None
    val = float(m.group(1))
    return val / 100.0

def _df_from_two_col_table(rows: list[tuple[str,str]]) -> pd.DataFrame:
    out = []
    for k, v in rows:
        if not k or not v:
            continue
        w = _clean_pct(v)
        if w is None:
            continue
        out.append({"label": k.strip(), "weight": w})
    if not out:
        return pd.DataFrame(columns=["label","weight"])
    df = pd.DataFrame(out)
    # collassa duplicati tipo "Other"
    df = df.groupby("label", as_index=False)["weight"].sum()
    tot = df["weight"].sum()
    if tot > 0:
        df["weight"] = df["weight"] / tot
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    return df

def _parse_tables_and_legends(html: str) -> dict:
    """
    Prende l'HTML della pagina ETFdb (tab Charts già aperto & scrollato)
    e ritorna un dict con potenziali candidate:
      {
        "country_tables": [df1, df2, ...],
        "country_legends": [df3, ...],
        "sector_tables": [...],
        "sector_legends": [...]
      }
    dove cada df ha colonne ['label','weight'] (0..1)
    """
    soup = BeautifulSoup(html, "html.parser")

    out = {
        "country_tables": [],
        "country_legends": [],
        "sector_tables": [],
        "sector_legends": [],
    }

    # ---- TABELLE CLASSICHE <table> ----
    # cerchiamo le tabelle con 2 colonne "Country | Percentage" o "Sector | Percentage"
    for tb in soup.find_all("table"):
        # prendi header <th>
        headers = [th.get_text(" ", strip=True).lower() for th in tb.find_all("th")]
        body_rows = []
        for tr in tb.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) != 2:
                continue
            k = tds[0].get_text(" ", strip=True)
            v = tds[1].get_text(" ", strip=True)
            if k and v:
                body_rows.append((k, v))
        if len(body_rows) >= 3:
            df_try = _df_from_two_col_table(body_rows)
            if df_try.empty:
                continue

            # heuristics: se header contiene "country", è country table
            if any("country" in h for h in headers) or any("geographic" in h for h in headers):
                out["country_tables"].append(df_try)
            # se header contiene "sector", è settori
            elif any("sector" in h for h in headers) or any("holdings analysis" in h for h in headers):
                out["sector_tables"].append(df_try)
            else:
                # fallback ulteriore:
                # se le label sembrano paesi tipici, metti in country_tables
                if (df_try["label"].str.contains("united states", case=False).any()
                    or df_try["label"].str.contains("japan", case=False).any()):
                    out["country_tables"].append(df_try)
                # se le label sembrano settori tech/finance ecc -> settori
                elif (df_try["label"].str.contains("technology", case=False).any()
                      or df_try["label"].str.contains("finance", case=False).any()
                      or df_try["label"].str.contains("services", case=False).any()):
                    out["sector_tables"].append(df_try)

    # ---- LEGENDS (div.chart-legend) ----
    legends = soup.find_all("div", class_=re.compile(r"(chart-legend|highcharts-legend)"))
    for lg in legends:
        # capiamo il titolo guardando heading vicino (h3/h4 precedente)
        # e prendiamo tutte le (label -> pct) coppie
        # pattern: dentro la legend ci sono vari <div> con testo label e percentuale
        text_chunks = []
        # catturiamo tutti i div figli che abbiano testo non vuoto
        for d in lg.find_all("div"):
            txt = d.get_text(" ", strip=True)
            if txt:
                text_chunks.append(txt)

        pairs = []
        last_label = None
        for t in text_chunks:
            if "%" in t:
                # è percentuale
                if last_label:
                    pairs.append((last_label, t))
                last_label = None
            else:
                # è label candidato
                last_label = t

        if not pairs:
            continue
        df_leg = _df_from_two_col_table(pairs)
        if df_leg.empty:
            continue

        # heuristic per capire se è country o sector:
        # country se troviamo "United States", "Japan", "China", etc.
        if (df_leg["label"].str.contains("united states", case=False).any()
            or df_leg["label"].str.contains("japan", case=False).any()
            or df_leg["label"].str.contains("china", case=False).any()
            or df_leg["label"].str.contains("korea", case=False).any()):
            out["country_legends"].append(df_leg)
        # sector se trovi parole tipo Technology, Finance, Services, etc.
        if (df_leg["label"].str.contains("technology", case=False).any()
            or df_leg["label"].str.contains("finance", case=False).any()
            or df_leg["label"].str.contains("services", case=False).any()
            or df_leg["label"].str.contains("manufacturing", case=False).any()
            or df_leg["label"].str.contains("utilities", case=False).any()
            or df_leg["label"].str.contains("energy", case=False).any()):
            out["sector_legends"].append(df_leg)

    return out

def _pick_best(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Scegli la df 'migliore'. Heuristica semplice: più righe = meglio."""
    if not df_list:
        return pd.DataFrame(columns=["label","weight"])
    best = sorted(df_list, key=lambda d: len(d), reverse=True)[0].copy()
    # normalizza somma=1
    if "weight" in best.columns and best["weight"].sum() > 0:
        best["weight"] = best["weight"] / best["weight"].sum()
    return best.reset_index(drop=True)

def scrape_ticker_allocations_etfdb(
        ticker: str,
        outdir: str,
        headless: bool = True,
        scroll_pause_ms: int = 800
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apre https://etfdb.com/etf/<ticker>/#charts con Playwright,
    scrolla per far caricare i grafici,
    cattura HTML,
    estrae:
      - df_sectors ['label','weight']
      - df_countries ['label','weight']
    dove weight è frazione [0..1].
    Salva raw HTML e CSV (se trovati).
    """
    os.makedirs(outdir, exist_ok=True)
    url = f"https://etfdb.com/etf/{ticker}/#charts"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1400, "height": 900},
            java_script_enabled=True,
        )
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        # prova a cliccare "Charts" tab se esiste
        try:
            charts_tab_button = page.locator("a[href='#charts_tab']")
            if charts_tab_button.count() > 0:
                charts_tab_button.first.click()
        except Exception:
            pass

        # scrolla giù parecchio per far caricare tutte le tabelle/grafici
        try:
            for _ in range(10):
                page.mouse.wheel(0, 1200)
                page.wait_for_timeout(scroll_pause_ms)
        except Exception:
            pass

        html = page.content()
        browser.close()

    raw_path = os.path.join(outdir, f"{ticker}_raw.html")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(html)

    parsed = _parse_tables_and_legends(html)

    df_countries = _pick_best(parsed["country_tables"] + parsed["country_legends"])
    df_sectors   = _pick_best(parsed["sector_tables"]  + parsed["sector_legends"])

    # salva csv singolo-ticker per debug/cache
    if not df_countries.empty:
        df_countries.to_csv(os.path.join(outdir, f"{ticker}_countries.csv"), index=False)
    if not df_sectors.empty:
        df_sectors.to_csv(os.path.join(outdir, f"{ticker}_sectors.csv"), index=False)

    return df_sectors, df_countries

# =============================================================================
# ----------------------- AGGREGAZIONE ALLOCAZIONI PORTAFOGLIO ----------------
# =============================================================================

def aggregate_allocations_portfolio_etfdb(
        shares_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        etfs: list[str],
        outdir: str,
        headless: bool
    ):
    """
    Calcola:
      - peso di ogni ETF nel portafoglio alla data finale
      - per ciascun ETF, scarica allocazioni da ETFdb (settori & paesi)
      - aggrega pesando per il peso ETF
    Ritorna:
      sectors_portfolio, countries_portfolio
      (column 'label', 'weight_portfolio' in frazione [0..1])
    """
    end_date = shares_df.index[-1]
    mv = shares_df.loc[end_date] * prices_df.loc[end_date, shares_df.columns]
    w_etf = (mv / mv.sum()).fillna(0.0)
    w_etf = w_etf[w_etf > 0]
    w_etf = w_etf[w_etf.index.isin(etfs)]

    sector_rows = []
    country_rows = []

    for etf, w in w_etf.items():
        try:
            sec_df, ctry_df = scrape_ticker_allocations_etfdb(
                etf, outdir=outdir, headless=headless
            )
        except Exception as e:
            print(f"⚠️ Errore scrape ETFdb per {etf}: {e}")
            sec_df = pd.DataFrame(columns=["label","weight"])
            ctry_df = pd.DataFrame(columns=["label","weight"])

        if sec_df is None or sec_df.empty:
            print(f"⚠️ Settori non disponibili per {etf}")
        else:
            tmp = sec_df.copy()
            tmp["weight_portfolio"] = w * tmp["weight"]  # contrib ETF -> portafoglio
            tmp["etf"] = etf
            sector_rows.append(tmp[["etf","label","weight_portfolio"]])

        if ctry_df is None or ctry_df.empty:
            print(f"⚠️ Paesi non disponibili per {etf}")
        else:
            tmp2 = ctry_df.copy()
            tmp2["weight_portfolio"] = w * tmp2["weight"]
            tmp2["etf"] = etf
            country_rows.append(tmp2[["etf","label","weight_portfolio"]])

        time.sleep(0.25)

    if sector_rows:
        sectors = (
            pd.concat(sector_rows)
            .groupby("label", as_index=False)["weight_portfolio"]
            .sum()
            .sort_values("weight_portfolio", ascending=False)
        )
    else:
        sectors = pd.DataFrame(columns=["label","weight_portfolio"])

    if country_rows:
        countries = (
            pd.concat(country_rows)
            .groupby("label", as_index=False)["weight_portfolio"]
            .sum()
            .sort_values("weight_portfolio", ascending=False)
        )
    else:
        countries = pd.DataFrame(columns=["label","weight_portfolio"])

    # normalizza in modo che sommino a 1
    def _norm(df):
        if df.empty:
            return df
        tot = df["weight_portfolio"].sum()
        if tot > 0:
            df = df.copy()
            df["weight_portfolio"] = df["weight_portfolio"] / tot
        return df

    sectors = _norm(sectors)
    countries = _norm(countries)

    return sectors, countries

def compare_allocations(port_df: pd.DataFrame, bench_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    p = port_df.set_index(label_col)["weight_portfolio"] if not port_df.empty else pd.Series(dtype=float)
    b = bench_df.set_index(label_col)["weight_portfolio"] if not bench_df.empty else pd.Series(dtype=float)
    idx = p.index.union(b.index)
    comp = pd.DataFrame({
        "Portfolio": p.reindex(idx).fillna(0.0),
        "Benchmark": b.reindex(idx).fillna(0.0),
    })
    comp["Diff (Port - Bench)"] = comp["Portfolio"] - comp["Benchmark"]
    comp = comp.sort_values("Diff (Port - Bench)", ascending=False).reset_index().rename(columns={"index": label_col})
    return comp

def bar_diff_plot(comp_df: pd.DataFrame, label_col: str, title: str, outpath: str, topn: int = 12):
    if comp_df is None or comp_df.empty:
        return
    d = comp_df.copy()
    d = d.iloc[d["Diff (Port - Bench)"].abs().sort_values(ascending=False).index][:topn]
    d = d.sort_values("Diff (Port - Bench)", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(d[label_col], d["Diff (Port - Bench)"])
    plt.axvline(0, linestyle="--")
    plt.title(title)
    plt.xlabel("Differenza di peso (frazione: 0.05 = 5%)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)

# =============================================================================
# ----------------------- MAIN -----------------------
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Portafoglio TWR vs Benchmark + SUMMARY + Rischio + Allocazioni ETFdb.")
    ap.add_argument("--lots", type=str, default="lots.txt")
    ap.add_argument("--bench", type=str, default="VT")
    ap.add_argument("--outdir", type=str, default="backtest_outputs")
    ap.add_argument("--start-buffer-days", type=int, default=7)
    ap.add_argument("--use-adjclose", action="store_true",
                    help="Usa Adj Close per tutti e ignora il prezzo del file (usa solo data/qty). [Default OFF: usa Close e prezzo file al trade day]")
    ap.add_argument("--rf-source", type=str, default="fred_1y",
                    choices=["fred_1y", "irx_13w", "fixed"], help="Sorgente risk-free per Sharpe/Var.")
    ap.add_argument("--rf", type=float, default=0.0, help="Risk-free annuo (solo se --rf-source fixed).")
    ap.add_argument("--alloc", action="store_true", help="Calcola allocazioni (settori/paesi) e confronto col benchmark.")
    ap.add_argument("--headless", dest="headless", action="store_true", default=True,
                    help="Browser invisibile (default).")
    ap.add_argument("--no-headless", dest="headless", action="store_false",
                    help="Mostra browser (debug).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------- LOTTI ----------------
    lots = read_lots(args.lots)
    first_tx_date = lots["data"].min()
    bench = args.bench.upper()

    # ---------------- RISOLUZIONE TICKER SU YAHOO ----------------
    tickers = sorted(set(lots["ticker"].tolist()) | {bench})
    resolved: Dict[str, str] = {}
    print("=== Risoluzione ticker su Yahoo ===")
    for s in tickers:
        rs = try_symbol(s)
        resolved[s] = rs
        print(("✅" if rs else "⚠️"), f"{s} -> {rs or 'NON TROVATO'}")
    if not resolved.get(bench):
        raise SystemExit(f"Benchmark {bench} non risolto su Yahoo")

    # ---------------- PREZZI ----------------
    start = pd.Timestamp(first_tx_date.date() - timedelta(days=args.start_buffer_days))
    end = pd.Timestamp(datetime.today().date() + timedelta(days=2))
    series = {}
    print("\n=== Download prezzi (Close/AdjClose) ===")
    for k, y in resolved.items():
        if not y:
            continue
        h = yf.Ticker(y).history(start=start, end=end, interval="1d",
                                 auto_adjust=True if args.use_adjclose else False)
        col = "Close"
        if h is None or h.empty or col not in h.columns:
            print(f"⚠️ {k}: vuoto")
            continue
        s = h[col].rename(k).sort_index().ffill()
        series[k] = to_naive(s)
        print(f"→ {k} ok (last {s.index[-1].date()} {float(s.iloc[-1]):.4f})")
        time.sleep(0.1)
    if not series:
        raise SystemExit("Nessun prezzo scaricato.")
    px = pd.concat(series.values(), axis=1).sort_index().ffill()
    px = to_naive(px).asfreq("B").ffill()

    # ---------------- COSTRUZIONE POSIZIONI (shares nel tempo) ----------------
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
        px_eff_trade = float(px.loc[d_eff, sym]) if args.use_adjclose else px_file
        trades.append((sym, d, d_eff, qty, px_file, px_eff_trade))

    px_eff = px.copy()
    if not args.use_adjclose:
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

    # ---------------- VALORE PORTAFOGLIO vs BENCH ----------------
    port_val = (shares * px_eff[present]).sum(axis=1)
    first_mv = port_val[port_val > 0].first_valid_index()
    port_val = port_val.loc[first_mv:].dropna()

    bench_val = px[bench].loc[port_val.index[0]:].dropna()
    idx_common = port_val.index.intersection(bench_val.index)
    port_val = port_val.loc[idx_common]
    bench_val = bench_val.loc[idx_common]

    # ---------------- CASH FLOWS ----------------
    cf = pd.Series(0.0, index=port_val.index)
    for sym, d0, d_eff, qty, px_file, px_eff_trade in trades:
        if d_eff in cf.index:
            cf.loc[d_eff] += qty * (px_eff_trade)

    # ---------------- TWR base 100 ----------------
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

    # Benchmark base 100 (prezzi)
    bench_ret = bench_val.pct_change()
    bench_idx = (1 + bench_ret.iloc[1:]).cumprod() * 100.0
    bench_idx = pd.concat([pd.Series([100.0], index=bench_val.index[:1]), bench_idx])

    # ---------------- RF daily ----------------
    rf_daily, rf_meta = build_rf_daily_series(twr_ret.index, args.rf_source, args.rf)

    # ---------------- PME (replica flussi sul benchmark) ----------------
    bench_pme_val = []
    units = 0.0
    for t in port_val.index:
        px_b = float(bench_val.loc[t])
        invest_today = float(cf.loc[t])
        if px_b != 0:
            units += invest_today / px_b
        bench_pme_val.append(units * px_b)
    bench_pme_val = pd.Series(bench_pme_val, index=port_val.index)

    # ---------------- RISK METRICS 12m ----------------
    lb = TRADING_DAYS

    # --- Portafoglio ---
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

    # --- Benchmark ---
    bench_r_12m = (bench_ret.iloc[-lb:] if len(bench_ret) >= lb else bench_ret.dropna()).copy()
    rf_12m_b = rf_daily.reindex(bench_r_12m.index).ffill().fillna(0.0)
    vol_bench_12m = np.nan if bench_r_12m.empty else float(bench_r_12m.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe_bench_12m = np.nan
    if not bench_r_12m.empty and bench_r_12m.std(ddof=1) > 0:
        exb = bench_r_12m - rf_12m_b
        sharpe_bench_12m = float(np.sqrt(TRADING_DAYS) * exb.mean() / exb.std(ddof=1))
    sigma_1d_b = float(bench_r_12m.std(ddof=1)) if not bench_r_12m.empty else np.nan
    var95_bench_pct = np.nan if np.isnan(sigma_1d_b) else z_95 * sigma_1d_b
    current_value_bench_pme = float(bench_pme_val.iloc[-1])
    var95_bench_usd = np.nan if np.isnan(var95_bench_pct) else var95_bench_pct * current_value_bench_pme

    # ---------------- GRAFICO storico Port vs Bench ----------------
    out_hist = os.path.join(args.outdir, "01_crescita_cumulata.png")
    plt.figure(figsize=(10, 6))
    plt.plot(port_idx, label="Portafoglio (TWR, base 100)")
    plt.plot(bench_idx, label=f"Benchmark {bench} (prezzo, base 100)")
    mode_label = "AdjClose (ignora prezzo file)" if args.use_adjclose else "Close (usa prezzo file al trade day)"
    plt.title(f"Andamento storico (base 100)\nMode: {mode_label} | {rf_meta}")
    plt.ylabel("Indice (base 100)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_hist, dpi=160)

    # ---------------- SUMMARY ----------------
    contrib = cf.clip(upper=0) * -1.0
    withdrw = cf.clip(lower=0)
    gross_contrib = float(contrib.sum())
    gross_withdrw = float(withdrw.sum())
    net_invested = float(cf.sum())
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

    # ---------------- ALLOCATIONS (ETFdb) ----------------
    if args.alloc:
        print("\n=== Allocazioni (settori/paesi) da ETFdb ===")

        # Portafoglio
        sectors_p, countries_p = aggregate_allocations_portfolio_etfdb(
            shares[present], px_eff, present,
            outdir=args.outdir,
            headless=args.headless
        )

        # Benchmark singolo (es. VT)
        try:
            s_bench, c_bench = scrape_ticker_allocations_etfdb(
                bench,
                outdir=args.outdir,
                headless=args.headless
            )
        except Exception as e:
            print(f"⚠️ Errore scrape benchmark {bench}: {e}")
            s_bench = pd.DataFrame(columns=["label","weight"])
            c_bench = pd.DataFrame(columns=["label","weight"])

        # normalizza benchmark per coerenza col portafoglio
        def _bench_norm(df):
            if df.empty:
                return pd.DataFrame(columns=["label","weight_portfolio"])
            d = df.copy()
            tot = d["weight"].sum()
            if tot > 0:
                d["weight"] = d["weight"] / tot
            d["weight_portfolio"] = d["weight"]
            return d[["label","weight_portfolio"]]

        sectors_bench = _bench_norm(s_bench)
        countries_bench = _bench_norm(c_bench)

        # salva CSV allocazioni assolute
        sectors_p.to_csv(os.path.join(args.outdir, "alloc_sectors_portfolio.csv"), index=False)
        countries_p.to_csv(os.path.join(args.outdir, "alloc_countries_portfolio.csv"), index=False)
        sectors_bench.to_csv(os.path.join(args.outdir, "alloc_sectors_benchmark.csv"), index=False)
        countries_bench.to_csv(os.path.join(args.outdir, "alloc_countries_benchmark.csv"), index=False)

        # confronto e differenze
        comp_sect = compare_allocations(sectors_p, sectors_bench, "label")
        comp_ctry = compare_allocations(countries_p, countries_bench, "label")
        comp_sect.to_csv(os.path.join(args.outdir, "alloc_diff_sectors.csv"), index=False)
        comp_ctry.to_csv(os.path.join(args.outdir, "alloc_diff_countries.csv"), index=False)

        # grafici differenze
        bar_diff_plot(
            comp_sect, "label",
            "Differenze allocazione SETTORI (Port - Bench)",
            os.path.join(args.outdir, "alloc_sectors_diff.png")
        )
        bar_diff_plot(
            comp_ctry, "label",
            "Differenze allocazione PAESI (Port - Bench)",
            os.path.join(args.outdir, "alloc_countries_diff.png")
        )

        # testo sintetico nel SUMMARY
        summary_lines.append("\n--- Confronto allocazioni (fonte: ETFdb.com 'Charts') ---")

        def top_lines(comp_df, what):
            if comp_df is None or comp_df.empty:
                return [f"Nessun dato {what} disponibile."]
            head = comp_df.copy()
            head["pp"] = (head["Diff (Port - Bench)"] * 100).round(2)
            plus = head.sort_values("pp", ascending=False).head(3)
            minus = head.sort_values("pp", ascending=True).head(3)
            lines = [f"Top sovrappesi {what}:"] + \
                    [f"  + {row[what]}: +{row['pp']} pp" for _, row in plus.iterrows()] + \
                    [f"Top sottopesi {what}:"] + \
                    [f"  - {row[what]}: {row['pp']} pp" for _, row in minus.iterrows()]
            return lines

        comp_sect_disp = comp_sect.rename(columns={"label": "Settore"})
        comp_ctry_disp = comp_ctry.rename(columns={"label": "Paese"})
        summary_lines += top_lines(comp_sect_disp, "Settore")
        summary_lines += top_lines(comp_ctry_disp, "Paese")

    # ---------------- Salva SUMMARY ----------------
    out_sum = os.path.join(args.outdir, "02_summary.txt")
    with open(out_sum, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # ---------------- PRINT finale ----------------
    print("\n".join(summary_lines))
    print(f"\nFile salvati in: {args.outdir}")
    print(" -", out_hist)
    print(" -", out_sum)
    if args.alloc:
        print(" - alloc_sectors_portfolio.csv")
        print(" - alloc_sectors_benchmark.csv")
        print(" - alloc_diff_sectors.csv")
        print(" - alloc_countries_portfolio.csv")
        print(" - alloc_countries_benchmark.csv")
        print(" - alloc_diff_countries.csv")
        print(" - alloc_sectors_diff.png")
        print(" - alloc_countries_diff.png")

if __name__ == "__main__":
    main()
