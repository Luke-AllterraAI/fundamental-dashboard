#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║           FUNDAMENTAL TRADING DASHBOARD  v3                          ║
║  EdgeFinder-style macro scoring · Edgeful-style stat reports         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  🎯 Scorecard   — composite directional bias per instrument          ║
║  🌍 Macro/Carry — G7 economic heatmap + rate differentials           ║
║  😨 Sentiment   — fear/greed gauge, put/call, retail positioning     ║
║  ⏱️  Stat Edge   — ORB, gap fills, IB stats (day-trading layer)      ║
║  🤖 AI Analysis — Claude generates full macro report                 ║
║  📋 COT         — CFTC speculator positioning (futures)              ║
║  📊 Valuation   — over/undervalued, any instrument, daily TF         ║
║  💱 FX Strength — relative strength + correlation matrix             ║
║  🏦 Bonds       — US yield curve + context table                     ║
║  📅 Seasonality — monthly historical bias for ANY market             ║
║  📰 News        — RSS headlines + session clock                      ║
╠═══════════════════════════════════════════════════════════════════════╣
║  pip install -r requirements.txt                                     ║
║  streamlit run fundamental_dashboard.py                              ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations  # allow "float | None" hints on Python 3.9

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
import feedparser
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fundamental Dashboard v3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  .stApp { background-color: #0d1117; }
  div[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 10px 14px;
  }
  .score-cell-bull  { background: #1a472a; color: #56d364; font-weight:700; border-radius:4px; padding:3px 8px; }
  .score-cell-bear  { background: #3d1a1a; color: #f85149; font-weight:700; border-radius:4px; padding:3px 8px; }
  .score-cell-neut  { background: #21262d; color: #e3b341; font-weight:700; border-radius:4px; padding:3px 8px; }
  .stTabs [data-baseweb="tab"] { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
FOREX_PAIRS = {
    "EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"USDJPY=X",
    "USD/CHF":"USDCHF=X","AUD/USD":"AUDUSD=X","USD/CAD":"USDCAD=X",
    "NZD/USD":"NZDUSD=X","EUR/GBP":"EURGBP=X","GBP/JPY":"GBPJPY=X",
    "EUR/JPY":"EURJPY=X","AUD/JPY":"AUDJPY=X","EUR/AUD":"EURAUD=X",
    "GBP/AUD":"GBPAUD=X","EUR/CAD":"EURCAD=X","GBP/CAD":"GBPCAD=X",
    "AUD/CAD":"AUDCAD=X","NZD/JPY":"NZDJPY=X",
}
INDICES = {
    "S&P 500":"^GSPC","Nasdaq 100":"^NDX","Dow Jones":"^DJI",
    "Russell 2000":"^RUT","FTSE 100":"^FTSE","DAX":"^GDAXI",
    "Nikkei 225":"^N225","ASX 200":"^AXJO","Euro Stoxx 50":"^STOXX50E",
}
METALS = {
    "Gold":"GC=F","Silver":"SI=F","Platinum":"PL=F","Copper":"HG=F",
}
BOND_YIELDS = {
    "US 2-Year":"^IRX","US 5-Year":"^FVX",
    "US 10-Year":"^TNX","US 30-Year":"^TYX",
}
COT_CODES = {
    "EUR":        ("099741","EURO FX"),
    "GBP":        ("096742","BRITISH POUND"),
    "JPY":        ("097741","JAPANESE YEN"),
    "CHF":        ("092741","SWISS FRANC"),
    "AUD":        ("232741","AUSTRALIAN DOLLAR"),
    "CAD":        ("090741","CANADIAN DOLLAR"),
    "NZD":        ("112741","NZ DOLLAR"),
    "USD Index":  ("098662","U.S. DOLLAR INDEX"),
    "Gold":       ("088691","GOLD"),
    "Silver":     ("084691","SILVER"),
    "S&P 500":    ("13874A","E-MINI S&P"),
    "Nasdaq":     ("209742","NASDAQ-100"),
    "10Y T-Notes":("043602","10-YEAR T-NOTES"),
}
# Maps instrument name → COT key + direction multiplier
# direction -1 = COT is priced in USD/base (USD pairs where USD is base)
INSTRUMENT_COT_MAP = {
    "EUR/USD":("EUR",1),"GBP/USD":("GBP",1),"AUD/USD":("AUD",1),
    "NZD/USD":("NZD",1),"USD/JPY":("JPY",-1),"USD/CHF":("CHF",-1),
    "USD/CAD":("CAD",-1),"EUR/GBP":("EUR",1),"GBP/JPY":("GBP",1),
    "EUR/JPY":("EUR",1),"AUD/JPY":("AUD",1),"EUR/AUD":("EUR",1),
    "GBP/AUD":("GBP",1),"Gold":("Gold",1),"Silver":("Silver",1),
    "S&P 500":("S&P 500",1),"Nasdaq 100":("Nasdaq",1),
}
# G7 currencies → World Bank country codes
G7_COUNTRIES = {
    "USD":("US","United States","🇺🇸"),
    "EUR":("DE","Germany / EU","🇪🇺"),
    "GBP":("GB","United Kingdom","🇬🇧"),
    "JPY":("JP","Japan","🇯🇵"),
    "AUD":("AU","Australia","🇦🇺"),
    "CAD":("CA","Canada","🇨🇦"),
    "NZD":("NZ","New Zealand","🇳🇿"),
    "CHF":("CH","Switzerland","🇨🇭"),
}
INDEX_COUNTRY = {
    "S&P 500":"US","Nasdaq 100":"US","Dow Jones":"US","Russell 2000":"US",
    "FTSE 100":"GB","DAX":"DE","Nikkei 225":"JP",
    "ASX 200":"AU","Euro Stoxx 50":"DE",
}
# Default CB rates (approximate — user updates in sidebar)
CB_RATES_DEFAULT = {
    "USD":4.50,"EUR":2.40,"GBP":4.25,"JPY":0.50,
    "AUD":3.85,"CAD":2.75,"NZD":3.25,"CHF":0.25,
}
# Trading sessions in UTC
SESSION_UTC = [
    ("Sydney",       22, 7),
    ("Tokyo",        0,  9),
    ("London",       7,  16),
    ("New York",     13, 21),
    ("London/NY ★",  13, 16),
]
ALL_CURRENCIES = list(G7_COUNTRIES.keys())
CROSS_PAIRS = [
    ("EUR","USD","EURUSD=X"),("GBP","USD","GBPUSD=X"),("USD","JPY","USDJPY=X"),
    ("AUD","USD","AUDUSD=X"),("USD","CAD","USDCAD=X"),("NZD","USD","NZDUSD=X"),
    ("USD","CHF","USDCHF=X"),("EUR","GBP","EURGBP=X"),("EUR","JPY","EURJPY=X"),
    ("GBP","JPY","GBPJPY=X"),("EUR","AUD","EURAUD=X"),("AUD","JPY","AUDJPY=X"),
]
CROSS_IMPLIED = {
    "GBP/JPY":("GBPUSD=X","USDJPY=X","multiply"),
    "EUR/JPY":("EURUSD=X","USDJPY=X","multiply"),
    "AUD/JPY":("AUDUSD=X","USDJPY=X","multiply"),
    "NZD/JPY":("NZDUSD=X","USDJPY=X","multiply"),
    "EUR/GBP":("EURUSD=X","GBPUSD=X","divide"),
    "EUR/AUD":("EURUSD=X","AUDUSD=X","divide"),
    "EUR/CAD":("EURUSD=X","USDCAD=X","multiply"),
    "GBP/AUD":("GBPUSD=X","AUDUSD=X","divide"),
    "GBP/CAD":("GBPUSD=X","USDCAD=X","multiply"),
    "AUD/CAD":("AUDUSD=X","USDCAD=X","multiply"),
    "AUD/NZD":("AUDUSD=X","NZDUSD=X","divide"),
}
NEWS_FEEDS = [
    ("Reuters Business","https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters Finance", "https://feeds.reuters.com/reuters/companyNews"),
    ("MarketWatch",     "https://feeds.marketwatch.com/marketwatch/topstories/"),
]
RISK_ON_CURRENCIES  = {"AUD","NZD","CAD","GBP"}
RISK_OFF_CURRENCIES = {"JPY","CHF"}
RISK_OFF_METALS     = {"Gold","Silver"}


# ══════════════════════════════════════════════════════════════
#  CORE DATA FUNCTIONS
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def get_prices(symbol: str, period: str = "1y", interval: str = "1d"):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        return None if df.empty else df
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_cot_data():
    """Unified COT positioning from two CFTC reports (column layouts verified
    against live rows):
      • FinFutWk.txt  — financial futures (FX, indices, rates)
      • f_disagg.txt  — disaggregated (metals & physical commodities)
    Normalised to commercials / non-commercials / retail (nonreportable)."""
    headers = {"User-Agent": "Mozilla/5.0 (FundamentalDashboardV3)"}

    def parse_csv(line):
        parts, cur, in_q = [], "", False
        for ch in line:
            if ch == '"':
                in_q = not in_q
            elif ch == "," and not in_q:
                parts.append(cur.strip().strip('"')); cur = ""
            else:
                cur += ch
        parts.append(cur.strip().strip('"'))
        return parts

    def fetch(url, spec):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code != 200:
                return None
            df = pd.DataFrame([parse_csv(l) for l in resp.text.split("\n") if l.strip()])
            if df.shape[1] <= spec["max_idx"]:
                return None
            def col_sum(idxs):
                s = None
                for i in idxs:
                    v = pd.to_numeric(df[i].astype(str).str.replace(",", ""), errors="coerce")
                    s = v if s is None else s + v
                return s
            out = pd.DataFrame()
            out["market_name"]   = df[0].astype(str).str.strip()
            out["contract_code"] = df[3].astype(str).str.strip()
            out["open_interest"] = col_sum([spec["oi"]])
            out["comm_long"]     = col_sum(spec["comm_long"])
            out["comm_short"]    = col_sum(spec["comm_short"])
            out["nc_long"]       = col_sum(spec["nc_long"])
            out["nc_short"]      = col_sum(spec["nc_short"])
            out["retail_long"]   = col_sum(spec["retail_long"])
            out["retail_short"]  = col_sum(spec["retail_short"])
            return out
        except Exception:
            return None

    # Traders in Financial Futures (FX / indices / rates): Dealer = commercial,
    # AssetMgr+Leveraged+Other = non-commercial, Nonreportable(22/23) = retail.
    TFF = {"oi":7, "comm_long":[8], "comm_short":[9],
           "nc_long":[11,14,17], "nc_short":[12,15,18],
           "retail_long":[22], "retail_short":[23], "max_idx":23}
    # Disaggregated (metals): Producer+Swap = commercial, ManagedMoney+Other =
    # non-commercial, Nonreportable(21/22) = retail.
    DIS = {"oi":7, "comm_long":[8,10], "comm_short":[9,11],
           "nc_long":[13,16], "nc_short":[14,17],
           "retail_long":[21], "retail_short":[22], "max_idx":22}

    fin = fetch("https://www.cftc.gov/dea/newcot/FinFutWk.txt", TFF)
    com = fetch("https://www.cftc.gov/dea/newcot/f_disagg.txt",  DIS)
    parts = [p for p in (fin, com) if p is not None and not p.empty]
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["comm_net"]   = df["comm_long"]   - df["comm_short"]
    df["nc_net"]     = df["nc_long"]     - df["nc_short"]
    df["retail_net"] = df["retail_long"] - df["retail_short"]
    return df

def cot_leans(r):
    """Directional lean of each group as (net / gross), range -1..+1."""
    def lean(l, s):
        l = 0.0 if pd.isna(l) else float(l)
        s = 0.0 if pd.isna(s) else float(s)
        t = l + s
        return (l - s) / t if t > 0 else 0.0
    return (lean(r.get("retail_long"), r.get("retail_short")),
            lean(r.get("comm_long"),   r.get("comm_short")),
            lean(r.get("nc_long"),     r.get("nc_short")))

def cot_raw_signal(retail_lean, comm_lean, nc_lean):
    """Trade AGAINST retail; amplify when commercials or non-commercials sit on
    the opposite side of retail (confirming the fade). Returns -1..+1."""
    base = -retail_lean
    amp  = 1.0
    if comm_lean * retail_lean < 0: amp += abs(comm_lean) * 0.5
    if nc_lean   * retail_lean < 0: amp += abs(nc_lean)   * 0.5
    return float(np.clip(base * amp, -1.0, 1.0))

# ── COT Index (Commercials vs Retail) — historical normalisation ──────────
# Each group's net (as % of open interest) is placed within its OWN range over
# a lookback window: 0 = most bearish in range, 100 = most bullish in range.
# Defaults: 26-week responsive index, 156-week (3y) historical hi/lo extreme.
COT_LB_SHORT = 26
COT_LB_LONG  = 156
CFTC_DISAGG_RES = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
CFTC_TFF_RES    = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
COMMODITY_COT_KEYS = {"Gold", "Silver"}

@st.cache_data(ttl=3600, show_spinner=False)
def get_cot_history(code: str, report: str, weeks: int = 170):
    """Weekly net positioning (% of OI) for a contract over ~3+ years."""
    url = CFTC_DISAGG_RES if report == "disagg" else CFTC_TFF_RES
    params = {"cftc_contract_market_code": code,
              "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": str(weeks)}
    try:
        r = requests.get(url, headers={"User-Agent": "FundamentalDashboardV3"},
                         params=params, timeout=30)
        if not r.ok:
            return None
        rows = r.json()
    except Exception:
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    def num(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns \
               else pd.Series([np.nan] * len(df))
    if report == "disagg":
        comm_l = num("prod_merc_positions_long")  + num("swap_positions_long_all")
        comm_s = num("prod_merc_positions_short") + num("swap__positions_short_all")
        nc_l   = num("m_money_positions_long_all")  + num("other_rept_positions_long")
        nc_s   = num("m_money_positions_short_all") + num("other_rept_positions_short")
    else:
        comm_l = num("dealer_positions_long_all")
        comm_s = num("dealer_positions_short_all")
        nc_l   = num("asset_mgr_positions_long")  + num("lev_money_positions_long")  + num("other_rept_positions_long")
        nc_s   = num("asset_mgr_positions_short") + num("lev_money_positions_short") + num("other_rept_positions_short")
    rl = num("nonrept_positions_long_all")
    rs = num("nonrept_positions_short_all")
    oi = num("open_interest_all").replace(0, np.nan)
    out = pd.DataFrame({
        "date":       pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce"),
        "comm_net":   (comm_l - comm_s) / oi * 100.0,
        "nc_net":     (nc_l - nc_s)     / oi * 100.0,
        "retail_net": (rl - rs)         / oi * 100.0,
    }).dropna().sort_values("date").reset_index(drop=True)
    return out if len(out) >= 20 else None

def cot_index(series, lookback):
    """Position of the latest value within its [min,max] over `lookback` -> 0..100."""
    w = series.tail(lookback)
    lo, hi = float(w.min()), float(w.max())
    cur = float(series.iloc[-1])
    return (cur - lo) / (hi - lo) * 100.0 if hi > lo else 50.0

def resolve_cot(name):
    entry = INSTRUMENT_COT_MAP.get(name)
    if entry is None:
        return None
    key, direction = entry
    cc = COT_CODES.get(key)
    if not cc:
        return None
    code, _kw = cc
    report = "disagg" if key in COMMODITY_COT_KEYS else "tff"
    return code, report, direction

@st.cache_data(ttl=3600, show_spinner=False)
def get_cot_index_signal(name: str):
    """Commercials-vs-Retail COT Index signal for an instrument.
    Follows commercials (smart money) and fades retail, both measured as an
    index within their own 26w / 156w range. Returns -1..+1 (direction-applied)."""
    r = resolve_cot(name)
    if r is None:
        return None
    code, report, direction = r
    hist = get_cot_history(code, report)
    if hist is None:
        return None
    out = {"direction": direction}
    for lb, tag in [(COT_LB_SHORT, "s"), (COT_LB_LONG, "l")]:
        out[f"comm_{tag}"]   = cot_index(hist["comm_net"],   lb)
        out[f"retail_{tag}"] = cot_index(hist["retail_net"], lb)
    # Commercials vs Retail: bullish when commercials high in range AND retail low.
    # The 26-week index is the signal (responsive); 156-week is historical context.
    out["div_s"] = (out["comm_s"] - out["retail_s"]) / 100.0   # 26w — primary signal
    out["div_l"] = (out["comm_l"] - out["retail_l"]) / 100.0   # 156w — long-range context
    out["raw"]   = float(np.clip(out["div_s"], -1.0, 1.0))
    out["score"] = round(out["raw"] * direction, 2)
    return out

def cot_index_label(o):
    if o is None:
        return "— N/A"
    raw = o["raw"]                       # from the market's own perspective
    conf = (o["retail_s"] <= 15 or o["retail_s"] >= 85 or
            o["comm_s"]   >= 85 or o["comm_s"]   <= 15)
    tag = " ✓extreme" if conf else ""
    if   raw >  0.33: return "🟢 BULLISH (comm long / retail short)" + tag
    elif raw >  0.10: return "🟩 lean bull" + tag
    elif raw < -0.33: return "🔴 BEARISH (comm short / retail long)" + tag
    elif raw < -0.10: return "🟥 lean bear" + tag
    else:             return "🟡 neutral"

@st.cache_data(ttl=86400, show_spinner=False)
def get_world_bank(country_code: str, indicator: str) -> dict:
    """Fetch latest macro data from World Bank (free, no key)."""
    url = (f"https://api.worldbank.org/v2/country/{country_code}"
           f"/indicator/{indicator}?format=json&mrv=5&per_page=5")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {}
        data = r.json()
        if len(data) < 2 or not data[1]:
            return {}
        vals = [(d["date"], d["value"]) for d in data[1] if d.get("value") is not None]
        if not vals:
            return {}
        latest_year, latest_val = vals[0]
        trend = None
        if len(vals) >= 2:
            trend = "up" if latest_val > vals[1][1] else "down"
        return {"value": round(latest_val, 2), "year": latest_year, "trend": trend}
    except Exception:
        return {}

@st.cache_data(ttl=86400, show_spinner=False)
def get_all_macro_scores() -> dict:
    """G7 macro scores: GDP growth, CPI, unemployment → hawkishness score."""
    WB_INDICATORS = {
        "gdp":          "NY.GDP.MKTP.KD.ZG",   # annual GDP growth %
        "cpi":          "FP.CPI.TOTL.ZG",       # CPI inflation annual %
        "unemployment": "SL.UEM.TOTL.ZS",        # unemployment %
    }
    results = {}
    for currency, (cc, label, _) in G7_COUNTRIES.items():
        data = {}
        for key, indicator in WB_INDICATORS.items():
            data[key] = get_world_bank(cc, indicator)

        gdp_v   = data["gdp"].get("value")
        cpi_v   = data["cpi"].get("value")
        unem_v  = data["unemployment"].get("value")

        scores = {}
        if gdp_v is not None:
            if   gdp_v >  3:   scores["gdp"] =  1.0
            elif gdp_v >  2:   scores["gdp"] =  0.5
            elif gdp_v >  0:   scores["gdp"] =  0.0
            elif gdp_v > -1:   scores["gdp"] = -0.5
            else:              scores["gdp"] = -1.0
        if cpi_v is not None:
            if   cpi_v >  4:   scores["cpi"] =  0.7
            elif cpi_v >  2:   scores["cpi"] =  0.2
            elif cpi_v >  0:   scores["cpi"] = -0.3
            else:              scores["cpi"] = -0.7
        if unem_v is not None:
            if   unem_v <  4:  scores["unem"] =  0.3
            elif unem_v <  6:  scores["unem"] =  0.0
            else:              scores["unem"] = -0.3

        total = sum(scores.values()) if scores else 0.0
        results[cc] = {
            "currency": currency, "label": label,
            "score": round(total, 2),
            "gdp":   gdp_v, "cpi": cpi_v, "unemployment": unem_v,
            "gdp_year":  data["gdp"].get("year"),
            "cpi_year":  data["cpi"].get("year"),
            "unem_year": data["unemployment"].get("year"),
        }
    return results

@st.cache_data(ttl=600, show_spinner=False)
def get_put_call_ratio():
    """Approximate equity put/call ratio from SPY & QQQ options OI."""
    results = {}
    for ticker in ["SPY", "QQQ"]:
        try:
            t = yf.Ticker(ticker)
            dates = t.options
            if not dates:
                continue
            chain = t.option_chain(dates[0])
            c_oi  = chain.calls["openInterest"].sum()
            p_oi  = chain.puts["openInterest"].sum()
            if c_oi > 0:
                results[ticker] = round(p_oi / c_oi, 3)
        except Exception:
            pass
    if results:
        avg = round(sum(results.values()) / len(results), 3)
        return {"pcr": avg, "by_ticker": results}
    return None

@st.cache_data(ttl=300, show_spinner=False)
def get_sentiment_gauge():
    """
    Multi-factor Fear/Greed index (0 = extreme fear, 100 = extreme greed).
    Components:
      1. VIX level
      2. VIX vs 50-day MA (momentum of fear)
      3. SPY market momentum (vs 125-day MA)
      4. HYG/LQD credit spread proxy
      5. Gold / SPY ratio (safe-haven flow)
      6. Put/Call ratio
    """
    components = {}

    # 1. VIX level  (lower VIX = more greed)
    vd = get_prices("^VIX", period="1y")
    if vd is not None and len(vd) > 50:
        vix    = vd["Close"].iloc[-1]
        vix_ma = vd["Close"].rolling(50).mean().iloc[-1]
        # scale: vix 10→100, 40→0
        vix_score = np.clip((40 - vix) / 30 * 100, 0, 100)
        components["VIX Level"] = round(vix_score, 1)
        # 2. VIX vs MA  (rising vix = fear)
        if vix > vix_ma * 1.15:
            components["VIX Momentum"] = 15.0   # fear
        elif vix < vix_ma * 0.85:
            components["VIX Momentum"] = 85.0   # greed
        else:
            components["VIX Momentum"] = 50.0

    # 3. SPY momentum vs 125-day MA
    sd = get_prices("^GSPC", period="1y")
    if sd is not None and len(sd) > 130:
        spy     = sd["Close"].iloc[-1]
        spy_ma  = sd["Close"].rolling(125).mean().iloc[-1]
        dev_pct = (spy / spy_ma - 1) * 100
        # -5% below MA = 10, flat = 50, +5% above = 90
        spym_score = np.clip(50 + dev_pct * 8, 0, 100)
        components["Market Momentum"] = round(spym_score, 1)

    # 4. Credit spread proxy: HYG / LQD ratio vs 50-day MA
    hd = get_prices("HYG", period="6mo")
    ld = get_prices("LQD", period="6mo")
    if hd is not None and ld is not None and len(hd) > 50 and len(ld) > 50:
        combined    = pd.DataFrame({"HYG": hd["Close"], "LQD": ld["Close"]}).dropna()
        ratio       = combined["HYG"] / combined["LQD"]
        ratio_ma    = ratio.rolling(50).mean()
        dev         = (ratio.iloc[-1] / ratio_ma.iloc[-1] - 1) * 100
        # positive ratio deviation = tightening spreads = greed
        credit_score = np.clip(50 + dev * 20, 0, 100)
        components["Credit Spread"] = round(credit_score, 1)

    # 5. Gold/SPY ratio (rising = risk-off = fear)
    gd = get_prices("GC=F", period="6mo")
    if gd is not None and sd is not None and len(gd) > 50:
        gs_combined = pd.DataFrame({"Gold": gd["Close"], "SPY": sd["Close"]}).dropna()
        gs_ratio    = gs_combined["Gold"] / gs_combined["SPY"]
        gs_ma       = gs_ratio.rolling(50).mean()
        gs_dev      = (gs_ratio.iloc[-1] / gs_ma.iloc[-1] - 1) * 100
        # rising gold/spy = fear (lower score)
        gold_score  = np.clip(50 - gs_dev * 15, 0, 100)
        components["Safe Haven Flow"] = round(gold_score, 1)

    # 6. Put/Call ratio
    pcr_data = get_put_call_ratio()
    if pcr_data:
        pcr   = pcr_data["pcr"]
        # PCR 0.6=100 (greed), 0.8=70, 1.0=50, 1.2=30, 1.4=0 (fear)
        pc_score = np.clip((1.5 - pcr) / 0.9 * 100, 0, 100)
        components["Put/Call Ratio"] = round(pc_score, 1)

    if not components:
        return None

    composite = round(sum(components.values()) / len(components), 1)

    if   composite > 75: label, emoji = "Extreme Greed", "💚"
    elif composite > 55: label, emoji = "Greed",         "🟢"
    elif composite > 45: label, emoji = "Neutral",       "🟡"
    elif composite > 25: label, emoji = "Fear",          "🟠"
    else:                label, emoji = "Extreme Fear",  "🔴"

    return {
        "composite":  composite,
        "label":      label,
        "emoji":      emoji,
        "components": components,
    }

@st.cache_data(ttl=300, show_spinner=False)
def get_currency_strength(lookback: int = 14):
    strength = {c: 0.0 for c in ALL_CURRENCIES}
    counts   = {c: 0   for c in ALL_CURRENCIES}
    for base, quote, sym in CROSS_PAIRS:
        df = get_prices(sym, period="3mo")
        if df is None or len(df) < lookback + 1:
            continue
        pct = (df["Close"].iloc[-1] / df["Close"].iloc[-lookback] - 1) * 100
        if quote == "JPY":
            pct = -pct
        if base  in strength: strength[base]  += pct;  counts[base]  += 1
        if quote in strength: strength[quote] -= pct;  counts[quote] += 1
    return {c: round(strength[c]/counts[c], 3) if counts[c] > 0 else 0.0
            for c in ALL_CURRENCIES}

def _valuation_stats(close, lookback: int):
    close = close.dropna()
    if len(close) < 50:
        return None
    current = close.iloc[-1]
    hist    = close.tail(min(lookback, len(close)))
    # Simple over/under valuation: where does the current price sit within its
    # own recent range? (percentile 0-100). No z-score / std-dev involved.
    pct     = float((hist < current).sum()) / len(hist) * 100
    hi, lo  = hist.max(), hist.min()
    rpos    = (current - lo) / (hi - lo) * 100 if hi != lo else 50.0
    span    = min(200, max(20, len(close)//2))
    ema     = close.ewm(span=span, adjust=False).mean().iloc[-1]
    ema_dev = (current - ema) / ema * 100 if ema else 0.0
    if   pct >= 85: sig, col = "🔴 OVERVALUED",  "#f85149"
    elif pct >= 65: sig, col = "🟠 SLIGHT OV",   "#f0883e"
    elif pct <= 15: sig, col = "🟢 UNDERVALUED", "#56d364"
    elif pct <= 35: sig, col = "🟡 SLIGHT UV",   "#e3b341"
    else:           sig, col = "⚪ FAIR VALUE",   "#8b949e"
    # scorecard score: cheap = bullish (+1), rich = bearish (-1)
    val_score = round((50.0 - pct) / 50.0, 2)
    return {"current":current,"percentile":round(pct,1),"range_pos":round(rpos,1),
            "ema_dev":round(ema_dev,2),"hi":hi,"lo":lo,
            "signal":sig,"color":col,"val_score":val_score}

@st.cache_data(ttl=600, show_spinner=False)
def get_valuation(symbol: str, lookback: int = 252):
    # Raw price valuation. For an FX pair the price already IS base-vs-quote,
    # so this measures the base currency's value against the quote.
    df = get_prices(symbol, period="3y", interval="1d")
    if df is None or len(df) < 50:
        return None
    return _valuation_stats(df["Close"], lookback)

@st.cache_data(ttl=600, show_spinner=False)
def get_valuation_ratio(symbol: str, denom_symbol: str, lookback: int = 252):
    """Value `symbol` AGAINST `denom_symbol` by z-scoring the price ratio.
    e.g. US100 vs the Dollar (DXY) or vs Treasuries (TLT)."""
    a = get_prices(symbol, period="3y", interval="1d")
    b = get_prices(denom_symbol, period="3y", interval="1d")
    if a is None or b is None:
        return None
    combined = pd.DataFrame({"a": a["Close"], "b": b["Close"]}).dropna()
    if len(combined) < 50 or (combined["b"] <= 0).all():
        return None
    return _valuation_stats(combined["a"] / combined["b"], lookback)

def valuation_for_instrument(name: str, symbol: str, lookback: int,
                             ref_symbol: str | None, ref_label: str):
    """Relative valuation. FX pairs value base-vs-quote (the pair itself);
    everything else is valued against ref_symbol (Dollar/Bonds) if given."""
    if symbol.endswith("=X"):                      # forex pair
        vs = name.split("/")[1] if "/" in name else "quote"
        return get_valuation(symbol, lookback), vs
    if ref_symbol:                                 # index / metal / stock vs reference
        return get_valuation_ratio(symbol, ref_symbol, lookback), ref_label
    return get_valuation(symbol, lookback), "USD"

@st.cache_data(ttl=300, show_spinner=False)
def get_cross_implied(pair_name: str):
    if pair_name not in CROSS_IMPLIED:
        return None
    sym1, sym2, op = CROSS_IMPLIED[pair_name]
    d1 = get_prices(sym1, period="6mo")
    d2 = get_prices(sym2, period="6mo")
    if d1 is None or d2 is None or len(d1) < 20 or len(d2) < 20:
        return None
    combined = pd.DataFrame({"p1":d1["Close"],"p2":d2["Close"]}).dropna()
    if len(combined) < 20:
        return None
    if   op == "multiply": implied = combined["p1"] * combined["p2"]
    elif op == "divide":   implied = combined["p1"] / combined["p2"]
    else:                  return None
    base, quote = pair_name.split("/")
    actual_df   = get_prices(f"{base}{quote}=X", period="6mo")
    if actual_df is None:
        return None
    actual   = actual_df["Close"].reindex(combined.index, method="ffill").dropna()
    imp_aln  = implied.reindex(actual.index).dropna()
    if len(actual) < 20:
        return None
    div = (actual / imp_aln - 1) * 100
    z   = (div.iloc[-1] - div.mean()) / div.std() if div.std() > 0 else 0
    if   div.iloc[-1] >  0.5: sig = "🔴 ABOVE IMPLIED"
    elif div.iloc[-1] < -0.5: sig = "🟢 BELOW IMPLIED"
    else:                     sig = "⚪ AT IMPLIED"
    return {"actual":round(actual.iloc[-1],5),"implied":round(imp_aln.iloc[-1],5),
            "div_pct":round(div.iloc[-1],3),"div_z":round(z,2),
            "signal":sig,"history_div":div}

@st.cache_data(ttl=600, show_spinner=False)
def get_seasonality(symbol: str, years: int = 10):
    try:
        end   = datetime.now()
        start = end - timedelta(days=years*366)
        df    = yf.Ticker(symbol).history(start=start, end=end, interval="1mo")
        if len(df) < 12:
            return None
    except Exception:
        return None
    df["month"] = df.index.month
    df["ret"]   = df["Close"].pct_change() * 100
    df = df.dropna(subset=["ret"])
    grp = df.groupby("month")["ret"]
    result = pd.DataFrame({
        "avg":      grp.mean(),
        "std":      grp.std(),
        "n":        grp.count(),
        "win_rate": grp.apply(lambda x: (x>0).sum()/len(x)*100),
    })
    result.index = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"][:len(result)]
    return result

@st.cache_data(ttl=1800, show_spinner=False)
def get_news():
    items = []
    for source, url in NEWS_FEEDS:
        try:
            for e in feedparser.parse(url).entries[:5]:
                items.append({
                    "source":  source,
                    "title":   e.get("title",""),
                    "summary": (e.get("summary") or "")[:250],
                    "link":    e.get("link",""),
                })
        except Exception:
            pass
    return items

# ── Statistical Edge (Edgeful-style) ─────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def get_atr_data(symbol: str):
    df = get_prices(symbol, period="1y", interval="1d")
    if df is None or len(df) < 20:
        return None
    df = df.copy()
    df["tr"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr50"] = df["tr"].rolling(50).mean()
    a14, a50    = df["atr14"].iloc[-1], df["atr50"].iloc[-1]
    regime      = ("EXPANDING ↑" if a14 > a50*1.1 else
                   "CONTRACTING ↓" if a14 < a50*0.9 else "NORMAL ↔")
    today_r     = df["High"].iloc[-1] - df["Low"].iloc[-1]
    r_used      = today_r / a14 * 100 if a14 > 0 else 0
    return {"atr14":a14,"atr50":a50,"regime":regime,
            "today_range":today_r,"range_used":round(r_used,1),
            "close":df["Close"].iloc[-1],"atr_series":df["atr14"]}

@st.cache_data(ttl=600, show_spinner=False)
def get_dow_stats(symbol: str):
    df = get_prices(symbol, period="2y", interval="1d")
    if df is None or len(df) < 100:
        return None
    df = df.copy()
    df["ret"]   = df["Close"].pct_change() * 100
    df["range"] = (df["High"] - df["Low"]) / df["Close"].shift(1) * 100
    df["dow"]   = df.index.dayofweek
    df = df.dropna()
    days = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
    grp  = df.groupby("dow")
    result = pd.DataFrame({
        "avg_return": grp["ret"].mean().round(3),
        "win_rate":   grp["ret"].apply(lambda x:(x>0).mean()*100).round(1),
        "avg_range":  grp["range"].mean().round(3),
        "n":          grp["ret"].count(),
    })
    result.index = [days.get(i,str(i)) for i in result.index]
    return result

@st.cache_data(ttl=600, show_spinner=False)
def get_gap_stats(symbol: str, years: int = 2):
    df = get_prices(symbol, period=f"{years}y", interval="1d")
    if df is None or len(df) < 80:
        return None
    df = df.copy()
    df["prev_close"] = df["Close"].shift(1)
    df["gap"]        = df["Open"] - df["prev_close"]
    df["gap_pct"]    = df["gap"] / df["prev_close"] * 100
    df = df.dropna()
    # Only count meaningful gaps
    df = df[df["gap_pct"].abs() > 0.05]
    if len(df) < 20:
        return None
    df["filled"] = (
        ((df["gap"] > 0) & (df["Low"]  <= df["prev_close"])) |
        ((df["gap"] < 0) & (df["High"] >= df["prev_close"]))
    )
    total     = len(df)
    up_gaps   = df[df["gap"] > 0]
    down_gaps = df[df["gap"] < 0]
    # Bucket by size
    df["size_bucket"] = pd.cut(
        df["gap_pct"].abs(),
        bins=[0, 0.3, 0.75, 1.5, 100],
        labels=["Small <0.3%","Medium 0.3–0.75%","Large 0.75–1.5%","Huge >1.5%"]
    )
    by_size = df.groupby("size_bucket", observed=True)["filled"].agg(
        fill_rate=lambda x: round(x.mean()*100,1), count="count"
    )
    return {
        "total":          total,
        "fill_rate":      round(df["filled"].mean()*100, 1),
        "up_fill":        round(up_gaps["filled"].mean()*100, 1) if len(up_gaps) > 5 else None,
        "down_fill":      round(down_gaps["filled"].mean()*100, 1) if len(down_gaps) > 5 else None,
        "by_size":        by_size,
        "n_years":        years,
    }

@st.cache_data(ttl=600, show_spinner=False)
def get_orb_stats(symbol: str):
    """Opening Range Breakout stats from hourly data (2 years)."""
    try:
        df = yf.Ticker(symbol).history(period="2y", interval="1h")
        if df.empty or len(df) < 200:
            return None
    except Exception:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df["date"] = df.index.date
    results = []
    for date, ddf in df.groupby("date"):
        if len(ddf) < 4:
            continue
        or_high  = ddf.iloc[0]["High"]
        or_low   = ddf.iloc[0]["Low"]
        or_open  = ddf.iloc[0]["Open"]
        rest     = ddf.iloc[1:]
        if rest.empty:
            continue
        day_high  = rest["High"].max()
        day_low   = rest["Low"].min()
        day_close = ddf.iloc[-1]["Close"]
        broke_up  = day_high  > or_high
        broke_dn  = day_low   < or_low
        results.append({
            "broke_up":broke_up,"broke_dn":broke_dn,
            "both":broke_up and broke_dn,
            "or_range": or_high - or_low,
            "bull_day": day_close > or_open,
        })
    if len(results) < 30:
        return None
    rdf = pd.DataFrame(results)
    up_days = rdf[rdf["broke_up"]]
    return {
        "n":                len(rdf),
        "break_up_pct":     round(rdf["broke_up"].mean()*100, 1),
        "break_dn_pct":     round(rdf["broke_dn"].mean()*100, 1),
        "both_break_pct":   round(rdf["both"].mean()*100, 1),
        "neither_pct":      round((~rdf["broke_up"] & ~rdf["broke_dn"]).mean()*100, 1),
        "bull_day_pct":     round(rdf["bull_day"].mean()*100, 1),
        "up_break_bull_pct":round(up_days["bull_day"].mean()*100, 1) if len(up_days)>10 else None,
    }


# ══════════════════════════════════════════════════════════════
#  SCORECARD SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _score_cot(name: str, cot_df=None) -> float | None:
    # COT Index (Commercials vs Retail) over 26w / 156w — follow commercials,
    # fade retail, each measured within its own historical range.
    sig = get_cot_index_signal(name)
    return sig["score"] if sig else None

def _score_carry(name: str, cb_rates: dict) -> float | None:
    if "/" not in name:
        return None
    base, quote = name.split("/")
    br = cb_rates.get(base.strip())
    qr = cb_rates.get(quote.strip())
    if br is None or qr is None:
        return None
    return round(float(np.clip((br - qr) / 5.0, -1, 1)), 2)

def _score_macro(name: str, macro_scores: dict) -> float | None:
    if "/" in name:
        base, quote = name.split("/")
        bc = G7_COUNTRIES.get(base.strip(),  (None,))[0]
        qc = G7_COUNTRIES.get(quote.strip(), (None,))[0]
        if not bc or not qc:
            return None
        bs = macro_scores.get(bc, {}).get("score")
        qs = macro_scores.get(qc, {}).get("score")
        if bs is None or qs is None:
            return None
        return round(float(np.clip((bs - qs) / 2.0, -1, 1)), 2)
    cc = INDEX_COUNTRY.get(name)
    if cc:
        s = macro_scores.get(cc, {}).get("score")
        return round(float(np.clip(s / 2.0, -1, 1)), 2) if s is not None else None
    if name in ("Gold", "Silver"):
        us = macro_scores.get("US", {})
        cpi = us.get("cpi")
        sc  = us.get("score")
        if cpi is None or sc is None:
            return None
        return round(float(np.clip(((cpi - 2) / 4.0) + (-sc / 4.0), -1, 1)), 2)
    return None

VAL_SCORECARD_REF = "DX-Y.NYB"   # non-FX instruments valued vs the US Dollar (DXY)

def _score_valuation(name: str, symbol: str, lookback: int) -> float | None:
    v, _vs = valuation_for_instrument(name, symbol, lookback, VAL_SCORECARD_REF, "USD")
    if v is None:
        return None
    return v["val_score"]

def _score_seasonality(symbol: str, years: int) -> float | None:
    sd = get_seasonality(symbol, years)
    if sd is None:
        return None
    m = datetime.now().strftime("%b")
    if m not in sd.index:
        return None
    wr  = sd.loc[m, "win_rate"]
    avg = sd.loc[m, "avg"]
    return round(float(np.clip((wr - 50) / 50 * 0.6 + np.clip(avg / 3, -1, 1) * 0.4, -1, 1)), 2)

def _score_sentiment(name: str, symbol: str, fg: float) -> float | None:
    # fg: 0-100 → normalised to -1..+1
    norm = (fg - 50) / 50
    if "/" in name:
        base, quote = name.split("/")
        b_ro = 1 if base.strip()  in RISK_ON_CURRENCIES  else (-1 if base.strip()  in RISK_OFF_CURRENCIES else 0)
        q_ro = 1 if quote.strip() in RISK_OFF_CURRENCIES else (-1 if quote.strip() in RISK_ON_CURRENCIES  else 0)
        d    = b_ro + q_ro
        return round(float(np.clip(norm * np.sign(d), -1, 1)), 2) if d != 0 else 0.0
    if name in RISK_OFF_METALS:
        return round(float(np.clip(-norm, -1, 1)), 2)
    if symbol.startswith("^") or symbol in ["SPY","QQQ"]:
        return round(float(np.clip(norm, -1, 1)), 2)
    return None

def compute_scorecard(instruments: dict, cb_rates: dict, cot_df,
                      macro_scores: dict, fg: float,
                      val_lookback: int, season_years: int) -> pd.DataFrame:
    rows = []
    for name, sym in instruments.items():
        cot_s  = _score_cot(name, cot_df)
        carry_s= _score_carry(name, cb_rates)
        macro_s= _score_macro(name, macro_scores)
        val_s  = _score_valuation(name, sym, val_lookback)
        seas_s = _score_seasonality(sym, season_years)
        sent_s = _score_sentiment(name, sym, fg) if fg is not None else None
        # Weighted scoring — COT 10x dominant (#1), Valuation 3x (#2), Seasonality 2x (#3), others 1x
        WEIGHTS = {
            "cot":  10.0,   # #1 dominant — Commercials-vs-Retail COT index (~55% of weight)
            "val":   3.0,   # #2 — over/undervaluation
            "seas":  2.0,   # #3 — historical seasonal bias
            "carry": 1.0,   # Secondary
            "macro": 1.0,   # Secondary
            "sent":  1.0,   # Secondary
        }
        scored = [
            (cot_s,   WEIGHTS["cot"]),
            (carry_s, WEIGHTS["carry"]),
            (macro_s, WEIGHTS["macro"]),
            (val_s,   WEIGHTS["val"]),
            (seas_s,  WEIGHTS["seas"]),
            (sent_s,  WEIGHTS["sent"]),
        ]
        filtered     = [(v, w) for v, w in scored if v is not None]
        weight_sum   = sum(w for _, w in filtered)
        weighted_avg = sum(v * w for v, w in filtered) / weight_sum if weight_sum > 0 else 0.0
        total        = round(weighted_avg * 5, 2)
        if   total >  3.5: bias = "🟢🟢 STRONG BULL"
        elif total >  1.5: bias = "🟢 BULL"
        elif total > -1.5: bias = "🟡 NEUTRAL"
        elif total > -3.5: bias = "🔴 BEAR"
        else:              bias = "🔴🔴 STRONG BEAR"
        rows.append({
            "Instrument": name,
            "COT":        cot_s,
            "Carry":      carry_s,
            "Macro":      macro_s,
            "Valuation":  val_s,
            "Seasonality":seas_s,
            "Sentiment":  sent_s,
            "Score /5":   total,
            "Bias":       bias,
        })
    return pd.DataFrame(rows).sort_values("Score /5", ascending=False)

def cot_bias_label(raw, retail_lean, comm_lean, nc_lean):
    """Fade-retail bias for the COT tab (from the market's own perspective)."""
    if raw is None:
        return "— N/A"
    smart_opp = (comm_lean * retail_lean < 0) or (nc_lean * retail_lean < 0)
    tag = " ✓smart" if (smart_opp and abs(retail_lean) > 0.05) else ""
    if   raw >  0.33: return "🟢 LONG (fade retail)" + tag
    elif raw >  0.10: return "🟩 lean long" + tag
    elif raw < -0.33: return "🔴 SHORT (fade retail)" + tag
    elif raw < -0.10: return "🟥 lean short" + tag
    else:             return "🟡 neutral"


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️  Settings")
    api_key = st.text_input("Anthropic API Key", type="password",
                            placeholder="sk-ant-...", help="AI Analysis tab only")
    st.divider()
    st.markdown("### Markets")
    sel_forex   = st.multiselect("Forex",   list(FOREX_PAIRS),
                                 default=["EUR/USD","GBP/USD","GBP/JPY","AUD/USD","USD/JPY"])
    sel_indices = st.multiselect("Indices", list(INDICES),
                                 default=["S&P 500","Nasdaq 100"])
    sel_metals  = st.multiselect("Metals",  list(METALS), default=["Gold","Silver"])
    custom_raw  = st.text_area("Stocks (one ticker per line)",
                               placeholder="AAPL\nNVDA\nMSFT", height=80)
    st.divider()
    st.markdown("### Central Bank Rates (%)")
    st.caption("Update when rates change. Used for carry & scorecard.")
    cb_rates = {}
    r1, r2 = st.columns(2)
    cb_rates["USD"] = r1.number_input("USD",  value=CB_RATES_DEFAULT["USD"], step=0.25, format="%.2f")
    cb_rates["EUR"] = r2.number_input("EUR",  value=CB_RATES_DEFAULT["EUR"], step=0.25, format="%.2f")
    r3, r4 = st.columns(2)
    cb_rates["GBP"] = r3.number_input("GBP",  value=CB_RATES_DEFAULT["GBP"], step=0.25, format="%.2f")
    cb_rates["JPY"] = r4.number_input("JPY",  value=CB_RATES_DEFAULT["JPY"], step=0.25, format="%.2f")
    r5, r6 = st.columns(2)
    cb_rates["AUD"] = r5.number_input("AUD",  value=CB_RATES_DEFAULT["AUD"], step=0.25, format="%.2f")
    cb_rates["CAD"] = r6.number_input("CAD",  value=CB_RATES_DEFAULT["CAD"], step=0.25, format="%.2f")
    r7, r8 = st.columns(2)
    cb_rates["NZD"] = r7.number_input("NZD",  value=CB_RATES_DEFAULT["NZD"], step=0.25, format="%.2f")
    cb_rates["CHF"] = r8.number_input("CHF",  value=CB_RATES_DEFAULT["CHF"], step=0.25, format="%.2f")
    st.divider()
    st.markdown("### Parameters")
    cs_lookback  = st.slider("FX Strength lookback (days)", 5, 50, 14)
    val_lookback = st.slider("Valuation window (days)", 63, 504, 252)
    season_years = st.slider("Seasonality history (years)", 3, 15, 5)
    st.divider()
    if st.button("🔄  Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

custom_stocks   = [s.strip().upper() for s in custom_raw.split("\n") if s.strip()]
ALL_INSTRUMENTS = (
    {k: FOREX_PAIRS[k]  for k in sel_forex}
    | {k: INDICES[k]    for k in sel_indices}
    | {k: METALS[k]     for k in sel_metals}
    | {s: s             for s in custom_stocks}
)

# ══════════════════════════════════════════════════════════════
#  HEADER SNAPSHOT
# ══════════════════════════════════════════════════════════════
st.title("📊  Fundamental Trading Dashboard  v3")
st.caption(f"⏱  {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
snap_cols = st.columns(6)
for col, (label, sym, fmt) in zip(snap_cols, [
    ("DXY",     "DX-Y.NYB","   {:.2f}"),
    ("10Y Yld", "^TNX",    "   {:.3f}%"),
    ("Gold",    "GC=F",    "   ${:.0f}"),
    ("VIX",     "^VIX",    "   {:.2f}"),
    ("S&P 500", "^GSPC",   "   {:.0f}"),
    ("GBP/USD", "GBPUSD=X","   {:.4f}"),
]):
    d = get_prices(sym, period="5d")
    if d is not None and len(d) > 1:
        cur  = d["Close"].iloc[-1]
        pchg = (cur / d["Close"].iloc[-2] - 1) * 100
        col.metric(label, fmt.format(cur).strip(), f"{pchg:+.2f}%")


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
(tab_score, tab_macro, tab_sent, tab_stat,
 tab_ai, tab_cot, tab_val, tab_cs,
 tab_bonds, tab_season, tab_news) = st.tabs([
    "🎯 Scorecard",
    "🌍 Macro & Carry",
    "😨 Sentiment",
    "⏱️  Stat Edge",
    "🤖 AI Analysis",
    "📋 COT Report",
    "📊 Valuation",
    "💱 FX Strength",
    "🏦 Bonds",
    "📅 Seasonality",
    "📰 News",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SCORECARD  (EdgeFinder-style composite)
# ══════════════════════════════════════════════════════════════
with tab_score:
    st.header("🎯  Asset Scorecard — Directional Bias")
    st.markdown(
        "Composite score across COT positioning · Macro backdrop · Carry · "
        "Valuation · Seasonality · Sentiment. "
        "**Score –5 = strong bear, 0 = neutral, +5 = strong bull.**"
    )
    with st.expander("📐  How each component is scored"):
        st.markdown("""
| Component | Source | Logic |
|---|---|---|
| **COT** | CFTC futures | Fade retail (nonreportable) extremes; amplified when commercials/non-comms sit opposite. +1 = fade-long, −1 = fade-short |
| **Carry** | CB rates (sidebar) | Rate differential ±5% → ±1 |
| **Macro** | World Bank GDP/CPI/unemployment | G7 hawkishness score per country |
| **Valuation** | yfinance daily | Z-score inverted: cheap = bullish |
| **Seasonality** | yfinance 10Y monthly | Win-rate + avg return for current month |
| **Sentiment** | Fear/Greed gauge | Risk-on assets benefit from greed; risk-off from fear |
| **Total /5** | Weighted average | Sum of available components, scaled to –5/+5 |
        """)

    if not ALL_INSTRUMENTS:
        st.info("Select instruments in the sidebar.")
    else:
        with st.spinner("Computing scorecard (first run may take ~30 seconds)…"):
            cot_df_sc = None   # COT now uses per-instrument historical index (get_cot_index_signal)
            macro_sc  = get_all_macro_scores()
            sg        = get_sentiment_gauge()
            fg_val    = sg["composite"] if sg else None
            sc_df     = compute_scorecard(
                ALL_INSTRUMENTS, cb_rates, cot_df_sc,
                macro_sc, fg_val, val_lookback, season_years
            )

        # ── AUTO EXPORT TO MT5 ────────────────────────────────────────
        # Writes fundamental_bias.json every time scores are computed
        # All EAs read this file for directional bias
        try:
            import json, os, pathlib

            export_data = {
                "timestamp":   datetime.now().isoformat(),
                "instruments": {}
            }

            for _, row in sc_df.iterrows():
                inst      = row["Instrument"]
                score     = row["Score /5"]
                bias_raw  = row["Bias"]

                # Strip emoji for EA parsing
                bias_clean = (bias_raw
                    .replace("🟢🟢 ", "").replace("🟢 ", "")
                    .replace("🔴🔴 ", "").replace("🔴 ", "")
                    .replace("🟡 ", "").strip())

                # Direction for EA
                if   score >  1.5: direction = "LONG"
                elif score < -1.5: direction = "SHORT"
                else:              direction = "NEUTRAL"

                export_data["instruments"][inst] = {
                    "score":     round(float(score), 2),
                    "bias":      bias_clean,
                    "direction": direction
                }

            json_str = json.dumps(export_data, indent=2)
            paths_to_write = []

            # 1. Same folder as dashboard script
            script_dir = pathlib.Path(__file__).parent
            paths_to_write.append(script_dir / "fundamental_bias.json")

            # 2. MT5 Common Files folder (Windows)
            mt5_common = os.environ.get("APPDATA", "")
            if mt5_common:
                mt5_path = (pathlib.Path(mt5_common)
                            / "MetaQuotes" / "Terminal" / "Common" / "Files")
                if mt5_path.exists():
                    paths_to_write.append(mt5_path / "fundamental_bias.json")

            # 3. Search for MQL5/Files folder
            home = pathlib.Path(os.path.expanduser("~"))
            for candidate in home.rglob("MQL5/Files"):
                paths_to_write.append(candidate / "fundamental_bias.json")
                break

            written = []
            for p in paths_to_write:
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json_str)
                    written.append(str(p))
                except Exception:
                    pass

            if written:
                st.success(
                    f"✅ Bias exported to MT5 — "
                    f"{datetime.now().strftime('%H:%M:%S')} | "
                    f"{len(export_data['instruments'])} instruments"
                )
                with st.expander("View exported bias"):
                    st.json(export_data)
            else:
                st.warning("⚠️ Could not write to MT5 folder")

        except Exception as e:
            st.error(f"Export error: {e}")

        # ── Summary row of counts ──────────────────────────────
        bull_n   = (sc_df["Score /5"] >  1.5).sum()
        bear_n   = (sc_df["Score /5"] < -1.5).sum()
        neut_n   = len(sc_df) - bull_n - bear_n
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Bullish", bull_n, help="Score > +1.5")
        sc2.metric("Neutral", neut_n, help="-1.5 to +1.5")
        sc3.metric("Bearish", bear_n, help="Score < -1.5")
        sc4.metric("F/G Gauge", f"{fg_val:.0f}/100" if fg_val else "N/A",
                   sg["label"] if sg else "")

        # ── Heatmap table ──────────────────────────────────────
        num_cols = ["COT","Carry","Macro","Valuation","Seasonality","Sentiment","Score /5"]
        display_df = sc_df.copy()

        def colorize(val):
            if pd.isna(val) or val is None:
                return "color: #4a5568"
            if   val > 0.3: return "color: #56d364; font-weight:700"
            elif val < -0.3: return "color: #f85149; font-weight:700"
            else:            return "color: #e3b341"

        styled = display_df.style.map(
            colorize,
            subset=num_cols
        ).format(
            {c: lambda x: f"{x:+.2f}" if pd.notna(x) and x is not None else "—"
             for c in num_cols}
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

        # ── Score bar chart ────────────────────────────────────
        fig = go.Figure(go.Bar(
            x=sc_df["Instrument"],
            y=sc_df["Score /5"],
            marker_color=[
                "#1a472a" if s > 3.5 else "#56d364" if s > 1.5
                else "#f85149" if s < -1.5 else "#3d1a1a" if s < -3.5
                else "#e3b341"
                for s in sc_df["Score /5"]
            ],
            text=[f"{s:+.1f}" for s in sc_df["Score /5"]],
            textposition="outside",
        ))
        fig.add_hline(y= 1.5, line_dash="dot", line_color="#56d364", annotation_text="Bull threshold")
        fig.add_hline(y=-1.5, line_dash="dot", line_color="#f85149", annotation_text="Bear threshold")
        fig.update_layout(
            title="Composite Fundamental Score (–5 to +5)",
            template="plotly_dark", height=400,
            yaxis=dict(range=[-5.5, 5.5], title="Score"),
            xaxis_tickangle=-30, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Radar for top instrument ───────────────────────────
        if len(sc_df) > 0:
            top = sc_df.iloc[0]
            radar_vals = [top[c] for c in ["COT","Carry","Macro","Valuation","Seasonality","Sentiment"]
                          if pd.notna(top[c]) and top[c] is not None]
            radar_cats = [c for c in ["COT","Carry","Macro","Valuation","Seasonality","Sentiment"]
                          if pd.notna(top[c]) and top[c] is not None]
            if len(radar_vals) >= 3:
                r_col1, r_col2 = st.columns([1,1])
                with r_col1:
                    st.subheader(f"Top Setup: {top['Instrument']}")
                    st.metric("Score", f"{top['Score /5']:+.2f}")
                    st.markdown(f"**{top['Bias']}**")
                with r_col2:
                    fig_r = go.Figure(go.Scatterpolar(
                        r=radar_vals + [radar_vals[0]],
                        theta=radar_cats + [radar_cats[0]],
                        fill="toself",
                        fillcolor="rgba(86,211,100,0.15)",
                        line=dict(color="#56d364", width=2),
                    ))
                    fig_r.update_layout(
                        polar=dict(radialaxis=dict(range=[-1,1], showticklabels=False)),
                        template="plotly_dark", height=280,
                        margin=dict(l=20,r=20,t=20,b=20),
                    )
                    st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — MACRO & CARRY
# ══════════════════════════════════════════════════════════════
with tab_macro:
    st.header("🌍  G7 Macro Heatmap & Carry Trade")
    st.caption("Macro data: World Bank (annual, free API). Carry: sidebar CB rates.")

    with st.spinner("Fetching G7 macro data from World Bank…"):
        macro_data = get_all_macro_scores()

    # ── Economic heatmap ──────────────────────────────────────
    st.subheader("G7 Economic Scorecard")
    macro_rows = []
    for cc, d in macro_data.items():
        gdp  = f"{d['gdp']:+.1f}% ({d['gdp_year']})"   if d.get("gdp")  is not None else "N/A"
        cpi  = f"{d['cpi']:+.1f}% ({d['cpi_year']})"   if d.get("cpi")  is not None else "N/A"
        unem = f"{d['unemployment']:.1f}% ({d['unem_year']})" if d.get("unemployment") is not None else "N/A"
        score_v = d.get("score", 0)
        if   score_v >  1: bias = "🟢 HAWKISH"
        elif score_v >  0: bias = "🟩 MILD HAWK"
        elif score_v > -1: bias = "🟥 MILD DOVE"
        else:              bias = "🔴 DOVISH"
        macro_rows.append({
            "Currency": f"{G7_COUNTRIES.get(d['currency'],(None,None,''))[2]} {d['currency']}",
            "Country":  d["label"],
            "GDP Growth": gdp,
            "CPI Inflation": cpi,
            "Unemployment": unem,
            "Score": round(score_v, 2),
            "CB Rate %": cb_rates.get(d["currency"], "?"),
            "Macro Bias": bias,
        })
    macro_df = pd.DataFrame(macro_rows).sort_values("Score", ascending=False)
    st.dataframe(macro_df, use_container_width=True, hide_index=True)

    # ── Score bar chart ────────────────────────────────────────
    fig_m = go.Figure(go.Bar(
        x=macro_df["Currency"], y=macro_df["Score"],
        marker_color=["#56d364" if s > 0 else "#f85149" for s in macro_df["Score"]],
        text=[f"{s:+.2f}" for s in macro_df["Score"]],
        textposition="outside",
    ))
    fig_m.update_layout(title="G7 Macro Hawkishness Score",
                        template="plotly_dark", height=320,
                        yaxis_title="Score (positive = hawkish)",
                        showlegend=False)
    st.plotly_chart(fig_m, use_container_width=True)

    st.info(
        "**Note:** World Bank data is annual and may lag 1–2 years. "
        "It reflects structural macro backdrop. For latest prints (CPI, NFP), "
        "cross-reference with your economic calendar."
    )

    st.markdown("---")
    # ── Carry Trade Scanner ────────────────────────────────────
    st.subheader("Carry Trade Scanner — Rate Differentials")
    st.caption("Higher carry = base currency pays more interest than quote. Positive = favorable carry for long.")

    carry_rows = []
    for pair_name in list(FOREX_PAIRS.keys()):
        if "/" not in pair_name:
            continue
        base, quote = pair_name.split("/")
        br = cb_rates.get(base.strip())
        qr = cb_rates.get(quote.strip())
        if br is None or qr is None:
            continue
        diff = round(br - qr, 2)
        if   diff >  2.0: label = "🟢 HIGH CARRY"
        elif diff >  0.5: label = "🟩 MILD CARRY"
        elif diff > -0.5: label = "🟡 NEGLIGIBLE"
        elif diff > -2.0: label = "🟥 MILD NEG"
        else:             label = "🔴 HIGH NEG CARRY"
        carry_rows.append({
            "Pair":       pair_name,
            "Base Rate":  f"{br:.2f}%",
            "Quote Rate": f"{qr:.2f}%",
            "Differential": f"{diff:+.2f}%",
            "Carry Label":  label,
        })
    carry_df = pd.DataFrame(carry_rows)
    if not carry_df.empty:
        carry_df["diff_num"] = carry_df["Differential"].str.replace("%","").str.replace("+","").astype(float)
        carry_df = carry_df.sort_values("diff_num", ascending=False)
        st.dataframe(carry_df.drop("diff_num", axis=1), use_container_width=True, hide_index=True)

        fig_c = go.Figure(go.Bar(
            x=carry_df["Pair"],
            y=carry_df["diff_num"],
            marker_color=["#56d364" if d > 0 else "#f85149" for d in carry_df["diff_num"]],
            text=carry_df["Differential"],
            textposition="outside",
        ))
        fig_c.update_layout(title="Interest Rate Differentials (Base – Quote)",
                            template="plotly_dark", height=380,
                            yaxis_title="Rate Differential (%)",
                            xaxis_tickangle=-30, showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)

    # ── Rate Differential Matrix ───────────────────────────────
    st.subheader("Rate Differential Matrix")
    currs = list(cb_rates.keys())
    rates = [cb_rates[c] for c in currs]
    matrix = np.array(rates)[:,None] - np.array(rates)[None,:]
    fig_hm = go.Figure(go.Heatmap(
        z=matrix, x=currs, y=currs,
        colorscale="RdYlGn",
        text=[[f"{v:+.2f}%" for v in row] for row in matrix],
        texttemplate="%{text}",
        colorbar=dict(title="%"),
    ))
    fig_hm.update_layout(
        title="Row currency rate − Column currency rate (positive = row has higher rate)",
        template="plotly_dark", height=380,
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — SENTIMENT & RISK
# ══════════════════════════════════════════════════════════════
with tab_sent:
    st.header("😨  Sentiment & Risk Gauge")

    with st.spinner("Calculating sentiment components…"):
        sg = get_sentiment_gauge()

    if sg is None:
        st.warning("Sentiment data unavailable. Check connection.")
    else:
        # ── Gauge ──────────────────────────────────────────────
        g1, g2 = st.columns([1, 2])
        with g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sg["composite"],
                title={"text": f"Fear & Greed<br>{sg['emoji']} {sg['label']}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#58a6ff", "thickness": 0.3},
                    "steps": [
                        {"range":[0,25],  "color":"#3d1a1a"},
                        {"range":[25,45], "color":"#5a2d0c"},
                        {"range":[45,55], "color":"#21262d"},
                        {"range":[55,75], "color":"#1a3d1a"},
                        {"range":[75,100],"color":"#0d2d0d"},
                    ],
                    "threshold":{"line":{"color":"white","width":3},
                                 "thickness":0.75,"value":sg["composite"]},
                },
            ))
            fig_g.update_layout(template="plotly_dark", height=270,
                                margin=dict(l=20,r=20,t=50,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        with g2:
            st.subheader("Component Breakdown")
            for comp, val in sg["components"].items():
                if   val > 60: c_col, c_lbl = "#56d364", "Greed"
                elif val < 40: c_col, c_lbl = "#f85149", "Fear"
                else:          c_col, c_lbl = "#e3b341", "Neutral"
                pct = int(val)
                bar_html = (
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
                    f"<span style='width:140px;font-size:13px'>{comp}</span>"
                    f"<div style='flex:1;background:#21262d;border-radius:4px;height:18px'>"
                    f"<div style='width:{pct}%;background:{c_col};height:18px;border-radius:4px'></div></div>"
                    f"<span style='width:60px;color:{c_col};font-weight:700'>{pct}/100 {c_lbl}</span>"
                    f"</div>"
                )
                st.markdown(bar_html, unsafe_allow_html=True)

    st.markdown("---")
    # ── Risk On / Off by asset class ──────────────────────────
    st.subheader("Risk On / Off Implications")
    fg_now = sg["composite"] if sg else 50
    risk_state = ("RISK-ON 🟢" if fg_now > 55 else
                  "RISK-OFF 🔴" if fg_now < 45 else "NEUTRAL 🟡")
    st.markdown(f"### Current Regime: {risk_state}")
    ri1, ri2, ri3 = st.columns(3)
    ri1.markdown("""
**If Risk-ON (greed > 55)**
- ✅ Long AUD, NZD, CAD
- ✅ Long equity indices (S&P, Nasdaq)
- ✅ Short JPY, CHF (funding)
- ⚠️ Gold may lag
""")
    ri2.markdown("""
**If Risk-OFF (fear < 45)**
- ✅ Long JPY, CHF, USD
- ✅ Long Gold / safe havens
- ✅ Long bonds (yields fall)
- ⚠️ Fade equity strength
""")
    ri3.markdown("""
**If Neutral (45–55)**
- ⚪ Range / mean-reversion
- ⚪ Favour carry & seasonality
- ⚪ Trade the stat edge
- ⚠️ Lower conviction
""")


# ══════════════════════════════════════════════════════════════
# TAB 4 — STAT EDGE  (Edgeful-style day-trading stats)
# ══════════════════════════════════════════════════════════════
with tab_stat:
    st.header("⏱️  Statistical Edge — Day-Trading Layer")
    st.caption("ATR regime · Day-of-week bias · Gap fills · Opening-range breakout. Pick an instrument.")

    stat_choices = list(ALL_INSTRUMENTS.keys()) or ["S&P 500"]
    stat_name = st.selectbox("Instrument", stat_choices, key="stat_sel")
    stat_sym  = ALL_INSTRUMENTS.get(stat_name, "^GSPC")

    with st.spinner(f"Crunching stats for {stat_name}…"):
        atr = get_atr_data(stat_sym)
        dow = get_dow_stats(stat_sym)
        gap = get_gap_stats(stat_sym)
        orb = get_orb_stats(stat_sym)

    # ── ATR / volatility regime ────────────────────────────────
    st.subheader("Volatility Regime (ATR)")
    if atr:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("ATR(14)", f"{atr['atr14']:.4f}")
        a2.metric("ATR(50)", f"{atr['atr50']:.4f}")
        a3.metric("Regime", atr["regime"])
        a4.metric("Today's range used", f"{atr['range_used']:.0f}%",
                  help="Today's high-low as a % of ATR(14)")
        if atr.get("atr_series") is not None:
            fig_atr = go.Figure(go.Scatter(
                y=atr["atr_series"].dropna().tail(120).values,
                mode="lines", line=dict(color="#58a6ff")))
            fig_atr.update_layout(title="ATR(14) — last 120 bars",
                                  template="plotly_dark", height=240,
                                  margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_atr, use_container_width=True)
    else:
        st.info("Not enough data for ATR on this instrument.")

    st.markdown("---")
    c_dow, c_gap = st.columns(2)

    # ── Day-of-week seasonality ────────────────────────────────
    with c_dow:
        st.subheader("Day-of-Week Bias (2y)")
        if dow is not None:
            st.dataframe(dow, use_container_width=True)
            fig_dow = go.Figure(go.Bar(
                x=list(dow.index), y=dow["avg_return"],
                marker_color=["#56d364" if v > 0 else "#f85149" for v in dow["avg_return"]],
                text=[f"{v:+.2f}%" for v in dow["avg_return"]], textposition="outside"))
            fig_dow.update_layout(title="Avg daily return by weekday",
                                  template="plotly_dark", height=280, showlegend=False)
            st.plotly_chart(fig_dow, use_container_width=True)
        else:
            st.info("Not enough data.")

    # ── Gap fill stats ─────────────────────────────────────────
    with c_gap:
        st.subheader("Gap Fill Stats")
        if gap is not None:
            g1, g2, g3 = st.columns(3)
            g1.metric("Overall fill rate", f"{gap['fill_rate']:.0f}%",
                      help=f"{gap['total']} gaps over {gap['n_years']}y")
            g2.metric("Up-gap fill", f"{gap['up_fill']:.0f}%" if gap["up_fill"] else "N/A")
            g3.metric("Down-gap fill", f"{gap['down_fill']:.0f}%" if gap["down_fill"] else "N/A")
            if gap.get("by_size") is not None and not gap["by_size"].empty:
                st.dataframe(gap["by_size"], use_container_width=True)
        else:
            st.info("Not enough data.")

    st.markdown("---")
    # ── Opening Range Breakout ─────────────────────────────────
    st.subheader("Opening Range Breakout (hourly, 2y)")
    if orb is not None:
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Breaks UP",   f"{orb['break_up_pct']:.0f}%")
        o2.metric("Breaks DOWN", f"{orb['break_dn_pct']:.0f}%")
        o3.metric("Breaks BOTH", f"{orb['both_break_pct']:.0f}%")
        o4.metric("Neither",     f"{orb['neither_pct']:.0f}%")
        st.caption(
            f"Bullish-day rate: {orb['bull_day_pct']:.0f}% · "
            + (f"After up-break, close bullish: {orb['up_break_bull_pct']:.0f}% · "
               if orb.get("up_break_bull_pct") else "")
            + f"sample: {orb['n']} days"
        )
    else:
        st.info("ORB needs intraday history — unavailable for this instrument/timeframe.")


# ══════════════════════════════════════════════════════════════
# TAB 5 — AI ANALYSIS  (Claude macro report)
# ══════════════════════════════════════════════════════════════
with tab_ai:
    st.header("🤖  AI Macro Analysis")
    st.caption("Claude synthesises the current macro/sentiment picture into a written report.")

    ai_model = st.selectbox(
        "Model",
        ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
        index=0,
        help="Sonnet 5 is a good balance of quality and speed.",
    )
    ai_focus = st.text_input("Focus (optional)",
                             placeholder="e.g. GBP/JPY swing bias into next week")

    if not api_key:
        st.info("Add your Anthropic API key in the sidebar to enable AI analysis.")
    elif st.button("🧠  Generate Report", use_container_width=True):
        with st.spinner("Gathering context and calling Claude…"):
            macro_ctx = get_all_macro_scores()
            sg_ctx    = get_sentiment_gauge()
            strength  = get_currency_strength(cs_lookback)

            macro_lines = []
            for cc, d in macro_ctx.items():
                macro_lines.append(
                    f"- {d['currency']} ({d['label']}): score {d['score']:+.2f}, "
                    f"GDP {d.get('gdp')}, CPI {d.get('cpi')}, unemployment {d.get('unemployment')}"
                )
            strength_line = ", ".join(f"{c} {v:+.2f}" for c, v in
                                      sorted(strength.items(), key=lambda kv: kv[1], reverse=True))
            rates_line = ", ".join(f"{c} {r:.2f}%" for c, r in cb_rates.items())
            fg_line = (f"{sg_ctx['composite']:.0f}/100 ({sg_ctx['label']})"
                       if sg_ctx else "unavailable")

            prompt = f"""You are a senior macro strategist. Using ONLY the data below, write a concise
markdown trading report. Be specific and actionable, note conflicts between signals, and
avoid disclaimers.

Fear/Greed: {fg_line}
Central bank rates: {rates_line}
Currency strength (14d momentum, strongest first): {strength_line}

G7 macro scorecard (positive = hawkish):
{chr(10).join(macro_lines)}

Focus requested: {ai_focus or 'general — top 3 opportunities across FX, metals, indices'}

Structure:
## Regime
## Top Opportunities (3, each: instrument, direction, rationale, key risk)
## What Would Change My Mind
"""
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model=ai_model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}],
                )
                report = "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                )
                st.markdown(report)
            except Exception as e:
                st.error(f"AI request failed: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 6 — COT REPORT
# ══════════════════════════════════════════════════════════════
with tab_cot:
    st.header("📋  COT Index — Commercials vs Retail")
    st.caption(
        f"Each group's net (% of OI) placed within its **own** range: 0 = most bearish, "
        f"100 = most bullish. Index lookback **{COT_LB_SHORT}w**, historical hi/lo over "
        f"**{COT_LB_LONG}w**. Bullish when **commercials high (buying) & retail low (selling)** — "
        "this is why gold reads bullish even though commercials are net-short in absolute terms. "
        "Financials from the TFF report; metals from the disaggregated report."
    )

    cot_names = [n for n in ALL_INSTRUMENTS if n in INSTRUMENT_COT_MAP] or list(INSTRUMENT_COT_MAP.keys())
    with st.spinner("Building COT indices (first run pulls 3y history per market)…"):
        cot_rows = []
        for nm in cot_names:
            o = get_cot_index_signal(nm)
            if o is None:
                continue
            cot_rows.append({
                "Instrument":   nm,
                f"Comm {COT_LB_SHORT}w":   f"{o['comm_s']:.0f}",
                f"Comm {COT_LB_LONG}w":    f"{o['comm_l']:.0f}",
                f"Retail {COT_LB_SHORT}w": f"{o['retail_s']:.0f}",
                f"Retail {COT_LB_LONG}w":  f"{o['retail_l']:.0f}",
                "Signal":       cot_index_label(o),
                "_score":       o["score"],
                "_comm":        o["comm_l"],
                "_retail":      o["retail_l"],
            })

    if not cot_rows:
        st.warning("COT history unavailable (CFTC API may be down).")
    else:
        cdf = pd.DataFrame(cot_rows).sort_values("_score", ascending=False)
        st.dataframe(cdf.drop(columns=["_score", "_comm", "_retail"]),
                     use_container_width=True, hide_index=True, height=460)

        st.subheader(f"Commercials vs Retail — {COT_LB_LONG}-week COT Index")
        fig_cot = go.Figure()
        fig_cot.add_trace(go.Bar(name="Commercials", x=cdf["Instrument"], y=cdf["_comm"],
                                 marker_color="#56d364"))
        fig_cot.add_trace(go.Bar(name="Retail", x=cdf["Instrument"], y=cdf["_retail"],
                                 marker_color="#f85149"))
        fig_cot.add_hline(y=80, line_dash="dot", line_color="#8b949e", annotation_text="bullish extreme")
        fig_cot.add_hline(y=20, line_dash="dot", line_color="#8b949e", annotation_text="bearish extreme")
        fig_cot.update_layout(barmode="group", template="plotly_dark", height=420,
                              yaxis=dict(range=[0, 100], title="COT Index (0-100)"),
                              xaxis_tickangle=-30,
                              title="High commercials + low retail = bullish setup")
        st.plotly_chart(fig_cot, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 7 — VALUATION
# ══════════════════════════════════════════════════════════════
with tab_val:
    st.header("📊  Valuation — Over / Undervalued")
    st.caption(
        "Is each instrument rich or cheap **relative to what it's priced against**? "
        "FX pairs are valued base-vs-quote (the pair itself); indices/metals/stocks "
        "are valued against the reference you pick below. "
        f"Measured by position in its own {val_lookback}-day range — no z-score."
    )

    REF_OPTIONS = {
        "US Dollar (DXY)":     "DX-Y.NYB",
        "Treasury Bonds (TLT)":"TLT",
        "Raw USD price":       None,
    }
    ref_choice = st.selectbox("Value indices / metals / stocks against", list(REF_OPTIONS),
                              index=0, key="val_ref")
    ref_symbol = REF_OPTIONS[ref_choice]
    ref_label  = ref_choice.split(" (")[0]   # e.g. "US Dollar"

    if not ALL_INSTRUMENTS:
        st.info("Select instruments in the sidebar.")
    else:
        with st.spinner("Scoring valuation…"):
            val_rows = []
            for name, sym in ALL_INSTRUMENTS.items():
                v, vs = valuation_for_instrument(name, sym, val_lookback, ref_symbol, ref_label)
                if v is None:
                    continue
                val_rows.append({
                    "Instrument":   name,
                    "Valued vs":    vs,
                    "Level":        round(v["current"], 4),
                    "Range Pos":    f"{v['range_pos']:.0f}%",
                    "Percentile":   f"{v['percentile']:.0f}%",
                    "EMA Dev %":    v["ema_dev"],
                    "Signal":       v["signal"],
                    "_pct":         v["percentile"],
                })
        if val_rows:
            vdf = pd.DataFrame(val_rows).sort_values("_pct")
            show = vdf.drop(columns=["_pct"])
            st.dataframe(show, use_container_width=True, hide_index=True, height=420)

            fig_val = go.Figure(go.Bar(
                x=vdf["Instrument"], y=vdf["_pct"],
                marker_color=["#f85149" if p >= 85 else "#f0883e" if p >= 65
                              else "#56d364" if p <= 15 else "#e3b341" if p <= 35
                              else "#8b949e" for p in vdf["_pct"]],
                text=[f"{p:.0f}%" for p in vdf["_pct"]], textposition="outside"))
            fig_val.add_hline(y=85, line_dash="dot", line_color="#f85149", annotation_text="Overvalued")
            fig_val.add_hline(y=15, line_dash="dot", line_color="#56d364", annotation_text="Undervalued")
            fig_val.update_layout(title="Position in range (100% = top / rich, 0% = bottom / cheap)",
                                  template="plotly_dark", height=380,
                                  xaxis_tickangle=-30, showlegend=False,
                                  yaxis=dict(range=[0, 105], title="Percentile"))
            st.plotly_chart(fig_val, use_container_width=True)
        else:
            st.info("No valuation data available.")


# ══════════════════════════════════════════════════════════════
# TAB 8 — FX STRENGTH
# ══════════════════════════════════════════════════════════════
with tab_cs:
    st.header("💱  Relative Currency Strength & Correlation")
    st.caption(f"Strength = averaged {cs_lookback}-day momentum across major crosses.")

    with st.spinner("Computing currency strength…"):
        strength = get_currency_strength(cs_lookback)

    sdf = (pd.DataFrame({"Currency": list(strength.keys()),
                         "Strength": list(strength.values())})
           .sort_values("Strength", ascending=False))
    fig_s = go.Figure(go.Bar(
        x=sdf["Currency"], y=sdf["Strength"],
        marker_color=["#56d364" if v > 0 else "#f85149" for v in sdf["Strength"]],
        text=[f"{v:+.2f}" for v in sdf["Strength"]], textposition="outside"))
    fig_s.update_layout(title="Currency Strength Ranking",
                        template="plotly_dark", height=340, showlegend=False,
                        yaxis_title="Relative strength")
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Matrix (60-day returns)")
    corr_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
                  "USD/CAD", "NZD/USD", "USD/CHF"]
    closes = {}
    for pn in corr_pairs:
        sym = FOREX_PAIRS.get(pn)
        d = get_prices(sym, period="3mo") if sym else None
        if d is not None and len(d) > 20:
            closes[pn] = d["Close"]
    if len(closes) >= 3:
        rets = pd.DataFrame(closes).pct_change().dropna()
        corr = rets.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(template="plotly_dark", height=420,
                               title="Pairwise return correlation")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough data for a correlation matrix right now.")

    st.markdown("---")
    st.subheader("Cross Implied vs Actual")
    cross_name = st.selectbox("Cross", list(CROSS_IMPLIED.keys()), key="cross_sel")
    ci = get_cross_implied(cross_name)
    if ci:
        ci1, ci2, ci3, ci4 = st.columns(4)
        ci1.metric("Actual",   f"{ci['actual']:.5f}")
        ci2.metric("Implied",  f"{ci['implied']:.5f}")
        ci3.metric("Divergence", f"{ci['div_pct']:+.3f}%")
        ci4.metric("Signal",   ci["signal"])
        if ci.get("history_div") is not None:
            fig_ci = go.Figure(go.Scatter(
                y=ci["history_div"].tail(120).values, mode="lines",
                line=dict(color="#58a6ff")))
            fig_ci.add_hline(y=0, line_dash="dot", line_color="#8b949e")
            fig_ci.update_layout(title=f"{cross_name} actual vs implied divergence (%)",
                                 template="plotly_dark", height=260,
                                 margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_ci, use_container_width=True)
    else:
        st.info("Implied cross data unavailable.")


# ══════════════════════════════════════════════════════════════
# TAB 9 — BONDS
# ══════════════════════════════════════════════════════════════
with tab_bonds:
    st.header("🏦  US Treasury Yields & Curve")
    st.caption("Yields from Yahoo (^IRX 3M, ^FVX 5Y, ^TNX 10Y, ^TYX 30Y).")

    labels, mats, yields, changes = [], [], [], []
    maturity_order = [("US 2-Year", "^IRX", "3M"), ("US 5-Year", "^FVX", "5Y"),
                      ("US 10-Year", "^TNX", "10Y"), ("US 30-Year", "^TYX", "30Y")]
    with st.spinner("Fetching yields…"):
        for name, sym, mat in maturity_order:
            d = get_prices(sym, period="5d")
            if d is not None and len(d) > 1:
                cur  = d["Close"].iloc[-1]
                prev = d["Close"].iloc[-2]
                labels.append(name); mats.append(mat)
                yields.append(round(cur, 3)); changes.append(round((cur - prev) * 100, 1))

    if yields:
        bcols = st.columns(len(yields))
        for col, name, y, ch in zip(bcols, labels, yields, changes):
            col.metric(name, f"{y:.3f}%", f"{ch:+.1f} bps")

        fig_curve = go.Figure(go.Scatter(
            x=mats, y=yields, mode="lines+markers",
            line=dict(color="#58a6ff", width=3), marker=dict(size=10)))
        fig_curve.update_layout(title="US Yield Curve",
                                template="plotly_dark", height=360,
                                xaxis_title="Maturity", yaxis_title="Yield %")
        st.plotly_chart(fig_curve, use_container_width=True)

        if len(yields) >= 3:
            spread = round(yields[-2] - yields[0], 3)  # 10Y - 3M proxy
            state  = "🔴 INVERTED (recession signal)" if spread < 0 else "🟢 NORMAL"
            st.metric("10Y − 3M spread", f"{spread:+.3f}%", state)
    else:
        st.warning("Yield data unavailable right now.")


# ══════════════════════════════════════════════════════════════
# TAB 10 — SEASONALITY
# ══════════════════════════════════════════════════════════════
with tab_season:
    st.header("📅  Seasonality — Monthly Historical Bias")
    st.caption(f"Average monthly return & win-rate over {season_years} years. Any market.")

    seas_choices = list(ALL_INSTRUMENTS.keys()) or ["Gold"]
    seas_name = st.selectbox("Instrument", seas_choices, key="seas_sel")
    seas_sym  = ALL_INSTRUMENTS.get(seas_name, "GC=F")

    with st.spinner("Building seasonality…"):
        sd = get_seasonality(seas_sym, season_years)

    if sd is None:
        st.info("Not enough monthly history for this instrument.")
    else:
        cur_month = datetime.now().strftime("%b")
        colors = ["#58a6ff" if m == cur_month else
                  ("#56d364" if v > 0 else "#f85149")
                  for m, v in zip(sd.index, sd["avg"])]
        fig_se = go.Figure(go.Bar(
            x=list(sd.index), y=sd["avg"], marker_color=colors,
            text=[f"{v:+.1f}%" for v in sd["avg"]], textposition="outside"))
        fig_se.update_layout(title=f"{seas_name} — avg monthly return (current month highlighted)",
                             template="plotly_dark", height=360, showlegend=False,
                             yaxis_title="Avg return %")
        st.plotly_chart(fig_se, use_container_width=True)

        show = sd.copy()
        show["avg"]      = show["avg"].round(2)
        show["std"]      = show["std"].round(2)
        show["win_rate"] = show["win_rate"].round(0)
        show.columns = ["Avg Return %", "Std Dev", "Samples", "Win Rate %"]
        st.dataframe(show, use_container_width=True)

        if cur_month in sd.index:
            wr = sd.loc[cur_month, "win_rate"]; av = sd.loc[cur_month, "avg"]
            st.metric(f"This month ({cur_month})", f"{av:+.2f}% avg", f"{wr:.0f}% win rate")


# ══════════════════════════════════════════════════════════════
# TAB 11 — NEWS + SESSION CLOCK
# ══════════════════════════════════════════════════════════════
with tab_news:
    st.header("📰  Market News & Session Clock")

    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour + now_utc.minute / 60.0
    st.subheader(f"Trading Sessions — {now_utc.strftime('%H:%M')} UTC")
    sess_cols = st.columns(len(SESSION_UTC))
    for col, (sname, o, c) in zip(sess_cols, SESSION_UTC):
        if o <= c:
            is_open = o <= hour < c
        else:  # wraps midnight (e.g. Sydney)
            is_open = hour >= o or hour < c
        col.metric(sname, "OPEN 🟢" if is_open else "closed",
                   f"{o:02d}:00–{c:02d}:00 UTC")

    st.markdown("---")
    st.subheader("Headlines")
    with st.spinner("Fetching RSS feeds…"):
        news = get_news()
    if not news:
        st.info("No headlines available (feeds may be rate-limited).")
    else:
        for item in news:
            title = item["title"] or "(untitled)"
            link  = item["link"]
            st.markdown(f"**[{title}]({link})**  ·  _{item['source']}_")
            if item["summary"]:
                st.caption(item["summary"])
            st.markdown("")


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Data: yfinance · World Bank · CFTC · RSS. For research/education only — not financial advice. "
    f"Rendered {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC."
)
