import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Zone Ticket", layout="centered")

NY = ZoneInfo("America/New_York")
SYD = ZoneInfo("Australia/Sydney")

# ================= scanner (zone detection, volume, VWAP, structure) =================
RV_STRONG = 2.0                      # relative volume at a zone touch that counts as strong interest
IDX_BARS, IDX_NET, IDX_BAR = 4, 3.0, 2.0   # SPY/QQQ: net move over 4 bars, or one bar, in ATR


def atr_series(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def completed(d15, minutes=15):
    """Drop the last bar when it is still forming, so its partial volume is not misread."""
    idx = d15.index
    if isinstance(idx, pd.DatetimeIndex) and len(d15) > 1:
        now = pd.Timestamp.now(tz=idx.tz) if idx.tz is not None else pd.Timestamp.now()
        if now - idx[-1] < pd.Timedelta(minutes=minutes):
            return d15.iloc[:-1]
    return d15


def session_vwap(d15):
    """Volume weighted average price of the latest session. None when there is no volume feed."""
    idx = d15.index
    if not isinstance(idx, pd.DatetimeIndex) or float(d15.Volume.sum()) <= 0:
        return None
    day = np.array(idx.date)
    last = d15[day == day[-1]]
    v = last.Volume.values.astype(float)
    if v.sum() <= 0:
        return None
    tp = ((last.High + last.Low + last.Close) / 3).values
    return float((tp * v).sum() / v.sum())


def bar_rvol_at(pv, day, slot):
    """One bar's volume divided by the average for the same time of day on other sessions."""
    if slot not in pv.columns:
        return None
    col = pv[slot]
    cur = col.get(day)
    others = col.drop(day, errors="ignore").dropna()
    if cur is None or pd.isna(cur) or len(others) < 3 or others.mean() <= 0:
        return None
    return float(cur / others.mean())


def intraday_rvol(d15c):
    """Time-of-day adjusted relative volume.
    Returns ({bar timestamp: rvol} for the last 8 bars, cumulative rvol for the session so far).
    Comparing with the same clock time on earlier sessions removes the open and close surges,
    so no fixed time-of-day multiplier is needed."""
    idx = d15c.index
    if not isinstance(idx, pd.DatetimeIndex) or len(d15c) < 30 or float(d15c.Volume.sum()) <= 0:
        return {}, None
    d = pd.DataFrame({"v": d15c.Volume.values.astype(float), "day": idx.date,
                      "slot": idx.hour * 60 + idx.minute})
    pv = d.pivot_table(index="day", columns="slot", values="v", aggfunc="sum")
    if len(pv) < 4:
        return {}, None
    today = pv.index[-1]
    bars = {ts: bar_rvol_at(pv, ts.date(), int(ts.hour * 60 + ts.minute)) for ts in idx[-8:]}
    last_slot = int(idx[-1].hour * 60 + idx[-1].minute)
    cum = pv[[c for c in pv.columns if c <= last_slot]].sum(axis=1, min_count=1)
    others = cum.drop(today).dropna()
    cum_rv = None
    if len(others) >= 3 and others.mean() > 0 and not pd.isna(cum.get(today)):
        cum_rv = float(cum[today] / others.mean())
    return bars, cum_rv


def structure(d1h, n=2):
    """1 hour market structure from swing points: 'up' (HH and HL), 'down' (LH and LL) or 'range'."""
    hi = (d1h["High"] if "High" in d1h else d1h["Close"]).values
    lo = (d1h["Low"] if "Low" in d1h else d1h["Close"]).values
    ph = [hi[i] for i in range(n, len(hi) - n) if hi[i] == max(hi[i - n:i + n + 1])]
    pl = [lo[i] for i in range(n, len(lo) - n) if lo[i] == min(lo[i - n:i + n + 1])]
    if len(ph) >= 2 and len(pl) >= 2:
        if ph[-1] < ph[-2] and pl[-1] < pl[-2]:
            return "down"
        if ph[-1] > ph[-2] and pl[-1] > pl[-2]:
            return "up"
    return "range"


def index_flags(d):
    """Is this index selling off or spiking right now? Uses 15 minute bars."""
    a = float(atr_series(d).iloc[-1])
    dc = completed(d)
    if not a or np.isnan(a) or len(dc) <= IDX_BARS:
        return None
    net = (dc.Close.iloc[-1] - dc.Close.iloc[-1 - IDX_BARS]) / a
    bar = (dc.Close.iloc[-1] - dc.Open.iloc[-1]) / a
    now = pd.Timestamp.now(tz=d.index.tz) if d.index.tz is not None else pd.Timestamp.now()
    return dict(open=bool(now - d.index[-1] < pd.Timedelta(minutes=40)),
                down=bool(net <= -IDX_NET or bar <= -IDX_BAR),
                up=bool(net >= IDX_NET or bar >= IDX_BAR))


# ---------- zone detection ----------
def find_zones(df, a, impulse=2.0):
    """A zone is 1-3 small candles (the base) followed by a big candle that
    leaves the base and travels at least `impulse` x ATR within 3 candles."""
    o, h, l, c, v = (df[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(df)
    body = np.abs(c - o)
    rng = h - l
    small = (body <= 0.5 * a) & (rng <= 1.2 * a)
    avgv = float(np.mean(v)) if n else 0.0
    zones = []
    for j in range(1, n - 1):
        k = 0
        while k < 3 and j - 1 - k >= 0 and small[j - 1 - k]:
            k += 1
        if k == 0 or body[j] < 0.8 * a:
            continue
        base = slice(j - k, j)
        end = min(j + 3, n)
        if c[j] > o[j]:
            low = l[base].min()
            move = h[j:end].max() - low
            if move < impulse * a:
                continue
            typ, prox, dist = "demand", max(o[base].max(), c[base].max()), low
        elif c[j] < o[j]:
            high = h[base].max()
            move = high - l[j:end].min()
            if move < impulse * a:
                continue
            typ, prox, dist = "supply", min(o[base].min(), c[base].min()), high
        else:
            continue
        height = abs(prox - dist)
        if height < 0.05 * a:
            continue
        after = slice(min(j + 3, n), n)
        if typ == "demand":
            broken = bool((c[after] < dist).any())
            touched = bool((l[after] <= prox).any())
        else:
            broken = bool((c[after] > dist).any())
            touched = bool((h[after] >= prox).any())
        zones.append(dict(type=typ, prox=float(prox), dist=float(dist), height=float(height),
                          idx=j, depart=float(move / a), volx=float(v[j] / avgv) if avgv else 0.0,
                          fresh=not touched, broken=broken))
    return zones


# ---------- play building ----------
ATR_RANGE = {"stock": (2, 8), "forex": (0.25, 2.5), "commodity": (0.8, 5),
             "index": (0.6, 6), "crypto": (2, 10)}


def reaction(d15c, prox, dist, long, look=4):
    """Did price reach the zone in the last `look` completed bars, and did it reject?
    Returns (state, time): 'rejected', 'touched' (no rejection yet) or 'waiting' (not there yet).
    Rejection = the bar tagged the zone and closed back out of it with a long wick
    or an engulfing candle in the trade direction."""
    t = d15c.tail(look + 1)
    if len(t) < 2:
        return "waiting", None
    o, h, l, c = (t[k].values for k in ("Open", "High", "Low", "Close"))
    for i in range(len(t) - 1, 0, -1):
        hit = (l[i] <= prox) if long else (h[i] >= prox)
        if not hit:
            continue
        rng = h[i] - l[i]
        if rng <= 0:
            return "touched", t.index[i]
        if long:
            out = c[i] > prox
            wick = (min(o[i], c[i]) - l[i]) >= 0.5 * rng
            engulf = c[i] > o[i] and c[i] >= o[i - 1] and o[i] <= c[i - 1] and c[i - 1] < o[i - 1]
        else:
            out = c[i] < prox
            wick = (h[i] - max(o[i], c[i])) >= 0.5 * rng
            engulf = c[i] < o[i] and c[i] <= o[i - 1] and o[i] >= c[i - 1] and c[i - 1] > o[i - 1]
        return ("rejected" if out and (wick or engulf) else "touched"), t.index[i]
    return "waiting", None


def decimals(price):
    return 2 if price >= 100 else 3 if price >= 10 else 5 if price >= 1 else 6


def build_plays(tk, d15, d1h, dd, spread=None, max_dist=8.0, buf_pct=10.0, kind="stock"):
    a = float(atr_series(d15).iloc[-1])
    price = float(d15.Close.iloc[-1])
    nd = decimals(price)
    adv = float(dd.Volume.iloc[-21:-1].mean())
    rvol_daily = float(dd.Volume.iloc[-1] / adv) if adv else 0.0
    atr_pct = float(atr_series(dd).iloc[-1] / dd.Close.iloc[-1] * 100)

    d15c = completed(d15)
    bars, cum_rv = intraday_rvol(d15c)
    rvol = cum_rv if cum_rv is not None else rvol_daily
    vwap = session_vwap(d15)
    htf = structure(d1h.tail(80))

    ema1h = d1h.Close.ewm(span=20, adjust=False).mean()
    up1h = d1h.Close.iloc[-1] > ema1h.iloc[-1] and ema1h.iloc[-1] > ema1h.iloc[-5]
    dn1h = d1h.Close.iloc[-1] < ema1h.iloc[-1] and ema1h.iloc[-1] < ema1h.iloc[-5]
    emad = dd.Close.ewm(span=20, adjust=False).mean()
    upd = dd.Close.iloc[-1] > emad.iloc[-1]
    dnd = dd.Close.iloc[-1] < emad.iloc[-1]

    lo, hi = ATR_RANGE.get(kind, ATR_RANGE["stock"])
    if kind == "stock":
        # the spread is judged later against your own setting
        screen = [price > 5, adv >= 1e6, rvol >= 1.5, lo <= atr_pct <= hi]
    else:
        # forex, commodities, indices and crypto have no reliable volume feed here,
        # so the screen only asks for enough daily movement to be worth trading.
        screen = [lo <= atr_pct <= hi]
    screen_ok = all(screen)

    zones = [z for z in find_zones(d15, a) if not z["broken"]]
    session = d15.tail(52)  # roughly the last two sessions
    plays = []
    for z in zones:
        long = z["type"] == "demand"
        entry = z["prox"]
        if long and not price > entry:
            continue
        if (not long) and not price < entry:
            continue
        if abs(price - entry) > max_dist * a:
            continue
        buf = z["height"] * buf_pct / 100
        stop = z["dist"] - buf if long else z["dist"] + buf
        # target: nearest opposing zone ahead of price, else the recent swing
        opp = [x for x in zones if x["type"] == ("supply" if long else "demand")]
        if long:
            ahead = [x["prox"] for x in opp if x["prox"] > price]
            tgt = min(ahead) if ahead else float(session.High.max())
            if tgt <= entry:
                continue
        else:
            ahead = [x["prox"] for x in opp if x["prox"] < price]
            tgt = max(ahead) if ahead else float(session.Low.min())
            if tgt >= entry:
                continue
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        rr = abs(tgt - entry) / risk

        # zone quality extras: age, and whether the entry sits inside a bigger opposing zone
        age_bars = len(d15) - 1 - z["idx"]
        nested = any(x["height"] > z["height"] and min(x["prox"], x["dist"]) <= entry <= max(x["prox"], x["dist"])
                     for x in opp)
        react, react_ts = reaction(d15c, entry, z["dist"], long, look=4)

        # volume at the zone: the bar that touched it, otherwise the last two bars as price approaches
        rv, basis = None, None
        if bars:
            for k in range(1, 5):
                r, ts = d15c.iloc[-k], d15c.index[-k]
                if (long and r.Low <= entry) or ((not long) and r.High >= entry):
                    rv, basis = bars.get(ts), "touch"
                    break
            if rv is None:
                vals = [bars[t] for t in d15c.index[-2:] if bars.get(t) is not None]
                if vals:
                    rv, basis = float(np.mean(vals)), "approach"

        checks = [
            z["depart"] >= 3.0,                                  # strong departure
            z["fresh"],                                          # untouched since forming
            z["height"] <= 1.0 * a,                              # tight base
            (up1h and upd) if long else (dn1h and dnd),          # higher timeframe agrees
            ((z["volx"] >= 1.5 or rvol >= 2.0 or (rv is not None and rv >= RV_STRONG))
             if kind == "stock" else z["depart"] >= 4.0),        # volume or momentum
        ]
        score = int(sum(checks))
        take = screen_ok and score >= 4 and rr >= 3
        plays.append(dict(
            tkr=tk, side="long" if long else "short", kind=kind, nd=nd,
            px=round(price, nd), adv=int(round(adv)), rvol=round(rvol, 2), atr=round(atr_pct, 2),
            spr=None if spread is None else round(spread, 3),
            prox=round(entry, nd), dist=round(z["dist"], nd), tgt=round(tgt, nd),
            rr=round(rr, 1), score=score, sc=[bool(x) for x in checks],
            screen=[bool(x) for x in screen], take=bool(take), tf="15m",
            atr15=a, vwap=None if vwap is None else round(vwap, nd),
            rv=None if rv is None else round(rv, 2), rv_basis=basis,
            htf=htf, depart=round(z["depart"], 1), age_bars=int(age_bars), nested=bool(nested),
            react=react, react_ts=react_ts,
        ))
    plays.sort(key=lambda p: (p["take"], p["score"], p["rr"]), reverse=True)
    return plays


# ---------- data ----------
def fetch(tk, with_info=True):
    import yfinance as yf
    t = yf.Ticker(tk)
    d15 = t.history(period="20d", interval="15m")
    d1h = t.history(period="1mo", interval="1h")
    dd = t.history(period="3mo", interval="1d")
    spread = None
    if with_info:
        try:
            info = t.info
            bid, ask = info.get("bid") or 0, info.get("ask") or 0
            if bid > 0 and ask > 0:
                spread = (ask - bid) / ((ask + bid) / 2) * 100
        except Exception:
            pass
    return d15.dropna(), d1h.dropna(), dd.dropna(), spread


# ================= news, events and earnings =================
NEG_RE = re.compile(
    r"\b(halt(ed|s)?|investigat\w+|probe[ds]?|subpoena\w*|lawsuit|sues?|sued|fraud\w*|bankrupt\w*|chapter 11|"
    r"delist\w*|recall\w*|downgrad\w+|plunge\w*|tumbl\w+|crash\w*|slump\w*|miss(es|ed)?|"
    r"(cuts?|lowers?|slashes?) (guidance|forecast|outlook)|warning|resigns?|steps? down|"
    r"(public |stock |share )?offering|dilut\w+|short[- ]seller|SEC charges|data breach|cyberattack|"
    r"complete response letter|rejects?|tariffs?)\b", re.I)
POS_RE = re.compile(
    r"\b(upgrad\w+|surg\w+|soar\w*|jump\w*|rall(y|ies|ied)|beats?|"
    r"raises? (guidance|forecast|outlook)|record (high|revenue|earnings)|approv\w+|acquir\w+|"
    r"acquisition|buyout|takeover|partnership|wins?|breakthrough)\b", re.I)
MAJOR_RE = re.compile(
    r"(\bcpi\b|consumer price|\bppi\b|producer price|non-?farm|payrolls|fomc|fed funds|"
    r"interest rate (decision|announcement)|rate decision|\bgdp\b|retail sales|\bpce\b|"
    r"unemployment rate|jobless claims)", re.I)
# Fed decision days, announced 2:00 pm New York time (federalreserve.gov calendar).
FOMC_DECISIONS = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
                  "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]
COUNTRY_CCY = {"US": "USD", "EU": "EUR", "EMU": "EUR", "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR",
               "GB": "GBP", "UK": "GBP", "JP": "JPY", "AU": "AUD", "NZ": "NZD", "CA": "CAD",
               "CH": "CHF", "CN": "CNY", "HK": "HKD"}


def _get_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "zone-ticket"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_manual(text):
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})\s*(.*)", line)
        if not m:
            continue
        try:
            t = datetime.strptime(f"{m[1]} {int(m[2]):02d}:{m[3]}", "%Y-%m-%d %H:%M").replace(tzinfo=SYD)
        except ValueError:
            continue
        out.append(dict(t=t.astimezone(timezone.utc), name=(m[4].strip() or "Your blackout"), ccy="ALL"))
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def fmp_events(key):
    """High impact economic releases from Financial Modeling Prep. Raises if nothing could be read."""
    today = datetime.now(timezone.utc).date()
    a, b = today - timedelta(days=1), today + timedelta(days=3)
    problem = "no response"
    for base in ("https://financialmodelingprep.com/stable/economic-calendar",
                 "https://financialmodelingprep.com/api/v3/economic_calendar"):
        try:
            data = _get_json(f"{base}?from={a}&to={b}&apikey={urllib.parse.quote(key)}")
        except urllib.error.HTTPError as e:
            problem = f"HTTP {e.code}"
            continue
        except Exception:
            problem = "network error"
            continue
        if not isinstance(data, list):
            problem = "the key was refused or the plan does not include the calendar"
            continue
        out = []
        for e in data:
            name = str(e.get("event", ""))
            if str(e.get("impact", "")).lower() != "high" and not MAJOR_RE.search(name):
                continue
            try:
                t = pd.Timestamp(e["date"])
                t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            except Exception:
                continue
            ccy = (e.get("currency") or "").upper() or COUNTRY_CCY.get(str(e.get("country", "")).upper(), "")
            if ccy:
                out.append(dict(t=t.to_pydatetime(), name=name, ccy=ccy))
        return out
    raise RuntimeError(problem)


def fmp_key():
    k = str(st.session_state.get("fmp_key_input", "")).strip()
    if k:
        return k
    try:
        return str(st.secrets.get("FMP_KEY", "")).strip()
    except Exception:
        return ""


def build_events(manual_text):
    ev = []
    for d in FOMC_DECISIONS:
        y, m, dd_ = (int(x) for x in d.split("-"))
        ev.append(dict(t=datetime(y, m, dd_, 14, 0, tzinfo=NY).astimezone(timezone.utc),
                       name="FOMC rate decision", ccy="USD"))
    ev += parse_manual(manual_text)
    status = "Event blackouts: built-in FOMC dates"
    if manual_text.strip():
        status += " and your own times"
    key = fmp_key()
    if key:
        try:
            more = fmp_events(key)
            ev += more
            status += f", plus {len(more)} high-impact releases from FMP"
        except Exception as e:
            status += f". FMP calendar not loaded ({e})"
    else:
        status += ". Add an FMP key in Filters to include CPI, PPI and jobs data"
    now = datetime.now(timezone.utc)
    return [e for e in ev if abs((e["t"] - now).total_seconds()) < 2 * 86400], status


@st.cache_data(ttl=300, show_spinner=False)
def index_state():
    """SPY and QQQ 15 minute state. None when neither could be read."""
    import yfinance as yf
    res = {}
    for sym in ("SPY", "QQQ"):
        try:
            f = index_flags(yf.Ticker(sym).history(period="5d", interval="15m").dropna())
            if f:
                res[sym] = f
        except Exception:
            pass
    if not res:
        return None
    return dict(open=any(v["open"] for v in res.values()),
                down=[k for k, v in res.items() if v["open"] and v["down"]],
                up=[k for k, v in res.items() if v["open"] and v["up"]])


def fetch_news(tk):
    import yfinance as yf
    out = []
    for it in yf.Ticker(tk).news or []:
        c = it.get("content") if isinstance(it.get("content"), dict) else it
        title = c.get("title") or it.get("title")
        ts = c.get("pubDate") or c.get("displayTime") or it.get("providerPublishTime")
        if not title or ts is None:
            continue
        try:
            t = pd.Timestamp(ts, unit="s", tz="UTC") if isinstance(ts, (int, float)) else pd.Timestamp(ts)
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        except Exception:
            continue
        out.append((t.to_pydatetime(), str(title)))
    return out


def fetch_earnings(tk):
    import yfinance as yf
    cal = yf.Ticker(tk).calendar
    vals = []
    if isinstance(cal, dict):
        vals = cal.get("Earnings Date") or []
    elif cal is not None and hasattr(cal, "loc"):
        try:
            vals = list(cal.loc["Earnings Date"])
        except Exception:
            vals = []
    out = []
    for v in vals:
        try:
            out.append(pd.Timestamp(v).date())
        except Exception:
            pass
    return out


# ================= app =================
DEFAULT_WATCH = ("AAPL MSFT NVDA AMD TSLA META AMZN GOOGL NFLX AVGO PLTR SMCI COIN MSTR HOOD SOFI "
                 "RKLB ASTS LUNR IONQ RGTI AVAV KTOS MRNA CRSP")

# name -> (screen kind, default leverage limit, [(yahoo symbol, label)])
MARKETS = {
    "Forex": ("forex", 30.0, [
        ("EURUSD=X", "EUR/USD"), ("GBPUSD=X", "GBP/USD"), ("USDJPY=X", "USD/JPY"),
        ("AUDUSD=X", "AUD/USD"), ("NZDUSD=X", "NZD/USD"), ("USDCAD=X", "USD/CAD"),
        ("USDCHF=X", "USD/CHF"), ("EURJPY=X", "EUR/JPY"), ("GBPJPY=X", "GBP/JPY"),
        ("AUDJPY=X", "AUD/JPY"), ("EURGBP=X", "EUR/GBP"), ("EURAUD=X", "EUR/AUD")]),
    "Commodities": ("commodity", 10.0, [
        ("GC=F", "Gold"), ("SI=F", "Silver"), ("PL=F", "Platinum"), ("HG=F", "Copper"),
        ("CL=F", "WTI Oil"), ("BZ=F", "Brent Oil"), ("NG=F", "Natural Gas"),
        ("ZW=F", "Wheat"), ("ZC=F", "Corn"), ("KC=F", "Coffee")]),
    "Indices": ("index", 20.0, [
        ("ES=F", "S&P 500"), ("NQ=F", "Nasdaq 100"), ("YM=F", "Dow Jones"), ("RTY=F", "Russell 2000"),
        ("^GDAXI", "Germany 40"), ("^FTSE", "UK 100"), ("^N225", "Japan 225"),
        ("^AXJO", "Australia 200"), ("^HSI", "Hong Kong 50")]),
    "Crypto": ("crypto", 2.0, [
        ("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"), ("SOL-USD", "Solana"),
        ("XRP-USD", "XRP"), ("DOGE-USD", "Dogecoin"), ("ADA-USD", "Cardano")]),
    "ETFs": ("index", 5.0, [
        ("SPY", "SPY"), ("QQQ", "QQQ"), ("IWM", "IWM"), ("DIA", "DIA"), ("TQQQ", "TQQQ"),
        ("SQQQ", "SQQQ"), ("SOXL", "SOXL"), ("GLD", "GLD"), ("SLV", "SLV"), ("TLT", "TLT")]),
    "Stocks": ("stock", 5.0, []),
}
US_MARKETS = ("Stocks", "ETFs")
LOOKUP = {sym: (label, MARKETS[m][0], m) for m in MARKETS for sym, label in MARKETS[m][2]}
INDEX_CCY = {"^GDAXI": "EUR", "^FTSE": "GBP", "^N225": "JPY", "^AXJO": "AUD", "^HSI": "HKD"}

# ---------------- saved settings (kept in the page address) ----------------
FILTERS = {"f_events": True, "ev_min": 15.0, "manual_ev": "", "f_news": True, "news_hrs": 3.0,
           "f_earn": True, "f_rvol": True, "rv_weak": 0.8, "f_vwap": True, "vwap_tol": 1.0, "f_htf": True,
           "f_index": True, "max_spread": 0.05,
           "f_depart": True, "zq_depart": 4.0, "f_age": True, "zq_age": 48.0, "f_nest": True,
           "f_react": True}
DEFAULTS = {"acct": "AUD", "equity": 10000.0, "risk_pct": 2.0, "sel": ["Forex", "Commodities", "Indices"],
            "watch": DEFAULT_WATCH, "extras": "", "show_close": False, "auto": False}
DEFAULTS.update(FILTERS)
for _m in MARKETS:
    DEFAULTS[f"lev_{_m}"] = MARKETS[_m][1]


def from_q(raw, default):
    if isinstance(default, bool):
        return raw == "1"
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return [x for x in raw.split(",") if x in MARKETS]
    if default in ("AUD", "USD"):
        return raw if raw in ("AUD", "USD") else default
    return raw


def to_q(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, list):
        return ",".join(v)
    return str(v)


if "_init" not in st.session_state:
    q = st.query_params
    for k, d in DEFAULTS.items():
        val = d
        if k in q:
            try:
                val = from_q(q[k], d)
            except Exception:
                val = d
        st.session_state[k] = val
    st.session_state["_init"] = True


def reset_settings():
    for k, d in DEFAULTS.items():
        st.session_state[k] = d


# ---------------- helpers ----------------
def md(s):
    """Streamlit treats a pair of dollar signs as maths, so escape them."""
    return str(s).replace("$", "\\$")


def infer(sym):
    if sym in LOOKUP:
        return LOOKUP[sym]
    if sym.endswith("=X"):
        return sym, "forex", "Forex"
    if sym.endswith("=F"):
        return sym, "commodity", "Commodities"
    if sym.endswith("-USD"):
        return sym, "crypto", "Crypto"
    if sym.startswith("^"):
        return sym, "index", "Indices"
    return sym, "stock", "Stocks"


def fmt(v, nd):
    return f"{v:,.{nd}f}"


@st.cache_data(ttl=600, show_spinner=False)
def usd_rate(ccy):
    """USD value of one unit of ccy, or None if it cannot be read."""
    if ccy == "USD":
        return 1.0
    for sym, inv in ((f"{ccy}USD=X", False), (f"USD{ccy}=X", True)):
        try:
            import yfinance as yf
            c = yf.Ticker(sym).history(period="5d")["Close"].dropna()
            if len(c):
                return 1 / float(c.iloc[-1]) if inv else float(c.iloc[-1])
        except Exception:
            pass
    return None


def acct_rate(ccy, acct):
    """Value of one unit of ccy in the account currency, or None."""
    a, b = usd_rate(ccy), usd_rate(acct)
    return None if a is None or b is None else a / b


def quote_ccy(p):
    sym = p["tkr"]
    if p.get("kind") == "forex" and len(sym) >= 6:
        return sym[3:6]
    return INDEX_CCY.get(sym, "USD")


def size(p, equity, risk_pct, lev_cap, acct="AUD", buf_pct=10.0):
    long = p["side"] == "long"
    height = abs(p["prox"] - p["dist"])
    stop = p["dist"] - height * buf_pct / 100 if long else p["dist"] + height * buf_pct / 100
    ccy = quote_ccy(p)
    f = acct_rate(ccy, acct)
    approx = f is None
    f = 1.0 if approx else f
    per_unit = abs(p["prox"] - stop) * f
    units = math.floor(equity * risk_pct / 100 / per_unit) if per_unit > 0 else 0
    notional = units * p["prox"] * f
    margin = notional / lev_cap if lev_cap > 0 else float("inf")
    return dict(stop=stop, units=units, notional=notional, margin=margin, approx=approx, ccy=ccy,
                acct=acct, cur="A$" if acct == "AUD" else "$",
                margin_pct=margin / equity * 100, lev=notional / equity,
                loss=units * per_unit, gain=units * abs(p["tgt"] - p["prox"]) * f)


def reasons(p, s, lev_cap):
    """Reasons a play fails on its own numbers."""
    out = []
    if not all(p["screen"]):
        out.append("fails the screen")
    if p["score"] < 4:
        out.append(f"zone score {p['score']}/5, needs 4")
    if p["rr"] < 3:
        out.append(f"reward to risk {p['rr']:.1f}, needs 3")
    if s["units"] < 1:
        out.append("risk amount too small for one unit")
    if s["lev"] > lev_cap:
        out.append(f"needs {s['lev']:.1f}x leverage, limit {lev_cap:.0f}x")
    if s["margin_pct"] > 50:
        out.append(f"margin is {s['margin_pct']:.0f}% of equity, keep under 50%")
    return out


def ccy_applies(ccy, p):
    if ccy in ("ALL", "USD"):
        return True
    sym = p["tkr"]
    if p.get("kind") == "forex" and len(sym) >= 6:
        return ccy in (sym[:3], sym[3:6])
    return INDEX_CCY.get(sym) == ccy


def prev_bday(d):
    return (pd.Timestamp(d) - pd.offsets.BDay(1)).date()


def filter_reasons(p, S, ctx, now=None):
    """Reasons a play is held back by the context filters (events, news, volume, VWAP, structure, index)."""
    now = now or datetime.now(timezone.utc)
    out = []
    long = p["side"] == "long"
    if S["f_events"]:
        for e in ctx.get("events", []):
            if not ccy_applies(e["ccy"], p):
                continue
            mins = (e["t"] - now).total_seconds() / 60
            if -S["ev_min"] <= mins <= S["ev_min"]:
                out.append(f"news blackout: {e['name']} {e['t'].astimezone(SYD):%H:%M} Sydney ({mins:+.0f} min)")
                break
    if p.get("us"):
        if p.get("spr") is not None and p["spr"] > S["max_spread"]:
            out.append(f"spread {p['spr']:.3f}% is over your {S['max_spread']:.3f}% limit")
        if S["f_earn"]:
            today = now.astimezone(NY).date()
            for d in p.get("earn") or []:
                if d == today:
                    out.append(f"earnings today ({d:%d %b})")
                    break
                if d == prev_bday(today):
                    out.append(f"earnings {d:%d %b}, gap risk the day after")
                    break
        if S["f_news"]:
            bad = NEG_RE if long else POS_RE
            for t, title in sorted(p.get("news") or [], reverse=True):
                age = (now - t).total_seconds() / 3600
                if 0 <= age <= S["news_hrs"] and bad.search(title):
                    out.append(f"{'negative' if long else 'positive'} headline {age:.1f}h ago: \"{title[:70]}\"")
                    break
        if S["f_rvol"] and p.get("rv") is not None and p["rv"] < S["rv_weak"]:
            out.append(f"weak volume at the zone ({p['rv']:.1f}x normal for this time)")
        idx = ctx.get("index")
        if S["f_index"] and idx and idx["open"]:
            if long and idx["down"]:
                out.append(f"{'/'.join(idx['down'])} selling off, wait for it to settle")
            if (not long) and idx["up"]:
                out.append(f"{'/'.join(idx['up'])} spiking, wait for it to settle")
    if S["f_vwap"] and p.get("vwap") is not None and p.get("atr15"):
        gap = (p["prox"] - p["vwap"]) / p["atr15"]
        if long and gap < -S["vwap_tol"]:
            out.append(f"demand zone is {abs(gap):.1f} ATR below session VWAP")
        if (not long) and gap > S["vwap_tol"]:
            out.append(f"supply zone is {gap:.1f} ATR above session VWAP")
    if S["f_htf"] and p.get("htf") == ("down" if long else "up"):
        out.append("counter-trend: 1h makes " + ("lower highs and lower lows" if long else "higher highs and higher lows"))
    if S["f_depart"] and p.get("depart") is not None and p["depart"] < S["zq_depart"]:
        out.append(f"weak departure: the zone only moved {p['depart']:.1f} ATR away (needs {S['zq_depart']:.1f})")
    if S["f_age"] and p.get("age_bars") is not None and p["age_bars"] > S["zq_age"] * 4:
        out.append(f"old zone: formed about {p['age_bars'] / 4:.0f}h of trading ago (limit {S['zq_age']:.0f}h)")
    if S["f_nest"] and p.get("nested"):
        out.append("the zone sits inside a bigger " + ("supply" if long else "demand") + " zone")
    if S["f_react"] and p.get("react") is not None:
        if p["react"] == "waiting":
            out.append("price has not reached the zone yet, wait for a rejection candle")
        elif p["react"] == "touched":
            out.append("price tagged the zone but has not printed a rejection candle")
    return out


def context_lines(p, S):
    out = []
    if p.get("vwap") is not None and p.get("atr15"):
        gap = (p["prox"] - p["vwap"]) / p["atr15"]
        out.append(f"Session VWAP {fmt(p['vwap'], p['nd'])}, zone is {gap:+.1f} ATR from it")
    if p.get("rv") is not None:
        out.append(f"Volume {'at the touch' if p.get('rv_basis') == 'touch' else 'approaching the zone'}: "
                   f"{p['rv']:.1f}x normal for this time of day")
    if p.get("depart") is not None:
        extra = " · inside a bigger opposing zone" if p.get("nested") else ""
        out.append(f"Zone left by {p['depart']:.1f} ATR, formed about {p.get('age_bars', 0) / 4:.0f}h of trading ago{extra}")
    if p.get("react"):
        out.append({"rejected": "Rejection candle printed at the zone",
                    "touched": "Price tagged the zone, no rejection candle yet",
                    "waiting": "Price has not reached the zone yet"}[p["react"]])
    if p.get("htf"):
        out.append("1h structure: " + {"up": "higher highs and higher lows", "down": "lower highs and lower lows",
                                       "range": "sideways"}[p["htf"]])
    if p.get("us"):
        out.append("Spread: " + ("unknown" if p.get("spr") is None else f"{p['spr']:.3f}%"))
        if p.get("news_ok") is False:
            out.append("Headlines could not be read, so the headline check was skipped for this one")
        now = datetime.now(timezone.utc)
        recent = [(t, ti) for t, ti in (p.get("news") or []) if 0 <= (now - t).total_seconds() < 86400]
        if recent:
            t, ti = max(recent)
            out.append(f"Latest headline ({(now - t).total_seconds() / 3600:.1f}h ago): \"{ti[:100]}\"")
        soon = [d for d in (p.get("earn") or []) if 0 <= (d - now.astimezone(NY).date()).days <= 3]
        if soon:
            out.append(f"Earnings {soon[0]:%d %b}")
    return out


def play_body(p, price, s, why, S):
    long = p["side"] == "long"
    nd = p.get("nd", 2)
    dist = abs(price - p["prox"]) / price * 100
    st.markdown(f"### {'BUY' if long else 'SELL SHORT'} at {fmt(p['prox'], nd)}")
    st.caption(f"{'Demand' if long else 'Supply'} zone, {dist:.1f}% from price. Use a limit order.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stop", fmt(s["stop"], nd))
    c2.metric("Sell target" if long else "Cover target", fmt(p["tgt"], nd))
    c3.metric("Reward : risk", f"{p['rr']:.1f} : 1")
    c4, c5, c6 = st.columns(3)
    c4.metric("Units", f"{s['units']:,}")
    cur = s["cur"]
    c5.metric("Margin", f"{cur}{s['margin']:,.0f}")
    c6.metric("Loss at stop", f"{cur}{s['loss']:,.0f}")
    extra = f" · about {s['units'] / 100000:.2f} standard lots" if p.get("kind") == "forex" else ""
    st.write(md(f"Zone score {p['score']}/5 · gain at target {cur}{s['gain']:,.0f} · leverage used {s['lev']:.1f}x{extra}"))
    if s["approx"]:
        st.caption(f"Could not read the {s['ccy']} to {s['acct']} rate, so size and margin are approximate.")
    lines = context_lines(p, S)
    if lines:
        st.caption(md("\n\n".join(lines)))
    if not why:
        st.success("TAKE: every check passes.")
    else:
        st.warning(md("SKIP for now: " + "; ".join(why) + "."))


# ---------------- scanning ----------------
def get_ctx(S):
    events, status = build_events(S["manual_ev"])
    try:
        idx = index_state()
    except Exception:
        idx = None
    return dict(events=events, status=status, index=idx)


def enrich(p, us, stock, news, earn, news_ok=True):
    p["us"] = us
    p["news"] = news if us else []
    p["news_ok"] = news_ok
    p["earn"] = earn if stock else []


def do_scan(selected, watchlist, extras, S, max_dist=8.0):
    jobs, seen = [], set()

    def add(sym, label, m, kind):
        if sym and sym not in seen:
            seen.add(sym)
            jobs.append((sym, label, m, kind))

    for m in selected:
        kind = MARKETS[m][0]
        if m == "Stocks":
            for sym in watchlist.replace(",", " ").upper().split():
                add(sym, sym, m, kind)
        else:
            for sym, label in MARKETS[m][2]:
                add(sym, label, m, kind)
    for sym in extras.replace(",", " ").upper().split():
        label, kind, m = infer(sym)
        add(sym, label, m, kind)

    ctx = get_ctx(S)

    def work(job):
        sym, label, m, kind = job
        us = m in US_MARKETS
        try:
            d15, d1h, dd, spr = fetch(sym, with_info=us)
            if len(d15) < 60 or len(d1h) < 10 or len(dd) < 25:
                return sym, [], "not enough data"
            ps = build_plays(sym, d15, d1h, dd, spr, max_dist, kind=kind)[:2]
            news, earn, news_ok = [], [], True
            if us and ps and ps[0]["score"] >= 3:
                try:
                    news = fetch_news(sym)
                except Exception:
                    news_ok = False
                if m == "Stocks":
                    try:
                        earn = fetch_earnings(sym)
                    except Exception:
                        pass
            for p in ps:
                p["label"], p["market"] = label, m
                enrich(p, us, m == "Stocks", news, earn, news_ok)
            return sym, ps, None
        except Exception as e:
            return sym, [], str(e)[:80]

    plays, errors = [], []
    bar = st.progress(0.0, text=f"Scanning {len(jobs)} instruments...")
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            sym, ps, err = fu.result()
            plays += ps
            if err:
                errors.append(f"{sym}: {err}")
            bar.progress(i / len(jobs), text=f"Scanned {i} of {len(jobs)}")
    bar.empty()
    st.session_state["scan"] = dict(ts=time.time(), plays=plays, errors=errors, n=len(jobs), ctx=ctx)


def render_banner(ctx, S):
    now = datetime.now(timezone.utc)
    idx = ctx.get("index")
    if idx is None:
        st.caption("SPY and QQQ could not be read, so the index check is skipped.")
    elif not idx["open"]:
        st.caption("US market is closed. Volume, VWAP and index checks use the last session.")
    elif idx["down"]:
        st.warning(f"{'/'.join(idx['down'])} selling off. Buy signals on US stocks are held back.")
    elif idx["up"]:
        st.warning(f"{'/'.join(idx['up'])} spiking. Short signals on US stocks are held back.")
    m = S["ev_min"]
    if S["f_events"]:
        for e in sorted(ctx.get("events", []), key=lambda x: x["t"]):
            mins = (e["t"] - now).total_seconds() / 60
            when = f"{e['name']} at {e['t'].astimezone(SYD):%a %H:%M} Sydney"
            if -m <= mins <= m:
                st.error(f"Blackout active: {when} ({mins:+.0f} min). Signals for affected markets are suppressed.")
            elif m < mins <= 180:
                st.warning(f"Coming up: {when}, in {int(mins // 60)}h {int(mins % 60)}m.")
    st.caption(ctx["status"])


def render_scan(equity, risk_pct, levs, show_close, acct, S):
    s = st.session_state.get("scan")
    if not s:
        st.info("Pick your markets and tap Scan now.")
        return
    ctx = s["ctx"]
    when = datetime.fromtimestamp(s["ts"], SYD).strftime("%a %H:%M")
    st.caption(f"Last scan {when} Sydney time · {s['n']} instruments · prices delayed")
    render_banner(ctx, S)
    skipped = sorted({p["tkr"] for p in s["plays"] if p.get("news_ok") is False})
    if skipped and S["f_news"]:
        st.warning("Headlines could not be read for " + ", ".join(skipped[:8]) +
                   (" and more" if len(skipped) > 8 else "") + ". The headline check was skipped for them.")
    rows = []
    for p in s["plays"]:
        lev = levs.get(p["market"], 5.0)
        sz = size(p, equity, risk_pct, lev, acct)
        rows.append((p, sz, reasons(p, sz, lev), filter_reasons(p, S, ctx)))
    rows.sort(key=lambda r: (bool(r[2] or r[3]), -r[0]["score"], -r[0]["rr"]))
    takes = [r for r in rows if not r[2] and not r[3]]
    blocked = [r for r in rows if not r[2] and r[3]]
    close = [r for r in rows if r[2] and r[0]["score"] >= 3 and r[0]["rr"] >= 2]

    def show(items, tag):
        for p, sz, tech, flt in items:
            long = p["side"] == "long"
            title = (f"{tag} · {'BUY' if long else 'SELL SHORT'} {p['label']} @ {fmt(p['prox'], p['nd'])}"
                     f" · {p['rr']:.1f}R · {p['score']}/5")
            if tag == "BLOCKED":
                title += f" · {flt[0][:48]}"
            with st.expander(md(title)):
                play_body(p, p["px"], sz, tech + flt, S)

    st.subheader(f"Take ({len(takes)})")
    if takes:
        show(takes, "TAKE")
    else:
        st.info("Nothing passes every check right now. That is normal. Waiting is part of the strategy.")
    if blocked:
        st.subheader(f"Blocked by filters ({len(blocked)})")
        st.caption("These pass the zone rules but a filter says wait. Open one to see why.")
        show(blocked, "BLOCKED")
    if show_close:
        st.subheader(f"Close calls ({len(close)})")
        show(close, "WATCH")
    if s["errors"]:
        with st.expander(f"{len(s['errors'])} instruments could not load"):
            st.write("\n".join(s["errors"]))


# ---------------- page ----------------
st.title("Zone Ticket")
st.caption("Supply and demand day-trade levels. Scan the markets or look up one ticker.")

with st.expander("Settings"):
    st.selectbox("Account currency", ["AUD", "USD"], key="acct")
    st.number_input("Account equity", min_value=100.0, step=500.0, key="equity")
    st.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, step=0.25, key="risk_pct")
    st.caption("Leverage limits. Defaults are typical Australian retail limits. Use your broker's real numbers.")
    for m in MARKETS:
        st.number_input(f"{m} max leverage (x)", min_value=1.0, max_value=500.0, step=1.0, key=f"lev_{m}")
    st.text_area("Stock watchlist (tickers separated by spaces)", key="watch")
    st.text_area("Extra symbols to scan every time", key="extras",
                 help="Any Yahoo Finance symbol, for example AUDNZD=X, NG=F, BTC-USD or a stock like UBER.")
    st.button("Reset to defaults", on_click=reset_settings)
    st.caption("Settings save in this page's web address. After you change them, bookmark the page "
               "or add it to your home screen again so it opens with your settings.")

with st.expander("Filters"):
    st.caption("Each filter can hold a play back. Turn one off to see what it was hiding.")
    st.toggle("Event blackouts (FOMC, CPI, jobs, your own times)", key="f_events")
    st.number_input("Blackout window, minutes either side", min_value=0.0, max_value=120.0, step=5.0, key="ev_min")
    st.text_area("Your own blackout times, Sydney time, one per line", key="manual_ev",
                 placeholder="2026-10-22 23:30 US CPI",
                 help="Format: YYYY-MM-DD HH:MM then a name. Affects every market.")
    st.text_input("FMP API key for CPI, PPI and jobs data (this visit only)", type="password", key="fmp_key_input",
                 help="Not saved. To keep it, add FMP_KEY in the app's Secrets settings on Streamlit.")
    st.toggle("Headline check for US stocks and ETFs", key="f_news")
    st.number_input("Headline window, hours", min_value=0.5, max_value=24.0, step=0.5, key="news_hrs")
    st.toggle("Block earnings day and the day after", key="f_earn")
    st.toggle("Block weak volume at the zone", key="f_rvol")
    st.number_input("Weak volume below (x normal for the time of day)", min_value=0.1, max_value=2.0,
                    step=0.1, key="rv_weak")
    st.number_input("Max bid-ask spread for US stocks and ETFs (%)", min_value=0.0, max_value=2.0, step=0.01,
                    format="%.3f", key="max_spread")
    st.toggle("Session VWAP side", key="f_vwap")
    st.number_input("How far on the wrong side of VWAP is allowed (15m ATR)", min_value=0.0, max_value=5.0,
                    step=0.25, key="vwap_tol")
    st.toggle("Block counter-trend plays (1h structure)", key="f_htf")
    st.toggle("Hold plays while SPY or QQQ is selling off or spiking", key="f_index")
    st.toggle("Zone must have left strongly", key="f_depart")
    st.number_input("Minimum departure (ATR)", min_value=2.0, max_value=10.0, step=0.5, key="zq_depart")
    st.toggle("Block old zones", key="f_age")
    st.number_input("Oldest zone allowed (hours of trading)", min_value=4.0, max_value=120.0, step=4.0, key="zq_age")
    st.toggle("Block zones inside a bigger opposing zone", key="f_nest")
    st.toggle("Only take after a rejection candle at the zone", key="f_react",
              help="Price must have tagged the zone and closed back out with a long wick or engulfing candle. "
                   "With this on, plays waiting for price to arrive show under Blocked until it happens.")

S = {k: st.session_state[k] for k in FILTERS}
acct = st.session_state["acct"]
equity = st.session_state["equity"]
risk_pct = st.session_state["risk_pct"]
levs = {m: st.session_state[f"lev_{m}"] for m in MARKETS}

tab_scan, tab_one = st.tabs(["Scan markets", "One ticker"])

with tab_scan:
    selected = st.multiselect("Markets", list(MARKETS), key="sel")
    watchlist, extras = st.session_state["watch"], st.session_state["extras"]
    show_close = st.toggle("Show close calls", key="show_close")
    auto = st.toggle("Rescan every hour while this page is open", key="auto")
    can_scan = bool(selected) or bool(extras.strip())
    if st.button("Scan now", type="primary", use_container_width=True, disabled=not can_scan):
        do_scan(selected, watchlist, extras, S)

    @st.fragment(run_every=3600 if auto else None)
    def results():
        s = st.session_state.get("scan")
        if auto and s and time.time() - s["ts"] > 3300 and can_scan:
            do_scan(selected, watchlist, extras, S)
        render_scan(equity, risk_pct, levs, show_close, acct, S)

    results()
    if auto:
        st.caption("Auto scan only runs while this page is open on your phone. It does not run in the background.")


@st.cache_data(ttl=300, show_spinner=False)
def load(tk, us):
    return fetch(tk, with_info=us)


def snapshot(d15, d1h, dd, spread, kind, S):
    price = float(d15.Close.iloc[-1])
    atr_pct = float(atr_series(dd).iloc[-1] / dd.Close.iloc[-1] * 100)
    lo, hi = ATR_RANGE[kind]
    nd = decimals(price)
    if kind != "stock":
        return price, nd, [(f"Daily range {lo}-{hi}%", f"{atr_pct:.2f}%", lo <= atr_pct <= hi)]
    adv = float(dd.Volume.iloc[-21:-1].mean())
    rvd = float(dd.Volume.iloc[-1] / adv) if adv else 0.0
    _, cum = intraday_rvol(completed(d15))
    rv = cum if cum is not None else rvd
    rv_label = "Relative volume 1.5x+ (same time of day)" if cum is not None else "Relative volume 1.5x+ (daily)"
    return price, nd, [
        ("Price above $5", f"${price:,.2f}", price > 5),
        ("Avg volume 1M+", f"{adv:,.0f}", adv >= 1e6),
        (rv_label, f"{rv:.1f}x", rv >= 1.5),
        ("Daily range 2-8%", f"{atr_pct:.1f}%", lo <= atr_pct <= hi),
        (f"Spread {S['max_spread']:.3f}% or less", "n/a" if spread is None else f"{spread:.3f}%",
         None if spread is None else spread <= S["max_spread"]),
    ]


with tab_one:
    tk = st.text_input("Ticker", placeholder="NVDA, EURUSD=X, GC=F, BTC-USD").strip().upper()
    if tk:
        label, kind, market = infer(tk)
        us = market in US_MARKETS
        with st.spinner(f"Reading {tk}..."):
            try:
                d15, d1h, dd, spread = load(tk, us)
                ok = len(d15) >= 60 and len(d1h) >= 10 and len(dd) >= 25
            except Exception:
                ok = False
        if not ok:
            st.error(f"Could not get enough data for {tk}. Check the ticker.")
        else:
            ctx = get_ctx(S)
            price, nd, rows = snapshot(d15, d1h, dd, spread, kind, S)
            st.subheader(f"{label}  {fmt(price, nd)}")
            st.caption("Prices are delayed. Confirm on your chart before ordering.")
            render_banner(ctx, S)
            with st.expander("Screen", expanded=any(r[2] is False for r in rows)):
                for lab, val, res in rows:
                    mark = ":gray[n/a]" if res is None else (":green[PASS]" if res else ":red[FAIL]")
                    st.write(md(f"{mark}  {lab}: **{val}**"))
            plays = build_plays(tk, d15, d1h, dd, spread, max_dist=15, kind=kind)
            news, earn, news_ok = [], [], True
            if us and plays:
                try:
                    news = fetch_news(tk)
                except Exception:
                    news_ok = False
                if market == "Stocks":
                    try:
                        earn = fetch_earnings(tk)
                    except Exception:
                        pass
            for head, side in (("Buy levels", "long"), ("Sell short levels", "short")):
                st.markdown(f"## {head}")
                items = [p for p in plays if p["side"] == side][:2]
                if not items:
                    st.info("No zone with a usable target right now.")
                for p in items:
                    enrich(p, us, market == "Stocks", news, earn, news_ok)
                    lev = levs.get(market, 5.0)
                    sz = size(p, equity, risk_pct, lev, acct)
                    with st.container(border=True):
                        play_body(p, price, sz, reasons(p, sz, lev) + filter_reasons(p, S, ctx), S)

st.divider()
st.caption("Leveraged products lose money for most retail traders. Test on a demo account first. "
           "This is a calculator, not financial advice.")

# Save the current settings into the page address.
try:
    st.query_params.from_dict({k: to_q(st.session_state[k]) for k in DEFAULTS})
except Exception:
    pass
