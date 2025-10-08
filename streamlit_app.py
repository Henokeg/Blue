import time
import math
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

# History providers
import yfinance as yf
from pandas_datareader import data as pdr


# -----------------------------
# Helpers: indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(period).mean() / down.rolling(period).mean()
    out = 100 - (100 / (1 + rs))
    return out

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # expects columns: High, Low, Close
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# -----------------------------
# Finviz snapshot (name/price/ratios)
# -----------------------------
def fetch_finviz_snapshot(ticker: str) -> tuple[dict, str | None]:
    """
    Scrapes Finviz quote page snapshot table.
    Returns (data_dict, error_message)
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finviz.com/"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return {}, f"Finviz returned HTTP {resp.status_code}."
        soup = BeautifulSoup(resp.text, "lxml")

        # Title (e.g., Company name / ticker)
        title_el = soup.select_one(".fullview-title")
        title = title_el.get_text(strip=True) if title_el else ticker.upper()

        # Snapshot table
        snap = {}
        for row in soup.select("table.snapshot-table2 tr"):
            cols = [c.get_text(strip=True) for c in row.select("td")]
            # cells come in pairs: Key | Val | Key | Val | ...
            for i in range(0, len(cols), 2):
                if i + 1 < len(cols):
                    k, v = cols[i], cols[i+1]
                    if k: snap[k] = v
        return {"title": title, **snap}, None
    except Exception as e:
        return {}, f"Finviz error: {e}"


# -----------------------------
# Price history (Yahoo ➜ Stooq fallback)
# -----------------------------
def fetch_history(ticker: str, days: int = 365, interval: str = "1d") -> pd.DataFrame | None:
    """
    Try yfinance first; if empty, fall back to Stooq (pandas-datareader).
    Returns DataFrame with columns [Open, High, Low, Close, Volume]
    """
    # --- Yahoo
    try:
        df = yf.download(
            tickers=ticker,
            period=f"{days}d",
            interval=interval,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            out.dropna(how="any", inplace=True)
            if not out.empty:
                return out
    except Exception:
        pass

    # --- Stooq fallback
    try:
        df = pdr.DataReader(ticker, "stooq")
        df = df.sort_index()
        if days:
            df = df.tail(days)
        out = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].copy()
        out.dropna(how="any", inplace=True)
        if not out.empty:
            return out
    except Exception:
        pass

    return None


# -----------------------------
# Signals / levels from history
# -----------------------------
def generate_signal(df: pd.DataFrame,
                    short_ema: int = 50,
                    long_ema: int = 200,
                    rsi_period: int = 14,
                    rsi_buy: float = 40,
                    rsi_sell: float = 70,
                    atr_mult: float = 1.5) -> dict:
    """
    Compute EMA crossover + RSI filter; derive stop & targets from ATR.
    """
    close = df["Close"]
    need = max(short_ema, long_ema, rsi_period) + 5
    if len(close) < need:
        return {"error": f"Not enough bars ({len(close)}) for chosen periods (need ≥ {need})."}

    se = ema(close, short_ema)
    le = ema(close, long_ema)
    r = rsi(close, rsi_period)
    a = atr(df, 14)

    # crossovers
    state = "neutral"
    if se.iloc[-1] > le.iloc[-1] and se.iloc[-2] <= le.iloc[-2]:
        state = "bullish_crossover"
    elif se.iloc[-1] < le.iloc[-1] and se.iloc[-2] >= le.iloc[-2]:
        state = "bearish_crossover"

    price = float(close.iloc[-1])
    last_rsi = float(r.iloc[-1])
    last_atr = float(a.iloc[-1]) if not math.isnan(a.iloc[-1]) else float("nan")

    if state == "bullish_crossover" and last_rsi <= rsi_buy:
        signal = "BUY"
    elif state == "bearish_crossover" and last_rsi >= rsi_sell:
        signal = "SELL"
    else:
        if last_rsi < 30:
            signal = "BUY (RSI oversold)"
        elif last_rsi > 75:
            signal = "SELL (RSI overbought)"
        else:
            signal = "HOLD"

    if not math.isnan(last_atr):
        if signal.startswith("BUY"):
            stop = round(price - atr_mult * last_atr, 2)
            tp1 = round(price + 2 * last_atr, 2)
            tp2 = round(price + 4 * last_atr, 2)
        elif signal.startswith("SELL"):
            stop = round(price + atr_mult * last_atr, 2)
            tp1 = round(price - 1.5 * last_atr, 2)
            tp2 = round(price - 3 * last_atr, 2)
        else:
            stop = tp1 = tp2 = float("nan")
    else:
        stop = tp1 = tp2 = float("nan")

    return {
        "signal": signal,
        "state": state,
        "price": round(price, 2),
        "short_ema": round(float(se.iloc[-1]), 2),
        "long_ema": round(float(le.iloc[-1]), 2),
        "rsi": round(last_rsi, 1),
        "atr": None if math.isnan(last_atr) else round(last_atr, 2),
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2
    }


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Stock Watcher", page_icon="📈", layout="wide")
st.title("📈 Stock Watcher — Watchlist signals & levels")
st.caption("Signals are educational only — not financial advice.")

with st.sidebar:
    st.header("Settings")

    watchlist_str = st.text_input("Watchlist (comma-separated)", "AAPL, NVDA, TSLA")
    interval = st.selectbox("Interval", ["1d"], index=0)
    history_days = st.number_input("History (days)", min_value=60, max_value=2000, value=365, step=5)

    short_ema = st.number_input("Short EMA", min_value=5, max_value=200, value=20, step=1)
    long_ema = st.number_input("Long EMA", min_value=10, max_value=400, value=50, step=5)
    rsi_period = st.number_input("RSI period", min_value=5, max_value=50, value=14, step=1)
    rsi_buy = st.number_input("RSI buy ≤", min_value=5, max_value=60, value=40, step=1)
    rsi_sell = st.number_input("RSI sell ≥", min_value=40, max_value=95, value=70, step=1)
    atr_mult = st.number_input("ATR multiple (stop)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

    run = st.button("Run")

tickers = [t.strip().upper() for t in watchlist_str.split(",") if t.strip()]
if not tickers:
    st.info("Add at least one ticker in the sidebar to begin.")
    st.stop()

if run:
    results = []

    for t in tickers:
        st.markdown(f"### {t}")
        with st.spinner(f"Fetching Finviz snapshot for {t}…"):
            snap, err = fetch_finviz_snapshot(t)
        if err:
            st.warning(f"{t}: {err}")

        if snap:
            # show a small snapshot subset if available
            cols = st.columns(3)
            cols[0].metric("Name", snap.get("title", t))
            cols[1].metric("P/E", snap.get("P/E", "—"))
            cols[2].metric("EPS (ttm)", snap.get("EPS (ttm)", "—"))

        st.caption(f"{t}: requesting **{interval}** for ~**{history_days}** days")
        df = fetch_history(t, days=history_days, interval=interval)

        if df is None or df.empty:
            st.error(f"{t}: No/insufficient data for this interval/period (Yahoo & Stooq both failed).")
            st.divider()
            continue

        sig = generate_signal(
            df,
            short_ema=short_ema,
            long_ema=long_ema,
            rsi_period=rsi_period,
            rsi_buy=rsi_buy,
            rsi_sell=rsi_sell,
            atr_mult=atr_mult
        )
        if "error" in sig:
            st.error(f"{t}: {sig['error']}")
            st.divider()
            continue

        # Last 10 bars table (quick look)
        last = df.tail(10).copy()
        last["ShortEMA"] = ema(df["Close"], short_ema).tail(10).round(2)
        last["LongEMA"] = ema(df["Close"], long_ema).tail(10).round(2)
        last["RSI"] = rsi(df["Close"], rsi_period).tail(10).round(1)
        st.dataframe(last[["Open", "High", "Low", "Close", "ShortEMA", "LongEMA", "RSI"]])

        # Summary row
        row = {
            "Ticker": t,
            "Signal": sig["signal"],
            "State": sig["state"],
            "Price": sig["price"],
            "ShortEMA": sig["short_ema"],
            "LongEMA": sig["long_ema"],
            "RSI": sig["rsi"],
            "ATR": sig["atr"],
            "Stop": sig["stop"],
            "TP1": sig["tp1"],
            "TP2": sig["tp2"],
        }
        results.append(row)
        st.divider()

    if results:
        st.subheader("Watchlist Summary")
        st.dataframe(pd.DataFrame(results)[
            ["Ticker","Signal","State","Price","ShortEMA","LongEMA","RSI","ATR","Stop","TP1","TP2"]
        ])
