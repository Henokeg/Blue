import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# -------------------
# Finviz Scraper
# -------------------
def fetch_stock_df(ticker: str):
    """Fetch stock snapshot data from Finviz"""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        return None, f"Error fetching data: {e}"

    soup = BeautifulSoup(response.text, "lxml")

    # Find Finviz table
    table = soup.find("table", class_="snapshot-table2")
    if table is None:
        return None, "No data table found on Finviz page."

    data = {}
    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        for i in range(0, len(cells), 2):
            key = cells[i].text.strip()
            val = cells[i+1].text.strip()
            data[key] = val

    # Convert to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["Metric", "Value"])
    return df, None
# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_df(ticker: str, interval: str, lookback_days: int) -> pd.DataFrame:
    """Fetch OHLCV for a single ticker. Returns empty df on failure."""
    period = f"{min(lookback_days, 3640)}d"
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index().rename(columns=str)
    # Ensure standard column names exist
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    # Drop rows with missing price values
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = up.rolling(period).mean() / dn.rolling(period).mean()
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def analyze(df: pd.DataFrame,
            short_ema_period: int,
            long_ema_period: int,
            rsi_period: int,
            rsi_buy_thresh: int,
            rsi_sell_thresh: int,
            atr_mult_for_stop: float):
    """Return dict with signal/levels or None if not enough data."""
    if df is None or df.empty:
        return None

    # Need enough bars for indicators
    needed = max(short_ema_period, long_ema_period, rsi_period) + 5
    if len(df) < needed:
        return None

    close = df["Close"]
    se = ema(close, short_ema_period)
    le = ema(close, long_ema_period)
    rsi = compute_rsi(close, rsi_period)
    atr = compute_atr(df, 14)

    price = float(close.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else np.nan

    # EMA cross state
    ema_state = "neutral"
    if se.iloc[-1] > le.iloc[-1] and se.iloc[-2] <= le.iloc[-2]:
        ema_state = "bullish_crossover"
    elif se.iloc[-1] < le.iloc[-1] and se.iloc[-2] >= le.iloc[-2]:
        ema_state = "bearish_crossover"

    # Signal logic
    if ema_state == "bullish_crossover" and last_rsi <= rsi_buy_thresh:
        signal = "BUY"
    elif ema_state == "bearish_crossover" and last_rsi >= rsi_sell_thresh:
        signal = "SELL"
    else:
        if last_rsi < 30:
            signal = "BUY (RSI oversold)"
        elif last_rsi > 75:
            signal = "SELL (RSI overbought)"
        else:
            signal = "HOLD"

    # Levels from ATR
    if np.isnan(last_atr):
        stop = tp1 = tp2 = np.nan
    else:
        if signal.startswith("BUY"):
            stop = price - atr_mult_for_stop * last_atr
            tp1 = price + 2 * last_atr
            tp2 = price + 4 * last_atr
        elif signal.startswith("SELL"):
            stop = price + atr_mult_for_stop * last_atr
            tp1 = price - 2 * last_atr
            tp2 = price - 4 * last_atr
        else:
            stop = tp1 = tp2 = np.nan

    return {
        "signal": signal,
        "price": price,
        "short_ema": float(se.iloc[-1]),
        "long_ema": float(le.iloc[-1]),
        "rsi": last_rsi,
        "atr": last_atr,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "ema_state": ema_state,
    }

# ---------- Run ----------
if run or show_last:
    tickers = [t.strip().upper() for t in watchlist_raw.split(",") if t.strip()]
    if not tickers:
        st.warning("Add at least one ticker in the sidebar.")
    else:
        results = []
        for t in tickers:
            st.subheader(t)
            st.caption(f"requesting interval **{interval}** for ~**{lookback_days}** days")
            df = fetch_df(t, interval, lookback_days)

            if df.empty:
                st.warning(f"{t}: No/insufficient data for this interval/period.")
                continue

            s = analyze(
                df,
                short_ema_period=short_ema,
                long_ema_period=long_ema,
                rsi_period=rsi_period,
                rsi_buy_thresh=rsi_buy,
                rsi_sell_thresh=rsi_sell,
                atr_mult_for_stop=atr_mult,
            )

            if s is None:
                st.warning(f"{t}: Not enough bars to compute indicators yet.")
                continue

            st.markdown(
                f"**Signal:** `{s['signal']}`  •  "
                f"**Price:** {s['price']:.2f}  •  **RSI:** {s['rsi']:.1f}  •  "
                f"**EMA(Short/Long):** {s['short_ema']:.2f} / {s['long_ema']:.2f}"
            )
            st.markdown(
                f"**Stop:** {s['stop'] if np.isnan(s['stop']) else f'{s['stop']:.2f}'}  "
                f"• **TP1:** {s['tp1'] if np.isnan(s['tp1']) else f'{s['tp1']:.2f}'}  "
                f"• **TP2:** {s['tp2'] if np.isnan(s['tp2']) else f'{s['tp2']:.2f}'}"
            )

            if show_last:
                st.dataframe(df.tail(10))

            results.append({"Ticker": t, **s})

        if results:
            st.markdown("## Watchlist Summary")
            st.dataframe(
                pd.DataFrame(results)[
                    ["Ticker", "signal", "price", "rsi", "short_ema", "long_ema", "stop", "tp1", "tp2"]
                ]
            )

st.markdown("---")
st.caption("Made with Streamlit + Yahoo Finance. Use at your own risk.")
