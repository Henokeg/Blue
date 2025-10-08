# streamlit_app.py
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Watcher", page_icon="📈", layout="wide")
st.title("📈 Stock Watcher — Watchlist signals & levels")
st.caption("Signals are educational only — not financial advice.")

# ---------- helpers ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = (up.rolling(period).mean()) / (dn.rolling(period).mean())
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _period_for(interval: str, lookback_days: int) -> str:
    # Yahoo caps for intraday:
    # 1m  → max 7d
    # 5m  → max 60d
    # 15m → max 60d
    # 30m → max 60d
    # 1h  → max ~730d (2 years)
    caps = {
        "1m": 7,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "1h": 730,
        "60m": 730,   # if you ever use 60m internally
    }
    if interval == "1d":
        days = max(lookback_days, 2)
    else:
        cap = caps.get(interval, 60)
        days = min(max(lookback_days, 2), cap)
    return f"{days}d"

def fetch_stock_df(ticker: str, interval: str, lookback_days: int) -> pd.DataFrame | None:
    ticker = ticker.strip().upper()
    period = _period_for(interval, lookback_days)

    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False,
        threads=False,
        auto_adjust=False,
    )
    req_period = _period_for(interval, lookback_days)
st.caption(f"{t}: requesting interval {interval} with period {req_period}")
df = fetch_stock_df(t, interval, lookback_days)
if df is None:
    st.warning(f"{t}: No data returned for {interval}/{req_period}.")
    continue
st.caption(f"{t}: got {len(df)} bars")

if df is None or df.empty:
    return None   # <-- this line indented by exactly 4 spaces
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df if not df.empty else None

def generate_signal(df, short_ema_period=50, long_ema_period=200,
                    rsi_period=14, rsi_buy_thresh=40, rsi_sell_thresh=70,
                    atr_mult_for_stop=1.5):

    # ✅ Make sure this IF is aligned at the first indentation level inside the function
    if df is None or df.empty:
        raise ValueError("No or insufficient data for this interval/period.")

    close = df['Close']
    ...

    needed = max(short_ema_period, long_ema_period, rsi_period) + 5
    if len(close) < needed:
        raise ValueError(f"not_enough_bars: have={len(close)} need>={needed}")

    se = ema(close, short_ema_period)
    le = ema(close, long_ema_period)
    rsi = compute_rsi(close, rsi_period)
    atr_series = atr(df, 14)

    # crossover state
    ema_state = "neutral"
    if se.iloc[-1] > le.iloc[-1] and se.iloc[-2] <= le.iloc[-2]:
        ema_state = "bullish_crossover"
    elif se.iloc[-1] < le.iloc[-1] and se.iloc[-2] >= le.iloc[-2]:
        ema_state = "bearish_crossover"

    price = float(close.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_se = float(se.iloc[-1])
    last_le = float(le.iloc[-1])
    last_atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else np.nan

    # signal
    if ema_state == "bullish_crossover" and last_rsi <= rsi_buy_thresh:
        signal = "BUY"
    elif ema_state == "bearish_crossover" and last_rsi >= rsi_sell_thresh:
        signal = "SELL"
    else:
        signal = "HOLD"

    # levels
    if np.isnan(last_atr):
        stop = tp1 = tp2 = np.nan
    else:
        if signal == "BUY":
            stop = price - atr_mult_for_stop * last_atr
            tp1 = price + 2 * atr_mult_for_stop * last_atr
            tp2 = price + 4 * atr_mult_for_stop * last_atr
        elif signal == "SELL":
            stop = price + atr_mult_for_stop * last_atr
            tp1 = price - 1.5 * atr_mult_for_stop * last_atr
            tp2 = price - 3 * atr_mult_for_stop * last_atr
        else:
            stop = tp1 = tp2 = np.nan

    return {
        "signal": signal, "price": price,
        "short_ema": last_se, "long_ema": last_le,
        "rsi": last_rsi, "atr": last_atr,
        "stop": stop, "tp1": tp1, "tp2": tp2, "ema_state": ema_state
    }

# ---------- UI (form in main page – works better on mobile) ----------
st.markdown("### Settings")
with st.form("settings", clear_on_submit=False):
    tickers_input = st.text_area("Watchlist (comma-separated)", "AAPL, NVDA, TSLA", height=70)
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m", "1m"], index=0)
    lookback_days = st.number_input("History (days)", min_value=2, max_value=3650, value=200)
    short_ema = st.number_input("Short EMA", min_value=5, max_value=200, value=50)
    long_ema  = st.number_input("Long EMA",  min_value=20, max_value=500, value=200)
    rsi_period = st.number_input("RSI period", min_value=5, max_value=50, value=14)
    rsi_buy = st.number_input("RSI buy ≤", min_value=1, max_value=100, value=40)
    rsi_sell = st.number_input("RSI sell ≥", min_value=1, max_value=100, value=70)
    atr_mult = st.number_input("ATR multiple (stop)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

    cols = st.columns(2)
    run = cols[0].form_submit_button("Run")
    show_table = cols[1].form_submit_button("Run & show last 10 bars")

# ---------- run ----------
if run or show_table:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker.")
    else:
        if interval != "1d" and long_ema > 100:
            st.info("For intraday intervals, consider Long EMA ≤ 100 and History ≤ 60 for reliable data.")

        results = []
        for t in tickers:
            with st.spinner(f"Fetching {t}…"):
                df = fetch_stock_df(t, interval, lookback_days)

            if df is None or len(df) < 2:
                st.warning(f"{t}: No/insufficient data for this interval/period.")
                continue

            # ensure enough bars for chosen lookbacks
            needed = max(short_ema, long_ema, rsi_period) + 5
            if len(df) < needed:
                st.warning(f"{t}: Not enough candles ({len(df)}) for EMA/RSI (need ≥ {needed}). "
                           "Try Interval = 1d, or reduce Long EMA / increase History.")
                continue

            # add indicators for display table (optional)
            se = ema(df["Close"], short_ema)
            le = ema(df["Close"], long_ema)
            rsi_ser = compute_rsi(df["Close"], rsi_period)

            try:
                s = generate_signal(
                    df,
                    short_ema_period=short_ema,
                    long_ema_period=long_ema,
                    rsi_period=rsi_period,
                    rsi_buy_thresh=rsi_buy,
                    rsi_sell_thresh=rsi_sell,
                    atr_mult_for_stop=atr_mult,
                )
            except Exception as e:
                st.error(f"{t}: {type(e).__name__}: {e}")
                continue

            st.markdown(f"### {t} — **{s['signal']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last", f"${s['price']:.2f}")
            c2.metric(f"EMA {short_ema}", f"${s['short_ema']:.2f}")
            c3.metric(f"EMA {long_ema}", f"${s['long_ema']:.2f}")
            c4.metric("RSI", f"{s['rsi']:.1f}")

            st.write(f"Stops/Targets → Stop: `{s['stop']:.4f}` · TP1: `{s['tp1']:.4f}` · TP2: `{s['tp2']:.4f}`")

            st.line_chart(df["Close"].tail(200))

            if show_table:
                last = df[["Open", "High", "Low", "Close"]].copy()
                last["ShortEMA"] = se
                last["LongEMA"] = le
                last["RSI"] = rsi_ser
                st.dataframe(last[["Open","High","Low","Close","ShortEMA","LongEMA","RSI"]].tail(10))

            results.append({"Ticker": t, **s})

        if results:
            st.markdown("## Watchlist Summary")
            summary = pd.DataFrame(results)[
                ["Ticker","signal","price","rsi","short_ema","long_ema","stop","tp1","tp2"]
            ]
            st.dataframe(summary)

st.markdown("---")
st.caption("Made with Streamlit + Yahoo Finance. Use at your own risk.")
