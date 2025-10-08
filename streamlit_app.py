# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ---------- Indicator helpers ----------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=period).mean()
    ma_down = down.rolling(period, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def generate_signal(df, short_ema_period=50, long_ema_period=200,
                    rsi_period=14, rsi_buy_thresh=40, rsi_sell_thresh=70,
                    atr_mult_for_stop=1.5):
    close = df['Close']
    needed = max(short_ema_period, long_ema_period, rsi_period) + 5
    if len(close) < needed:
        raise ValueError(f"not_enough_bars: have={len(close)} need>={needed}")

    se = close.ewm(span=short_ema_period, adjust=False).mean()
    le = close.ewm(span=long_ema_period, adjust=False).mean()

    # RSI
    d = close.diff()
    up = d.clip(lower=0); dn = -d.clip(upper=0)
    rsi = 100 - (100/(1 + up.rolling(rsi_period).mean() / dn.rolling(rsi_period).mean()))

    # ATR
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    ema_state = "neutral"
    if se.iloc[-1] > le.iloc[-1] and se.iloc[-2] <= le.iloc[-2]:
        ema_state = "bullish_crossover"
    elif se.iloc[-1] < le.iloc[-1] and se.iloc[-2] >= le.iloc[-2]:
        ema_state = "bearish_crossover"

    price = float(close.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else np.nan

    if ema_state == "bullish_crossover" and last_rsi <= rsi_buy_thresh:
        signal = "BUY"
    elif ema_state == "bearish_crossover" and last_rsi >= rsi_sell_thresh:
        signal = "SELL"
    else:
        signal = "BUY (RSI oversold)" if last_rsi < 30 else ("SELL (RSI overbought)" if last_rsi > 75 else "HOLD")

    if np.isnan(last_atr):
        stop = tp1 = tp2 = np.nan
    else:
        if signal.startswith("BUY"):
            stop = price - 1.5*last_atr
            tp1  = price + 3*last_atr
            tp2  = price + 6*last_atr
        elif signal.startswith("SELL"):
            stop = price + 1.5*last_atr
            tp1  = price - 2.25*last_atr
            tp2  = price - 4.5*last_atr
        else:
            stop = tp1 = tp2 = np.nan

    return {"signal": signal, "price": price, "short_ema": float(se.iloc[-1]),
            "long_ema": float(le.iloc[-1]), "rsi": last_rsi, "atr": last_atr,
            "stop": stop, "tp1": tp1, "tp2": tp2, "ema_state": ema_state}

    # crossover detection
    ema_state = "neutral"
    if len(df) >= 2:
        if se.iloc[-1] > le.iloc[-1] and se.iloc[-2] <= le.iloc[-2]:
            ema_state = "bullish_crossover"
        elif se.iloc[-1] < le.iloc[-1] and se.iloc[-2] >= le.iloc[-2]:
            ema_state = "bearish_crossover"

    # combine into signal
    if ema_state == "bullish_crossover" and last_rsi <= rsi_buy_thresh:
        signal = "BUY"
    elif ema_state == "bearish_crossover" and last_rsi >= rsi_sell_thresh:
        signal = "SELL"
    else:
        signal = "BUY (RSI oversold)" if last_rsi < 30 else ("SELL (RSI overbought)" if last_rsi > 75 else "HOLD")

    # levels
    if np.isnan(last_atr):
        stop = tp1 = tp2 = np.nan
    else:
        if signal.startswith("BUY"):
            stop = last_close - atr_mult_for_stop * last_atr
            tp1 = last_close + 2 * atr_mult_for_stop * last_atr
            tp2 = last_close + 4 * atr_mult_for_stop * last_atr
        elif signal.startswith("SELL"):
            stop = last_close + atr_mult_for_stop * last_atr
            tp1 = last_close - 1.5 * atr_mult_for_stop * last_atr
            tp2 = last_close - 3 * atr_mult_for_stop * last_atr
        else:
            stop = tp1 = tp2 = np.nan

    return {
        "signal": signal,
        "price": last_close,
        "short_ema": last_se,
        "long_ema": last_le,
        "rsi": last_rsi,
        "atr": last_atr,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "ema_state": ema_state
    }

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Stock Watcher", layout="wide")
st.title("📈 Stock Watcher — Watchlist signals & levels")

with st.sidebar:
    st.header("Settings")
    tickers_input = st.text_input("Watchlist (comma-separated)", "AAPL, NVDA, TSLA")
    interval = st.selectbox("Interval", ["1d","1h","30m","15m","5m","1m"], index=0)
    lookback_days = st.number_input("History (days)", min_value=2, max_value=3650, value=200)
    short_ema = st.number_input("Short EMA", min_value=5, max_value=200, value=50)
    long_ema  = st.number_input("Long EMA",  min_value=20, max_value=500, value=200)
    rsi_period = st.number_input("RSI period", min_value=5, max_value=50, value=14)
    rsi_buy = st.number_input("RSI buy ≤", min_value=1, max_value=100, value=40)
    rsi_sell = st.number_input("RSI sell ≥", min_value=1, max_value=100, value=70)
    atr_mult = st.number_input("ATR multiple (stop)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
    run = st.button("Run")

def fetch_stock_df(ticker, interval, lookback_days):
    period = f"{max(lookback_days, 2)}d" if interval == "1d" else f"{max(min(lookback_days, 60), 2)}d"
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
    if df is None or df.empty:
        return None
    return df.dropna()

if run:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker.")
    else:
        results = []
        for t in tickers:
            with st.spinner(f"Fetching {t}…"):
                df = fetch_stock_df(t, interval, lookback_days)
            if df is None:
                st.warning(f"No data for {t}. Try a different interval/period.")
                continue

            s = generate_signal(df,
                                short_ema_period=short_ema,
                                long_ema_period=long_ema,
                                rsi_period=rsi_period,
                                rsi_buy_thresh=rsi_buy,
                                rsi_sell_thresh=rsi_sell,
                                atr_mult_for_stop=atr_mult)

            st.markdown(f"### {t} — **{s['signal']}**")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Last", f"${s['price']:.2f}")
            c2.metric(f"EMA {short_ema}", f"${s['short_ema']:.2f}")
            c3.metric(f"EMA {long_ema}", f"${s['long_ema']:.2f}")
            c4.metric("RSI", f"{s['rsi']:.1f}")
            st.write(f"EMA state: **{s['ema_state']}** · ATR: **{s['atr']:.4f}**")
            st.write(f"Stop: `{s['stop']:.4f}` · TP1: `{s['tp1']:.4f}` · TP2: `{s['tp2']:.4f}`")

            st.line_chart(df['Close'].tail(200))

            last = df.tail(10).copy()
            last['ShortEMA'] = ema(df['Close'], short_ema).tail(10)
            last['LongEMA']  = ema(df['Close'], long_ema).tail(10)
            last['RSI']      = compute_rsi(df['Close'], rsi_period).tail(10)
            st.dataframe(last[['Open','High','Low','Close','ShortEMA','LongEMA','RSI']])

            results.append({"Ticker": t, **s})

        if results:
            st.markdown("## Watchlist Summary")
            st.dataframe(pd.DataFrame(results)[
                ["Ticker","signal","price","rsi","short_ema","long_ema","stop","tp1","tp2"]
            ])

st.markdown("---")
st.caption("Signals are educational only — not financial advice.")
