
return "1y"import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ---------- helpers ----------
def _period_for(interval: str, lookback_days: int) -> str:
    """Map interval + lookback to a yfinance 'period' that returns enough bars."""
    if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m"):
        # intraday data is limited to ~30 days on Yahoo
        return "30d" if lookback_days > 30 else f"{lookback_days}d"
    if interval in ("1h",):
        return "730d"  # 2y is usually plenty
    if interval in ("1d", "5d", "1wk", "1mo"):
        # use months/years for longer windows
        if lookback_days <= 31:
            return "1mo"
        if lookback_days <= 93:
            return "3mo"
        if lookback_days <= 186:
            return "6mo"
        if lookback_days <= 365:
        if lookback_days <= 730:
            return "2y"
        return "5y"
    return "1y"

def fetch_stock_df(ticker: str, interval: str, lookback_days: int) -> pd.DataFrame | None:
    """Download OHLCV with sensible defaults and return a clean dataframe."""
    period = _period_for(interval, lookback_days)
    try:
        df = yf.download(
            tickers=ticker.strip().upper(),
            period=period,
            interval=interval,
            progress=False,
            threads=False,
            auto_adjust=False,
        )
    except Exception:
        return None

    if isinstance(df, pd.DataFrame) and not df.empty:
        # Handle multi-index that yfinance returns for multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker.strip().upper(), level=1, axis=1)

        need = {"Open", "High", "Low", "Close"}
        if not need.issubset(df.columns):
            return None

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        return df
    return None

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(period).mean() / down.rolling(period).mean()
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift()
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def generate_signal(
    df: pd.DataFrame,
    short_ema_period: int = 50,
    long_ema_period: int = 200,
    rsi_period: int = 14,
    rsi_buy_thresh: int = 40,
    rsi_sell_thresh: int = 70,
    atr_mult_for_stop: float = 1.5,
) -> dict:
    close = df["Close"]

    # Need at least this many bars to be safe
    needed = max(short_ema_period, long_ema_period, rsi_period) + 5
    if len(close) < needed:
        return {"signal": "N/A (not enough data)"}

    se = ema(close, short_ema_period)
    le = ema(close, long_ema_period)
    rsi = compute_rsi(close, rsi_period)
    atr_series = atr(df, 14)

    # simple EMA crossover state
    ema_state = "neutral"
    if se.iloc[-1] > le.iloc[-1] and se.iloc[-2] <= le.iloc[-2]:
        ema_state = "bullish_crossover"
    elif se.iloc[-1] < le.iloc[-1] and se.iloc[-2] >= le.iloc[-2]:
        ema_state = "bearish_crossover"

    price = float(close.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else np.nan

    # signal logic
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

    # levels
    if np.isnan(last_atr):
        stop = tp1 = tp2 = np.nan
    else:
        if signal.startswith("BUY"):
            stop = price - atr_mult_for_stop * last_atr
            tp1 = price + 2 * last_atr
            tp2 = price + 4 * last_atr
        elif signal.startswith("SELL"):
            stop = price + atr_mult_for_stop * last_atr
            tp1 = price - 1.5 * last_atr
            tp2 = price - 3 * last_atr
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

# ---------- UI ----------
st.set_page_config(page_title="Stock Watcher", layout="centered")
st.title("📈 Stock Watcher — Watchlist signals & levels")
st.caption("Signals are educational only — not financial advice.")

with st.sidebar:
    st.header("Settings")
    watchlist = st.text_input("Watchlist (comma-separated)", "AAPL, NVDA, TSLA")
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"])
    history_days = st.slider("History (days)", 60, 365*2, 200, step=10)

    short_ema = st.number_input("Short EMA", 5, 1000, 50, step=5)
    long_ema = st.number_input("Long EMA", 10, 2000, 200, step=10)
    rsi_period = st.number_input("RSI period", 2, 100, 14, step=1)
    rsi_buy = st.number_input("RSI buy ≤", 1, 99, 40, step=1)
    rsi_sell = st.number_input("RSI sell ≥", 1, 99, 70, step=1)
    atr_mult = st.number_input("ATR multiple (stop)", 0.5, 5.0, 1.5, step=0.1)

    run = st.button("Run")

if run:
    symbols = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    results = []

    for ticker in symbols:
        st.caption(f"{ticker}: requesting interval {interval}")
        df = fetch_stock_df(ticker, interval, history_days)

        if df is None or df.empty:
            st.warning(f"{ticker}: No/insufficient data for this interval/period.")
            continue

        sig = generate_signal(
            df,
            short_ema_period=short_ema,
            long_ema_period=long_ema,
            rsi_period=rsi_period,
            rsi_buy_thresh=rsi_buy,
            rsi_sell_thresh=rsi_sell,
            atr_mult_for_stop=atr_mult,
        )

        # Show last 10 bars quick peek
        with st.expander(f"{ticker} — details"):
            st.dataframe(df.tail(10)[["Open", "High", "Low", "Close"]])

        results.append({"Ticker": ticker, **sig})

    if results:
        st.subheader("Watchlist Summary")
        st.dataframe(
            pd.DataFrame(results)[
                ["Ticker", "signal", "price", "rsi", "short_ema", "long_ema", "atr", "stop", "tp1", "tp2"]
            ]
        )

st.markdown("———")
st.caption("Made with Streamlit + Yahoo Finance. Use at your own risk.")
