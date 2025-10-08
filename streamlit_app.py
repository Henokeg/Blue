import time
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st

# -------------------- Page --------------------
st.set_page_config(page_title="Stock Watcher (Alpha Vantage)", page_icon="📈", layout="wide")
st.title("📈 Stock Watcher — Alpha Vantage")
st.caption("Educational only — not financial advice.")

API_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", None)

# -------------------- Helpers --------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = up.rolling(period).mean() / dn.rolling(period).mean()
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

@st.cache_data(show_spinner=False)
def fetch_alpha_daily_csv(symbol: str, outputsize: str = "compact") -> pd.DataFrame:
    """
    Uses Alpha Vantage TIME_SERIES_DAILY_ADJUSTED in CSV mode (fast, clean).
    Returns DataFrame indexed by date with Open/High/Low/Close/Volume.
    outputsize: 'compact' (~100 bars) or 'full' (20+ years).
    """
    if not API_KEY:
        raise RuntimeError("Alpha Vantage API key missing. Add ALPHA_VANTAGE_KEY in Streamlit secrets.")

    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}"
        f"&outputsize={outputsize}&datatype=csv&apikey={API_KEY}"
    )
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} from Alpha Vantage")

    # If rate-limited, Alpha returns a small CSV with "Note"
    if r.text.strip().lower().startswith("note"):
        raise RuntimeError("Rate limit: Alpha Vantage limit reached. Try fewer tickers or wait a minute.")

    df = pd.read_csv(pd.compat.StringIO(r.text))
    # Expected cols: timestamp,open,high,low,close,adjusted_close,volume,dividend_amount,split_coefficient
    need_cols = {"timestamp","open","high","low","close","adjusted_close","volume"}
    if not need_cols.issubset(set(df.columns)):
        raise RuntimeError("Unexpected CSV schema from Alpha Vantage")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    # Use adjusted close for signals
    out = pd.DataFrame({
        "Open": df["open"].astype(float),
        "High": df["high"].astype(float),
        "Low": df["low"].astype(float),
        "Close": df["adjusted_close"].astype(float),
        "Volume": df["volume"].astype(float),
    })
    return out

def generate_signal(df: pd.DataFrame,
                    short_ema: int = 20,
                    long_ema: int = 50,
                    rsi_period: int = 14,
                    rsi_buy: float = 40,
                    rsi_sell: float = 70,
                    atr_mult: float = 1.5) -> dict:
    close = df["Close"]
    needed = max(short_ema, long_ema, rsi_period) + 5
    if len(close) < needed:
        return {"error": f"Not enough bars ({len(close)}) for chosen periods (need ≥ {needed})."}

    se = ema(close, short_ema)
    le = ema(close, long_ema)
    r = rsi(close, rsi_period)
    a = atr(df, 14)

    # Crossovers
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
            stop = price - atr_mult * last_atr
            tp1 = price + 2 * last_atr
            tp2 = price + 4 * last_atr
        elif signal.startswith("SELL"):
            stop = price + atr_mult * last_atr
            tp1 = price - 1.5 * last_atr
            tp2 = price - 3 * last_atr
        else:
            stop = tp1 = tp2 = float("nan")
    else:
        stop = tp1 = tp2 = float("nan")

    return {
        "signal": signal, "state": state,
        "price": round(price, 2),
        "short_ema": round(float(se.iloc[-1]), 2),
        "long_ema": round(float(le.iloc[-1]), 2),
        "rsi": round(last_rsi, 1),
        "atr": None if math.isnan(last_atr) else round(last_atr, 2),
        "stop": None if math.isnan(stop) else round(stop, 2),
        "tp1": None if math.isnan(tp1) else round(tp1, 2),
        "tp2": None if math.isnan(tp2) else round(tp2, 2),
    }

# -------------------- UI --------------------
with st.sidebar:
    st.header("Settings")
    watchlist = st.text_area("Watchlist (comma-separated)", "AAPL, NVDA, MSFT", height=70)
    outputsize = st.selectbox("History size", ["compact (≈100 bars)", "full (all)"], index=0)
    short_ema = st.number_input("Short EMA", 5, 200, 20)
    long_ema  = st.number_input("Long EMA", 10, 400, 50)
    rsi_period = st.number_input("RSI period", 5, 50, 14)
    rsi_buy = st.number_input("RSI buy ≤", 5, 60, 40)
    rsi_sell = st.number_input("RSI sell ≥", 40, 95, 70)
    atr_mult = st.number_input("ATR multiple (stop)", 0.5, 5.0, 1.5, step=0.1)
    run = st.button("Run")

if not API_KEY:
    st.error("Add ALPHA_VANTAGE_KEY in Streamlit secrets to proceed.")
    st.stop()

if run:
    symbols = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    if not symbols:
        st.warning("Add at least one ticker.")
        st.stop()

    outsize = "compact" if outputsize.startswith("compact") else "full"
    results = []
    requests_made = 0

    for sym in symbols:
        st.markdown(f"### {sym}")
        st.caption(f"Fetching Alpha Vantage daily {outsize} …")

        # Rate-limit: free tier allows 5 requests/minute. Pace calls for safety.
        if requests_made and requests_made % 5 == 0:
            with st.spinner("Cooling down to respect API rate limits…"):
                time.sleep(15)

        try:
            df = fetch_alpha_daily_csv(sym, outputsize=outsize)
        except Exception as e:
            st.error(f"{sym}: {e}")
            continue
        finally:
            requests_made += 1

        if df is None or df.empty:
            st.warning(f"{sym}: no data returned.")
            continue

        sig = generate_signal(
            df,
            short_ema=short_ema, long_ema=long_ema,
            rsi_period=rsi_period, rsi_buy=rsi_buy, rsi_sell=rsi_sell,
            atr_mult=atr_mult
        )
        if "error" in sig:
            st.warning(f"{sym}: {sig['error']}")
            continue

        # Quick peek
        st.line_chart(df["Close"].tail(200))

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"{sig['price']:.2f}")
        c2.metric(f"EMA {short_ema}", f"{sig['short_ema']:.2f}")
        c3.metric(f"EMA {long_ema}", f"{sig['long_ema']:.2f}")
        c4.metric("RSI", f"{sig['rsi']:.1f}")
        st.caption(f"State: **{sig['state']}**  ·  ATR: {sig['atr']}  ·  "
                   f"Stop: {sig['stop']}  ·  TP1: {sig['tp1']}  ·  TP2: {sig['tp2']}")

        results.append({
            "Ticker": sym, "Signal": sig["signal"], "State": sig["state"],
            "Price": sig["price"], "RSI": sig["rsi"],
            "EMA_short": sig["short_ema"], "EMA_long": sig["long_ema"],
            "ATR": sig["atr"], "Stop": sig["stop"], "TP1": sig["tp1"], "TP2": sig["tp2"]
        })

    if results:
        st.subheader("Watchlist Summary")
        st.dataframe(pd.DataFrame(results))
