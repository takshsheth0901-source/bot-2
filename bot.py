"""
GOLD TRADING BOT - OANDA VERSION (single-run, for GitHub Actions)
"""

import csv
import os
import time
from datetime import date, datetime

import requests
import pandas as pd
import numpy as np

USE_DEMO_ACCOUNT = True
INSTRUMENT = "XAU_USD"
GRANULARITY = "M5"
CANDLE_COUNT = 200

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

BUY_SCORE_THRESHOLD = 2
SELL_SCORE_THRESHOLD = -2

BB_PERIOD = 20
BB_STD_DEV = 2
ATR_PERIOD = 14
VOLUME_SPIKE_MULT = 1.5
FIB_LOOKBACK = 40
FIB_TOLERANCE_PCT = 0.3

MAX_TRADES_PER_DAY = 10
MAX_POSITION_FRACTION = 0.10
DAILY_LOSS_LIMIT_FRACTION = 0.03
COOLDOWN_SECONDS_AFTER_TRADE = 300
KILL_SWITCH_FILE = "STOP"

LOG_FILE = "bot_activity.csv"
STATE_FILE = "bot_state.csv"

OANDA_API_TOKEN = os.environ.get("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")

BASE_URL = "https://api-fxpractice.oanda.com" if USE_DEMO_ACCOUNT else "https://api-fxtrade.oanda.com"


def oanda_session():
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {OANDA_API_TOKEN}",
        "Content-Type": "application/json",
    })
    return s


def oanda_request(session, method, path, **kwargs):
    url = f"{BASE_URL}{path}"
    resp = session.request(method, url, timeout=15, **kwargs)
    if resp.status_code == 429:
        time.sleep(5)
        resp = session.request(method, url, timeout=15, **kwargs)
    resp.raise_for_status()
    return resp.json() if resp.content else None


def get_account_summary(session):
    data = oanda_request(session, "GET", f"/v3/accounts/{OANDA_ACCOUNT_ID}/summary")
    return data["account"]


def get_open_position(session, instrument):
    try:
        data = oanda_request(session, "GET", f"/v3/accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return 0.0
        raise
    pos = data.get("position", {})
    long_units = float(pos.get("long", {}).get("units", 0))
    short_units = float(pos.get("short", {}).get("units", 0))
    return long_units + short_units


def place_market_order(session, instrument, units):
    body = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
    }
    return oanda_request(session, "POST", f"/v3/accounts/{OANDA_ACCOUNT_ID}/orders", json=body)


def get_candles(session, instrument=INSTRUMENT, granularity=GRANULARITY, count=CANDLE_COUNT):
    params = {"granularity": granularity, "count": count, "price": "M"}
    data = oanda_request(session, "GET", f"/v3/instruments/{instrument}/candles", params=params)
    rows = []
    for c in data["candles"]:
        if not c.get("complete", False):
            continue
        mid = c["mid"]
        rows.append({
            "time": c["time"],
            "Open": float(mid["o"]),
            "High": float(mid["h"]),
            "Low": float(mid["l"]),
            "Close": float(mid["c"]),
            "Volume": int(c.get("volume", 0)),
        })
    df = pd.DataFrame(rows).set_index("time")
    return df


def add_indicators(df):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    ema_fast = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def add_candle_patterns(df):
    body = (df["Close"] - df["Open"]).abs()
    rng = (df["High"] - df["Low"]).replace(0, 1e-9)
    lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)

    prev_open, prev_close = df["Open"].shift(1), df["Close"].shift(1)
    df["pat_bull_engulf"] = (prev_close < prev_open) & (df["Close"] > df["Open"]) & \
                             (df["Open"] <= prev_close) & (df["Close"] >= prev_open)
    df["pat_bear_engulf"] = (prev_close > prev_open) & (df["Close"] < df["Open"]) & \
                             (df["Open"] >= prev_close) & (df["Close"] <= prev_open)
    df["pat_hammer"] = (body <= 0.35*rng) & (lower_wick >= 2*body.replace(0, 1e-9)) & (upper_wick <= 0.15*rng)
    df["pat_shooting_star"] = (body <= 0.35*rng) & (upper_wick >= 2*body.replace(0, 1e-9)) & (lower_wick <= 0.15*rng)
    return df


def add_bollinger_bands(df):
    mid = df["Close"].rolling(BB_PERIOD).mean()
    std = df["Close"].rolling(BB_PERIOD).std()
    df["BB_mid"] = mid
    df["BB_upper"] = mid + BB_STD_DEV * std
    df["BB_lower"] = mid - BB_STD_DEV * std
    band_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_percent_b"] = (df["Close"] - df["BB_lower"]) / band_range
    return df


def add_atr(df):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean()
    df["ATR_pct"] = df["ATR"] / df["Close"] * 100
    return df


def add_volume_signal(df):
    avg_volume = df["Volume"].rolling(20).mean()
    df["volume_spike"] = df["Volume"] > (avg_volume * VOLUME_SPIKE_MULT)
    return df


def add_fibonacci_signal(df):
    swing_high = df["High"].rolling(FIB_LOOKBACK).max()
    swing_low = df["Low"].rolling(FIB_LOOKBACK).min()
    diff = swing_high - swing_low

    fib_382 = swing_high - 0.382 * diff
    fib_500 = swing_high - 0.5 * diff
    fib_618 = swing_high - 0.618 * diff

    def near(level):
        return (df["Close"] - level).abs() / df["Close"] * 100 <= FIB_TOLERANCE_PCT

    df["near_fib_382"] = near(fib_382)
    df["near_fib_500"] = near(fib_500)
    df["near_fib_618"] = near(fib_618)
    return df


def add_ict_signals(df, lookback=20):
    recent_low = df["Low"].rolling(lookback).min().shift(1)
    recent_high = df["High"].rolling(lookback).max().shift(1)
    df["bullish_sweep"] = (df["Low"] < recent_low) & (df["Close"] > recent_low)
    df["bearish_sweep"] = (df["High"] > recent_high) & (df["Close"] < recent_high)

    next_close = df["Close"].shift(-1)
    move_pct = (next_close - df["Close"]) / df["Close"] * 100
    df["bull_order_block"] = (df["Close"] < df["Open"]) & (move_pct >= 0.5)
    df["bear_order_block"] = (df["Close"] > df["Open"]) & (move_pct <= -0.5)
    return df


def score_row(row):
    score = 0
    if row["RSI"] < RSI_OVERSOLD: score += 1
    elif row["RSI"] > RSI_OVERBOUGHT: score -= 1
    if row["MACD"] > row["MACD_signal"] and row["MACD_hist"] > 0: score += 1
    elif row["MACD"] < row["MACD_signal"] and row["MACD_hist"] < 0: score -= 1
    if row.get("pat_bull_engulf") or row.get("pat_hammer"): score += 1
    if row.get("pat_bear_engulf") or row.get("pat_shooting_star"): score -= 1
    if row.get("bullish_sweep"): score += 1
    if row.get("bearish_sweep"): score -= 1
    if row.get("bull_order_block"): score += 1
    if row.get("bear_order_block"): score -= 1

    bb = row.get("BB_percent_b")
    if bb is not None and not pd.isna(bb):
        if bb <= 0: score += 1
        elif bb >= 1: score -= 1

    if row.get("volume_spike"):
        if score > 0: score += 1
        elif score < 0: score -= 1

    if row.get("near_fib_382") or row.get("near_fib_500") or row.get("near_fib_618"):
        if score > 0: score += 1
        elif score < 0: score -= 1

    atr_pct = row.get("ATR_pct")
    if atr_pct is not None and not pd.isna(atr_pct) and atr_pct < 0.02:
        score = 0

    return score


def latest_decision(df):
    df = add_indicators(df)
    df = add_candle_patterns(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_signal(df)
    df = add_fibonacci_signal(df)
    df = add_ict_signals(df)
    df = df.dropna(subset=["RSI", "MACD", "MACD_signal", "BB_percent_b", "ATR_pct"])
    if df.empty:
        return "hold", 0, None
    last_row = df.iloc[-1]
    score = score_row(last_row)
    if score >= BUY_SCORE_THRESHOLD:
        decision = "buy"
    elif score <= SELL_SCORE_THRESHOLD:
        decision = "sell"
    else:
        decision = "hold"
    return decision, score, last_row


def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": str(date.today()), "trades_today": 0, "last_trade_time": 0,
                "starting_equity_today": None}
    with open(STATE_FILE) as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if not row:
        return {"date": str(date.today()), "trades_today": 0, "last_trade_time": 0,
                "starting_equity_today": None}
    state = {
        "date": row["date"],
        "trades_today": int(row["trades_today"]),
        "last_trade_time": float(row["last_trade_time"]),
        "starting_equity_today": float(row["starting_equity_today"]) if row["starting_equity_today"] else None,
    }
    if state["date"] != str(date.today()):
        state = {"date": str(date.today()), "trades_today": 0, "last_trade_time": 0,
                  "starting_equity_today": None}
    return state


def save_state(state):
    with open(STATE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "trades_today", "last_trade_time", "starting_equity_today"])
        writer.writeheader()
        writer.writerow(state)


def approve_trade(state, equity):
    if state["starting_equity_today"] is None:
        state["starting_equity_today"] = equity

    if os.path.exists(KILL_SWITCH_FILE):
        return False, "Kill switch active"
    if (time.time() - state["last_trade_time"]) < COOLDOWN_SECONDS_AFTER_TRADE:
        return False, "In cooldown"
    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        return False, "Max trades/day reached"
    if state["starting_equity_today"] > 0:
        drawdown = (state["starting_equity_today"] - equity) / state["starting_equity_today"]
        if drawdown >= DAILY_LOSS_LIMIT_FRACTION:
            return False, "Daily loss limit reached"
    return True, "OK"


def log_row(row):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    ts = datetime.utcnow().isoformat()

    if not OANDA_API_TOKEN or not OANDA_ACCOUNT_ID:
        print("ERROR: OANDA_API_TOKEN / OANDA_ACCOUNT_ID not set. Check GitHub Secrets.")
        return

    session = oanda_session()
    state = load_state()

    try:
        df = get_candles(session)
        decision, score, last_row = latest_decision(df)

        account = get_account_summary(session)
        equity = float(account["NAV"])

        approved, reason = approve_trade(state, equity)

        action = "none"
        if decision in ("buy", "sell") and approved:
            price = float(last_row["Close"])
            current_position = get_open_position(session, INSTRUMENT)

            if decision == "buy":
                if current_position > 0:
                    action = "skipped (already long)"
                    units = 0
                else:
                    cash_to_risk = equity * MAX_POSITION_FRACTION
                    units = int(cash_to_risk // price)
            else:
                if current_position <= 0:
                    action = "skipped (nothing to sell)"
                    units = 0
                else:
                    units = -abs(current_position)

            if units != 0:
                place_market_order(session, INSTRUMENT, units)
                state["trades_today"] += 1
                state["last_trade_time"] = time.time()
                action = f"{decision} units={units}"

        elif decision in ("buy", "sell") and not approved:
            action = f"blocked: {reason}"

        log_row({"timestamp": ts, "decision": decision, "score": score,
                  "equity": equity, "action": action})
        print(f"[{ts}] decision={decision} score={score} equity={equity:.2f} action={action}")

    except Exception as e:
        log_row({"timestamp": ts, "decision": "error", "score": "", "equity": "", "action": str(e)})
        print(f"[{ts}] ERROR: {e}")

    save_state(state)


if __name__ == "__main__":
    main()
