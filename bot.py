"""
Automated Gold (XAU/USD) trading bot for OANDA.
Runs once per invocation — intended to be triggered every 5 minutes by GitHub Actions.
"""

import os
import sys
import json
import csv
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

USE_DEMO_ACCOUNT = True

OANDA_API_TOKEN = os.environ.get("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")

INSTRUMENT = "XAU_USD"
GRANULARITY = "M1"
CANDLE_COUNT = 300

STOP_LOSS_PCT = 0.15
TAKE_PROFIT_PCT = 0.30
ALLOW_SHORTING = True

BUY_SCORE_THRESHOLD = 1
SELL_SCORE_THRESHOLD = -1

MAX_TRADES_PER_DAY = 30
MAX_POSITION_FRACTION = 0.05
DAILY_LOSS_LIMIT_FRACTION = 0.03
COOLDOWN_SECONDS_AFTER_TRADE = 120
KILL_SWITCH_FILE = "STOP"

STATE_FILE = "bot_state.csv"
ACTIVITY_FILE = "bot_activity.csv"

NEWS_FILTER_ENABLED = True
NEWS_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_BLACKOUT_MINUTES_BEFORE = 30
NEWS_BLACKOUT_MINUTES_AFTER = 30
NEWS_CURRENCIES = ["USD"]
NEWS_IMPACT_LEVELS = ["High"]

OANDA_HOSTS = {
    True: "https://api-fxpractice.oanda.com",
    False: "https://api-fxtrade.oanda.com",
}

# ══════════════════════════════════════════════════════════════════════════
# OANDA API HELPERS
# ══════════════════════════════════════════════════════════════════════════

def oanda_session():
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {OANDA_API_TOKEN}",
        "Content-Type": "application/json",
    })
    return s


def oanda_base_url():
    return OANDA_HOSTS[USE_DEMO_ACCOUNT]


def oanda_request(session, method, path, **kwargs):
    url = f"{oanda_base_url()}{path}"
    resp = session.request(method, url, timeout=20, **kwargs)
    resp.raise_for_status()
    if resp.text:
        return resp.json()
    return {}


def get_account_summary(session):
    data = oanda_request(session, "GET", f"/v3/accounts/{OANDA_ACCOUNT_ID}/summary")
    acc = data["account"]
    return {
        "balance": float(acc["balance"]),
        "nav": float(acc["NAV"]),
        "unrealized_pl": float(acc.get("unrealizedPL", 0.0)),
        "margin_available": float(acc.get("marginAvailable", 0.0)),
    }


def get_open_position(session, instrument):
    try:
        data = oanda_request(
            session, "GET",
            f"/v3/accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}"
        )
        pos = data["position"]
        long_units = float(pos["long"]["units"])
        short_units = float(pos["short"]["units"])
        return long_units + short_units
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return 0.0
        raise


def place_market_order(session, instrument, units, stop_loss_price=None, take_profit_price=None):
    order = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
    }
    if stop_loss_price is not None:
        order["order"]["stopLossOnFill"] = {"price": f"{stop_loss_price:.2f}"}
    if take_profit_price is not None:
        order["order"]["takeProfitOnFill"] = {"price": f"{take_profit_price:.2f}"}

    return oanda_request(
        session, "POST",
        f"/v3/accounts/{OANDA_ACCOUNT_ID}/orders",
        data=json.dumps(order),
    )


def get_candles(session, instrument, granularity, count):
    params = {"granularity": granularity, "count": count, "price": "M"}
    data = oanda_request(
        session, "GET",
        f"/v3/instruments/{instrument}/candles",
        params=params,
    )
    rows = []
    for c in data["candles"]:
        if not c["complete"]:
            continue
        rows.append({
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": int(c["volume"]),
        })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df


# ══════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════

def add_indicators(df):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    return df


def add_bollinger_bands(df, period=20, std_mult=2):
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + std_mult * std
    df["bb_lower"] = mid - std_mult * std
    return df


def add_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return df


def add_volume_signal(df, period=20):
    avg_vol = df["volume"].rolling(period).mean()
    df["vol_ratio"] = df["volume"] / avg_vol.replace(0, np.nan)
    df["vol_spike"] = df["vol_ratio"] > 1.5
    return df


def add_fibonacci_signal(df, lookback=50):
    roll_high = df["high"].rolling(lookback).max()
    roll_low = df["low"].rolling(lookback).min()
    rng = (roll_high - roll_low).replace(0, np.nan)
    fib_618 = roll_low + 0.618 * rng
    fib_50 = roll_low + 0.5 * rng
    df["near_fib_618"] = (df["close"] - fib_618).abs() / rng < 0.01
    df["near_fib_50"] = (df["close"] - fib_50).abs() / rng < 0.01
    return df


def add_ict_signals(df):
    prev_low = df["low"].shift(1)
    prev_high = df["high"].shift(1)
    df["liquidity_sweep_low"] = (df["low"] < prev_low) & (df["close"] > prev_low)
    df["liquidity_sweep_high"] = (df["high"] > prev_high) & (df["close"] < prev_high)

    bull_ob = (df["close"].shift(1) < df["open"].shift(1)) & (df["close"] > df["high"].shift(1))
    bear_ob = (df["close"].shift(1) > df["open"].shift(1)) & (df["close"] < df["low"].shift(1))
    df["order_block_bull"] = bull_ob
    df["order_block_bear"] = bear_ob
    return df


def add_adx(df, period=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr_w = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period).mean() / atr_w.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / period, min_periods=period).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def add_stochastic(df, k_period=14, d_period=3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


def add_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci"] = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return df


def add_mfi(df, period=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = tp * df["volume"]
    positive_flow = np.where(tp > tp.shift(1), raw_money_flow, 0.0)
    negative_flow = np.where(tp < tp.shift(1), raw_money_flow, 0.0)
    pos_sum = pd.Series(positive_flow, index=df.index).rolling(period).sum()
    neg_sum = pd.Series(negative_flow, index=df.index).rolling(period).sum()
    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    df["mfi"] = 100 - (100 / (1 + money_ratio))
    df["mfi"] = df["mfi"].fillna(50)
    return df


def add_vwap(df, period=50):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    df["vwap"] = pv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)
    return df


def get_htf_bias(session, instrument):
    bias = 0
    try:
        for gran, weight in [("D", 1), ("H4", 1), ("H1", 1)]:
            df = get_candles(session, instrument, gran, 60)
            if len(df) < 20:
                continue
            ema20 = df["close"].ewm(span=20, adjust=False).mean()
            if df["close"].iloc[-1] > ema20.iloc[-1]:
                bias += weight
            else:
                bias -= weight
    except Exception:
        return 0
    return bias


def in_active_session():
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    london = 7 <= hour <= 16
    ny = 12 <= hour <= 21
    return london or ny


def fetch_news_events():
    if not NEWS_FILTER_ENABLED:
        return []
    try:
        resp = requests.get(NEWS_CALENDAR_URL, timeout=10)
        resp.raise_for_status()
        events = resp.json()
        filtered = []
        for ev in events:
            if ev.get("country") not in NEWS_CURRENCIES:
                continue
            if ev.get("impact") not in NEWS_IMPACT_LEVELS:
                continue
            try:
                ev_time = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
            except Exception:
                continue
            filtered.append({"time": ev_time, "title": ev.get("title", "")})
        return filtered
    except Exception:
        return []


def in_news_blackout(news_events):
    now_utc = datetime.now(timezone.utc)
    for ev in news_events:
        window_start = ev["time"] - timedelta(minutes=NEWS_BLACKOUT_MINUTES_BEFORE)
        window_end = ev["time"] + timedelta(minutes=NEWS_BLACKOUT_MINUTES_AFTER)
        if window_start <= now_utc <= window_end:
            return True, ev
    return False, None


# ══════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════

def score_row(row, htf_bias=0, in_session=False):
    score = 0

    if row["close"] > row["ema20"]:
        score += 1
    else:
        score -= 1

    if row["macd_hist"] > 0:
        score += 1
    else:
        score -= 1

    if row["rsi"] < 30:
        score += 1
    elif row["rsi"] > 70:
        score -= 1

    if not pd.isna(row.get("bb_lower", np.nan)) and row["close"] < row["bb_lower"]:
        score += 1
    if not pd.isna(row.get("bb_upper", np.nan)) and row["close"] > row["bb_upper"]:
        score -= 1

    if not pd.isna(row.get("adx", np.nan)) and row["adx"] > 20:
        if row.get("plus_di", 0) > row.get("minus_di", 0):
            score += 1
        else:
            score -= 1

    if not pd.isna(row.get("stoch_k", np.nan)):
        if row["stoch_k"] < 20 and row["stoch_k"] > row.get("stoch_d", 0):
            score += 1
        elif row["stoch_k"] > 80 and row["stoch_k"] < row.get("stoch_d", 100):
            score -= 1

    if not pd.isna(row.get("cci", np.nan)):
        if row["cci"] < -100:
            score += 1
        elif row["cci"] > 100:
            score -= 1

    if not pd.isna(row.get("mfi", np.nan)):
        if row["mfi"] < 20:
            score += 1
        elif row["mfi"] > 80:
            score -= 1

    if not pd.isna(row.get("vwap", np.nan)):
        if row["close"] > row["vwap"]:
            score += 1
        else:
            score -= 1

    if row.get("liquidity_sweep_low", False) or row.get("order_block_bull", False):
        score += 1
    if row.get("liquidity_sweep_high", False) or row.get("order_block_bear", False):
        score -= 1

    if row.get("near_fib_618", False) or row.get("near_fib_50", False):
        if score > 0:
            score += 1
        elif score < 0:
            score -= 1

    if row.get("vol_spike", False):
        if score > 0:
            score += 1
        elif score < 0:
            score -= 1

    if htf_bias > 0 and score > 0:
        score += 1
    elif htf_bias < 0 and score < 0:
        score -= 1

    if in_session:
        if score > 0:
            score += 1
        elif score < 0:
            score -= 1

    if not pd.isna(row.get("atr", np.nan)) and not pd.isna(row.get("close", np.nan)):
        atr_pct = row["atr"] / row["close"] * 100
        if atr_pct < 0.02:
            score = 0

    return score


def latest_decision(df, htf_bias=0):
    df = add_indicators(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_signal(df)
    df = add_fibonacci_signal(df)
    df = add_ict_signals(df)
    df = add_adx(df)
    df = add_stochastic(df)
    df = add_cci(df)
    df = add_mfi(df)
    df = add_vwap(df)

    session_active = in_active_session()
    last_row = df.iloc[-1]
    score = score_row(last_row, htf_bias=htf_bias, in_session=session_active)

    if score >= BUY_SCORE_THRESHOLD:
        decision = "buy"
    elif score <= SELL_SCORE_THRESHOLD:
        decision = "sell"
    else:
        decision = "none"

    return {
        "decision": decision,
        "score": score,
        "price": float(last_row["close"]),
        "atr": float(last_row["atr"]) if not pd.isna(last_row.get("atr", np.nan)) else None,
    }


# ══════════════════════════════════════════════════════════════════════════
# STATE / RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

def load_state():
    default = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "trades_today": 0,
        "start_of_day_equity": None,
        "last_trade_time": None,
    }
    if not os.path.exists(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return default
            row = rows[-1]
            date_val = row.get("date")
            trades_val = row.get("trades_today")
            equity_val = row.get("start_of_day_equity")
            last_trade_val = row.get("last_trade_time")
            if not date_val or trades_val is None:
                return default
            return {
                "date": date_val,
                "trades_today": int(trades_val),
                "start_of_day_equity": float(equity_val) if equity_val else None,
                "last_trade_time": last_trade_val if last_trade_val else None,
            }
    except Exception:
        return default


def save_state(state):
    file_exists = os.path.exists(STATE_FILE)
    with open(STATE_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "trades_today", "start_of_day_equity", "last_trade_time"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(state)


def approve_trade(state, equity):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if state["date"] != today:
        state = {
            "date": today,
            "trades_today": 0,
            "start_of_day_equity": equity,
            "last_trade_time": None,
        }

    if state["start_of_day_equity"] is None:
        state["start_of_day_equity"] = equity

    if os.path.exists(KILL_SWITCH_FILE):
        return False, state, "kill switch active"

    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        return False, state, "daily trade limit reached"

    if state["last_trade_time"]:
        last_time = datetime.fromisoformat(state["last_trade_time"])
        elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
        if elapsed < COOLDOWN_SECONDS_AFTER_TRADE:
            return False, state, "cooldown active"

    daily_loss = state["start_of_day_equity"] - equity
    if daily_loss > state["start_of_day_equity"] * DAILY_LOSS_LIMIT_FRACTION:
        return False, state, "daily loss limit reached"

    return True, state, "ok"


def log_row(row_dict):
    file_exists = os.path.exists(ACTIVITY_FILE)
    with open(ACTIVITY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    if not OANDA_API_TOKEN or not OANDA_ACCOUNT_ID:
        print("Missing OANDA_API_TOKEN or OANDA_ACCOUNT_ID environment variables.")
        sys.exit(1)

    session = oanda_session()

    account = get_account_summary(session)
    equity = account["nav"]

    df = get_candles(session, INSTRUMENT, GRANULARITY, CANDLE_COUNT)
    if len(df) < 60:
        print("Not enough candle data yet.")
        return

    htf_bias = get_htf_bias(session, INSTRUMENT)
    result = latest_decision(df, htf_bias=htf_bias)

    decision = result["decision"]
    score = result["score"]
    price = result["price"]

    state = load_state()
    current_position = get_open_position(session, INSTRUMENT)

    news_events = fetch_news_events()
    blackout_active, blackout_event = in_news_blackout(news_events)

    action = "none"
    reason = ""
    units_placed = 0

    if decision == "none":
        reason = "no signal"
    elif blackout_active:
        reason = f"news blackout: {blackout_event['title']}"
    elif decision == "sell" and not ALLOW_SHORTING:
        reason = "shorting disabled"
    else:
        approved, state, approve_reason = approve_trade(state, equity)
        if not approved:
            reason = approve_reason
        else:
            stop_distance = price * STOP_LOSS_PCT / 100
            risk_amount = equity * MAX_POSITION_FRACTION
            units = int(risk_amount / stop_distance) if stop_distance > 0 else 0

            if units <= 0:
                reason = "computed 0 units (risk_amount too small vs stop_distance)"
            elif decision == "buy" and current_position <= 0:
                sl_price = price * (1 - STOP_LOSS_PCT / 100)
                tp_price = price * (1 + TAKE_PROFIT_PCT / 100)
                place_market_order(session, INSTRUMENT, units, sl_price, tp_price)
                action = "buy"
                units_placed = units
                reason = "buy order placed"
            elif decision == "sell" and current_position >= 0:
                sl_price = price * (1 + STOP_LOSS_PCT / 100)
                tp_price = price * (1 - TAKE_PROFIT_PCT / 100)
                place_market_order(session, INSTRUMENT, -units, sl_price, tp_price)
                action = "sell"
                units_placed = -units
                reason = "sell order placed"
            else:
                reason = "already in matching position"

            if action != "none":
                state["trades_today"] += 1
                state["last_trade_time"] = datetime.now(timezone.utc).isoformat()

    save_state(state)

    log_row({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": price,
        "score": score,
        "decision": decision,
        "action": action,
        "units": units_placed,
        "equity": equity,
        "position": current_position,
        "htf_bias": htf_bias,
        "reason": reason,
    })

    print(f"[{datetime.now(timezone.utc).isoformat()}] price={price} score={score} decision={decision} action={action} units={units_placed} reason={reason}")


if __name__ == "__main__":
    main()
