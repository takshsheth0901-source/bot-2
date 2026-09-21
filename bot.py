"""
Automated multi-pair trend-breakout trading bot for OANDA (Gold + EUR/USD + GBP/USD).
Runs once per invocation — intended to be triggered every 5 minutes by GitHub Actions.

STRATEGY (rewritten from scratch — simple trend-following breakout, not a
multi-indicator voting scheme):
  - Only trade WITH the trend: EMA50 vs EMA200, confirmed by ADX > ADX_MIN
    (avoids trading in a flat/choppy market).
  - Entry: price breaks out beyond its recent N-candle high/low in the
    trend's direction — a real structural signal, not indicator noise.
  - Stop-loss and take-profit are both based on ATR (current volatility),
    not a fixed tiny percentage — this avoids getting stopped out by normal
    noise, and keeps a healthy ~2:1 reward:risk ratio (TP_ATR_MULT / SL_ATR_MULT).
  - A daily-timeframe bias filter must not contradict the trade direction.
  - Fewer, higher-quality trades — not a high-frequency scalp count.
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

INSTRUMENTS = ["XAU_USD", "EUR_USD", "GBP_USD"]
GRANULARITY = "M15"        # 15-minute candles — enough structure for real breakouts
CANDLE_COUNT = 500

# ── Strategy parameters ──
BREAKOUT_LOOKBACK = 20      # N-candle high/low breakout
ADX_MIN = 20                # minimum trend strength to trade at all
RSI_UPPER = 75              # skip new longs if RSI already this overbought
RSI_LOWER = 25              # skip new shorts if RSI already this oversold
SL_ATR_MULT = 1.5           # stop-loss = 1.5x ATR
TP_ATR_MULT = 3.0           # take-profit = 3.0x ATR  (~2:1 reward:risk)

ALLOW_SHORTING = True

MAX_TRADES_PER_DAY = 30                # shared across all instruments
MAX_POSITION_FRACTION = 0.05           # 5% of equity RISKED (lost if SL hit) per trade
DAILY_LOSS_LIMIT_FRACTION = 0.10       # shared across all instruments (£100 on a £1000 account)
COOLDOWN_SECONDS_AFTER_TRADE = 120     # per-instrument cooldown
KILL_SWITCH_FILE = "STOP"

STATE_FILE = "bot_state_v2.csv"
ACTIVITY_FILE = "bot_activity_v2.csv"

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


def place_market_order(session, instrument, units, stop_loss_price=None, take_profit_price=None, price_precision=2):
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
        order["order"]["stopLossOnFill"] = {"price": f"{stop_loss_price:.{price_precision}f}"}
    if take_profit_price is not None:
        order["order"]["takeProfitOnFill"] = {"price": f"{take_profit_price:.{price_precision}f}"}

    return oanda_request(
        session, "POST",
        f"/v3/accounts/{OANDA_ACCOUNT_ID}/orders",
        data=json.dumps(order),
    )


def get_instruments_details(session, instruments):
    """Returns {instrument: {marginRate: float, displayPrecision: int}} for all requested instruments in one call."""
    details = {}
    try:
        data = oanda_request(
            session, "GET",
            f"/v3/accounts/{OANDA_ACCOUNT_ID}/instruments",
            params={"instruments": ",".join(instruments)},
        )
        for inst in data.get("instruments", []):
            details[inst["name"]] = {
                "marginRate": float(inst.get("marginRate", 0.05)),
                "displayPrecision": int(inst.get("displayPrecision", 5)),
            }
    except Exception:
        pass
    for instrument in instruments:
        if instrument not in details:
            details[instrument] = {"marginRate": 0.05, "displayPrecision": 5}
    return details


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
# INDICATORS (kept minimal and purposeful — no indicator-zoo voting)
# ══════════════════════════════════════════════════════════════════════════

def add_trend(df):
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    return df


def add_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)
    return df


def add_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / period, min_periods=period).mean()
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
    return df


def add_breakout_levels(df, lookback=BREAKOUT_LOOKBACK):
    # prior N-candle high/low, EXCLUDING the current candle (shift(1)) to avoid look-ahead
    df["breakout_high"] = df["high"].rolling(lookback).max().shift(1)
    df["breakout_low"] = df["low"].rolling(lookback).min().shift(1)
    return df


def get_htf_bias(session, instrument):
    """Daily-timeframe bias: +1 if price above daily EMA20, -1 if below, 0 if unclear."""
    try:
        df = get_candles(session, instrument, "D", 60)
        if len(df) < 20:
            return 0
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        return 1 if df["close"].iloc[-1] > ema20.iloc[-1] else -1
    except Exception:
        return 0


def in_active_session():
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    london = 7 <= hour <= 16
    ny = 12 <= hour <= 21
    return london or ny


# ── NEWS FILTER ─────────────────────────────────────────────────────────

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
        return []  # fail open — never block trading due to a feed outage


def in_news_blackout(news_events):
    now_utc = datetime.now(timezone.utc)
    for ev in news_events:
        window_start = ev["time"] - timedelta(minutes=NEWS_BLACKOUT_MINUTES_BEFORE)
        window_end = ev["time"] + timedelta(minutes=NEWS_BLACKOUT_MINUTES_AFTER)
        if window_start <= now_utc <= window_end:
            return True, ev
    return False, None


# ══════════════════════════════════════════════════════════════════════════
# DECISION LOGIC
# ══════════════════════════════════════════════════════════════════════════

def decide_from_row(row, daily_bias=0):
    """Trend + breakout + strength filter. Returns 'buy' / 'sell' / 'none'."""
    required = ["ema50", "ema200", "adx", "rsi", "atr", "breakout_high", "breakout_low", "close"]
    for col in required:
        if pd.isna(row.get(col, np.nan)):
            return "none"

    if row["adx"] < ADX_MIN:
        return "none"  # market isn't trending enough — sit out

    trend_up = row["close"] > row["ema50"] > row["ema200"]
    trend_down = row["close"] < row["ema50"] < row["ema200"]

    if trend_up and daily_bias >= 0:
        if row["close"] > row["breakout_high"] and row["rsi"] < RSI_UPPER:
            return "buy"

    if trend_down and daily_bias <= 0:
        if row["close"] < row["breakout_low"] and row["rsi"] > RSI_LOWER:
            return "sell"

    return "none"


def latest_decision(df, daily_bias=0):
    df = add_trend(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_adx(df)
    df = add_breakout_levels(df)

    last_row = df.iloc[-1]
    decision = decide_from_row(last_row, daily_bias=daily_bias)

    def safe_float(val):
        return float(val) if not pd.isna(val) else None

    return {
        "decision": decision,
        "price": float(last_row["close"]),
        "atr": safe_float(last_row.get("atr", np.nan)),
        "adx": safe_float(last_row.get("adx", np.nan)),
        "rsi": safe_float(last_row.get("rsi", np.nan)),
        "ema50": safe_float(last_row.get("ema50", np.nan)),
        "ema200": safe_float(last_row.get("ema200", np.nan)),
        "breakout_high": safe_float(last_row.get("breakout_high", np.nan)),
        "breakout_low": safe_float(last_row.get("breakout_low", np.nan)),
    }


# ══════════════════════════════════════════════════════════════════════════
# STATE / RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

def load_state():
    default = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "trades_today": 0,
        "start_of_day_equity": None,
        "last_trade_time_by_instrument": {},
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
            last_trade_json = row.get("last_trade_time_by_instrument")
            if not date_val or trades_val is None:
                return default
            try:
                last_trade_dict = json.loads(last_trade_json) if last_trade_json else {}
            except Exception:
                last_trade_dict = {}
            return {
                "date": date_val,
                "trades_today": int(trades_val),
                "start_of_day_equity": float(equity_val) if equity_val else None,
                "last_trade_time_by_instrument": last_trade_dict,
            }
    except Exception:
        return default


def save_state(state):
    file_exists = os.path.exists(STATE_FILE)
    with open(STATE_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "trades_today", "start_of_day_equity", "last_trade_time_by_instrument"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "date": state["date"],
            "trades_today": state["trades_today"],
            "start_of_day_equity": state["start_of_day_equity"],
            "last_trade_time_by_instrument": json.dumps(state["last_trade_time_by_instrument"]),
        })


def approve_trade(state, equity, instrument):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if state["date"] != today:
        state = {
            "date": today,
            "trades_today": 0,
            "start_of_day_equity": equity,
            "last_trade_time_by_instrument": {},
        }

    if state["start_of_day_equity"] is None:
        state["start_of_day_equity"] = equity

    if os.path.exists(KILL_SWITCH_FILE):
        return False, state, "kill switch active"

    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        return False, state, "daily trade limit reached"

    last_trade_iso = state["last_trade_time_by_instrument"].get(instrument)
    if last_trade_iso:
        last_time = datetime.fromisoformat(last_trade_iso)
        elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
        if elapsed < COOLDOWN_SECONDS_AFTER_TRADE:
            return False, state, "cooldown active for this pair"

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
# PER-INSTRUMENT PROCESSING
# ══════════════════════════════════════════════════════════════════════════

def process_instrument(session, instrument, instrument_details, state, news_events):
    account = get_account_summary(session)
    equity = account["nav"]
    margin_available = account["margin_available"]

    df = get_candles(session, instrument, GRANULARITY, CANDLE_COUNT)
    if len(df) < 210:  # need enough for EMA200 to be meaningful
        log_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instrument": instrument,
            "price": None,
            "decision": "none",
            "action": "none",
            "units": 0,
            "equity": equity,
            "position": None,
            "daily_bias": None,
            "reason": "not enough candle data yet",
            "adx": None,
            "rsi": None,
            "ema50": None,
            "ema200": None,
            "breakout_high": None,
            "breakout_low": None,
        })
        return state

    daily_bias = get_htf_bias(session, instrument)
    result = latest_decision(df, daily_bias=daily_bias)

    decision = result["decision"]
    price = result["price"]
    atr = result["atr"]
    diag_adx = result["adx"]
    diag_rsi = result["rsi"]
    diag_ema50 = result["ema50"]
    diag_ema200 = result["ema200"]
    diag_breakout_high = result["breakout_high"]
    diag_breakout_low = result["breakout_low"]

    current_position = get_open_position(session, instrument)
    blackout_active, blackout_event = in_news_blackout(news_events)

    action = "none"
    reason = ""
    units_placed = 0

    margin_rate = instrument_details.get(instrument, {}).get("marginRate", 0.05)
    price_precision = instrument_details.get(instrument, {}).get("displayPrecision", 5)

    if decision == "none":
        reason = "no signal"
    elif blackout_active:
        reason = f"news blackout: {blackout_event['title']}"
    elif decision == "sell" and not ALLOW_SHORTING:
        reason = "shorting disabled"
    elif atr is None or atr <= 0:
        reason = "invalid ATR, skipping"
    else:
        approved, state, approve_reason = approve_trade(state, equity, instrument)
        if not approved:
            reason = approve_reason
        else:
            stop_distance = atr * SL_ATR_MULT
            tp_distance = atr * TP_ATR_MULT
            risk_amount = equity * MAX_POSITION_FRACTION
            risk_based_units = int(risk_amount / stop_distance) if stop_distance > 0 else 0

            margin_safety_buffer = 0.9
            margin_based_units = int((margin_available * margin_safety_buffer) / (price * margin_rate)) if price > 0 and margin_rate > 0 else 0

            units = min(risk_based_units, margin_based_units)
            was_margin_capped = margin_based_units < risk_based_units

            if units <= 0:
                if was_margin_capped:
                    reason = f"computed 0 units — insufficient margin (available={margin_available:.2f}, needed for {risk_based_units} units)"
                else:
                    reason = "computed 0 units (risk_amount too small vs stop_distance)"
            elif decision == "buy" and current_position <= 0:
                sl_price = price - stop_distance
                tp_price = price + tp_distance
                order_response = place_market_order(session, instrument, units, sl_price, tp_price, price_precision)
                if "orderFillTransaction" in order_response:
                    action = "buy"
                    units_placed = units
                    reason = "buy order FILLED" + (" (margin-capped size)" if was_margin_capped else "")
                elif "orderCancelTransaction" in order_response:
                    cancel_reason = order_response["orderCancelTransaction"].get("reason", "UNKNOWN")
                    reason = f"buy order REJECTED by OANDA: {cancel_reason}"
                else:
                    reason = f"buy order response unclear: {json.dumps(order_response)[:300]}"
            elif decision == "sell" and current_position >= 0:
                sl_price = price + stop_distance
                tp_price = price - tp_distance
                order_response = place_market_order(session, instrument, -units, sl_price, tp_price, price_precision)
                if "orderFillTransaction" in order_response:
                    action = "sell"
                    units_placed = -units
                    reason = "sell order FILLED" + (" (margin-capped size)" if was_margin_capped else "")
                elif "orderCancelTransaction" in order_response:
                    cancel_reason = order_response["orderCancelTransaction"].get("reason", "UNKNOWN")
                    reason = f"sell order REJECTED by OANDA: {cancel_reason}"
                else:
                    reason = f"sell order response unclear: {json.dumps(order_response)[:300]}"
            else:
                reason = "already in matching position"

            if action != "none":
                state["trades_today"] += 1
                state["last_trade_time_by_instrument"][instrument] = datetime.now(timezone.utc).isoformat()

    log_row({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instrument": instrument,
        "price": price,
        "decision": decision,
        "action": action,
        "units": units_placed,
        "equity": equity,
        "position": current_position,
        "daily_bias": daily_bias,
        "reason": reason,
        "adx": diag_adx,
        "rsi": diag_rsi,
        "ema50": diag_ema50,
        "ema200": diag_ema200,
        "breakout_high": diag_breakout_high,
        "breakout_low": diag_breakout_low,
    })

    print(f"[{datetime.now(timezone.utc).isoformat()}] {instrument} price={price} decision={decision} action={action} units={units_placed} reason={reason}")

    return state


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    if not OANDA_API_TOKEN or not OANDA_ACCOUNT_ID:
        print("Missing OANDA_API_TOKEN or OANDA_ACCOUNT_ID environment variables.")
        sys.exit(1)

    session = oanda_session()

    instrument_details = get_instruments_details(session, INSTRUMENTS)
    news_events = fetch_news_events()
    state = load_state()

    for instrument in INSTRUMENTS:
        state = process_instrument(session, instrument, instrument_details, state, news_events)

    save_state(state)


if __name__ == "__main__":
    main()
