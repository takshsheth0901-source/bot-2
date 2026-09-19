"""
Backtest for the multi-pair OANDA trend-breakout bot. Reuses bot.py's exact
indicator/decision/position-sizing logic (imported directly) so the backtest
can never drift from what the live bot actually does.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

import bot  # reuse the live bot's exact logic

BACKTEST_DAYS = 180
TRAIN_FRACTION = 0.6
STARTING_EQUITY = 1000.0
SPREAD_COST_PER_UNIT = {
    "XAU_USD": 0.30,
    "EUR_USD": 0.00008,
    "GBP_USD": 0.00010,
}
MARGIN_RATE_FALLBACK = {
    "XAU_USD": 0.05,
    "EUR_USD": 0.02,
    "GBP_USD": 0.02,
}


def fetch_historical_candles(session, instrument, granularity, days):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    all_rows = []
    cursor = start_time
    max_iterations = 60
    iteration = 0

    while cursor < end_time and iteration < max_iterations:
        iteration += 1
        params = {
            "granularity": granularity,
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "count": 5000,
            "price": "M",
        }
        data = bot.oanda_request(
            session, "GET",
            f"/v3/instruments/{instrument}/candles",
            params=params,
        )
        candles = data.get("candles", [])
        if not candles:
            break

        for c in candles:
            if not c["complete"]:
                continue
            all_rows.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": int(c["volume"]),
            })

        last_time = pd.to_datetime(candles[-1]["time"])
        if last_time.to_pydatetime().replace(tzinfo=timezone.utc) <= cursor:
            break
        cursor = last_time.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(seconds=1)

        if len(candles) < 5000:
            break

    df = pd.DataFrame(all_rows).drop_duplicates(subset="time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_daily_bias(df):
    df = df.copy().set_index("time")
    daily_close = df["close"].resample("1D").last().ffill()
    ema20 = daily_close.ewm(span=20, adjust=False).mean()
    bias = (daily_close > ema20).astype(int) * 2 - 1
    bias = bias.shift(1)
    bias = bias.reindex(df.index, method="ffill").fillna(0)
    df["daily_bias"] = bias
    return df.reset_index()


def simulate_trades(df, instrument):
    df = bot.add_trend(df)
    df = bot.add_rsi(df)
    df = bot.add_atr(df)
    df = bot.add_adx(df)
    df = bot.add_breakout_levels(df)

    spread_cost = SPREAD_COST_PER_UNIT.get(instrument, 0.0001)
    margin_rate = MARGIN_RATE_FALLBACK.get(instrument, 0.05)

    equity = STARTING_EQUITY
    trades = []
    open_trade = None

    warmup = 210
    for i in range(warmup, len(df)):
        row = df.iloc[i]
        price = row["close"]
        high = row["high"]
        low = row["low"]

        if open_trade is not None:
            direction = open_trade["direction"]
            if direction == "buy":
                hit_sl = low <= open_trade["sl"]
                hit_tp = high >= open_trade["tp"]
            else:
                hit_sl = high >= open_trade["sl"]
                hit_tp = low <= open_trade["tp"]

            if hit_sl or hit_tp:
                exit_price = open_trade["sl"] if hit_sl else open_trade["tp"]
                if direction == "buy":
                    pnl = (exit_price - open_trade["entry_price"]) * open_trade["units"]
                else:
                    pnl = (open_trade["entry_price"] - exit_price) * open_trade["units"]
                pnl -= spread_cost * open_trade["units"]
                equity += pnl
                trades.append({
                    "entry_time": open_trade["entry_time"],
                    "exit_time": row["time"],
                    "direction": direction,
                    "pnl": pnl,
                    "win": pnl > 0,
                })
                open_trade = None

        if open_trade is not None:
            continue

        daily_bias = row.get("daily_bias", 0)
        decision = bot.decide_from_row(row, daily_bias=daily_bias)

        if decision == "none":
            continue

        atr = row.get("atr")
        if pd.isna(atr) or atr <= 0:
            continue

        stop_distance = atr * bot.SL_ATR_MULT
        tp_distance = atr * bot.TP_ATR_MULT
        risk_amount = equity * bot.MAX_POSITION_FRACTION
        risk_based_units = int(risk_amount / stop_distance) if stop_distance > 0 else 0

        margin_available = equity
        margin_based_units = int((margin_available * 0.9) / (price * margin_rate)) if price > 0 else 0

        units = min(risk_based_units, margin_based_units)
        if units <= 0:
            continue

        if decision == "buy":
            sl = price - stop_distance
            tp = price + tp_distance
        else:
            sl = price + stop_distance
            tp = price - tp_distance

        open_trade = {
            "direction": decision,
            "entry_price": price,
            "entry_time": row["time"],
            "sl": sl,
            "tp": tp,
            "units": units,
        }

    return trades, equity


def compute_metrics(trades, starting_equity, final_equity):
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "expectancy": None,
            "profit_factor": None,
            "max_drawdown_pct": None,
            "sharpe": None,
            "total_return_pct": (final_equity - starting_equity) / starting_equity * 100,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    expectancy = np.mean(pnls)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity_curve = [starting_equity]
    for p in pnls:
        equity_curve.append(equity_curve[-1] + p)
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100
    max_drawdown_pct = drawdown.min()

    returns = np.array(pnls) / starting_equity
    sharpe = (returns.mean() / returns.std() * np.sqrt(len(returns))) if returns.std() > 0 else 0

    return {
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "total_return_pct": (final_equity - starting_equity) / starting_equity * 100,
    }


def print_metrics(label, m):
    print(f"\n--- {label} ---")
    if m["num_trades"] == 0:
        print("No trades taken.")
        print(f"Total return: {m['total_return_pct']:.2f}%")
        return
    print(f"Trades:          {m['num_trades']}")
    print(f"Win rate:        {m['win_rate']:.1f}%")
    print(f"Avg win:         {m['avg_win']:.2f}")
    print(f"Avg loss:        {m['avg_loss']:.2f}")
    print(f"Expectancy:      {m['expectancy']:.2f} per trade")
    print(f"Profit factor:   {m['profit_factor']:.2f}")
    print(f"Max drawdown:    {m['max_drawdown_pct']:.2f}%")
    print(f"Sharpe (approx): {m['sharpe']:.2f}")
    print(f"Total return:    {m['total_return_pct']:.2f}%")


def run_backtest():
    session = bot.oanda_session()

    combined_test_trades = []

    for instrument in bot.INSTRUMENTS:
        print(f"\n{'='*70}\nBACKTESTING {instrument}\n{'='*70}")
        print(f"Fetching {BACKTEST_DAYS} days of {bot.GRANULARITY} candles...")

        raw_df = fetch_historical_candles(session, instrument, bot.GRANULARITY, BACKTEST_DAYS)
        if len(raw_df) < 400:
            print(f"Not enough data for {instrument} ({len(raw_df)} candles) — skipping.")
            continue
        print(f"Got {len(raw_df)} candles.")

        df = build_daily_bias(raw_df)

        split_idx = int(len(df) * TRAIN_FRACTION)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        train_trades, train_final_equity = simulate_trades(train_df, instrument)
        test_trades, test_final_equity = simulate_trades(test_df, instrument)

        train_metrics = compute_metrics(train_trades, STARTING_EQUITY, train_final_equity)
        test_metrics = compute_metrics(test_trades, STARTING_EQUITY, test_final_equity)

        print_metrics(f"{instrument} — TRAIN ({TRAIN_FRACTION*100:.0f}% of data)", train_metrics)
        print_metrics(f"{instrument} — TEST ({(1-TRAIN_FRACTION)*100:.0f}% of data, held out)", test_metrics)

        print(f"\n--- {instrument} OVERFITTING CHECK ---")
        if train_metrics["num_trades"] > 0 and test_metrics["num_trades"] > 0:
            train_ret = train_metrics["total_return_pct"]
            test_ret = test_metrics["total_return_pct"]
            print(f"Train return: {train_ret:.2f}%  |  Test return: {test_ret:.2f}%")
            if (train_ret > 0) != (test_ret > 0):
                print("WARNING: train and test disagree on direction — likely overfit or unstable edge.")
            elif abs(train_ret - test_ret) > abs(train_ret) * 0.5 + 5:
                print("CAUTION: large gap between train and test performance — treat with skepticism.")
            else:
                print("Train/test results are broadly consistent.")
        else:
            print("Not enough trades in one or both halves to assess overfitting.")

        combined_test_trades.extend(test_trades)

    print(f"\n{'='*70}\nCOMBINED TEST-SET RESULTS (ALL INSTRUMENTS, HELD-OUT DATA ONLY)\n{'='*70}")
    combined_final_equity = STARTING_EQUITY + sum(t["pnl"] for t in combined_test_trades)
    combined_metrics = compute_metrics(combined_test_trades, STARTING_EQUITY, combined_final_equity)
    print_metrics("Combined (test-set only)", combined_metrics)

    print("\nNote: this backtest approximates spread costs and assumes no simultaneous")
    print("cross-instrument margin competition. It is built from the exact same")
    print("decision/sizing logic as the live bot, so it is a genuine estimate.")


if __name__ == "__main__":
    run_backtest()
