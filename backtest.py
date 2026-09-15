"""
REAL BACKTEST ENGINE

Pulls genuine historical OANDA gold price data and runs it through the
EXACT SAME strategy code as the live bot (imported directly from bot.py,
not reimplemented), simulates trades using the same stop-loss/take-profit/
position-sizing/risk rules as live trading, and reports honest performance
metrics with an out-of-sample train/test split.

Usage:
    Set OANDA_API_TOKEN and OANDA_ACCOUNT_ID as environment variables, then:
    python backtest.py
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import bot   # reuses the live bot's exact indicator/scoring/risk logic


BACKTEST_DAYS = 30
BACKTEST_GRANULARITY = "M5"
TRAIN_FRACTION = 0.6
STARTING_EQUITY = 100000.0


def fetch_historical_candles(session, instrument, granularity, days):
    all_rows = []
    to_time = datetime.utcnow()
    earliest_needed = to_time - timedelta(days=days)
    iterations = 0
    max_iterations = 20

    while to_time > earliest_needed and iterations < max_iterations:
        params = {
            "granularity": granularity,
            "count": 5000,
            "price": "M",
            "to": to_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        }
        data = bot.oanda_request(session, "GET", f"/v3/instruments/{instrument}/candles", params=params)
        candles = data["candles"]
        if not candles:
            break

        for c in candles:
            if not c.get("complete", False):
                continue
            mid = c["mid"]
            all_rows.append({
                "time": c["time"],
                "Open": float(mid["o"]),
                "High": float(mid["h"]),
                "Low": float(mid["l"]),
                "Close": float(mid["c"]),
                "Volume": int(c.get("volume", 0)),
            })

        earliest_in_batch = datetime.strptime(candles[0]["time"][:19], "%Y-%m-%dT%H:%M:%S")
        if earliest_in_batch >= to_time.replace(microsecond=0):
            break
        to_time = earliest_in_batch
        iterations += 1
        time.sleep(0.2)

    df = pd.DataFrame(all_rows).drop_duplicates(subset="time").sort_values("time")
    df = df.set_index("time")
    return df


def build_features_with_htf_bias(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    df = bot.add_indicators(df)
    df = bot.add_candle_patterns(df)
    df = bot.add_bollinger_bands(df)
    df = bot.add_atr(df)
    df = bot.add_volume_signal(df)
    df = bot.add_fibonacci_signal(df)
    df = bot.add_ict_signals(df)
    df = bot.add_adx(df)
    df = bot.add_stochastic(df)
    df = bot.add_cci(df)
    df = bot.add_mfi(df)
    df = bot.add_vwap(df)

    for label, rule in [("D", "1D"), ("H4", "4h"), ("H1", "1h")]:
        htf = df["Close"].resample(rule).last().ffill()
        htf_ema20 = htf.ewm(span=20, adjust=False).mean()
        htf_bull = (htf > htf_ema20).shift(1)
        df[f"htf_bull_{label}"] = htf_bull.reindex(df.index, method="ffill")

    votes = df[["htf_bull_D", "htf_bull_H4", "htf_bull_H1"]]
    all_bull = votes.all(axis=1)
    all_bear = (~votes.fillna(True)).all(axis=1) & votes.notna().all(axis=1)
    df["htf_bias"] = 0
    df.loc[all_bull, "htf_bias"] = 1
    df.loc[all_bear, "htf_bias"] = -1

    df["in_session"] = [bot.in_active_session(str(t)) for t in df.index]

    df["score"] = df.apply(
        lambda row: bot.score_row(row, htf_bias=row["htf_bias"], in_session=row["in_session"]),
        axis=1
    )
    return df


def simulate_trades(df, starting_equity=STARTING_EQUITY):
    equity = starting_equity
    position = 0.0
    entry_price = None
    entry_time = None
    stop_loss = None
    take_profit = None
    trades = []
    equity_curve = []

    trades_today = 0
    current_day = None
    last_trade_bar = -999
    starting_equity_today = equity

    for i, (ts, row) in enumerate(df.iterrows()):
        day = ts.date()
        if day != current_day:
            current_day = day
            trades_today = 0
            starting_equity_today = equity

        price = row["Close"]

        if position != 0:
            hit_sl = hit_tp = False
            if position > 0:
                hit_sl = row["Low"] <= stop_loss
                hit_tp = row["High"] >= take_profit
            else:
                hit_sl = row["High"] >= stop_loss
                hit_tp = row["Low"] <= take_profit

            exit_price = None
            outcome = None
            if hit_sl and hit_tp:
                exit_price = stop_loss
                outcome = "loss"
            elif hit_sl:
                exit_price = stop_loss
                outcome = "loss"
            elif hit_tp:
                exit_price = take_profit
                outcome = "win"

            if exit_price is not None:
                pnl = (exit_price - entry_price) * position
                equity += pnl
                trades.append({
                    "entry_time": entry_time, "exit_time": ts,
                    "direction": "long" if position > 0 else "short",
                    "entry_price": entry_price, "exit_price": exit_price,
                    "units": abs(position), "pnl": pnl, "outcome": outcome,
                })
                position = 0.0
                entry_price = stop_loss = take_profit = None

        if starting_equity_today > 0:
            drawdown_today = (starting_equity_today - equity) / starting_equity_today
        else:
            drawdown_today = 0
        daily_limit_hit = drawdown_today >= bot.DAILY_LOSS_LIMIT_FRACTION

        cooldown_bars = max(1, bot.COOLDOWN_SECONDS_AFTER_TRADE // (5 * 60))
        in_cooldown = (i - last_trade_bar) < cooldown_bars

        if (position == 0 and not daily_limit_hit and not in_cooldown
                and trades_today < bot.MAX_TRADES_PER_DAY):
            score = row["score"]
            if score >= bot.BUY_SCORE_THRESHOLD:
                cash_to_risk = equity * bot.MAX_POSITION_FRACTION
                units = int(cash_to_risk // price)
                if units > 0:
                    position = units
                    entry_price = price
                    entry_time = ts
                    stop_loss = price * (1 - bot.STOP_LOSS_PCT / 100)
                    take_profit = price * (1 + bot.TAKE_PROFIT_PCT / 100)
                    trades_today += 1
                    last_trade_bar = i
            elif score <= bot.SELL_SCORE_THRESHOLD and bot.ALLOW_SHORTING:
                cash_to_risk = equity * bot.MAX_POSITION_FRACTION
                units = int(cash_to_risk // price)
                if units > 0:
                    position = -units
                    entry_price = price
                    entry_time = ts
                    stop_loss = price * (1 + bot.STOP_LOSS_PCT / 100)
                    take_profit = price * (1 - bot.TAKE_PROFIT_PCT / 100)
                    trades_today += 1
                    last_trade_bar = i

        equity_curve.append({"time": ts, "equity": equity})

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


def compute_metrics(trades_df, equity_df, starting_equity):
    if trades_df.empty:
        return {
            "num_trades": 0, "win_rate": None, "avg_win": None, "avg_loss": None,
            "expectancy": None, "profit_factor": None, "max_drawdown_pct": None,
            "sharpe": None, "final_equity": starting_equity, "return_pct": 0.0,
        }

    wins = trades_df[trades_df["pnl"] > 0]["pnl"]
    losses = trades_df[trades_df["pnl"] <= 0]["pnl"]

    win_rate = len(wins) / len(trades_df) * 100
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    expectancy = trades_df["pnl"].mean()
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    equity_series = equity_df["equity"]
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100

    trade_returns = trades_df["pnl"] / starting_equity
    sharpe = (trade_returns.mean() / trade_returns.std() * np.sqrt(len(trade_returns))
              if trade_returns.std() > 0 and len(trade_returns) > 1 else 0)

    final_equity = equity_series.iloc[-1]
    return_pct = (final_equity / starting_equity - 1) * 100

    return {
        "num_trades": len(trades_df), "win_rate": win_rate, "avg_win": avg_win,
        "avg_loss": avg_loss, "expectancy": expectancy, "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct, "sharpe": sharpe,
        "final_equity": final_equity, "return_pct": return_pct,
    }


def print_report(label, metrics):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    if metrics["num_trades"] == 0:
        print("  No trades taken in this period.")
        return
    print(f"  Number of trades:    {metrics['num_trades']}")
    print(f"  Win rate:            {metrics['win_rate']:.1f}%")
    print(f"  Average win:         {metrics['avg_win']:.2f}")
    print(f"  Average loss:        {metrics['avg_loss']:.2f}")
    print(f"  Expectancy/trade:    {metrics['expectancy']:.2f}")
    print(f"  Profit factor:       {metrics['profit_factor']:.2f}")
    print(f"  Max drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe (per-trade):  {metrics['sharpe']:.2f}")
    print(f"  Final equity:        {metrics['final_equity']:.2f}")
    print(f"  Total return:        {metrics['return_pct']:.2f}%")


def run_backtest():
    if not bot.OANDA_API_TOKEN or not bot.OANDA_ACCOUNT_ID:
        print("ERROR: OANDA_API_TOKEN / OANDA_ACCOUNT_ID not set.")
        return

    session = bot.oanda_session()
    print(f"Fetching {BACKTEST_DAYS} days of {BACKTEST_GRANULARITY} gold data from OANDA...")
    df = fetch_historical_candles(session, bot.INSTRUMENT, BACKTEST_GRANULARITY, BACKTEST_DAYS)
    print(f"Got {len(df)} candles, from {df.index.min()} to {df.index.max()}")

    print("Computing indicators and scores (same logic as the live bot)...")
    feat = build_features_with_htf_bias(df)
    feat = feat.dropna(subset=["RSI", "MACD_signal", "BB_percent_b", "ATR_pct"])

    split_idx = int(len(feat) * TRAIN_FRACTION)
    train, test = feat.iloc[:split_idx], feat.iloc[split_idx:]

    print(f"\nTrain period: {train.index.min()} to {train.index.max()} ({len(train)} bars)")
    print(f"Test period:  {test.index.min()} to {test.index.max()} ({len(test)} bars)")

    train_trades, train_equity = simulate_trades(train)
    test_trades, test_equity = simulate_trades(test)

    train_metrics = compute_metrics(train_trades, train_equity, STARTING_EQUITY)
    test_metrics = compute_metrics(test_trades, test_equity, STARTING_EQUITY)

    print_report("TRAINING PERIOD (seen during tuning)", train_metrics)
    print_report("TEST PERIOD (out-of-sample - the honest number)", test_metrics)

    print(f"\n{'='*50}")
    print("  OVERFITTING CHECK")
    print(f"{'='*50}")
    if train_metrics["num_trades"] > 0 and test_metrics["num_trades"] > 0:
        gap = train_metrics["return_pct"] - test_metrics["return_pct"]
        print(f"  Train return: {train_metrics['return_pct']:.2f}%  |  Test return: {test_metrics['return_pct']:.2f}%")
        if gap > 15:
            print("  WARNING: training performance is much stronger than out-of-sample.")
            print("  This is a classic overfitting sign - treat the strategy with caution.")
        else:
            print("  Train/test performance is reasonably close - a healthier sign,")
            print("  though this is still a limited sample, not a guarantee.")
    else:
        print("  Not enough trades in one or both periods to assess overfitting.")
        print("  This itself is informative: the strategy may be too selective")
        print("  for this data window, or the market didn't offer qualifying setups.")

    if test_metrics["num_trades"] > 0 and test_metrics["num_trades"] < 20:
        print(f"\n  NOTE: only {test_metrics['num_trades']} out-of-sample trades.")
        print("  That's too small a sample to draw strong conclusions either way.")


if __name__ == "__main__":
    run_backtest()
