"""
Solana memecoin ("pump.fun") scanner + guarded-entry bot.

READ THIS FIRST — honest framing, not a sales pitch:

1. THIS IS NOT A TRUE "SNIPE". GitHub Actions' free tier can only run a job on
   a schedule as tight as every 5 minutes. Real sniping bots react in
   milliseconds using private low-latency infrastructure and compete directly
   against professional MEV bots. On a 5-minute loop, by the time this bot
   sees a new token, dozens of faster bots have already been in and out. What
   this bot actually does is: scan recently-launched tokens, apply safety
   filters, and take a very small, capped-risk position on ones that still
   look reasonable a few minutes after launch. That is a fundamentally
   different (slower, higher-information, still high-risk) activity than
   "sniping" — call it what it is when you think about expected outcomes.

2. New memecoin tokens fail overwhelmingly often — rug pulls, abandoned
   liquidity, and straight scams are the norm, not the exception, even with
   filters. Treat every dollar this bot risks as money you have already lost.

3. WALLET SECURITY — non-negotiable:
   - Create a BRAND NEW Phantom wallet used for nothing else ("burner
     wallet"). Never use your main wallet.
   - Fund it with only the total amount you are fully prepared to lose.
   - Export ONLY that burner wallet's private key (Phantom: Settings ->
     Manage Wallets -> select the burner -> Show Private Key -> it gives you
     a base58 string). Store it ONLY as a GitHub Actions secret
     (SOLANA_WALLET_PRIVATE_KEY). Never paste it into this chat, into a file
     that gets committed, or anywhere else.
   - If that key is ever exposed, it is compromised permanently — move any
     remaining funds out immediately and stop using that wallet.

4. THIS CODE HAS NOT EXECUTED A REAL TRANSACTION. It has not been tested
   against a live wallet because this sandbox cannot reach Solana RPC
   endpoints. Before funding it with real money: run it once with
   DRY_RUN = True (default) and read the logs carefully. Then fund the
   wallet with the smallest amount you can (e.g. 0.05-0.1 SOL total) and
   watch it for several days before trusting it with more.

Architecture:
  - Scan: pulls recently-launched pump.fun tokens from a launch-data API
    (Solana Tracker's public Pump.fun API — you need a free API key, see
    setup notes below).
  - Filter: age since launch, minimum liquidity, and a few other cheap
    sanity checks — NOT a guarantee against rugs, just a floor.
  - Execute: buys/sells are routed through Jupiter's swap API, which
    aggregates pump.fun/PumpSwap/Raydium routing and returns a ready-to-sign
    transaction, rather than hand-building raw pump.fun bonding-curve
    instructions (far more error-prone with real money on the line).
  - Position tracking: a small JSON state file tracks open positions across
    runs (since the process restarts every 5 minutes on GitHub Actions) and
    checks each one for a stop-loss / take-profit / max-hold-time exit.
"""

import base64
import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — all secrets come from environment variables (GitHub Secrets)
# ══════════════════════════════════════════════════════════════════════════

DRY_RUN = True  # MUST be flipped to False deliberately once you've reviewed logs

SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL", "")           # e.g. a Helius/QuickNode/Alchemy endpoint
WALLET_PRIVATE_KEY_B58 = os.environ.get("SOLANA_WALLET_PRIVATE_KEY", "")
SOLANA_TRACKER_API_KEY = os.environ.get("SOLANA_TRACKER_API_KEY", "")

JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
SOL_MINT = "So11111111111111111111111111111111111111112"

STATE_FILE = "sniper_state.json"
LOG_FILE = "sniper_activity.csv"

# --- risk controls (deliberately tiny) ---
POSITION_SIZE_SOL = 0.02          # SOL risked per new position — keep this small
MAX_CONCURRENT_POSITIONS = 2
MAX_DAILY_SPEND_SOL = 0.10        # hard cap across all positions opened today
SLIPPAGE_BPS = 500                # 5% — memecoin liquidity is thin and volatile

# --- entry filters (a floor, not a guarantee) ---
MIN_TOKEN_AGE_SECONDS = 90        # skip anything younger than this (avoid instant micro-rugs)
MAX_TOKEN_AGE_SECONDS = 600       # skip anything we're already this late to
MIN_LIQUIDITY_USD = 3000
MIN_MARKET_CAP_USD = 5000

# --- exit rules ---
TAKE_PROFIT_PCT = 60              # close if position is up this much
STOP_LOSS_PCT = 35                # close if position is down this much
MAX_HOLD_MINUTES = 120            # force-close regardless of P/L after this long


# ══════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════

def load_state():
    default = {"positions": [], "spent_today_sol": 0.0, "day": None}
    if not os.path.exists(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.get("day") != today:
            state["spent_today_sol"] = 0.0
            state["day"] = today
        return state
    except Exception:
        return default


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log_row(row):
    file_exists = os.path.exists(LOG_FILE)
    import csv
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════
# SCANNING NEW LAUNCHES (Solana Tracker Pump.fun API)
# https://docs.solanatracker.io/  — sign up for a free API key
# ══════════════════════════════════════════════════════════════════════════

def fetch_recent_launches():
    """Returns a list of dicts: {mint, symbol, created_at, liquidity_usd, market_cap_usd}."""
    if not SOLANA_TRACKER_API_KEY:
        raise RuntimeError("SOLANA_TRACKER_API_KEY is not set — sign up at solanatracker.io for a free key.")

    headers = {"x-api-key": SOLANA_TRACKER_API_KEY}
    resp = requests.get(
        "https://data.solanatracker.io/tokens/latest",
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    launches = []
    for item in data if isinstance(data, list) else data.get("tokens", []):
        try:
            pool = (item.get("pools") or [{}])[0]
            launches.append({
                "mint": item["token"]["mint"],
                "symbol": item["token"].get("symbol", "?"),
                "created_at": item["token"].get("createdOn") or item.get("createdAt"),
                "liquidity_usd": float(pool.get("liquidity", {}).get("usd", 0) or 0),
                "market_cap_usd": float(pool.get("marketCap", {}).get("usd", 0) or 0),
            })
        except Exception:
            continue
    return launches


def passes_filters(token, now):
    created_raw = token.get("created_at")
    if not created_raw:
        return False, "no_timestamp"
    try:
        created = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
    except Exception:
        return False, "bad_timestamp"

    age_seconds = (now - created).total_seconds()
    if age_seconds < MIN_TOKEN_AGE_SECONDS:
        return False, "too_new"
    if age_seconds > MAX_TOKEN_AGE_SECONDS:
        return False, "too_old"
    if token["liquidity_usd"] < MIN_LIQUIDITY_USD:
        return False, "low_liquidity"
    if token["market_cap_usd"] < MIN_MARKET_CAP_USD:
        return False, "low_market_cap"
    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════
# JUPITER SWAP EXECUTION
# ══════════════════════════════════════════════════════════════════════════

def get_jupiter_quote(input_mint, output_mint, amount_lamports, slippage_bps=SLIPPAGE_BPS):
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": amount_lamports,
        "slippageBps": slippage_bps,
    }
    resp = requests.get(JUPITER_QUOTE_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def build_and_send_swap(quote, wallet_pubkey):
    """
    Builds a signed swap transaction from a Jupiter quote and submits it.
    Requires the `solders` package (pip install solders) for keypair/tx handling.
    """
    from solders.keypair import Keypair
    from solders.transaction import VersionedTransaction
    import base58

    swap_resp = requests.post(
        JUPITER_SWAP_URL,
        json={
            "quoteResponse": quote,
            "userPublicKey": wallet_pubkey,
            "wrapAndUnwrapSol": True,
            "prioritizationFeeLamports": "auto",
        },
        timeout=20,
    )
    swap_resp.raise_for_status()
    swap_tx_b64 = swap_resp.json()["swapTransaction"]

    raw_tx = VersionedTransaction.from_bytes(base64.b64decode(swap_tx_b64))
    keypair = Keypair.from_bytes(base58.b58decode(WALLET_PRIVATE_KEY_B58))
    signed_tx = VersionedTransaction(raw_tx.message, [keypair])

    rpc_resp = requests.post(
        SOLANA_RPC_URL,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [
                base64.b64encode(bytes(signed_tx)).decode("utf-8"),
                {"encoding": "base64", "skipPreflight": False, "maxRetries": 3},
            ],
        },
        timeout=20,
    )
    rpc_resp.raise_for_status()
    result = rpc_resp.json()
    if "error" in result:
        raise RuntimeError(f"RPC error: {result['error']}")
    return result["result"]  # transaction signature


def get_wallet_pubkey():
    from solders.keypair import Keypair
    import base58
    keypair = Keypair.from_bytes(base58.b58decode(WALLET_PRIVATE_KEY_B58))
    return str(keypair.pubkey())


def get_token_price_usd(mint):
    """Quote 1 SOL -> token to derive an approximate price, via Jupiter."""
    quote = get_jupiter_quote(SOL_MINT, mint, 1_000_000_000)  # 1 SOL in lamports
    out_amount = float(quote["outAmount"])
    return out_amount  # tokens per SOL; used only for relative P/L tracking, not absolute USD


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

def main():
    now = datetime.now(timezone.utc)
    state = load_state()

    if not DRY_RUN:
        if not SOLANA_RPC_URL or not WALLET_PRIVATE_KEY_B58:
            print("Missing SOLANA_RPC_URL or SOLANA_WALLET_PRIVATE_KEY — cannot run live. Exiting.")
            return
        wallet_pubkey = get_wallet_pubkey()
    else:
        wallet_pubkey = "DRY_RUN_NO_WALLET"

    # --- 1. Manage existing positions: check exits ---
    still_open = []
    for pos in state["positions"]:
        opened_at = datetime.fromisoformat(pos["opened_at"])
        hold_minutes = (now - opened_at).total_seconds() / 60

        try:
            tokens_per_sol_now = get_token_price_usd(pos["mint"])
            # price went UP if it now takes MORE tokens to buy 1 SOL... no —
            # fewer tokens per SOL means the token got more expensive.
            pct_change = (pos["tokens_per_sol_at_entry"] - tokens_per_sol_now) / pos["tokens_per_sol_at_entry"] * 100
        except Exception as e:
            print(f"Could not price {pos['symbol']}: {e}")
            still_open.append(pos)
            continue

        exit_reason = None
        if pct_change >= TAKE_PROFIT_PCT:
            exit_reason = "take_profit"
        elif pct_change <= -STOP_LOSS_PCT:
            exit_reason = "stop_loss"
        elif hold_minutes >= MAX_HOLD_MINUTES:
            exit_reason = "max_hold_time"

        if exit_reason:
            print(f"Closing {pos['symbol']} ({exit_reason}, {pct_change:.1f}%)")
            if not DRY_RUN:
                try:
                    quote = get_jupiter_quote(pos["mint"], SOL_MINT, pos["token_amount"])
                    sig = build_and_send_swap(quote, wallet_pubkey)
                    print(f"Sell tx: {sig}")
                except Exception as e:
                    print(f"SELL FAILED for {pos['symbol']}: {e} — leaving position open, will retry next run.")
                    still_open.append(pos)
                    continue
            log_row({
                "time": now.isoformat(), "action": "sell", "symbol": pos["symbol"],
                "mint": pos["mint"], "reason": exit_reason, "pct_change": round(pct_change, 2),
                "dry_run": DRY_RUN,
            })
        else:
            still_open.append(pos)

    state["positions"] = still_open

    # --- 2. Look for new entries, if we have room ---
    if len(state["positions"]) < MAX_CONCURRENT_POSITIONS and state["spent_today_sol"] < MAX_DAILY_SPEND_SOL:
        try:
            launches = fetch_recent_launches()
        except Exception as e:
            print(f"Scan failed: {e}")
            launches = []

        held_mints = {p["mint"] for p in state["positions"]}

        for token in launches:
            if token["mint"] in held_mints:
                continue
            ok, reason = passes_filters(token, now)
            log_row({
                "time": now.isoformat(), "action": "scan", "symbol": token["symbol"],
                "mint": token["mint"], "reason": reason, "liquidity_usd": token["liquidity_usd"],
                "market_cap_usd": token["market_cap_usd"], "dry_run": DRY_RUN,
            })
            if not ok:
                continue
            if len(state["positions"]) >= MAX_CONCURRENT_POSITIONS:
                break
            if state["spent_today_sol"] + POSITION_SIZE_SOL > MAX_DAILY_SPEND_SOL:
                break

            print(f"Entering {token['symbol']} ({token['mint']})")
            amount_lamports = int(POSITION_SIZE_SOL * 1_000_000_000)

            if not DRY_RUN:
                try:
                    quote = get_jupiter_quote(SOL_MINT, token["mint"], amount_lamports)
                    sig = build_and_send_swap(quote, wallet_pubkey)
                    token_amount = int(quote["outAmount"])
                    print(f"Buy tx: {sig}")
                except Exception as e:
                    print(f"BUY FAILED for {token['symbol']}: {e}")
                    continue
            else:
                token_amount = 1  # placeholder in dry run

            tokens_per_sol_at_entry = get_token_price_usd(token["mint"]) if not DRY_RUN else 1.0

            state["positions"].append({
                "mint": token["mint"],
                "symbol": token["symbol"],
                "opened_at": now.isoformat(),
                "token_amount": token_amount,
                "tokens_per_sol_at_entry": tokens_per_sol_at_entry,
                "sol_spent": POSITION_SIZE_SOL,
            })
            state["spent_today_sol"] += POSITION_SIZE_SOL

            log_row({
                "time": now.isoformat(), "action": "buy", "symbol": token["symbol"],
                "mint": token["mint"], "reason": "entered", "sol_spent": POSITION_SIZE_SOL,
                "dry_run": DRY_RUN,
            })

    save_state(state)
    print(f"Run complete. Open positions: {len(state['positions'])}. Spent today: {state['spent_today_sol']:.3f} SOL.")


if __name__ == "__main__":
    main()
