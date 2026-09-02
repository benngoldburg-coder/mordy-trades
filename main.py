import os
import json
import logging
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

load_dotenv(override=True)

from market_scanner import scan_market
from discord_scraper import get_discord_signals
from claude_agent import analyze
from trader import execute_pick, monitor_positions, close_eod_positions, get_daily_trades, reset_daily_trades, get_client, get_account_info, is_market_open
from report import generate_eod_report
from notifier import send_sms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

DISCORD_CHANNEL_ID = os.environ["DISCORD_CHANNEL_ID"]
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]

# Backtested 2026-09-02 over 252 sessions / 1,415 trades: the EARLY bucket's mean
# excess return is -0.098% with the live bracket and -0.101% held to the close, at
# a day-level t of -1.69. That is no edge — a negative point estimate that cannot
# be distinguished from zero — and the near-identical bracketed and unbracketed
# figures say the entry is what fails, so no exit redesign rescues it. See
# backtest.py, whose pass criteria were fixed before it was first run.
#
# The scanner fix that made this measurable is worth deploying on its own: it
# repairs arithmetic that was silently wrong and turns a mute failure into a loud
# one. Trading on the result is not. So the scan still runs and still reports;
# it just cannot reach the broker until there is a signal worth trading.
EARLY_TRADING_ENABLED = os.getenv("EARLY_TRADING_ENABLED", "0") == "1"


def monitor_cycle():
    """Fast exit check between full cycles — a -2% stop checked every 30 min
    is routinely a -4% fill; 5-minute granularity keeps exits near their levels."""
    if not is_market_open():
        return
    closed = monitor_positions()
    for c in closed:
        log.info(f"Position closed: {json.dumps(c)}")
        send_sms(f"Mordy Trades: {c['symbol']} {c['reason']} | P&L: ${c.get('pnl_dollar', '?')}")


def trading_cycle():
    log.info("=== Trading cycle started ===")

    if not is_market_open():
        log.info("Market closed — skipping cycle (no trading on stale pre-market data).")
        return

    # Check existing positions first — take profits / cut losses
    closed = monitor_positions()
    for c in closed:
        log.info(f"Position closed: {json.dumps(c)}")
        send_sms(f"Mordy Trades: {c['symbol']} {c['reason']} | P&L: ${c.get('pnl_dollar', '?')}")

    log.info("Scanning market...")
    scan = scan_market()
    if scan.get("error"):
        # An empty bucket and a broken scan look identical downstream; the last
        # outage lasted six sessions because "no setups" was indistinguishable
        # from "the filter cannot fire". Say which one this is.
        log.error(f"Scan unusable ({scan['error']}) — skipping cycle without calling the model.")
        send_sms(f"Mordy Trades: scan unusable ({scan['error']}) — cycle skipped, no trades.")
        return
    log.info(f"Scanned {scan['scanned']} of {scan['universe']} tradable symbols "
             f"at {scan['pace']*100:.0f}% of a normal session's volume — "
             f"{len(scan['early'])} early (unusual volume for the hour, move not yet extended), "
             f"{len(scan['extended'])} already extended")

    if not EARLY_TRADING_ENABLED:
        # Say this on EVERY cycle. A bot that quietly places no orders looks
        # exactly like a bot whose filter is broken — that ambiguity is what let
        # the last outage run for six sessions, and it must not be recreated by
        # the fix for it. The candidate names are logged so the record of what
        # the scanner would have offered keeps accruing at zero cost.
        log.info(
            f"TRADING GATED OFF (EARLY_TRADING_ENABLED=0): scanner healthy, "
            f"{len(scan['early'])} early candidates "
            f"[{', '.join(r['symbol'] for r in scan['early']) or 'none'}], "
            f"no model call, no orders. The early bucket backtested to no edge "
            f"(-0.098% mean excess, day-level t -1.69 over 252 sessions); set "
            f"EARLY_TRADING_ENABLED=1 to trade it anyway."
        )
        return

    log.info("Fetching Discord signals...")
    discord_signals = get_discord_signals(DISCORD_CHANNEL_ID, DISCORD_TOKEN)
    log.info(f"Found {len(discord_signals)} Discord signals")

    log.info("Asking Claude for picks...")
    analysis = analyze(scan, discord_signals)
    log.info(f"Market summary: {analysis.get('market_summary', '')}")

    if analysis.get("pass"):
        log.info("Claude passed — no compelling trades this cycle.")
        return

    for pick in analysis.get("picks", []):
        log.info(f"Pick: {pick['symbol']} {pick['action']} (confidence: {pick['confidence']})")
        log.info(f"  Reasoning: {pick['reasoning']}")
        result = execute_pick(pick)
        log.info(f"  Execution: {json.dumps(result)}")
        if result.get("status") == "submitted":
            send_sms(f"Mordy Trades: {pick['action']} {pick['symbol']} | Confidence: {pick['confidence']}% | {pick['reasoning'][:80]}")

    log.info("=== Cycle complete ===")


def eod_close_and_report():
    log.info("End of day — closing all positions and generating report")
    close_eod_positions()

    client = get_client()
    account = get_account_info(client)
    trades = get_daily_trades()

    log.info(f"Generating EOD report for {len(trades)} trades...")
    report = generate_eod_report(trades, account)
    log.info(f"EOD Report: {report}")
    # Without this line the daily Discord post reads "no trades today", which is
    # indistinguishable from the failure it took six sessions to notice. State
    # the reason there are no trades, every day, in the place Ben actually looks.
    gate = "" if EARLY_TRADING_ENABLED else (
        "\n\n_Trading is GATED OFF: the early-bucket signal backtested to no edge "
        "(2026-09-02). The scanner runs and reports; it places no orders. This is "
        "expected, not a fault._"
    )
    send_sms(f"Mordy Trades EOD:\n{report}{gate}")

    reset_daily_trades()
    log.info("EOD complete")


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="America/New_York")

    scheduler.add_job(
        trading_cycle,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/30", timezone="America/New_York"),
        id="trading_cycle",
        name="Trading Cycle",
    )

    scheduler.add_job(
        monitor_cycle,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/5", timezone="America/New_York"),
        id="monitor_cycle",
        name="Position Monitor",
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        eod_close_and_report,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=55, timezone="America/New_York"),
        id="eod_report",
        name="EOD Close & Report",
    )

    log.info("Scheduler started.")
    scheduler.start()
