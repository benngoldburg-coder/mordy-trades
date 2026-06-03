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
from trader import execute_pick, monitor_positions, close_eod_positions, get_daily_trades, reset_daily_trades, get_client, get_account_info
from report import generate_eod_report
from notifier import send_sms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

DISCORD_CHANNEL_ID = os.environ["DISCORD_CHANNEL_ID"]
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]


def trading_cycle():
    log.info("=== Trading cycle started ===")

    # Check existing positions first — take profits / cut losses
    closed = monitor_positions()
    for c in closed:
        log.info(f"Position closed: {json.dumps(c)}")
        send_sms(f"Mordy Trades: {c['symbol']} {c['reason']} | P&L: ${c.get('pnl_dollar', '?')}")

    log.info("Scanning market movers...")
    market_movers = scan_market()
    log.info(f"Found {len(market_movers)} movers")

    log.info("Fetching Discord signals...")
    discord_signals = get_discord_signals(DISCORD_CHANNEL_ID, DISCORD_TOKEN)
    log.info(f"Found {len(discord_signals)} Discord signals")

    log.info("Asking Claude for picks...")
    analysis = analyze(market_movers, discord_signals)
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
    send_sms(f"Mordy Trades EOD:\n{report}")

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
        eod_close_and_report,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=55, timezone="America/New_York"),
        id="eod_report",
        name="EOD Close & Report",
    )

    log.info("Scheduler started.")
    scheduler.start()
