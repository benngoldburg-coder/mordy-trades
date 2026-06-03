import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from market_scanner import scan_market
from discord_scraper import get_discord_signals
from claude_agent import analyze
from trader import execute_pick, close_eod_positions

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

DISCORD_CHANNEL_ID = os.environ["DISCORD_CHANNEL_ID"]
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]


def trading_cycle():
    log.info("=== Trading cycle started ===")

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

    log.info("=== Cycle complete ===")


def eod_close():
    log.info("End of day — closing all positions")
    close_eod_positions()
    log.info("All positions closed")


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="America/New_York")

    # Run trading cycle every 30 minutes during market hours (Mon-Fri 9:30am-3:30pm ET)
    scheduler.add_job(
        trading_cycle,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/30", timezone="America/New_York"),
        id="trading_cycle",
        name="Trading Cycle",
    )

    # Close all positions at 3:45pm ET to avoid overnight exposure
    scheduler.add_job(
        eod_close,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=45, timezone="America/New_York"),
        id="eod_close",
        name="EOD Position Close",
    )

    log.info("Scheduler started. Running every 30 minutes during market hours.")
    scheduler.start()
