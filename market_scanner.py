"""Market scanner.

HISTORY OF THE TWO BUGS THIS MODULE HAS HAD, because the second was caused by
the fix for the first and the pattern is worth not repeating.

1. COVERAGE (fixed 2026-08-25). It sliced `min(len(tickers), 1000)` off the front
   of whatever order Alpaca returned assets in, so it scanned an arbitrary ~20%
   of the market and silently reported the result as "today's top movers". It now
   snapshots the entire filtered universe.

2. PREMISE (changed 2026-08-25). It ranked by `abs(change_pct)` on the daily bar
   — i.e. by how much a stock had ALREADY moved. Buying the top of the gainer
   list is a well-documented negative-expectancy trade: the move is in the price.
   The scanner now also surfaces an EARLY bucket — unusual volume with the price
   move still small — which is the only one of the two where the decision is
   still live.

3. PARTIAL-BAR ARITHMETIC (fixed 2026-09-02). The rewrite that introduced the
   EARLY bucket compared a PARTIAL day's volume against a FULL day's: the
   `daily_bar` at 11am holds three hours of trading, and it was divided by
   yesterday's complete session (and then by a 20-day average of complete
   sessions). Demanding a ratio of 2.5x from that arithmetic is very nearly
   impossible before the close, so the bucket was empty every cycle and the bot
   placed ZERO trades for six consecutive sessions while every log line said
   SUCCESS. Measured on 2026-09-01: 0 early names at every cycle up to 14:30 ET,
   1 at 15:00, and 4 only when re-run after the close — a scanner whose entire
   purpose is catching moves EARLY could only fire late.

   The same arithmetic error applied to the liquidity floor, so the scanned
   universe itself grew through the session (1,006 names at 12:00 ET, 1,631 at
   15:30, 1,885 after the close) — the morning was being judged against an
   afternoon's yardstick.

   Everything volume-derived is now divided by MARKET PACE (see below) so that
   today's partial figure is projected to a full session before being compared
   with one.

4. THIN-INSTRUMENT SELECTION (fixed 2026-09-02). `EARLY_MIN_PCT` was added to
   stop bond and sector ETFs filling the bucket on 45x relative volume and a
   0.01% move. It did not work: on 2026-09-01 the four names that survived every
   filter were AMDY ($2.2M), MOO ($6.8M), DBC ($4.5M) and GSG ($2.4M) — all thin
   commodity/yield ETFs — while JNJ ($105M) and EOG ($41M) were cut for having
   relative volume of "only" 1.6x.

   That is not a threshold that needs tuning, it is what the criteria select for:
   a liquid name essentially never trades 2.5x its 20-day volume while moving
   only 1.5-4%, and a thin one does it constantly. So the ranking reliably
   produced one unbuyable ETF per day, the model correctly refused it for
   illiquidity, and the pipeline deadlocked. A relative-volume ranking needs an
   ABSOLUTE liquidity floor applied after it, which is EARLY_MIN_DOLLAR_VOLUME.
"""
import datetime as _dt
import os
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

# Free-plan market data must request the IEX feed explicitly. Omitting it asks
# for recent SIP data, which returns an error the SDK surfaces as an empty result
# — so the 20-day baselines came back empty for EVERY symbol and the early bucket
# silently emptied itself. A data source that fails to nothing is worse than one
# that raises; assert coverage rather than trusting it.
DATA_FEED = DataFeed(os.getenv("ALPACA_DATA_FEED", "iex"))
SNAPSHOT_BATCH = 500          # Alpaca accepts up to 1000; 500 keeps URLs sane
BARS_BATCH = 200

MIN_DOLLAR_VOLUME = 2_000_000       # liquidity floor on PROJECTED full-day dollars
MIN_PRICE, MAX_PRICE = 2.0, 1000.0
EXTENDED_PCT = 8.0                  # a move this big is "already happened"
EARLY_MAX_PCT = 4.0                 # early bucket: move still modest...
EARLY_MIN_PCT = 1.5                 # ...but a REAL move: see below
EARLY_MIN_RELVOL = 2.5              # ...and volume is already unusual for the hour

# The floor that stops the ranking from being a thin-instrument detector. Applied
# to PROJECTED full-day dollar volume, after ranking, to the early bucket only.
# See note 4 above: without it the survivors are always $2-6M ETFs.
EARLY_MIN_DOLLAR_VOLUME = float(os.getenv("EARLY_MIN_DOLLAR_VOLUME", 20_000_000))

# EARLY_MIN_PCT exists because of a specific failure. Without a floor, the early
# bucket filled up with bond and sector ETFs — SGOV (a T-bill fund) showed 45x
# relative volume on a 0.01% move, JPIE 19x on 0.09%. Their volume ratios are
# huge because one block trade dwarfs a quiet prior session, not because anything
# happened. "Unusual volume with no price move at all" is a plumbing artifact,
# not an accumulation setup. It is necessary but, on its own, was not sufficient.

# Market pace is measured against a fixed basket of the most consistently traded
# names rather than the whole universe: it must not move because the composition
# of what clears a filter moved. These are mega caps and index ETFs that trade
# every session, so the basket's own volume is a clean clock.
REFERENCE_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "TSLA", "AVGO", "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA",
    "HD", "CVX", "MRK", "ABBV", "KO", "PEP", "BAC", "COST", "CSCO", "ADBE",
    "CRM", "AMD", "NFLX", "INTC", "T", "VZ", "PFE", "DIS", "WFC", "MCD",
    "BA", "CAT", "GE", "F", "GM", "NKE", "SBUX", "QCOM", "TXN", "MU",
]
MIN_PACE, MAX_PACE = 0.02, 1.5

# Alpaca stamps a daily bar at 04:00 UTC of its SESSION date, i.e. midnight ET.
# During market hours the UTC and ET dates happen to agree, so comparing against
# a UTC "today" works by coincidence — but only by coincidence. Everything that
# asks "is this bar today's?" uses the exchange's calendar explicitly.
ET = ZoneInfo("America/New_York")


def _session_today() -> _dt.date:
    return _dt.datetime.now(ET).date()

_ref_cache: dict = {"date": None, "avgs": {}}


def get_clients():
    trading = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    data = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    return trading, data


def get_active_tickers(trading_client: TradingClient) -> list[str]:
    assets = trading_client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    out = []
    for a in assets:
        if not (a.tradable and a.fractionable):
            continue
        sym = a.symbol
        # Warrants/rights/units are 5-letter symbols ending W/R/U. The old filter
        # dropped ANY symbol ending in W, R or P at any length, which silently
        # excluded ordinary names like SNAP, AAP and MP from the entire universe.
        if len(sym) == 5 and sym[-1] in ("W", "R", "U"):
            continue
        if "." in sym or len(sym) > 5:
            continue
        out.append(sym)
    return out


def _snapshot_all(data_client, tickers):
    """Snapshot the WHOLE universe, not an arbitrary prefix of it."""
    snapshots = {}
    for i in range(0, len(tickers), SNAPSHOT_BATCH):
        batch = tickers[i:i + SNAPSHOT_BATCH]
        try:
            snapshots.update(data_client.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=batch)))
        except Exception:
            continue  # one bad batch must not blind the whole scan
    return snapshots


def _avg_volumes(data_client, symbols, days=20):
    """20-day average volume. Symbols with too little history are simply absent.

    Measuring relative volume against YESTERDAY alone is what surfaced the ETF
    noise: a single quiet prior session turns any thin instrument into a 40x
    anomaly. A 20-day baseline is the honest denominator.
    """
    end = _dt.datetime.now(_dt.timezone.utc)
    start = end - _dt.timedelta(days=days * 2 + 15)  # calendar days -> ~20 sessions
    out = {}
    for i in range(0, len(symbols), BARS_BATCH):
        batch = symbols[i:i + BARS_BATCH]
        try:
            data = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                start=start, end=end, feed=DATA_FEED)).data
        except Exception:
            continue
        for sym, bars in data.items():
            # The final bar is TODAY, in progress. Including it in its own
            # baseline drags the average toward the partial figure being tested.
            vols = [b.volume for b in bars]
            if vols and bars[-1].timestamp.astimezone(ET).date() == _session_today():
                vols = vols[:-1]
            vols = vols[-days:]
            if len(vols) >= 10:
                out[sym] = sum(vols) / len(vols)
    return out


def market_pace(data_client, snapshots) -> float | None:
    """What fraction of a normal session's volume has traded so far.

    This is the fix for the partial-bar bug. Every volume figure on the snapshot
    is cumulative-so-far; every baseline is a complete session. Dividing the
    former by this number projects it to a full day, so the two are comparable
    and a threshold means the same thing at 09:45 as at 15:45.

    It is measured cross-sectionally rather than from a hardcoded U-shaped
    intraday curve, which means it also absorbs whether TODAY is a heavy or light
    session: on a quiet day the basket is behind its own average, pace comes in
    lower, and a name needs correspondingly less volume to look unusual. That is
    the intended behaviour — "unusual" should mean unusual relative to what the
    market is doing right now, not to a calendar.

    Returns None if the basket cannot be measured, which the caller must treat as
    a failure rather than silently substituting 1.0.
    """
    today = _session_today()
    if _ref_cache["date"] != today or not _ref_cache["avgs"]:
        _ref_cache["avgs"] = _avg_volumes(data_client, REFERENCE_SYMBOLS)
        _ref_cache["date"] = today
    avgs = _ref_cache["avgs"]
    if not avgs:
        return None

    traded = baseline = 0.0
    matched = 0
    for sym, avg in avgs.items():
        snap = snapshots.get(sym)
        if not snap or not snap.daily_bar or not avg:
            continue
        traded += snap.daily_bar.volume
        baseline += avg
        matched += 1

    # A handful of names is not a clock. Demand most of the basket.
    if matched < len(avgs) * 0.6 or baseline <= 0 or traded <= 0:
        return None
    return min(max(traded / baseline, MIN_PACE), MAX_PACE)


def _measure(symbol, snap, pace: float):
    """Turn a snapshot into the fields a decision actually needs.

    `pace` divides every volume-derived quantity, projecting today's
    cumulative figure to a full session so it can be compared with one.
    """
    daily, prev = snap.daily_bar, snap.previous_daily_bar
    if not daily or not prev or not prev.close or not prev.volume:
        return None
    if not (MIN_PRICE <= daily.close <= MAX_PRICE):
        return None

    projected_volume = daily.volume / pace
    projected_dollar_volume = daily.close * projected_volume
    # Floor on the PROJECTED figure, so the universe is the same size all day.
    if projected_dollar_volume < MIN_DOLLAR_VOLUME:
        return None

    change_pct = (daily.close - prev.close) / prev.close * 100
    gap_pct = (daily.open - prev.close) / prev.close * 100 if daily.open else 0.0
    rel_volume = projected_volume / prev.volume
    day_range = daily.high - daily.low
    # Where in today's range it is trading: 1.0 = at the high, 0.0 = at the low.
    range_pos = (daily.close - daily.low) / day_range if day_range > 0 else 0.5

    return {
        "symbol": symbol,
        "price": round(daily.close, 2),
        "change_pct": round(change_pct, 2),
        "gap_pct": round(gap_pct, 2),
        "rel_volume": round(rel_volume, 2),
        "range_position": round(range_pos, 2),
        "intraday_move_pct": round(change_pct - gap_pct, 2),
        "dollar_volume_m": round(daily.close * daily.volume / 1e6, 1),
        "projected_dollar_volume_m": round(projected_dollar_volume / 1e6, 1),
        "raw_volume": daily.volume,   # internal; stripped before returning
    }


def scan_market(top_n: int = 20) -> dict:
    """Return two labelled buckets rather than one undifferentiated mover list.

    `early`    — unusual volume FOR THE TIME OF DAY, move still small, holding
                 near the day's high, and liquid enough to actually trade. The
                 only bucket where a decision is still live.
    `extended` — the classic top-gainer list. Kept for context and explicitly
                 labelled, so the model can see what has already run instead of
                 mistaking it for an opportunity.
    """
    trading_client, data_client = get_clients()
    tickers = get_active_tickers(trading_client)
    snapshots = _snapshot_all(data_client, tickers)

    pace = market_pace(data_client, snapshots)
    if pace is None:
        # Failing to nothing is what let the last bug run for six sessions. A
        # scan with no clock cannot judge volume, so it returns no candidates
        # and says why, rather than returning a bucket that looks like "no
        # setups today".
        print("WARNING: market pace unmeasurable (reference basket returned no "
              "usable volume) — skipping this scan rather than comparing a "
              "partial session against full ones.")
        return {"scanned": 0, "universe": len(tickers), "pace": None,
                "early": [], "extended": [], "error": "pace_unavailable"}

    rows = []
    for symbol, snap in snapshots.items():
        try:
            m = _measure(symbol, snap, pace)
        except Exception:
            continue
        if m:
            rows.append(m)

    # Pass 1: cheap shortlist off the snapshot, using yesterday as a rough proxy.
    shortlist = [
        r for r in rows
        if r["rel_volume"] >= EARLY_MIN_RELVOL
        and EARLY_MIN_PCT <= abs(r["change_pct"]) <= EARLY_MAX_PCT
        and r["range_position"] >= 0.6
    ]

    # Pass 2: recompute relative volume against a 20-day baseline for the
    # shortlist only, then re-filter. This is the number that decides the ranking;
    # the pass-1 ratio only decides who is worth measuring properly.
    if shortlist:
        avgs = _avg_volumes(data_client, [r["symbol"] for r in shortlist])
        if not avgs:
            # Do not silently return an empty bucket: that looks identical to
            # "no setups today" and would hide a broken data feed indefinitely.
            print(f"WARNING: 20-day volume baselines unavailable for all "
                  f"{len(shortlist)} shortlisted symbols — falling back to the "
                  f"prior-day ratio, which overstates thin instruments.")
            for r in shortlist:
                r["rel_volume_20d"] = r["rel_volume"]
                r["baseline"] = "prior_day_fallback"
        for r in shortlist:
            avg = avgs.get(r["symbol"])
            if avg:
                r["rel_volume_20d"] = round((r["raw_volume"] / pace) / avg, 2)
        shortlist = [r for r in shortlist
                     if r.get("rel_volume_20d", 0) >= EARLY_MIN_RELVOL]

    for r in rows:
        r.pop("raw_volume", None)
    early = sorted(shortlist, key=lambda r: r.get("rel_volume_20d", 0), reverse=True)
    # Absolute liquidity floor AFTER the ranking. A relative measure will always
    # rank thin instruments first; this is what makes the bucket tradable.
    early = [r for r in early
             if r["projected_dollar_volume_m"] * 1e6 >= EARLY_MIN_DOLLAR_VOLUME]

    extended = [r for r in rows if abs(r["change_pct"]) >= EXTENDED_PCT]
    extended.sort(key=lambda r: abs(r["change_pct"]), reverse=True)

    return {
        "scanned": len(rows),
        "universe": len(tickers),
        "pace": round(pace, 3),
        "early": early[:top_n],
        "extended": extended[:top_n],
    }
