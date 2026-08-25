"""Market scanner.

Two things were wrong with the previous version and both are fixed here.

1. COVERAGE BUG. It sliced `min(len(tickers), 1000)` off the front of whatever
   order Alpaca returned assets in, so it scanned an arbitrary ~20% of the
   market and silently reported the result as "today's top movers". It now
   snapshots the entire filtered universe.

2. PREMISE. It ranked by `abs(change_pct)` on the daily bar — i.e. by how much
   a stock had ALREADY moved — and handed the top 30 to the model. Buying the
   top of the gainer list for a next-day hold is a well-documented
   negative-expectancy trade: the move is in the price, and the model is being
   asked to pick from a list where the information is spent. The scanner now
   also surfaces an EARLY bucket — unusual volume with the price move still
   small — which is the only one of the two where the decision is still live.
"""
import datetime as _dt
import os

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
MIN_DOLLAR_VOLUME = 2_000_000 # liquidity floor, in dollars traded today
MIN_PRICE, MAX_PRICE = 2.0, 1000.0
EXTENDED_PCT = 8.0            # a move this big is "already happened"
EARLY_MAX_PCT = 4.0           # early bucket: move still modest...
EARLY_MIN_PCT = 1.5           # ...but a REAL move: see below
EARLY_MIN_RELVOL = 2.5        # ...and volume is already unusual

# EARLY_MIN_PCT exists because of a specific failure. Without a floor, the early
# bucket filled up with bond and sector ETFs — SGOV (a T-bill fund) showed 45x
# relative volume on a 0.01% move, JPIE 19x on 0.09%. Their volume ratios are
# huge because one block trade dwarfs a quiet prior session, not because anything
# happened. "Unusual volume with no price move at all" is a plumbing artifact,
# not an accumulation setup, and it would have consumed every downstream LLM call.


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
    """20-day average volume for a shortlist. {} entries are simply absent.

    Measuring relative volume against YESTERDAY alone is what surfaced the ETF
    noise: a single quiet prior session turns any thin instrument into a 40x
    anomaly. A 20-day baseline is the honest denominator. It costs one extra
    round of requests, which is why it runs on the shortlist and not on all
    ~1,900 names that clear the liquidity floor.
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
            vols = [b.volume for b in bars][-days:]
            if len(vols) >= 10:
                out[sym] = sum(vols) / len(vols)
    return out


def _measure(symbol, snap):
    """Turn a snapshot into the fields a decision actually needs."""
    daily, prev = snap.daily_bar, snap.previous_daily_bar
    if not daily or not prev or not prev.close or not prev.volume:
        return None
    if not (MIN_PRICE <= daily.close <= MAX_PRICE):
        return None
    if daily.close * daily.volume < MIN_DOLLAR_VOLUME:
        return None

    change_pct = (daily.close - prev.close) / prev.close * 100
    gap_pct = (daily.open - prev.close) / prev.close * 100 if daily.open else 0.0
    rel_volume = daily.volume / prev.volume
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
        "raw_volume": daily.volume,   # internal; stripped before returning
    }


def scan_market(top_n: int = 20) -> dict:
    """Return two labelled buckets rather than one undifferentiated mover list.

    `early`    — unusual volume, move still small, holding near the day's high.
                 The only bucket where a next-day decision is still live.
    `extended` — the classic top-gainer list. Kept for context and explicitly
                 labelled, so the model can see what has already run instead of
                 mistaking it for an opportunity.
    """
    trading_client, data_client = get_clients()
    tickers = get_active_tickers(trading_client)
    snapshots = _snapshot_all(data_client, tickers)

    rows = []
    for symbol, snap in snapshots.items():
        try:
            m = _measure(symbol, snap)
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
                r["rel_volume_20d"] = round(r["raw_volume"] / avg, 2)
        shortlist = [r for r in shortlist
                     if r.get("rel_volume_20d", 0) >= EARLY_MIN_RELVOL]

    for r in rows:
        r.pop("raw_volume", None)
    early = sorted(shortlist, key=lambda r: r.get("rel_volume_20d", 0), reverse=True)

    extended = [r for r in rows if abs(r["change_pct"]) >= EXTENDED_PCT]
    extended.sort(key=lambda r: abs(r["change_pct"]), reverse=True)

    return {
        "scanned": len(rows),
        "universe": len(tickers),
        "early": early[:top_n],
        "extended": extended[:top_n],
    }
