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
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
import os

SNAPSHOT_BATCH = 500          # Alpaca accepts up to 1000; 500 keeps URLs sane
MIN_DOLLAR_VOLUME = 2_000_000 # liquidity floor, in dollars traded today
MIN_PRICE, MAX_PRICE = 2.0, 1000.0
EXTENDED_PCT = 8.0            # a move this big is "already happened"
EARLY_MAX_PCT = 4.0           # early bucket: move still modest...
EARLY_MIN_RELVOL = 2.5        # ...but volume is already unusual


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

    early = [
        r for r in rows
        if r["rel_volume"] >= EARLY_MIN_RELVOL
        and abs(r["change_pct"]) <= EARLY_MAX_PCT
        and r["range_position"] >= 0.6
    ]
    # Rank by volume anomaly — the part that has NOT yet been paid for in price.
    early.sort(key=lambda r: r["rel_volume"], reverse=True)

    extended = [r for r in rows if abs(r["change_pct"]) >= EXTENDED_PCT]
    extended.sort(key=lambda r: abs(r["change_pct"]), reverse=True)

    return {
        "scanned": len(rows),
        "universe": len(tickers),
        "early": early[:top_n],
        "extended": extended[:top_n],
    }
