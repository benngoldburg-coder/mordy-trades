from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
import os


def get_clients():
    trading = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    data = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    return trading, data


def get_active_tickers(trading_client: TradingClient) -> list[str]:
    assets = trading_client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    return [
        a.symbol for a in assets
        if a.tradable and a.fractionable and not a.symbol.endswith(("W", "R", "P"))
        and len(a.symbol) <= 4
    ]


def get_market_movers(data_client: StockHistoricalDataClient, tickers: list[str]) -> list[dict]:
    # Batch snapshot requests — Alpaca supports up to 1000 at a time
    batch_size = 500
    snapshots = {}
    for i in range(0, min(len(tickers), 1000), batch_size):
        batch = tickers[i:i + batch_size]
        req = StockSnapshotRequest(symbol_or_symbols=batch)
        snapshots.update(data_client.get_stock_snapshot(req))

    movers = []
    for symbol, snap in snapshots.items():
        try:
            daily = snap.daily_bar
            prev = snap.previous_daily_bar
            if not daily or not prev or prev.close == 0:
                continue
            change_pct = ((daily.close - prev.close) / prev.close) * 100
            movers.append({
                "symbol": symbol,
                "price": daily.close,
                "change_pct": round(change_pct, 2),
                "volume": daily.volume,
                "vwap": daily.vwap,
            })
        except Exception:
            continue

    # Return top 30 by absolute price change with meaningful volume
    movers = [m for m in movers if m["volume"] > 500_000 and 1 < m["price"] < 500]
    movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    return movers[:30]


def scan_market() -> list[dict]:
    trading_client, data_client = get_clients()
    tickers = get_active_tickers(trading_client)
    return get_market_movers(data_client, tickers)
