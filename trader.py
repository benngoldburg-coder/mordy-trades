from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
import os

TRADE_ALLOCATION = 0.05    # 5% of portfolio per trade
PROFIT_TARGET = 0.03       # Close at +3% gain
STOP_LOSS = -0.02          # Close at -2% loss
MAX_OPEN_POSITIONS = 6     # hard cap on simultaneous positions (5% x 1.5 x 6 = 45% max deployed)

# Tracks trades executed today for EOD report
daily_trades: list[dict] = []

# Symbols with a close order in flight — prevents double-submitting a close
# (which would flip the position) when monitoring runs again before the fill.
_closing: set[str] = set()


def get_client() -> TradingClient:
    return TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)


def get_data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def get_latest_price(symbol: str) -> float | None:
    try:
        trade = get_data_client().get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(trade[symbol].price)
    except Exception:
        return None


def is_market_open(client: TradingClient | None = None) -> bool:
    client = client or get_client()
    return bool(client.get_clock().is_open)


def get_portfolio_value(client: TradingClient) -> float:
    account = client.get_account()
    return float(account.portfolio_value)


def get_account_info(client: TradingClient) -> dict:
    account = client.get_account()
    return {
        "portfolio_value": float(account.portfolio_value),
        "cash": float(account.cash),
        "equity": float(account.equity),
    }


def get_open_positions(client: TradingClient) -> set[str]:
    positions = client.get_all_positions()
    return {p.symbol for p in positions}


def monitor_positions() -> list[dict]:
    """Check all open positions and close any that hit profit target or stop loss."""
    client = get_client()
    positions = client.get_all_positions()
    closed = []

    # Any symbol no longer held means its close order filled — stop tracking it.
    open_symbols = {p.symbol for p in positions}
    _closing.intersection_update(open_symbols)

    for pos in positions:
        symbol = pos.symbol
        if symbol in _closing:
            continue
        unrealized_plpc = float(pos.unrealized_plpc)  # e.g. 0.032 = +3.2%

        if unrealized_plpc >= PROFIT_TARGET:
            reason = f"profit target hit ({unrealized_plpc*100:.1f}%)"
        elif unrealized_plpc <= STOP_LOSS:
            reason = f"stop loss hit ({unrealized_plpc*100:.1f}%)"
        else:
            continue

        side = OrderSide.SELL if pos.side.value == "long" else OrderSide.BUY
        try:
            client.submit_order(MarketOrderRequest(
                symbol=symbol,
                qty=abs(float(pos.qty)),
                side=side,
                time_in_force=TimeInForce.DAY,
            ))
            _closing.add(symbol)
            result = {
                "symbol": symbol,
                "action": "CLOSED",
                "reason": reason,
                "pnl_pct": round(unrealized_plpc * 100, 2),
                "pnl_dollar": round(float(pos.unrealized_pl), 2),
            }
            daily_trades.append(result)
            closed.append(result)
        except Exception as e:
            closed.append({"symbol": symbol, "status": "error", "reason": str(e)})

    return closed


def execute_pick(pick: dict) -> dict:
    client = get_client()
    symbol = pick["symbol"]
    action = pick["action"]
    confidence = pick["confidence"]

    if confidence < 60:
        return {"symbol": symbol, "status": "skipped", "reason": f"confidence {confidence} below threshold"}

    open_positions = get_open_positions(client)
    if symbol in open_positions or symbol in _closing:
        return {"symbol": symbol, "status": "skipped", "reason": "position already open"}
    if len(open_positions) >= MAX_OPEN_POSITIONS:
        return {"symbol": symbol, "status": "skipped",
                "reason": f"max open positions reached ({len(open_positions)}/{MAX_OPEN_POSITIONS})"}

    portfolio_value = get_portfolio_value(client)
    dollar_amount = portfolio_value * TRADE_ALLOCATION

    if confidence >= 85:
        dollar_amount *= 1.5
    elif confidence < 70:
        dollar_amount *= 0.5

    side = OrderSide.BUY if action == "BUY" else OrderSide.SELL

    try:
        if side == OrderSide.SELL:
            # Shorts can't use notional orders on Alpaca — size in whole shares.
            price = get_latest_price(symbol)
            if not price:
                return {"symbol": symbol, "status": "skipped", "reason": "no price for short sizing"}
            qty = int(dollar_amount / price)
            if qty < 1:
                return {"symbol": symbol, "status": "skipped",
                        "reason": f"short budget ${dollar_amount:.0f} < 1 share at ${price:.2f}"}
            order = client.submit_order(MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            ))
            dollar_amount = qty * price
        else:
            order = client.submit_order(MarketOrderRequest(
                symbol=symbol,
                notional=round(dollar_amount, 2),
                side=side,
                time_in_force=TimeInForce.DAY,
            ))
        result = {
            "symbol": symbol,
            "status": "submitted",
            "action": action,
            "dollar_amount": round(dollar_amount, 2),
            "order_id": str(order.id),
        }
        daily_trades.append(result)
        return result
    except Exception as e:
        return {"symbol": symbol, "status": "error", "reason": str(e)}


def close_eod_positions():
    client = get_client()
    client.close_all_positions(cancel_orders=True)
    _closing.clear()


def get_daily_trades() -> list[dict]:
    return daily_trades


def reset_daily_trades():
    daily_trades.clear()
