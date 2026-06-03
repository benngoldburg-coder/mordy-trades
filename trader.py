from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
import os

TRADE_ALLOCATION = 0.05  # 5% of portfolio per trade


def get_client() -> TradingClient:
    return TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)


def get_portfolio_value(client: TradingClient) -> float:
    account = client.get_account()
    return float(account.portfolio_value)


def get_open_positions(client: TradingClient) -> set[str]:
    positions = client.get_all_positions()
    return {p.symbol for p in positions}


def execute_pick(pick: dict) -> dict:
    client = get_client()
    symbol = pick["symbol"]
    action = pick["action"]
    confidence = pick["confidence"]

    # Skip low-confidence picks
    if confidence < 65:
        return {"symbol": symbol, "status": "skipped", "reason": f"confidence {confidence} below threshold"}

    open_positions = get_open_positions(client)
    if symbol in open_positions:
        return {"symbol": symbol, "status": "skipped", "reason": "position already open"}

    portfolio_value = get_portfolio_value(client)
    dollar_amount = portfolio_value * TRADE_ALLOCATION

    # Scale allocation with confidence
    if confidence >= 85:
        dollar_amount *= 1.5
    elif confidence < 70:
        dollar_amount *= 0.5

    side = OrderSide.BUY if action == "BUY" else OrderSide.SELL

    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=symbol,
            notional=round(dollar_amount, 2),
            side=side,
            time_in_force=TimeInForce.DAY,
        ))
        return {
            "symbol": symbol,
            "status": "submitted",
            "action": action,
            "dollar_amount": round(dollar_amount, 2),
            "order_id": str(order.id),
        }
    except Exception as e:
        return {"symbol": symbol, "status": "error", "reason": str(e)}


def close_eod_positions():
    """Close all positions at end of day to avoid overnight risk."""
    client = get_client()
    client.close_all_positions(cancel_orders=True)
