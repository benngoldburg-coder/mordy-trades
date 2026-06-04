from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os

TRADE_ALLOCATION = 0.05  # 5% of portfolio per trade
PROFIT_TARGET = 0.03     # Close at +3% gain
STOP_LOSS = -0.02        # Close at -2% loss

# Tracks trades executed today for EOD report
daily_trades: list[dict] = []


def get_client() -> TradingClient:
    return TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)


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

    for pos in positions:
        symbol = pos.symbol
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

    if symbol in get_open_positions(client):
        return {"symbol": symbol, "status": "skipped", "reason": "position already open"}

    portfolio_value = get_portfolio_value(client)
    dollar_amount = portfolio_value * TRADE_ALLOCATION

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


def get_daily_trades() -> list[dict]:
    return daily_trades


def reset_daily_trades():
    daily_trades.clear()
