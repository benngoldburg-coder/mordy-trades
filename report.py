import anthropic
import os

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client

REPORT_PROMPT = """You are a trading bot reporting to your owner at end of day.
Summarize the day's activity in a short, clear SMS-friendly message (under 300 characters total).
Include: total P&L, number of trades, biggest win, biggest loss, and one sentence takeaway.
Be direct. No fluff. Format like a text message, not an essay."""


def generate_eod_report(trades: list[dict], account: dict) -> str:
    summary = f"""
Day's trades: {trades}
Account: portfolio_value={account.get('portfolio_value')}, cash={account.get('cash')}
"""
    message = _get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=REPORT_PROMPT,
        messages=[{"role": "user", "content": summary}],
    )
    return message.content[0].text.strip()
