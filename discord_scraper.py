import httpx
import re
import os

DISCORD_API = "https://discord.com/api/v9"
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')
# Also catch bare uppercase tickers like "NVDA" or "AVGO"
BARE_TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')
KEYWORD_PATTERN = re.compile(
    r'\b(buy|sell|long|short|calls|puts|yolo|bullish|bearish|moon|dump|squeeze|breakout|entry|target)\b',
    re.IGNORECASE
)
# Map common company names to tickers
NAME_TO_TICKER = {
    "broadcom": "AVGO", "nvidia": "NVDA", "nvda": "NVDA", "tesla": "TSLA",
    "apple": "AAPL", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL",
    "microsoft": "MSFT", "meta": "META", "palantir": "PLTR", "gamestop": "GME",
    "amc": "AMC", "coinbase": "COIN", "netflix": "NFLX", "uber": "UBER",
    "shopify": "SHOP", "snowflake": "SNOW", "arm": "ARM", "openai": "MSFT",
}


def get_headers(token: str) -> dict:
    return {
        "Authorization": token,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }


def fetch_messages(channel_id: str, token: str, limit: int = 50) -> list[dict]:
    url = f"{DISCORD_API}/channels/{channel_id}/messages?limit={limit}"
    with httpx.Client() as client:
        resp = client.get(url, headers=get_headers(token))
        resp.raise_for_status()
        return resp.json()


def extract_signals(messages: list[dict]) -> list[dict]:
    signals = []
    for msg in messages:
        content = msg.get("content", "")
        tickers = set(TICKER_PATTERN.findall(content))

        # Extract company name mentions
        for name, ticker in NAME_TO_TICKER.items():
            if name in content.lower():
                tickers.add(ticker)

        keywords = KEYWORD_PATTERN.findall(content)
        # Always include message if it has any trading-relevant content
        if tickers or keywords:
            signals.append({
                "author": msg["author"]["username"],
                "content": content,
                "tickers": list(tickers),
                "sentiment_keywords": [k.lower() for k in keywords],
                "timestamp": msg["timestamp"],
            })
    return signals


def get_discord_signals(channel_id: str, token: str) -> list[dict]:
    messages = fetch_messages(channel_id, token)
    return extract_signals(messages)
