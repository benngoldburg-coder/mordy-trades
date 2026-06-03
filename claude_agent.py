import anthropic
import json
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = """You are an elite quantitative trading analyst with deep expertise in momentum trading,
sentiment analysis, and technical analysis. You will receive:
1. A list of today's top market movers with price and volume data
2. Recent signals from a finance Discord channel (ticker mentions, sentiment keywords)

Your job is to identify the 1-3 highest conviction trade opportunities. For each pick:
- It must appear in the market data OR have strong Discord signal (ideally both)
- Assess momentum, volume confirmation, and social sentiment
- Assign a confidence score (0-100)
- Specify BUY or SELL (short)
- Give a clear, concise reasoning

Be selective. Only recommend trades with genuine edge. If nothing looks compelling, say so.

Respond in valid JSON only, with this structure:
{
  "picks": [
    {
      "symbol": "TICKER",
      "action": "BUY" or "SELL",
      "confidence": 0-100,
      "reasoning": "..."
    }
  ],
  "market_summary": "one sentence on overall market conditions",
  "pass": false
}

If no good trades exist, set "pass": true and return an empty picks array."""


def analyze(market_movers: list[dict], discord_signals: list[dict]) -> dict:
    market_text = json.dumps(market_movers, indent=2)
    discord_text = json.dumps(discord_signals, indent=2) if discord_signals else "No Discord signals this cycle."

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"## Market Movers\n{market_text}\n\n## Discord Signals\n{discord_text}\n\nProvide your trade picks."
            }
        ]
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
