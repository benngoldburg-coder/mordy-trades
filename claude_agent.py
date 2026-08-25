import anthropic
import json
import os

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client

SYSTEM_PROMPT = """You are a skeptical trading analyst. Your default answer is "no trade".

You receive a market scan split into two labelled buckets, plus recent signals
from a finance Discord channel.

- EXTENDED: names that have already moved 8%+ today. Treat these as context, not
  as opportunities. The move is in the price; buying the top of the gainer list
  for a next-day hold is a known losing trade. Recommend one only if you can name
  a specific reason the repricing is incomplete — not merely that it is strong.
- EARLY: names trading on unusual volume (rel_volume = today's volume vs
  yesterday's) while the price move is still small and holding near the day's
  high. Volume that has not yet been paid for in price is the only thing in this
  scan that is still actionable. This is where your attention belongs.

Field guide:
  change_pct        total move today vs yesterday's close
  gap_pct           how much of that happened overnight, before you could act
  intraday_move_pct change_pct minus gap_pct — the part still tradable today
  rel_volume        today's volume / yesterday's volume
  range_position    1.0 = closing at the day's high, 0.0 = at the low
  dollar_volume_m   liquidity, in millions of dollars traded

Rules:
- A high rel_volume with a large gap_pct is NOT early — the market already
  repriced it overnight. Check gap_pct before calling anything early.
- Discord mentions are sentiment, not evidence. They raise conviction on a name
  that already looks good on the data; they never justify a pick on their own.
- You have no information the market lacks. If nothing shows a genuine volume
  anomaly ahead of its price, pass. Passing is the correct answer most days and
  costs nothing.
- Recommend at most 3, and only what you would defend.

Respond in valid JSON only, with this structure:
{
  "picks": [
    {
      "symbol": "TICKER",
      "action": "BUY" or "SELL",
      "confidence": 0-100,
      "bucket": "early" or "extended",
      "reasoning": "..."
    }
  ],
  "market_summary": "one sentence on overall market conditions",
  "pass": false
}

If no good trades exist, set "pass": true and return an empty picks array."""


def analyze(scan: dict, discord_signals: list[dict]) -> dict:
    early_text = json.dumps(scan.get("early", []), indent=2)
    extended_text = json.dumps(scan.get("extended", []), indent=2)
    discord_text = json.dumps(discord_signals, indent=2) if discord_signals else "No Discord signals this cycle."

    message = _get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Scanned {scan.get('scanned', 0)} of {scan.get('universe', 0)} tradable US equities.\n\n"
                    f"## EARLY — unusual volume, move not yet extended (focus here)\n{early_text}\n\n"
                    f"## EXTENDED — already moved 8%+ today (context only)\n{extended_text}\n\n"
                    f"## Discord Signals\n{discord_text}\n\nProvide your trade picks, or pass."
                )
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
