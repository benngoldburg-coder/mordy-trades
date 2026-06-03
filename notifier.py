import httpx
import os

DISCORD_API = "https://discord.com/api/v9"
ALERT_CHANNEL_ID = "1511766243714011379"


def _headers():
    return {
        "Authorization": os.environ["DISCORD_TOKEN"],
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }


def send_sms(message: str):
    with httpx.Client() as client:
        resp = client.post(
            f"{DISCORD_API}/channels/{ALERT_CHANNEL_ID}/messages",
            headers=_headers(),
            json={"content": message}
        )
        resp.raise_for_status()
