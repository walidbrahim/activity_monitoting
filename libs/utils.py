import requests
from config import config


# TODO: Make it abstract
def send_watch_alert(state_message):
    """Sends a push notification in the background."""
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": config.pushover.api_token,
            "user": config.pushover.user_key,
            "message": f"Update: {state_message}",
            "title": "Room Activity Monitor",
            "sound": "intermission" # watch ping sound
        }, timeout=5)
    except Exception as e:
        print(f"Failed to send alert: {e}")