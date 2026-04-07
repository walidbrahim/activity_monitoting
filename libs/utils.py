import requests
from apps.common.gui_config import PushoverCredentials


# TODO: Make it abstract
def send_watch_alert(state_message):
    """Sends a push notification in the background."""
    try:
        creds = PushoverCredentials.load()
        if not creds:
            return
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": creds.api_token,
            "user": creds.user_key,
            "message": f"Update: {state_message}",
            "title": "Room Activity Monitor",
            "sound": "intermission" # watch ping sound
        }, timeout=5)
    except Exception as e:
        print(f"Failed to send alert: {e}")
