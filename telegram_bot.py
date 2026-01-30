# telegram_bot.py
import requests

def send_image(
    image_path: str,
    caption: str,
    bot_token: str,
    chat_id: str
):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"

    with open(image_path, "rb") as img:
        files = {"photo": img}
        data = {
            "chat_id": chat_id,
            "caption": caption
        }
        r = requests.post(url, data=data, files=files)

    if r.status_code != 200:
        raise RuntimeError(f"Telegram error: {r.text}")

