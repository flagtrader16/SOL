# telegram_bot.py
import requests
import pandas as pd

def send_image(
    image_path: str,
    caption: str,
    bot_token: str,
    chat_id: str,
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

def send_file(file_path, caption, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    with open(file_path, "rb") as f:
        files = {"document": f}
        data = {
            "chat_id": chat_id,
            "caption": caption
        }
        r = requests.post(url, data=data, files=files)

    if r.status_code != 200:
        raise RuntimeError(r.text)
        