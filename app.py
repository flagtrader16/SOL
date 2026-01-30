from get_data import get_binance_data_since
from hmm_z import apply_hmm_zscore
from graph import draw
from telegram_bot import send_image
import os

# =========================
# Config
# =========================
symbol = "SOLUSDT"
interval = "15m"
start_date = "2026-01-20"

BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
CHAT_ID   = os.getenv("TG_CHAT_ID")

# =========================
# Run pipeline
# =========================
df = get_binance_data_since(symbol, interval, start_date)

df_show = apply_hmm_zscore(df)

img_path = draw(df_show)

last = df_show.iloc[-1]

caption = (
    f"SOL 15m HMM Update\n"
    f"Price: {last['close']:.2f}\n"
    f"State: {int(last['state'])}\n"
    f"Conf : {last['state_prob']:.2f}"
)

send_image(
    image_path=img_path,
    caption=caption,
    bot_token=BOT_TOKEN,
    chat_id=CHAT_ID
)