
from get_data import get_binance_data_since
from hmm_z import apply_hmm_zscore
from graph import draw
from telegram_bot import send_image
from telegram_bot import send_file
from datetime import datetime
import os
import pandas as pd
# =========================
# Config
# =========================
INTERVAL = "15m"
START_DATE = "2026-01-21"

SYMBOLS = {
   # "BTCUSDT": "models/hmm_zscoreBTC_params.joblib",
    "ETHUSDT": "models/hmm_zscoreETH_params.joblib",
    "SOLUSDT": "models/hmm_zscoreSOL_params.joblib",
    "BNBUSDT": "models/hmm_zscoreBNB_params.joblib",
}

BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
CHAT_ID   = os.getenv("TG_CHAT_ID")   


# =========================
# Anti-sleep heartbeat
# =========================
def anti_sleep():
    now = datetime.utcnow().isoformat()
    with open("heartbeat.txt", "w") as f:
        f.write(f"Last run: {now}\n")


# =========================
# Main logic
# =========================
def run():
    for symbol, model_path in SYMBOLS.items():

        # --- Fetch data
        df = get_binance_data_since(
            symbol=symbol,
            interval=INTERVAL,
            start_date=START_DATE
        )

        # --- Apply HMM
        df_show = apply_hmm_zscore(df, model_path)

        # --- Save plot
        img_path = draw(
            df_show,
            path=f"HMM_Regime_{symbol}.png"
        )
        # --- Save Files
        df_show = df_show[['timestamp','close','z','state','state_prob']]
        df_show = df_show.round(1)
        df_show.tail(10).to_csv(f"{symbol}.csv",index=False)
        # --- Last state
        last = df_show.iloc[-1]

        caption = (
            f"{symbol} | {INTERVAL}\n"
            f"Price: {last['close']:.2f}\n"
            f"State: {int(last['state'])}\n"
            f"Conf : {last['state_prob']:.2f}"
        )
         
        # --- Send to Telegram
        send_image(
            image_path=img_path,
            caption=caption,
            bot_token=BOT_TOKEN,
            chat_id=CHAT_ID
        )
        #Send File To Telegram
        send_file(
           file_path=(f"{symbol}.csv"),
           caption="HMM Data Snapshot (last 50 rows)",
           bot_token=BOT_TOKEN,
           chat_id=CHAT_ID
)
    anti_sleep()

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    run()
