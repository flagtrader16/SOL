
import requests
import pandas as pd
import time

def fetch_binance_ohlcv(symbol="SOLUSDT", interval="15m", limit=1000, startTime=None):
    
# URL 
    BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.vision"
    ]
    BASE_URL = BASE_URLS[0]

    """
    Fetch OHLCV data from Binance Vision API.
    """
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if startTime:
        params["startTime"] = int(startTime)
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def get_binance_data_since(symbol="SOLUSDT", interval="15m", start_date="2023-01-01"):
    """
    Get OHLCV data from start_date until now, automatically handling pagination.
    """
    all_data = []
    startTime = int(pd.to_datetime(start_date).timestamp() * 1000)  # ms
    while True:
        data = fetch_binance_ohlcv(symbol, interval, limit=1000, startTime=startTime)
        if not data:
            break

        all_data.extend(data)
        last_close_time = data[-1][6]
        startTime = last_close_time + 1  # التالي بعد آخر شريط

        # إذا حصلنا أقل من 1000، انتهينا
        if len(data) < 1000:
            break

        time.sleep(0.1)  # لتجنب rate limit

    # DataFrame
    df = pd.DataFrame(all_data, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)

    return df
