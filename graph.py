import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("seaborn-v0_8-whitegrid")

def draw(df, path):
    df = df.copy()

    # EMAs
    df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()

    df = df.tail(300).reset_index(drop=True)

    # =========================
    # Figure
    # =========================
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [3.5, 1]},
        sharex=True
    )

    # =========================
    # Background regimes
    # =========================
    for i in range(len(df) - 1):
        color = "green" if df.loc[i, "state"] == 1 else "red"
        ax_price.axvspan(
            df.loc[i, "timestamp"],
            df.loc[i + 1, "timestamp"],
            color=color,
            alpha=0.08
        )

    # =========================
    # Price line
    # =========================
    ax_price.plot(
        df["timestamp"],
        df["close"],
        color="black",
        linewidth=1,
        label="Price"
    )

    # EMAs (اختياري)
    ax_price.plot(df["timestamp"], df["EMA_21"], color="blue", alpha=0.5, label="EMA 21")
    ax_price.plot(df["timestamp"], df["EMA_50"], color="red", alpha=0.5, label="EMA 50")

    # =========================
    # State change markers
    # =========================
    state_change = df["state"].diff()

    for i in state_change[state_change != 0].index:
        if i == 0:
            continue
        color = "green" if df.loc[i, "state"] == 1 else "red"
        ax_price.scatter(
            df.loc[i, "timestamp"],
            df.loc[i, "close"] * 1.003,
            color=color,
            s=25,
            zorder=40
        )

    # =========================
    # Last price dot
    # =========================
    last = df.iloc[-1]
    ax_price.scatter(
        last["timestamp"],
        last["close"],
        color="black",
        s=10,
        zorder=6
    )

    # =========================
    # Volume
    # =========================
    vol_colors = ["green" if s == 1 else "red" for s in df["state"]]

    ax_vol.bar(
        df["timestamp"],
        df["volume"],
        color=vol_colors,
        alpha=0.6,
        width=0.01
    )

    # =========================
    # Styling
    # =========================
    ax_price.set_title("HMM Regime – Price & Volume (15m)", fontsize=14)
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    ax_vol.grid(alpha=0.2)
    ax_vol.set_ylabel("Volume")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

    return path