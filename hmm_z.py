# hmm_z.py
import numpy as np
import pandas as pd
import joblib
from scipy.stats import multivariate_normal

# ================================
# Robust Z-score (IDENTICAL to training)
# ================================
def robust_zscore(series, window, clip=5.0):
    mean = series.rolling(window).mean()
    std  = series.rolling(window).std().replace(0, np.nan)

    z = (series - mean) / std
    z = z.clip(-clip, clip)

    return np.tanh(z)


# ================================
# Forward-only causal step
# ================================
def _forward_step(alpha_prev, obs, transmat, means, covars):
    emission = np.array([
        multivariate_normal.pdf(
            obs,
            mean=means[i],
            cov=covars[i],
            allow_singular=True
        )
        for i in range(len(means))
    ])

    alpha = emission * (alpha_prev @ transmat)
    s = alpha.sum()

    if s == 0 or not np.isfinite(s):
        return alpha_prev

    return alpha / s


# ================================
# Public API
# ================================
def apply_hmm_zscore(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Input df must contain:
    ['timestamp', 'close', 'volume']
    """

    params = joblib.load(path)

    N_STATES    = params["N_STATES"]
    WINDOW_FAST = params["WINDOW_FAST"]
    WINDOW_SLOW = params["WINDOW_SLOW"]
    CLIP_Z      = params["CLIP_Z"]

    transmat  = params["transmat"]
    startprob = params["startprob"]
    means     = params["means"]
    covars    = params["covars"]

    df = df.copy()

    # =========================
    # Feature engineering (causal)
    # =========================
    df["z_fast"] = robust_zscore(df["close"], WINDOW_FAST, CLIP_Z)
    df["z_slow"] = robust_zscore(df["close"], WINDOW_SLOW, CLIP_Z)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df[["z_fast", "z_slow"]].values

    # =========================
    # Forward-only inference
    # =========================
    alpha = startprob.copy()
    states_raw = []
    probs = []

    for obs in X:
        alpha = _forward_step(alpha, obs, transmat, means, covars)
        states_raw.append(int(np.argmax(alpha)))
        probs.append(float(np.max(alpha)))

    df["state_raw"]  = states_raw
    df["state_prob"] = probs

    # =========================
    # 🔑 Regime labeling (symbol-agnostic)
    # =========================
    df["ret"] = df["close"].pct_change()

    state_returns = (
        df.groupby("state_raw")["ret"]
        .mean()
        .sort_values()
    )

    state_map = {}
    state_map[state_returns.index[0]]  = "BEAR"
    state_map[state_returns.index[-1]] = "BULL"

    for s in state_returns.index[1:-1]:
        state_map[s] = "RANGE"

    df["state"] = df["state_raw"].map(state_map)

    return df