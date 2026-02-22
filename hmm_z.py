# hmm_z.py
import numpy as np
import pandas as pd
import joblib
from scipy.stats import multivariate_normal


# ================================
# Robust Z-score (IDENTICAL to training)
# ================================
def robust_zscore(series, window, clip=5.0):  # 👈 clip ثابت داخليًا
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

    WINDOW_Z = params["WINDOW_Z"]
    # ❌ حذف CLIP_Z من params

    transmat  = params["transmat"]
    startprob = params["startprob"]
    means     = params["means"]
    covars    = params["covars"]

    df = df.copy()

    # =========================
    # Feature engineering (causal)
    # =========================
    df["z"] = robust_zscore(df["close"], WINDOW_Z)  # 👈 بدون تمرير clip

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df[["z"]].values

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

    df["state"] = states_raw
    df["state_prob"] = probs

    return df