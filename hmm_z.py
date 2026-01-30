# hmm_zscore.py
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm

# ================================
# Load HMM parameters ONCE
# ================================
PARAMS_PATH = "models/hmm_zscore_params.joblib"

_params = joblib.load(PARAMS_PATH)

N_STATES  = _params["N_STATES"]
WINDOW_Z  = _params["WINDOW_Z"]
transmat  = _params["transmat"]
startprob = _params["startprob"]
means     = _params["means"]
vars_     = _params["vars"]

# ================================
# Forward-only step (causal)
# ================================
def _forward_step(alpha_prev, obs):
    emission = np.array([
        norm.pdf(obs, loc=means[i], scale=np.sqrt(vars_[i]))
        for i in range(N_STATES)
    ])

    alpha = emission * np.dot(alpha_prev, transmat)
    s = alpha.sum()
    return alpha_prev if s == 0 else alpha / s

# ================================
# Public function
# ================================
def apply_hmm_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes df with columns: ['timestamp', 'close']
    Returns df with ['z', 'state', 'state_prob']
    """

    df = df.copy()

    # ---- feature (causal) ----
    df["mean"] = df["close"].rolling(WINDOW_Z).mean()
    df["std"]  = df["close"].rolling(WINDOW_Z).std()
    df["z"]    = (df["close"] - df["mean"]) / df["std"]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---- causal inference ----
    alpha = startprob.copy()

    states = []
    probs  = []

    for z in df["z"].values:
        alpha = _forward_step(alpha, z)
        states.append(int(np.argmax(alpha)))
        probs.append(float(np.max(alpha)))

    df["state"] = states
    df["state_prob"] = probs
    return df