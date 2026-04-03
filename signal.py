"""
signal.py
Calcul du signal composite alpha_t pour la stratégie dynamic option selection.
On combine 3 indicateurs : IV-RV spread, skew implicite, momentum.
"""

import numpy as np
import pandas as pd


def compute_realized_volatility(df_spot, window=20):
    """Vol réalisée annualisée sur 'window' jours (formule Lecture 2)."""
    df = df_spot.sort_values("date").set_index("date")
    log_rets = np.log(df["spot"] / df["spot"].shift(1))
    variance = (252 / window) * (log_rets ** 2).rolling(window).sum()
    result = np.sqrt(variance).rename("realized_vol")
    return result[~result.index.duplicated(keep="first")]


def compute_iv_atm(df_options, target_dte=30):
    """IV ATM ~30j : on prend les calls avec delta entre 0.45 et 0.55."""
    mask = (
        (df_options["call_put"] == "C")
        & (df_options["delta"].abs() >= 0.45)
        & (df_options["delta"].abs() <= 0.55)
    )
    df_atm = df_options[mask].copy()
    df_atm["dte_distance"] = (df_atm["day_to_expiration"] - target_dte).abs()
    idx = df_atm.groupby("date")["dte_distance"].idxmin()
    iv = df_atm.loc[idx].set_index("date")["implied_volatility"]
    return iv[~iv.index.duplicated(keep="first")].rename("iv_atm")


def compute_skew(df_options, target_dte=30):
    """Skew = (IV 25delta call - IV 25delta put) / IV ATM."""
    df = df_options.copy()
    df["dte_distance"] = (df["day_to_expiration"] - target_dte).abs()
    best_dte = df.groupby("date")["dte_distance"].min().reset_index()
    best_dte.columns = ["date", "min_dte_distance"]
    df = df.merge(best_dte, on="date")
    df = df[df["dte_distance"] == df["min_dte_distance"]]

    call_25 = (
        df[(df["call_put"] == "C") & (df["delta"] >= 0.20) & (df["delta"] <= 0.30)]
        .groupby("date")["implied_volatility"].mean()
    )
    put_25 = (
        df[(df["call_put"] == "P") & (df["delta"] >= -0.30) & (df["delta"] <= -0.20)]
        .groupby("date")["implied_volatility"].mean()
    )
    atm = (
        df[(df["call_put"] == "C") & (df["delta"] >= 0.45) & (df["delta"] <= 0.55)]
        .groupby("date")["implied_volatility"].mean()
    )

    df_skew = pd.DataFrame({"iv_25c": call_25, "iv_25p": put_25, "iv_atm": atm}).dropna()
    skew = (df_skew["iv_25c"] - df_skew["iv_25p"]) / df_skew["iv_atm"]
    return skew[~skew.index.duplicated(keep="first")].rename("skew")


def compute_momentum(df_spot, window=10):
    """Momentum = log(S_t / S_{t-window}), rendement log sur 10 jours."""
    df = df_spot.sort_values("date").set_index("date")
    mom = np.log(df["spot"] / df["spot"].shift(window))
    return mom[~mom.index.duplicated(keep="first")].rename("momentum")


def rolling_zscore(series, window=252):
    """Z-score glissant sur 252j pour normaliser et rendre stationnaire."""
    mu = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mu) / std).rename(f"z_{series.name}")


def compute_composite_signal(df_spot, df_options, weights=None):
    """
    Signal composite : alpha_t = 0.5*z(IV-RV) + 0.3*z(Skew) + 0.2*z(Momentum)
    alpha > 0 = marché calme, alpha < 0 = marché stressé
    """
    if weights is None:
        weights = {"iv_rv": 0.5, "skew": 0.3, "momentum": 0.2}

    iv_atm = compute_iv_atm(df_options)
    rv = compute_realized_volatility(df_spot)
    skew = compute_skew(df_options)
    momentum = compute_momentum(df_spot)

    df = pd.DataFrame({"iv_atm": iv_atm, "realized_vol": rv})
    df["iv_rv_spread"] = df["iv_atm"] - df["realized_vol"]
    df["skew"] = -skew  # inversé : skew négatif = stress
    df["momentum"] = momentum

    df["z_iv_rv"] = rolling_zscore(df["iv_rv_spread"])
    df["z_skew"] = rolling_zscore(df["skew"])
    df["z_momentum"] = rolling_zscore(df["momentum"])

    df["alpha"] = (
        weights["iv_rv"] * df["z_iv_rv"]
        + weights["skew"] * df["z_skew"]
        + weights["momentum"] * df["z_momentum"]
    )

    return df
