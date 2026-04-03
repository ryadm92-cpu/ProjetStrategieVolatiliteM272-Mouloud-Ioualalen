"""
mapping.py
Fonctions de mapping f(alpha_t) -> strike et maturité cibles.
On transforme le signal (un réel quelconque) en paramètres bornés.
"""

import numpy as np


def sigmoid(x):
    """Sigmoid classique, résultat entre 0 et 1."""
    return 1.0 / (1.0 + np.exp(-x))


def map_strike_sigmoid(alpha, strike_min=0.90, strike_max=0.99, sensitivity=1.0):
    """Sigmoid : alpha haut -> strike proche ATM, alpha bas -> strike OTM."""
    normalized = sigmoid(sensitivity * np.asarray(alpha))
    return strike_min + normalized * (strike_max - strike_min)


def map_strike_tanh(alpha, strike_min=0.90, strike_max=0.99, sensitivity=1.0):
    """Tanh : même idée que sigmoid mais symétrique autour de 0."""
    normalized = (np.tanh(sensitivity * np.asarray(alpha)) + 1.0) / 2.0
    return strike_min + normalized * (strike_max - strike_min)


def map_strike_linear(alpha, strike_min=0.90, strike_max=0.99, clip_bound=2.0):
    """Linéaire borné : on clippe alpha entre -clip et +clip puis on interpole."""
    clipped = np.clip(np.asarray(alpha) / clip_bound, -1.0, 1.0)
    normalized = (clipped + 1.0) / 2.0
    return strike_min + normalized * (strike_max - strike_min)


def map_strike_step(alpha, strike_min=0.90, strike_max=0.99, threshold=0.5):
    """Step function : 3 régimes (baissier / neutre / haussier)."""
    mid = (strike_min + strike_max) / 2
    arr = np.asarray(alpha)
    return np.where(arr > threshold, strike_max,
           np.where(arr < -threshold, strike_min, mid))


def map_maturity(alpha, maturity_min=7, maturity_max=60, sensitivity=1.0):
    """Maturité cible. Sens inversé : alpha haut -> maturité courte."""
    normalized = sigmoid(sensitivity * np.asarray(alpha))
    result = maturity_max - normalized * (maturity_max - maturity_min)
    return np.round(result).astype(int)


# Dico pour itérer facilement sur les 4 fonctions
MAPPING_FUNCTIONS = {
    "sigmoid": map_strike_sigmoid,
    "tanh": map_strike_tanh,
    "linear": map_strike_linear,
    "step": map_strike_step,
}
