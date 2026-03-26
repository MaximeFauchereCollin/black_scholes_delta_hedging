import numpy as np
from scipy.stats import norm


def _check_inputs(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError("S must be strictly positive.")
    if K <= 0:
        raise ValueError("K must be strictly positive.")
    if T <= 0:
        raise ValueError("T must be strictly positive.")
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive.")


# d1 / d2


def d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """d1 of the Black-Scholes-Merton model (continuous dividend yield q)."""
    _check_inputs(S, K, T, sigma)
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """d2 = d1 - sigma * sqrt(T)."""
    return d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)


# Call


def call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Price of a European call option (BSM)."""
    D1, D2 = d1(S, K, T, r, q, sigma), d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def call_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """dC/dS = e^(-qT) * N(d1)."""
    return np.exp(-q * T) * norm.cdf(d1(S, K, T, r, q, sigma))


def call_theta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """dC/dt — time decay of the call (per unit of time)."""
    D1, D2 = d1(S, K, T, r, q, sigma), d2(S, K, T, r, q, sigma)
    return (
        -(S * np.exp(-q * T) * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))
        + q * S * np.exp(-q * T) * norm.cdf(D1)
        - r * K * np.exp(-r * T) * norm.cdf(D2)
    )


# Put


def put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Price of a European put option (BSM)."""
    D1, D2 = d1(S, K, T, r, q, sigma), d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * np.exp(-q * T) * norm.cdf(-D1)


def put_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """dP/dS = -e^(-qT) * N(-d1) -> always negative."""
    return -np.exp(-q * T) * norm.cdf(-d1(S, K, T, r, q, sigma))


def put_theta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """dP/dt — time decay of the put (per unit of time)."""
    D1, D2 = d1(S, K, T, r, q, sigma), d2(S, K, T, r, q, sigma)
    return (
        -(S * np.exp(-q * T) * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))
        - q * S * np.exp(-q * T) * norm.cdf(-D1)
        + r * K * np.exp(-r * T) * norm.cdf(-D2)
    )


# Common Greeks (call and put share gamma and vega)


def option_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """d²C/dS² = d²P/dS² — identical for call and put."""
    return np.exp(-q * T) * norm.pdf(d1(S, K, T, r, q, sigma)) / (S * sigma * np.sqrt(T))


def option_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """dC/dsigma = dP/dsigma — in price units per volatility point."""
    return S * np.exp(-q * T) * norm.pdf(d1(S, K, T, r, q, sigma)) * np.sqrt(T)


# Generic interfaces

_DISPATCH = {
    "call": {"price": call_price, "delta": call_delta, "theta": call_theta},
    "put": {"price": put_price, "delta": put_delta, "theta": put_theta},
}


def _get_fn(kind: str, greek: str):
    if kind not in _DISPATCH:
        raise ValueError(f"Invalid kind='{kind}' — choose 'call' or 'put'.")
    return _DISPATCH[kind][greek]


def option_price(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str = "call") -> float:
    return _get_fn(kind, "price")(S, K, T, r, q, sigma)


def option_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str = "call") -> float:
    return _get_fn(kind, "delta")(S, K, T, r, q, sigma)


def option_theta(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str = "call") -> float:
    return _get_fn(kind, "theta")(S, K, T, r, q, sigma)