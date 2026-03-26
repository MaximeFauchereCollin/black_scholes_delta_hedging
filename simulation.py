import numpy as np


def _check_inputs(S0: float, n_steps: int, n_paths: int, T: float) -> None:
    if S0 <= 0:
        raise ValueError("S0 must be strictly positive.")
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be > 0.")
    if T <= 0:
        raise ValueError("T must be strictly positive.")


def simulate_gbm(S0: float, r: float, q: float, sigma: float, T: float, n_steps: int, n_paths: int, seed: int | None = None) -> np.ndarray:
    """Simulate geometric Brownian motion price paths."""
    _check_inputs(S0, n_steps, n_paths, T)

    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))

    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)],
        axis=1,
    )

    return S0 * np.exp(log_paths)