import numpy as np
from hedging import hedge_one_path
from simulation import simulate_gbm


def run_backtest(S0: float, K: float, T: float, r: float, q: float, sigma: float,
                 n_paths: int, n_steps: int, rebal_freq: int, trans_cost: float,
                 kind: str = "call", seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a backtest over multiple simulated paths."""
    paths = simulate_gbm(S0, r, q, sigma, T, n_steps, n_paths, seed)
    results = [hedge_one_path(paths[i], K, T, r, q, sigma, rebal_freq, trans_cost, kind) for i in range(n_paths)]

    pnls = np.array([res["hedge_error"] for res in results])
    gamma_errors = np.array([res["gamma_error"] for res in results])

    return pnls, gamma_errors, paths


def run_trace(S0: float, K: float, T: float, r: float, q: float, sigma: float,
              n_steps: int, rebal_freq: int, trans_cost: float, kind: str = "call",
              seed: int | None = None) -> dict:
    """Run a single-path simulation with full trace output."""
    path = simulate_gbm(S0, r, q, sigma, T, n_steps, 1, seed)[0]
    trace = hedge_one_path(path, K, T, r, q, sigma, rebal_freq, trans_cost, kind, full_trace=True)

    return trace
