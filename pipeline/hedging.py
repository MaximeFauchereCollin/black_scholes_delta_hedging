import numpy as np
from black_scholes import option_delta, option_gamma, option_price


def hedge_one_path(path: np.ndarray, K: float, T: float, r: float, q: float, sigma: float, rebal_freq: int,
                   trans_cost: float, kind: str = "call", full_trace: bool = False) -> dict:
    n_steps = len(path) - 1
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)

    # Initial setup
    S0 = path[0]
    V0 = option_price(S0, K, T, r, q, sigma, kind)
    d0 = option_delta(S0, K, T, r, q, sigma, kind)
    tc_init = trans_cost * abs(d0) * S0
    cash = V0 - d0 * S0 - tc_init
    delta = d0

    # State used for gamma error tracking in hedge_one_path.
    gamma_error = 0.0
    hedge_start = 0
    S_hedge = S0
    tau_hedge = T

    # Output arrays
    delta_th = np.zeros(n_steps + 1)
    delta_held = np.zeros(n_steps + 1)
    option_val = np.zeros(n_steps + 1)
    portfolio_val = np.zeros(n_steps + 1)
    pnl = np.zeros(n_steps + 1)
    rebal_idx: list[int] = []

    # t = 0
    delta_th[0] = d0
    delta_held[0] = d0
    option_val[0] = V0
    portfolio_val[0] = cash + delta * S0  # Portfolio = V0
    pnl[0] = portfolio_val[0] - V0  # PnL = 0

    # Main loop
    for i in range(1, n_steps + 1):
        S_prev = path[i - 1]
        S = path[i]
        tau = T - i * dt

        # Cash account evolution over [t_{i-1}, t_i].
        cash *= np.exp(r * dt)
        cash += delta * S_prev * (np.exp(q * dt) - 1)  # Dividends received

        is_rebal = i % rebal_freq == 0
        is_expiry = i == n_steps

        # Gamma error over each rebalancing interval.
        if is_rebal or is_expiry:
            dt_hedge = (i - hedge_start) * dt
            dS = S - S_hedge
            gamma = option_gamma(S_hedge, K, tau_hedge, r, q, sigma)
            gamma_error += -0.5 * gamma * (dS**2 - sigma**2 * S_hedge**2 * dt_hedge)
            hedge_start = i
            S_hedge = S
            tau_hedge = tau

        # Current BSM value
        if is_expiry:
            if kind == "call":
                V_bs = max(S - K, 0.0)
                d_th = 1.0 if S > K else (0.5 if S == K else 0.0)
            else:
                V_bs = max(K - S, 0.0)
                d_th = -1.0 if S < K else (-0.5 if S == K else 0.0)
        else:
            V_bs = option_price(S, K, tau, r, q, sigma, kind)
            d_th = option_delta(S, K, tau, r, q, sigma, kind)

        # Rebalancing
        if is_rebal and not is_expiry:
            new_delta = d_th
            delta_change = new_delta - delta
            tc_cost = trans_cost * abs(delta_change) * S
            cash -= delta_change * S + tc_cost
            delta = new_delta
            rebal_idx.append(i)

        port = cash + delta * S

        delta_th[i] = d_th
        delta_held[i] = delta
        option_val[i] = V_bs
        portfolio_val[i] = port
        pnl[i] = port - V_bs

    hedge_error = pnl[-1]

    if full_trace:
        return {
            "times": times,
            "S": path,
            "delta_th": delta_th,
            "delta_held": delta_held,
            "option_val": option_val,
            "portfolio_val": portfolio_val,
            "pnl": pnl,
            "rebal_idx": rebal_idx,
            "hedge_error": hedge_error,
            "gamma_error": gamma_error,
        }

    return {"hedge_error": hedge_error, "gamma_error": gamma_error}
