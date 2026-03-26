from .backtest import run_backtest, run_trace
from .analysis import (plot_gamma_vs_pnl, plot_hedge_trace, plot_pnl_distribution,
                      plot_rebal_comparison, plot_sample_paths, print_stats)

PARAMS = {
    "S0": 100,
    "K": 100,
    "T": 0.25,
    "r": 0.03,
    "q": 0.00,
    "sigma": 0.20,
    "n_paths": 1000,
    "n_steps": 63,
    "trans_cost": 0.00,
    "seed": 42,
}

# 1. Call — daily rebalancing
pnls_call, gamma_errors_call, paths = run_backtest(
    **PARAMS,
    rebal_freq=1,
    kind="call",
)
print_stats(pnls_call, gamma_errors_call, "Call | daily rebalancing")
plot_gamma_vs_pnl(
    pnls_call,
    gamma_errors_call,
    figsize=(8, 8),
    style="dark_background",
)

# 2. Put — daily rebalancing
'''pnls_put, gamma_errors_put, _ = run_backtest(**PARAMS, rebal_freq=1, kind="put")
print_stats(pnls_put, gamma_errors_put, "Put | daily rebalancing")
plot_gamma_vs_pnl(pnls_put, gamma_errors_put, figsize=(8, 8), style="dark_background")'''

# 3. Rebalancing frequency comparison
pnls_dict = {}  # {freq -> pnls} for plot_rebal_comparison
gamma_dict = {}  # {freq -> gamma_errors} for stats
for freq in [1, 3, 7, 10]:
    pnls, gamma_errors, _ = run_backtest(**PARAMS, rebal_freq=freq, kind="call")
    pnls_dict[freq] = pnls
    gamma_dict[freq] = gamma_errors
    print_stats(pnls, gamma_errors, f"Call | freq={freq} steps")

# 4. Visualizations
plot_rebal_comparison(pnls_dict)
plot_pnl_distribution(pnls_call, gamma_errors_call)
plot_sample_paths(paths, K=PARAMS["K"])

# 5. Visualizing the hedge on one path
PARAMS_BIS = {
    "S0": 100,
    "K": 100,
    "T": 0.25,
    "r": 0.03,
    "q": 0.00,
    "sigma": 0.20,
    "n_steps": 63,
    "rebal_freq": 1,
    "trans_cost": 0.00,
    "seed": 42,
}

trace = run_trace(**PARAMS_BIS, kind="call")
plot_hedge_trace(trace, K=PARAMS_BIS["K"])