# Black-Scholes Delta Hedging

A rigorous Python implementation of discrete delta hedging under the Black-Scholes-Merton framework, with Monte Carlo simulation, hedging error analysis, and gamma error approximation.

---

## Overview

This project studies how a self-financing delta hedge behaves when rebalanced at **discrete** intervals rather than continuously. The central question is:

> *How does the discrete hedging error behave across simulated paths, and how well does the gamma-based second-order approximation explain it?*

Key features:
- Analytical BSM pricing and Greeks with continuous dividend yield
- Exact log-normal GBM simulation
- Discrete delta hedging engine with transaction cost support
- Gamma error approximation accumulated over rebalancing intervals
- Rebalancing frequency comparison
- Full single-path trace with three-panel visualisation
- Statistical summary: mean, std, VaR 95/99%, CVaR, skewness, kurtosis

---

## Project Structure

```
.
├── black_scholes.py      # BSM pricing formulas and Greeks
├── simulation.py         # GBM path simulation
├── hedging.py            # Discrete delta hedging engine (single path)
├── backtest.py           # Multi-path backtest and single-path trace runner
├── analysis.py           # Visualisations and descriptive statistics
├── main.py               # Main experiment script
├── hedging_study.ipynb        # Analytical narrative notebook
├── requirements.txt
└── README.md
```

---

## Theoretical Background

Under the BSM model, the underlying follows:

$$dS_t = (r - q)\,S_t\,dt + \sigma\,S_t\,dW_t$$

A delta-hedged portfolio replicates the option perfectly under **continuous** rebalancing. Under **discrete** rebalancing with step $\Delta t$, a residual hedging error accumulates. The leading-order contribution per interval is:

$$\epsilon_\Gamma = -\frac{1}{2}\,\Gamma\left[(\Delta S)^2 - \sigma^2 S^2 \Delta t\right]$$

---

## Quickstart

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/black-scholes-delta-hedging.git
cd black-scholes-delta-hedging
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the main experiment**
```bash
python main.py
```

Output figures are saved to `outputs/`.

**4. Explore the notebook**
```bash
jupyter notebook hedging_study.ipynb
```

---

## Parameters

The baseline experiment in `main.py` uses the following configuration:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `S0` | 100 | Initial spot price |
| `K` | 100 | Strike price (at-the-money) |
| `T` | 0.25 | Time to expiry (3 months) |
| `r` | 0.03 | Risk-free rate (3%) |
| `q` | 0.00 | Continuous dividend yield |
| `sigma` | 0.20 | Volatility (20%) |
| `n_paths` | 1000 | Number of Monte Carlo paths |
| `n_steps` | 63 | Time steps per path (daily) |
| `rebal_freq` | 1 | Rebalancing frequency (steps) |
| `trans_cost` | 0.00 | Proportional transaction cost |
| `seed` | 42 | Random seed |

---

## Outputs

Running `main.py` produces the following figures in `outputs/`:

| File | Description |
|------|-------------|
| `scatter_gamma_pnl.png` | Gamma error vs hedge P&L scatter (with OLS and R²) |
| `pnl_distribution.png` | Overlaid P&L and gamma error distributions |
| `sample_paths.png` | Sample GBM paths with strike level |
| `rebal_comparison.png` | Std and mean hedging error across rebalancing frequencies |
| `hedge_trace.png` | Single-path 3-panel trace (price, delta, cumulative P&L) |

---

## Requirements

- Python ≥ 3.10 (uses `int | None` union syntax)
- See `requirements.txt`
