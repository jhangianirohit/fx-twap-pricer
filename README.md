# FX TWAP Optimal-Window Pricer & Risk Manager

Streamlit webapp for pricing and risk-managing the "client-grants-bank window-stopping option" TWAP structure.

## Quick start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

## What's in the app

### 💰 Pricer tab
Three premium estimates side-by-side:
- **Trader heuristic** — Σ of N 1-step ATM puts on 1/N notional (your desk rule of thumb). Not a bound.
- **LSM Monte Carlo** ★ — the actual GBM price via Longstaff-Schwartz.
- **Perfect foresight** — pathwise max, strict upper bound.

Plus sample paths with optimal stopping points and the LSM stopping-time histogram.

### 📊 Risk tab
- Embedded-option Greeks at t=0 (delta, gamma, vega, theta-per-fixing) via bump-and-reprice with common random numbers.
- TWAP execution schedule showing per-fix buys, cumulative bought, and residual short.
- Block-execution risk callout — the operational risk to flag for the desk.
- Premium vs vol scan with both LSM and heuristic curves overlaid.

### 🎚 Decision Tool tab
Plug in current state `(k, A, S)` and get a STOP/CONTINUE call from the LSM regression. Includes a 2D heatmap of the stopping boundary in `(S, A)` space at the chosen `k`, with current state marked.

### ℹ️ About tab
Math + risk framework writeup, suitable for sharing context with someone reviewing the model.

## Key result for the boss

**Plain TWAP is self-hedging.** The bank's TWAP commitment has zero residual delta at every moment because future revenue and future cost both scale 1-for-1 with spot. All the actual risk lives in the embedded option — small delta, meaningful vega, and the operational risk of the block square at the stopping moment.

## Files

- `streamlit_app.py` — single-file app (pricing core + UI)
- `requirements.txt` — pip dependencies
- `README.md` — this file

## To extend before going live

1. **Vol smile** — replace flat σ with K-dependent surface (K = running average at decision time)
2. **Forward points** — `r_d − r_f` carry, especially for longer windows
3. **Intraday vol seasonality** — London/NY open spikes for sub-day fixings
4. **Min-window constraint** — `τ ≥ k_min` (one-line change in the LSM loop)
5. **Pathwise/LR Greeks** — replace bump-and-reprice for production-grade Greek stability
6. **Bid/ask & slippage** on the block square at stopping
7. **Live spot feed** — connect to your FX feed for real-time MTM and decision support
