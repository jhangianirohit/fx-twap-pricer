"""
FX TWAP Optimal-Window Pricer & Risk Manager
============================================
Single-file Streamlit app for the "client grants bank window-stopping option"
TWAP structure.  Run:

    pip install -r requirements.txt
    streamlit run streamlit_app.py

Trade structure
---------------
Client buys Notional EUR/USD over N hourly fixings.  At any fixing τ ∈ {1..N-1}
the bank may STOP the window: client pays the average of the τ kept fixings,
bank squares the unfilled (N-τ)/N at the prevailing spot.  Bank's edge per unit
notional at stopping is (N-τ)/N × (A_{τ-1} - S_τ).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
import plotly.graph_objects as go

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="FX TWAP Optimal-Window Pricer",
    layout="wide",
    initial_sidebar_state="expanded",
)

HOURS_PER_YEAR = 252 * 24  # FX convention: business days × 24h


# ============================================================
# Pricing core
# ============================================================
def bs_put(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    v = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / v
    d2 = d1 - v
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def heuristic_premium(S: float, sigma: float, dt_hours: float) -> float:
    """Trader heuristic: sum of N forward-start ATM puts on 1/N notional each.
    Collapses to bs_put(S, S, Δt) under r=q=0.  NOT a bound on the true price."""
    dt = dt_hours / HOURS_PER_YEAR
    return bs_put(S, S, dt, sigma)


@st.cache_data(show_spinner=False)
def simulate_paths(S0: float, sigma: float, dt_hours: float, N: int,
                   n_paths: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = dt_hours / HOURS_PER_YEAR
    z = rng.standard_normal((n_paths, N))
    log_steps = -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_steps, axis=1)
    return S0 * np.exp(log_paths)


def prior_avgs(paths: np.ndarray) -> np.ndarray:
    """A[:, k] = mean(path[:, 0..k-1]) for k ≥ 1; A[:, 0] is undefined (set to 0)."""
    N = paths.shape[1]
    cumsum = np.cumsum(paths, axis=1)
    pA = np.zeros_like(paths)
    pA[:, 1:] = cumsum[:, :-1] / np.arange(1, N)
    return pA


def exercise_value(prior_A, S, k: int, N: int):
    return (N - k) / N * (prior_A - S)


def perfect_foresight(paths: np.ndarray):
    n_paths, N = paths.shape
    pA = prior_avgs(paths)
    payoffs = np.zeros((n_paths, N))
    for k in range(1, N):
        payoffs[:, k] = np.maximum(0.0, exercise_value(pA[:, k], paths[:, k], k, N))
    best = payoffs.max(axis=1)
    best_k = payoffs.argmax(axis=1)
    best_k = np.where(best > 0, best_k, N)
    return float(best.mean()), best_k


def longstaff_schwartz(paths: np.ndarray):
    """Returns (premium, stop_k array, regression betas {k: beta vector})."""
    n_paths, N = paths.shape
    pA = prior_avgs(paths)
    cashflow = np.zeros(n_paths)
    stop_k = np.full(n_paths, N, dtype=int)  # N = never stopped
    betas: dict[int, np.ndarray] = {}

    for k in range(N - 1, 0, -1):
        S_k = paths[:, k]
        A_k = pA[:, k]
        ex = exercise_value(A_k, S_k, k, N)
        in_money = ex > 0
        if in_money.sum() < 12:
            continue

        S_im = S_k[in_money]
        A_im = A_k[in_money]
        ex_im = ex[in_money]
        cf_im = cashflow[in_money]

        # Basis: 1, S, A, S², A², S·A
        X = np.column_stack([
            np.ones_like(S_im), S_im, A_im,
            S_im ** 2, A_im ** 2, S_im * A_im,
        ])
        beta, *_ = np.linalg.lstsq(X, cf_im, rcond=None)
        betas[k] = beta
        cont = X @ beta

        exercise = ex_im > np.maximum(cont, 0.0)
        full_idx = np.where(in_money)[0][exercise]
        cashflow[full_idx] = ex_im[exercise]
        stop_k[full_idx] = k

    return float(cashflow.mean()), stop_k, betas


@st.cache_data(show_spinner=False)
def price_all(S0, sigma, dt_hours, N, n_paths, seed):
    paths = simulate_paths(S0, sigma, dt_hours, N, n_paths, seed)
    heur = heuristic_premium(S0, sigma, dt_hours)
    lsm_prem, lsm_stops, lsm_betas = longstaff_schwartz(paths)
    pf_prem, pf_stops = perfect_foresight(paths)
    return {
        "paths": paths,
        "heuristic": heur,
        "lsm": lsm_prem,
        "lsm_stops": lsm_stops,
        "lsm_betas": lsm_betas,
        "perfect": pf_prem,
        "pf_stops": pf_stops,
    }


def price_only(S0, sigma, dt_hours, N, n_paths, seed):
    """Used by Greek bumping (no need to keep stops/betas)."""
    paths = simulate_paths(S0, sigma, dt_hours, N, n_paths, seed)
    lsm, _, _ = longstaff_schwartz(paths)
    return lsm


@st.cache_data(show_spinner=False)
def compute_greeks(S0, sigma, dt_hours, N, n_paths, seed):
    """Bump-and-reprice greeks at t=0 with common random numbers."""
    base = price_only(S0, sigma, dt_hours, N, n_paths, seed)

    # Delta & gamma: bump spot by 5bps
    h_S = S0 * 5e-4
    p_up = price_only(S0 + h_S, sigma, dt_hours, N, n_paths, seed)
    p_dn = price_only(S0 - h_S, sigma, dt_hours, N, n_paths, seed)
    delta = (p_up - p_dn) / (2 * h_S)
    gamma = (p_up - 2 * base + p_dn) / (h_S ** 2)

    # Vega: bump vol by 1 vol point (1%)
    h_v = 0.01
    v_up = price_only(S0, sigma + h_v, dt_hours, N, n_paths, seed)
    v_dn = price_only(S0, sigma - h_v, dt_hours, N, n_paths, seed)
    vega = (v_up - v_dn) / (2 * h_v)

    # "Theta": value lost per fewer fixing (proxy for time decay)
    if N > 2:
        v_short = price_only(S0, sigma, dt_hours, N - 1, n_paths, seed)
        theta_per_fix = v_short - base  # negative — option decays
    else:
        theta_per_fix = 0.0

    return {
        "price": base,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta_per_fix": theta_per_fix,
    }


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Trade parameters")
    pair = st.text_input("Pair", value="EURUSD")
    spot = st.number_input("Spot", value=1.0850, step=0.0001, format="%.4f")
    vol_pct = st.number_input("ATM Vol (%)", value=8.0, step=0.1)
    N = int(st.number_input("Fixings (N)", value=11, min_value=2, max_value=100, step=1))
    dt_hours = st.number_input("Fixing interval (hours)", value=1.0, step=0.5)
    notional_mm = st.number_input("Notional (mm EUR)", value=250.0, step=10.0)

    st.divider()
    st.header("MC settings")
    n_paths = int(st.number_input("MC paths", value=8000, min_value=1000, step=1000))
    seed = int(st.number_input("Random seed", value=42, step=1))

sigma = vol_pct / 100.0
notional = notional_mm * 1e6
results = price_all(spot, sigma, dt_hours, N, n_paths, seed)

# ============================================================
# Header
# ============================================================
st.title("FX TWAP Optimal-Window Pricer & Risk Manager")
st.caption(
    f"**{pair}**  ·  spot {spot:.4f}  ·  σ {vol_pct:.1f}%  ·  "
    f"{N} fixings × {dt_hours:.1f}h  ·  notional {notional_mm:.0f}mm EUR"
)

# ============================================================
# Tabs
# ============================================================
tab_price, tab_risk, tab_decision, tab_about = st.tabs(
    ["💰 Pricer", "📊 Risk", "🎚 Decision Tool", "ℹ️ About"]
)

# ------------------------------------------------------------
# TAB 1 — PRICER
# ------------------------------------------------------------
with tab_price:
    heur_pips = results["heuristic"] * 10000
    lsm_pips = results["lsm"] * 10000
    pf_pips = results["perfect"] * 10000
    usd_per_pip = notional * 0.0001

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Trader heuristic",
        f"{heur_pips:.2f} pips",
        help="Sum of N 1-step ATM puts on 1/N notional. Not a bound — see About tab.",
    )
    c2.metric(
        "LSM Monte Carlo  ★",
        f"{lsm_pips:.2f} pips",
        delta=f"{lsm_pips - heur_pips:+.2f} vs heuristic",
        help="The actual price under GBM via Longstaff-Schwartz.",
    )
    c3.metric(
        "Perfect foresight (UB)",
        f"{pf_pips:.2f} pips",
        help="Pathwise max — strict upper bound (uses future info).",
    )

    st.success(
        f"**LSM premium** = ${lsm_pips * usd_per_pip:,.0f}  ·  "
        f"client fills at **average − {lsm_pips:.1f} pips** on "
        f"{notional_mm:.0f}mm {pair}  ·  1 pip ≈ ${usd_per_pip:,.0f}"
    )

    st.subheader("Sample paths with optimal stopping")
    fig = go.Figure()
    n_show = min(50, n_paths)
    for i in range(n_show):
        path = results["paths"][i]
        fig.add_trace(go.Scatter(
            x=list(range(1, N + 1)), y=path, mode="lines",
            line=dict(color="rgba(80,90,120,0.25)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    # Stop markers (separate trace so it's clean)
    stop_x, stop_y = [], []
    for i in range(n_show):
        sk = int(results["lsm_stops"][i])
        if 1 <= sk < N:
            stop_x.append(sk + 1)  # chart positions are 1-indexed
            stop_y.append(results["paths"][i][sk])
    if stop_x:
        fig.add_trace(go.Scatter(
            x=stop_x, y=stop_y, mode="markers",
            marker=dict(color="#dc2626", size=8),
            name="LSM stop", showlegend=True,
        ))
    fig.update_layout(
        xaxis_title="Fixing #", yaxis_title=pair,
        height=420, margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution of LSM stopping decisions")
    counts = np.bincount(results["lsm_stops"], minlength=N + 1)
    labels = [f"k={k}" for k in range(1, N)] + ["never"]
    probs = list(counts[1:N] / n_paths) + [counts[N] / n_paths]
    colors = ["#3b82f6"] * (N - 1) + ["#9ca3af"]
    fig2 = go.Figure(go.Bar(x=labels, y=probs, marker_color=colors))
    fig2.update_layout(
        height=300, margin=dict(l=40, r=20, t=20, b=40),
        yaxis_tickformat=".0%", xaxis_title="Stop after k fixings kept",
        yaxis_title="Probability",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# TAB 2 — RISK
# ------------------------------------------------------------
with tab_risk:
    st.subheader("Embedded option Greeks at t=0")

    with st.spinner("Bumping for Greeks…"):
        greeks = compute_greeks(spot, sigma, dt_hours, N, min(n_paths, 8000), seed)

    g1, g2, g3, g4 = st.columns(4)
    opt_delta_eur_mm = greeks["delta"] * notional / 1e6
    g1.metric(
        "Delta (∂V/∂S)",
        f"{greeks['delta']:.5f}",
        help="Per unit notional. Small & positive at t=0 by homogeneity (r_d=r_f).",
    )
    g2.metric(
        "Spot-equiv hedge",
        f"{opt_delta_eur_mm:+.2f}mm EUR",
        help="Additional spot position to hold beyond the TWAP execution itself.",
    )
    g3.metric(
        "Vega (per 1 vol-pt)",
        f"{greeks['vega'] * 100 * 10000:.1f} pips",
        help="dV/dσ — you are long vol on this trade.",
    )
    g4.metric(
        "θ per fixing decay",
        f"{greeks['theta_per_fix'] * 10000:+.2f} pips",
        help="Drop in V from removing one fixing — proxy for time decay.",
    )

    st.info(
        f"**Plain-TWAP delta is self-hedging.** The bank's TWAP commitment has zero "
        f"residual delta because future revenue ($N × E[A_N]$) and future cost "
        f"($\\sum E[S_k]$) both scale 1-for-1 with spot. The {opt_delta_eur_mm:+.2f}mm EUR "
        f"figure above is the **option-only** delta — additional spot to hold on top of "
        f"the execution schedule. It's small at inception and grows in magnitude as you "
        f"approach decision points where stopping becomes attractive."
    )

    st.divider()
    st.subheader("TWAP execution schedule")

    fix_eur = notional / N
    sched = pd.DataFrame({
        "Fix #": range(1, N + 1),
        "t (h)": [k * dt_hours for k in range(1, N + 1)],
        "Buy this fix": [f"{fix_eur/1e6:.2f}mm EUR" for _ in range(N)],
        "Cum. bought":  [f"{k * fix_eur/1e6:.2f}mm EUR" for k in range(1, N + 1)],
        "Residual short pre-fix": [f"{(N - k + 1) * fix_eur/1e6:.2f}mm EUR" for k in range(1, N + 1)],
        "Residual short post-fix": [f"{(N - k) * fix_eur/1e6:.2f}mm EUR" for k in range(1, N + 1)],
    })
    st.dataframe(sched, hide_index=True, use_container_width=True)

    st.warning(
        f"⚠️ **Block-execution risk at stopping.** If LSM signals stop at fix τ, the bank "
        f"must immediately source **(N−τ)/N × {notional_mm:.0f}mm EUR** at the prevailing "
        f"spot. Worst case is stop at τ=1, requiring a single block trade of "
        f"**{(N-1)/N * notional_mm:.1f}mm EUR**. Best case stop at τ=N−1 needs only "
        f"**{1/N * notional_mm:.1f}mm**. Operationally, work this carefully — slippage "
        f"on the block eats directly into the option premium."
    )

    st.divider()
    st.subheader("Premium vs ATM vol")

    @st.cache_data(show_spinner=False)
    def vol_scan(spot, dt_hours, N, n_paths, seed, vols_tuple):
        vols = np.array(vols_tuple)
        lsm_pts, heur_pts = [], []
        for v in vols:
            lsm, _, _ = longstaff_schwartz(simulate_paths(spot, v, dt_hours, N, n_paths, seed))
            lsm_pts.append(lsm * 10000)
            heur_pts.append(heuristic_premium(spot, v, dt_hours) * 10000)
        return vols, lsm_pts, heur_pts

    vols = tuple(np.linspace(0.04, 0.16, 13).round(4))
    with st.spinner("Vol scan…"):
        v_arr, lsm_pts, heur_pts = vol_scan(spot, dt_hours, N, min(n_paths, 5000), seed, vols)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=v_arr * 100, y=lsm_pts, mode="lines+markers",
                              name="LSM", line=dict(color="#2563eb", width=2)))
    fig3.add_trace(go.Scatter(x=v_arr * 100, y=heur_pts, mode="lines+markers",
                              name="Heuristic", line=dict(color="#9ca3af", dash="dash")))
    fig3.add_vline(x=vol_pct, line_dash="dot", line_color="#444",
                   annotation_text=f"current σ={vol_pct:.1f}%", annotation_position="top right")
    fig3.update_layout(
        xaxis_title="ATM vol (%)", yaxis_title="Premium (pips)",
        height=380, margin=dict(l=40, r=20, t=20, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------
# TAB 3 — DECISION TOOL
# ------------------------------------------------------------
with tab_decision:
    st.subheader("Live decision tool")
    st.caption(
        "Given current trade state, evaluate whether to STOP or CONTINUE. "
        "Uses the LSM regression coefficients fitted from the MC paths."
    )

    d1, d2, d3 = st.columns(3)
    cur_k = int(d1.number_input(
        "Fixings completed (k)", value=N // 2, min_value=1, max_value=N - 1, step=1,
    ))
    cur_avg = d2.number_input("Running prior average", value=spot, step=0.0001, format="%.4f")
    cur_spot = d3.number_input("Current spot", value=spot * 0.998, step=0.0001, format="%.4f")

    ex_v = exercise_value(cur_avg, cur_spot, cur_k, N)
    ex_pips = ex_v * 10000

    if cur_k in results["lsm_betas"]:
        beta = results["lsm_betas"][cur_k]
        X_now = np.array([1, cur_spot, cur_avg, cur_spot ** 2, cur_avg ** 2, cur_spot * cur_avg])
        cont_v = max(float(X_now @ beta), 0.0)
        cont_pips = cont_v * 10000

        r1, r2, r3 = st.columns(3)
        r1.metric("Stop now → payoff", f"{ex_pips:+.2f} pips")
        r2.metric("Continue → expected", f"{cont_pips:+.2f} pips")
        if ex_v > cont_v and ex_v > 0:
            r3.error("### 🛑 STOP NOW")
        else:
            r3.success("### ✅ CONTINUE")

        # Notional context
        unfilled_eur = (N - cur_k) / N * notional
        st.caption(
            f"If you stop now: source **{unfilled_eur/1e6:.2f}mm EUR** at spot {cur_spot:.4f}. "
            f"Realised P&L = {ex_pips:.2f} pips × {notional_mm:.0f}mm = "
            f"**${ex_v * notional:,.0f}**."
        )

        st.divider()
        st.subheader(f"Stopping boundary at k = {cur_k}")
        st.caption("Red region = stop, Blue = continue. The black ✕ is your current state.")

        S_grid = np.linspace(spot * 0.97, spot * 1.03, 60)
        A_grid = np.linspace(spot * 0.97, spot * 1.03, 60)
        SS, AA = np.meshgrid(S_grid, A_grid)
        ex_grid = exercise_value(AA, SS, cur_k, N)
        X_grid = np.column_stack([
            np.ones(SS.size), SS.ravel(), AA.ravel(),
            SS.ravel() ** 2, AA.ravel() ** 2, SS.ravel() * AA.ravel(),
        ])
        cont_grid = np.maximum(X_grid @ beta, 0.0).reshape(SS.shape)
        # 1 = stop, 0 = continue, masked region = ex<=0 (continue trivially)
        decision = np.where((ex_grid > cont_grid) & (ex_grid > 0), 1.0, 0.0)

        fig4 = go.Figure()
        fig4.add_trace(go.Heatmap(
            x=S_grid, y=A_grid, z=decision,
            colorscale=[[0, "#dbeafe"], [1, "#fca5a5"]],
            showscale=False,
            hovertemplate="S=%{x:.4f}<br>A=%{y:.4f}<br>%{z:.0f}<extra></extra>",
        ))
        fig4.add_trace(go.Scatter(
            x=[cur_spot], y=[cur_avg], mode="markers",
            marker=dict(color="black", size=14, symbol="x", line=dict(width=2)),
            name="current state", showlegend=False,
        ))
        # A=S diagonal (boundary of "in the money")
        fig4.add_trace(go.Scatter(
            x=S_grid, y=S_grid, mode="lines",
            line=dict(color="rgba(0,0,0,0.4)", dash="dot"),
            name="A = S (no edge)", showlegend=True,
        ))
        fig4.update_layout(
            xaxis_title="Current spot S",
            yaxis_title="Running prior average A",
            height=520, margin=dict(l=40, r=20, t=20, b=40),
            yaxis_scaleanchor="x", yaxis_scaleratio=1,
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning(f"No LSM regression available at k={cur_k} (too few in-money paths). Try a different k.")

# ------------------------------------------------------------
# TAB 4 — ABOUT
# ------------------------------------------------------------
with tab_about:
    st.markdown(r"""
### Trade structure

Client commits to buying `Notional` of EURUSD via `N` hourly fixings. **Crucially**, the
client grants the bank a window-stopping option: at any fixing $\tau \in \{1, \ldots, N-1\}$
the bank may stop. If stopped, the client pays the average of the $\tau$ kept fixings,
and the bank squares the unfilled $(N-\tau)/N$ at the prevailing spot.

**Bank's edge per unit notional at stopping time $\tau$:**

$$
V_\tau \;=\; \frac{N-\tau}{N}\,\big(A_{\tau-1} \;-\; S_\tau\big)
$$

where $A_{\tau-1} = \tfrac{1}{\tau}\sum_{i=0}^{\tau-1} S_i$ is the average of the
**kept** fixings (not including the stopping observation), and $S_\tau$ is the spot
at the stopping time (the rate at which we square the residual).

**Premium** = $\max_{\tau} \mathbb{E}[V_\tau]$, computed via Longstaff-Schwartz Monte Carlo.

### Why the trader heuristic isn't a bound

The intuition "I'm long N 1-step ATM puts on 1/N notional each" prices *N independent
options*. Under $r=q=0$ this collapses to $\text{bsPut}(S, S, \Delta t, \sigma)$ — a number
that depends only on the **fixing interval** $\Delta t$, not on $N$ or the total window $T$.

The actual trade has vega on the **whole window** $T$, because the optimal stopping rule
roams the full path looking for the best $(A - S)$ gap, and $\text{Var}(A_k - S_k) \approx T_k/3$
grows linearly with elapsed time. So:

- **Small $N$** (e.g. 3): heuristic *over*-states the price.
- **Large $N$** at fixed $\Delta t$: heuristic *under*-states (LSM grows like $\sqrt{T}$).
- **Large $N$** at fixed total window $T$: heuristic crashes (smaller $\Delta t$) while LSM
  is roughly stable (same total optionality, just sampled finer).

The relationship crosses around $N \approx 6{-}8$ for typical hourly-fixing parameters.

### Risk framework

**Plain TWAP is self-hedging.** At any time $t$ with $k$ fixings done:

$$\text{PV}_t^{\text{TWAP}} \;=\; (N-k)\,S_t \;-\; (N-k)\,S_t \;=\; 0$$

Future revenue and future cost both scale 1-for-1 with spot, so the TWAP commitment has
**zero net delta** at every moment. The act of executing 1/N at each fixing IS the hedge —
no extra position required.

**All risk lives in the embedded option.** What you actually manage:

- **Delta**: small, positive at $t=0$ by GBM homogeneity ($V/S_0$). Grows in magnitude
  near decision boundaries.
- **Gamma**: meaningful near the stopping boundary — the stop/continue decision causes
  a discrete jump in your hedge.
- **Vega**: you are long vol. Higher vol = wider $(A-S)$ distribution = more option value.
- **Theta**: option decays as remaining decision points elapse without exercise.
- **Block-execution risk at stopping**: when you stop at $\tau$, you must source
  $(N-\tau)/N$ × notional in one shot. This is the dominant operational risk and the
  most likely place to leak the option premium back to the market via slippage.

### Caveats / things to extend before going live

- $r_d = r_f = 0$ (fine for sub-day; trivial to add forward points)
- Flat ATM vol — no smile, no intraday vol seasonality
- GBM dynamics
- LSM polynomial basis is 6 terms — adequate for $N \le 20$, may want richer basis for larger $N$
- No bid/ask, no execution slippage on the block square
- Greeks computed by finite differences with common random numbers — pathwise / likelihood-ratio
  would be lower-variance for production
""")
