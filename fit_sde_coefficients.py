"""
NEW ECONOMETRIC MODEL (From Crimes CBL 22 SDE.pdf)
dE_i/dt = α_i·E_i + Σ (α_j / |B(i)|)·E_j - β_i·E_i·(P_i(t))^{-d}
"""

import numpy as np
import pandas as pd

def build_adj_matrix(pfa_names: list[str], adjacency_dict: dict) -> np.ndarray:
    N = len(pfa_names)
    idx = {name: i for i, name in enumerate(pfa_names)}
    ADJ = np.zeros((N, N), dtype=bool)
    for name, neighbours in adjacency_dict.items():
        if name not in idx: continue
        i = idx[name]
        for nb in neighbours:
            if nb in idx: ADJ[i, idx[nb]] = True
    return ADJ

def build_P_monthly(police_df: pd.DataFrame, months: list[str], pfa_names: list[str]) -> np.ndarray:
    K = len(months)
    N = len(pfa_names)
    P_data = np.zeros((K, N))
    month_frac = np.array([int(m[:4]) + (int(m[5:7]) - 1) / 12.0 for m in months])

    for j, pfa in enumerate(pfa_names):
        sub = police_df[police_df['PFA_Name'] == pfa].sort_values('Year')
        if sub.empty: continue
        years  = sub['Year'].values.astype(float)
        counts = sub['Police_Count'].values.astype(float)
        P_data[:, j] = np.interp(month_frac, years, counts)
    return P_data

def build_N_monthly(pop_df: pd.DataFrame, months: list[str], pfa_names: list[str]) -> np.ndarray:
    K = len(months)
    N = len(pfa_names)
    N_data = np.zeros((K, N))
    month_frac = np.array([int(m[:4]) + (int(m[5:7]) - 1) / 12.0 for m in months])

    for j, pfa in enumerate(pfa_names):
        sub = pop_df[pop_df['PFA_Name'] == pfa].sort_values('Year')
        if len(sub) < 2: continue
        years  = sub['Year'].values.astype(float)
        pops   = sub['Population'].values.astype(float)
        slope, intercept = np.polyfit(years, pops, 1)
        N_data[:, j] = slope * month_frac + intercept
    return N_data

def _trapz(f_a, f_b):
    """Trapezoid integral over one unit interval (one month = 1 unit)."""
    return 0.5 * (f_a + f_b)

# ─────────────────────────────────────────────────────────────────────────────
# NEW: ELASTICITY INTEGRAL LEAST SQUARES (PDF Option 2)
# ─────────────────────────────────────────────────────────────────────────────
def fit_elasticity_coefficient(
    E_data:    np.ndarray,        
    P_data:    np.ndarray,        
    ADJ:       np.ndarray,        
    alpha:     np.ndarray,        
    d:         float = 0.3,       # Elasticity exponent from empirical research
) -> np.ndarray:
    """
    Fits the β_i police suppression coefficient using Simultaneous Integral Equations.
    """
    K, N = E_data.shape
    intervals = K - 1
    beta_coeffs = np.zeros(N)

    for i in range(N):
        neighbours = np.where(ADJ[i])[0]
        n_nb = len(neighbours)

        # ── b_i: observed ΔE_i ───────────────────────────────────────────────
        b_i = np.diff(E_data[:, i])

        # ── subtract KNOWN α_i·∫E_i term ─────────────────────────────────────
        alpha_term = np.array([
            alpha[i] * _trapz(E_data[a, i], E_data[a+1, i])
            for a in range(intervals)
        ])

        # ── subtract KNOWN neighbour α_j·∫E_j / |B(i)| terms ─────────────────
        neighbour_term = np.zeros(intervals)
        if n_nb > 0:
            for j in neighbours:
                neighbour_term += (alpha[j] / n_nb) * np.array([
                    _trapz(E_data[a, j], E_data[a+1, j])
                    for a in range(intervals)
                ])

        b_adjusted = b_i - alpha_term - neighbour_term

        # ── single-column A_i: -∫ E_i * P_i^{-d} dt ──────────────────────────
        A_i = np.zeros((intervals, 1))
        for a in range(intervals):
            # Evaluate E_i * P_i^(-d) at month a and month a+1
            f_a = E_data[a, i]   * (P_data[a, i] ** -d)
            f_b = E_data[a+1, i] * (P_data[a+1, i] ** -d)
            A_i[a, 0] = -_trapz(f_a, f_b)

        # Solve for β_i
        x_i, _, _, _ = np.linalg.lstsq(A_i, b_adjusted, rcond=None)
        beta_coeffs[i] = x_i[0]

    return beta_coeffs

# ─────────────────────────────────────────────────────────────────────────────
# HOOKS & DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def print_diagnostics(alpha, beta_coeffs, pfa_names):
    N = len(pfa_names)
    print("\n" + "="*70)
    print(f"{'PFA':<35} {'α_i':>10} {'β_i (Police)':>12}  flags")
    print("-"*70)
    for i in range(N):
        flags = []
        if alpha[i] < 0: flags.append("α<0")
        if beta_coeffs[i] < 0: flags.append("β<0")
        flag_str = ", ".join(flags) if flags else "ok"
        print(f"{pfa_names[i]:<35} {alpha[i]:>10.5f} {beta_coeffs[i]:>12.5f}  {flag_str}")
    print("="*70)

def apply_fitted_coefficients(ts_data: pd.DataFrame, alpha: np.ndarray, beta_coeffs: np.ndarray, pfa_names: list[str]) -> pd.DataFrame:
    alpha_map = {name: alpha[i] for i, name in enumerate(pfa_names)}
    beta_map  = {name: beta_coeffs[i] for i, name in enumerate(pfa_names)}

    ts_data = ts_data.copy()
    ts_data['Alpha_i'] = ts_data['PFA_Name'].map(alpha_map).fillna(alpha.mean())
    ts_data['Beta_i']  = ts_data['PFA_Name'].map(beta_map).fillna(beta_coeffs.mean())

    print(f"\nFitted coefficients Alpha_i and Beta_i successfully applied to ts_data.")
    return ts_data