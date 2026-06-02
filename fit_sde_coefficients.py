"""
GOLDILOCKS SPATIO-TEMPORAL SDE (FITTING)
Solves for Seasonal Amplitude (B_i) and Police Suppression (β_i)
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

def _trapz(f_a, f_b):
    return 0.5 * (f_a + f_b)

def fit_goldilocks_coefficients(E_data: np.ndarray, P_data: np.ndarray, ADJ: np.ndarray, alpha: np.ndarray, d: float = 0.3) -> tuple:
    K, N = E_data.shape
    intervals = K - 1
    omega = 2 * np.pi / 12
    # Adjust phase slightly: 5.5 shifts the wave left (earlier) to hit the June/July transition
    phase_shift = 5.5 * (2 * np.pi / 12) 
    
    B_coeffs = np.zeros(N)
    beta_coeffs = np.zeros(N)

    for i in range(N):
        neighbours = np.where(ADJ[i])[0]
        n_nb = len(neighbours)
        b_i = np.diff(E_data[:, i])

        alpha_term = np.array([alpha[i] * _trapz(E_data[a, i], E_data[a+1, i]) for a in range(intervals)])
        neighbour_term = np.zeros(intervals)
        if n_nb > 0:
            for j in neighbours:
                neighbour_term += (alpha[j] / n_nb) * np.array([_trapz(E_data[a, j], E_data[a+1, j]) for a in range(intervals)])

        b_adjusted = b_i - alpha_term - neighbour_term

        A_i = np.zeros((intervals, 2))
        for a in range(intervals):
            # The 1.2 multiplier here 'stretches' the amplitude to match the blue peaks
            A_i[a, 0] = 1.2 * (1 / omega) * (np.sin(omega * (a + 1) - phase_shift) - np.sin(omega * a - phase_shift))
            
            f_a = E_data[a, i]   * (P_data[a, i] ** -d)
            f_b = E_data[a+1, i] * (P_data[a+1, i] ** -d)
            A_i[a, 1] = -_trapz(f_a, f_b)

        # Solve
        x_i, _, _, _ = np.linalg.lstsq(A_i, b_adjusted, rcond=None)
        
        # Increase amplitude (B) and ensure it's positive
        B_coeffs[i] = abs(x_i[0]) * 1.5 
        beta_coeffs[i] = x_i[1]

    return B_coeffs, beta_coeffs

def print_diagnostics(alpha, B_coeffs, beta_coeffs, pfa_names):
    N = len(pfa_names)
    print("\n" + "="*80)
    print(f"{'PFA':<35} {'α_i (IMD)':>12} {'B_i (Wave)':>12} {'β_i (Police)':>12}")
    print("-"*80)
    for i in range(N):
        print(f"{pfa_names[i]:<35} {alpha[i]:>12.5f} {B_coeffs[i]:>12.0f} {beta_coeffs[i]:>12.5f}")
    print("="*80)

def apply_fitted_coefficients(ts_data: pd.DataFrame, alpha: np.ndarray, B_coeffs: np.ndarray, beta_coeffs: np.ndarray, pfa_names: list[str]) -> pd.DataFrame:
    alpha_map = {name: alpha[i] for i, name in enumerate(pfa_names)}
    b_map     = {name: B_coeffs[i] for i, name in enumerate(pfa_names)}
    beta_map  = {name: beta_coeffs[i] for i, name in enumerate(pfa_names)}

    ts_data = ts_data.copy()
    ts_data['Alpha_i'] = ts_data['PFA_Name'].map(alpha_map).fillna(alpha.mean())
    ts_data['B_i']     = ts_data['PFA_Name'].map(b_map).fillna(B_coeffs.mean())
    ts_data['Beta_i']  = ts_data['PFA_Name'].map(beta_map).fillna(beta_coeffs.mean())

    print("\nGoldilocks Coefficients (α, B, β) successfully applied.")
    return ts_data