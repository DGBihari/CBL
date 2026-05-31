
'''print("""
─── INTEGRATION EXAMPLE ───────────────────────────────────────────────────────

# --- After Section 3 of run_sde_pipeline.py ---

from fit_sde_coefficients import (
    build_adj_matrix, build_P_monthly, build_N_monthly,
    fit_per_region, fit_global,
    print_diagnostics, apply_fitted_coefficients
)
from build_E_data_matrix import build_E_data, align_to_master_pfa_list, get_master_pfa_list_from_lookup

# 1. Get monthly crime data
E_raw, months, e_pfa_names = build_E_data()
master_pfas  = get_master_pfa_list_from_lookup()
E_data       = align_to_master_pfa_list(E_raw, e_pfa_names, master_pfas)  # (K=60, N=43)

# 2. Get area per PFA (from your existing pipeline)
area = (
    ts_data.groupby('PFA_Name')['Area_Sq_Km']
           .first()
           .reindex(master_pfas)
           .fillna(ts_data['Area_Sq_Km'].mean())
           .values
)   # (N,)

# 3. Build monthly police and population matrices
P_data = build_P_monthly(police_df, months, master_pfas)   # (K, N)
N_data = build_N_monthly(pfa_agg,   months, master_pfas)   # (K, N)  optional

# 4. Build adjacency matrix
ADJ = build_adj_matrix(master_pfas, adjacency_dict)         # (N, N)

# 5a. Fit with β fixed (fast, Option A)
alpha, gamma = fit_per_region(E_data, P_data, area, ADJ, beta=0.15)
beta = 0.15
print_diagnostics(alpha, gamma, master_pfas, beta)

# 5b. OR: Fit β from data too (Option B — recommended if you have 40+ months)
alpha, gamma, beta = fit_global(E_data, P_data, area, ADJ)
print_diagnostics(alpha, gamma, master_pfas, beta)

# 6. Slot back into ts_data (replaces mock Alpha_i and Gamma_i)
ts_data = apply_fitted_coefficients(ts_data, alpha, gamma, master_pfas, beta)

# Then continue to Section 6 (run_spatial_sde) as normal,
# but pass beta into calculate_spatial_spillover.
───────────────────────────────────────────────────────────────────────────────
""")'''
    
'''
# ─────────────────────────────────────────────────────────────────────────────
# OPTION A: Per-region fit with β fixed
# ─────────────────────────────────────────────────────────────────────────────

def fit_per_region(
    E_data:    np.ndarray,        # (K, N)
    P_data:    np.ndarray,        # (K, N)
    area:      np.ndarray,        # (N,)  Area_Sq_Km
    ADJ:       np.ndarray,        # (N, N) bool
    beta:      float = 0.15,      # fixed spatial weight
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit α_i and γ_i independently for each region i.

    For each consecutive month pair [m_a, m_{a+1}]:

        ΔC_i[a]  =  α_i · ∫C_i dt  -  γ_i · ∫ΔP_i dt  +  spatial_offset[a]

    where spatial_offset[a] = β · Σ_j ADJ[i,j] · ∫(C_j - C_i) dt  (known)

    Rearranging gives a 2-column least-squares system per region:
        b_i_adjusted = A_i @ [α_i, γ_i]

    Returns
    -------
    alpha : (N,)
    gamma : (N,)
    """
    K, N = E_data.shape
    intervals = K - 1

    # Crime DENSITY: C_i = E_i / area_i
    C_data  = E_data / area[np.newaxis, :]          # (K, N)

    # Month-to-month police CHANGE: ΔP_i[a] = P[a+1] - P[a]
    dP_data = np.diff(P_data, axis=0)               # (K-1, N)

    alpha = np.zeros(N)
    gamma = np.zeros(N)

    for i in range(N):
        neighbours = np.where(ADJ[i])[0]

        # ── b_i: observed crime density change ────────────────────────────────
        b_i = np.diff(C_data[:, i])                 # (K-1,)

        # ── subtract known spatial term from b_i ──────────────────────────────
        # spatial[a] = β · Σ_j ADJ[i,j] · trapz(C_j[a]-C_i[a], C_j[a+1]-C_i[a+1])
        spatial = np.zeros(intervals)
        for j in neighbours:
            diff_a   = C_data[:-1, j] - C_data[:-1, i]   # (K-1,)
            diff_b   = C_data[1:,  j] - C_data[1:,  i]
            spatial += beta * _trapz(diff_a, diff_b)

        b_adjusted = b_i - spatial                  # (K-1,)

        # ── A_i: 2-column matrix ──────────────────────────────────────────────
        # col 0:  ∫C_i dt          → α_i
        # col 1: -∫ΔP_i dt         → γ_i  (negative so γ comes out positive)
        A_i = np.zeros((intervals, 2))
        for a in range(intervals):
            A_i[a, 0] =  _trapz(C_data[a, i],  C_data[a+1, i])
            A_i[a, 1] = -_trapz(dP_data[a, i], dP_data[a+1, i])
            # Note: ΔP at m_{a+1} = P[a+2]-P[a+1], but ΔP at m_a = P[a+1]-P[a]
            # Using dP_data[a] as the value at m_a and dP_data[a+1] at m_{a+1}
            # is valid as long as K >= 3

        # ── solve ─────────────────────────────────────────────────────────────
        if intervals < 2:
            print(f"  Region {i}: too few months to fit — skipping")
            continue

        x_i, _, _, _ = np.linalg.lstsq(A_i, b_adjusted, rcond=None)
        alpha[i] = x_i[0]
        gamma[i] = x_i[1]

    return alpha, gamma
'''