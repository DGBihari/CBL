"""
dC_i/dt = α_i·C_i  -  γ_i·ΔP_i  +  β·Σ_j w_ij·(C_j - C_i)
 
where:
  C_i    = E_i / Area_i        (crime density, crimes/km²)
  ΔP_i   = dP_i/dt             (month-to-month change in police count)
  w_ij   = 1 if PFA i borders j (from adjacency_dict), else 0
  β      = global spatial weight (default 0.15 — can also be globally fitted)

Inputs (expected)
---------------
E_data   : np.ndarray (K, N)   monthly crime COUNTS per PFA
             rows = months (K total), cols = PFAs (N=43)
P_data   : np.ndarray (K, N)   monthly police headcount per PFA
             (interpolate your annual police_data.csv to monthly first —
              helper function build_P_monthly() is provided below)
area     : np.ndarray (N,)     Area_Sq_Km per PFA (constant)
adj_dict : dict[str, list[str]] adjacency_dict from run_sde_pipeline.py
pfa_names: list[str]           master PFA name list, length N
 
Returns
-------
alpha : (N,)  α_i — local crime growth rate per region
gamma : (N,)  γ_i — police suppression effectiveness per region
beta  : float β   — spatial spillover weight
              (fixed value if OPTION A, or fitted value if OPTION B)

"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_adj_matrix(pfa_names: list[str], adjacency_dict: dict) -> np.ndarray:
    """Convert adjacency_dict → boolean (N, N) matrix."""
    N = len(pfa_names)
    idx = {name: i for i, name in enumerate(pfa_names)}
    ADJ = np.zeros((N, N), dtype=bool)
    for name, neighbours in adjacency_dict.items():
        if name not in idx:
            continue
        i = idx[name]
        for nb in neighbours:
            if nb in idx:
                ADJ[i, idx[nb]] = True
    return ADJ


def build_P_monthly(
    police_df:  pd.DataFrame,
    months:     list[str],
    pfa_names:  list[str],
) -> np.ndarray:
    """
    Linearly interpolate annual police headcount to monthly resolution.

    police_df must have columns: ['PFA_Name', 'Year', 'Police_Count']
    months    : list of 'YYYY-MM' strings (same as from build_E_data)
    pfa_names : master PFA list

    Returns P_data (K, N) — monthly police counts per region.
    """
    K = len(months)
    N = len(pfa_names)
    P_data = np.zeros((K, N))

    # Convert months to fractional years for interpolation
    month_frac = np.array([
        int(m[:4]) + (int(m[5:7]) - 1) / 12.0
        for m in months
    ])

    for j, pfa in enumerate(pfa_names):
        sub = police_df[police_df['PFA_Name'] == pfa].sort_values('Year')
        if sub.empty:
            continue
        years  = sub['Year'].values.astype(float)
        counts = sub['Police_Count'].values.astype(float)
        # np.interp extrapolates as flat beyond the endpoints
        P_data[:, j] = np.interp(month_frac, years, counts)

    return P_data


def build_N_monthly(
    pop_df:    pd.DataFrame,
    months:    list[str],
    pfa_names: list[str],
) -> np.ndarray:
    """
    Linearly REGRESS annual population to monthly resolution (per PDF spec).

    pop_df must have columns: ['PFA_Name', 'Year', 'Population']
    (aggregate your lad_pop_long → pfa_agg first, as in run_sde_pipeline.py)

    Returns N_data (K, N) — monthly civilian population per region.
    """
    K = len(months)
    N = len(pfa_names)
    N_data = np.zeros((K, N))

    month_frac = np.array([
        int(m[:4]) + (int(m[5:7]) - 1) / 12.0
        for m in months
    ])

    for j, pfa in enumerate(pfa_names):
        sub = pop_df[pop_df['PFA_Name'] == pfa].sort_values('Year')
        if len(sub) < 2:
            continue
        years  = sub['Year'].values.astype(float)
        pops   = sub['Population'].values.astype(float)
        slope, intercept = np.polyfit(years, pops, 1)
        N_data[:, j] = slope * month_frac + intercept

    return N_data


def _trapz(f_a: float | np.ndarray, f_b: float | np.ndarray) -> float | np.ndarray:
    """Trapezoid integral over one unit interval (one month = 1 unit)."""
    return 0.5 * (f_a + f_b)


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


# ─────────────────────────────────────────────────────────────────────────────
# OPTION B: Global fit for α_i, γ_i, AND β simultaneously
# ─────────────────────────────────────────────────────────────────────────────

def fit_global(
    E_data: np.ndarray,      # (K, N)
    P_data: np.ndarray,      # (K, N)
    area:   np.ndarray,      # (N,)
    ADJ:    np.ndarray,      # (N, N) bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Stack all regions into one big least-squares system and fit
    [α_1,..,α_N, γ_1,..,γ_N, β] simultaneously.

    The system has (K-1)*N rows and (2N+1) unknowns.
    With 43 regions and 60 months → 2537 rows, 87 unknowns — very well posed.

    Returns
    -------
    alpha : (N,)
    gamma : (N,)
    beta  : float
    """
    K, N = E_data.shape
    intervals = K - 1
    C_data  = E_data / area[np.newaxis, :]
    dP_data = np.diff(P_data, axis=0)

    total_rows = intervals * N
    n_params   = 2 * N + 1          # N alphas + N gammas + 1 beta

    A_big = np.zeros((total_rows, n_params))
    b_big = np.zeros(total_rows)

    for i in range(N):
        neighbours = np.where(ADJ[i])[0]
        row_start  = i * intervals

        for a in range(intervals):
            row = row_start + a

            # b vector: observed ΔC_i
            b_big[row] = C_data[a+1, i] - C_data[a, i]

            # col for α_i
            A_big[row, i]       =  _trapz(C_data[a, i],   C_data[a+1, i])

            # col for γ_i  (stored in columns N..2N-1)
            A_big[row, N + i]   = -_trapz(dP_data[a, i],
                                           dP_data[min(a+1, intervals-1), i])

            # col for β  (last column, index 2N)
            spatial_a   = sum(C_data[a,   j] - C_data[a,   i] for j in neighbours)
            spatial_b   = sum(C_data[a+1, j] - C_data[a+1, i] for j in neighbours)
            A_big[row, 2*N] = _trapz(spatial_a, spatial_b)

    x, _, _, _ = np.linalg.lstsq(A_big, b_big, rcond=None)

    alpha = x[:N]
    gamma = x[N:2*N]
    beta  = x[2*N]

    return alpha, gamma, beta


# ─────────────────────────────────────────────────────────────────────────────
# OPTION C: PDF Option 2 — fix α from deprivation index, fit only γ per region
# ─────────────────────────────────────────────────────────────────────────────

def fit_gamma_only(
    E_data:    np.ndarray,        # (K, N)
    P_data:    np.ndarray,        # (K, N)
    area:      np.ndarray,        # (N,)
    ADJ:       np.ndarray,        # (N, N) bool
    alpha:     np.ndarray,        # (N,)  fixed α_i from deprivation index
    beta:      float = 0.15,      # fixed spatial weight
) -> np.ndarray:
    """
    PDF Option 2: α_i and α_j are known (set from deprivation index),
    so we only need to solve for γ_i per region.

    Each region i becomes a single-column least squares problem:
        b_i_adjusted = A_i @ [γ_i]

    where b_i_adjusted subtracts the known α terms from the observed ΔC_i.

    This is much more stable than fitting α and γ simultaneously because:
    - The system is better conditioned (1 unknown vs 2)
    - α is physically grounded in deprivation data
    - γ is cleanly identified from residual variation

    Returns
    -------
    gamma : (N,)  police effectiveness per region (should be positive)
    """
    K, N     = E_data.shape
    intervals = K - 1
    C_data   = E_data / area[np.newaxis, :]
    dP_data  = np.diff(P_data, axis=0)

    gamma = np.zeros(N)

    for i in range(N):
        neighbours = np.where(ADJ[i])[0]

        # ── b_i: observed ΔC_i ───────────────────────────────────────────────
        b_i = np.diff(C_data[:, i])

        # ── subtract KNOWN α_i·∫C_i term ─────────────────────────────────────
        alpha_term = np.array([
            alpha[i] * _trapz(C_data[a, i], C_data[a+1, i])
            for a in range(intervals)
        ])

        # ── subtract KNOWN neighbour α_j·∫C_j / |B(i)| terms ─────────────────
        # PDF Option 2 scales each neighbour by 1/|B(i)|
        n_nb = len(neighbours)
        neighbour_term = np.zeros(intervals)
        if n_nb > 0:
            for j in neighbours:
                neighbour_term += (alpha[j] / n_nb) * np.array([
                    _trapz(C_data[a, j], C_data[a+1, j])
                    for a in range(intervals)
                ])

        # ── subtract KNOWN spatial spillover β term ───────────────────────────
        spatial = np.zeros(intervals)
        for j in neighbours:
            diff_a = C_data[:-1, j] - C_data[:-1, i]
            diff_b = C_data[1:,  j] - C_data[1:,  i]
            spatial += beta * _trapz(diff_a, diff_b)

        b_adjusted = b_i - alpha_term - neighbour_term - spatial

        # ── single-column A_i: just -∫ΔP_i dt ────────────────────────────────
        A_i = np.array([
            [-_trapz(dP_data[a, i], dP_data[min(a+1, intervals-1), i])]
            for a in range(intervals)
        ])

        x_i, _, _, _ = np.linalg.lstsq(A_i, b_adjusted, rcond=None)
        gamma[i] = x_i[0]

    return gamma


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_diagnostics(alpha, gamma, pfa_names, beta=None):
    """Print a tidy summary table and flag suspicious values."""
    N = len(pfa_names)
    print("\n" + "="*70)
    print(f"{'PFA':<35} {'α_i':>10} {'γ_i':>10}  flags")
    print("-"*70)
    for i in range(N):
        flags = []
        if alpha[i] < 0:
            flags.append("α<0")
        if gamma[i] < 0:
            flags.append("γ<0")
        if abs(alpha[i]) > 1:
            flags.append("|α|>1")
        flag_str = ", ".join(flags) if flags else "ok"
        print(f"{pfa_names[i]:<35} {alpha[i]:>10.5f} {gamma[i]:>10.5f}  {flag_str}")

    print("="*70)
    print(f"\nα_i:  mean={alpha.mean():.5f}  std={alpha.std():.5f}  "
          f"negatives={( alpha<0).sum()}/{N}")
    print(f"γ_i:  mean={gamma.mean():.5f}  std={gamma.std():.5f}  "
          f"negatives={(gamma<0).sum()}/{N}")
    if beta is not None:
        print(f"β  :  {beta:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION HOOK: slot fitted coefficients back into run_sde_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def apply_fitted_coefficients(ts_data: pd.DataFrame,
                               alpha: np.ndarray,
                               gamma: np.ndarray,
                               pfa_names: list[str],
                               beta: float) -> pd.DataFrame:
    """
    Replace the mock coefficient columns in ts_data with fitted values.
    Call this just before Section 6 (run_spatial_sde) in run_sde_pipeline.py.

    Replaces:
        ts_data['Alpha_i']  ← fitted α_i  (was IMD_Score * 0.005)
        ts_data['Gamma_i']  ← fitted γ_i  (was 0.05 / pop_density)
    and returns the scalar β to pass into calculate_spatial_spillover().
    """
    alpha_map = {name: alpha[i] for i, name in enumerate(pfa_names)}
    gamma_map = {name: gamma[i] for i, name in enumerate(pfa_names)}

    ts_data = ts_data.copy()
    # Assign directly — these columns may not exist yet, so don't fillna from them.
    # Any PFA not in the fitted list gets the national mean as a safe fallback.
    ts_data['Alpha_i'] = ts_data['PFA_Name'].map(alpha_map).fillna(alpha.mean())
    ts_data['Gamma_i'] = ts_data['PFA_Name'].map(gamma_map).fillna(gamma.mean())

    print(f"\nCoefficients applied to ts_data. β = {beta:.5f}")
    print("Reminder: pass β into calculate_spatial_spillover(..., beta=beta)")

    return ts_data


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: example usage wired to run_sde_pipeline.py outputs
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── STEP 0: imports from your existing pipeline ───────────────────────────
    # Assumes you run this from the same directory as run_sde_pipeline.py
    # and that build_E_data-matrix.py has already built E_data.
    import sys
    sys.path.insert(0, ".")

    # You need to have already produced these from your pipeline:
    #   E_data       : (K, N)  from build-E_data-matrix.py
    #   months       : list[str]
    #   master_pfas  : list[str]  from get_master_pfa_list_from_lookup()
    #   adjacency_dict : dict     from run_sde_pipeline.py section 2
    #   pfa_agg      : DataFrame  from run_sde_pipeline.py section 3 (pop data)
    #   police_df    : DataFrame  from run_sde_pipeline.py section 3

    # ── STEP 1: build monthly P and N matrices ────────────────────────────────
    # Example — replace with your actual loaded variables:
    print("NOTE: Replace the placeholder variables below with your real data.")
    print("      This script is meant to be imported, not run standalone.")
    print("      See INTEGRATION EXAMPLE below.\n")

    print("""
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
""")