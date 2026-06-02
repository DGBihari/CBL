import pandas as pd
import numpy as np
import geopandas as gpd
import warnings
import glob
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from fit_sde_coefficients import (
    build_adj_matrix,
    build_P_monthly,
    fit_goldilocks_coefficients,
    print_diagnostics,
    apply_fitted_coefficients,
)

from build_E_data_matrix import (
    build_E_data,
    align_to_master_pfa_list,
    get_master_pfa_list_from_lookup,
)

warnings.filterwarnings('ignore', category=UserWarning)

print("Starting the Goldilocks Spatio-Temporal SDE Pipeline...")

# ==========================================
# 1. LOAD DATA & GEOMETRY
# ==========================================
imd_df        = pd.read_csv('csv/imd.csv')
lookup_df     = pd.read_csv('csv/lad_to_pfa_lookup.csv')
police_df     = pd.read_csv('csv/police_data.csv')
police_areas  = gpd.read_file('police_areas.geojson')

def standardize_pfa_names(series):
    series = series.astype(str).str.strip().str.replace('\n', '', regex=True)
    series = series.str.replace(' Police', '', regex=False)
    series = series.str.replace(' Constabulary', '', regex=False)
    series = series.str.replace(' Service', '', regex=False)
    series.loc[series.str.contains('Hampshire', case=False, na=False)] = 'Hampshire and Isle of Wight'
    series.loc[series.str.contains('Devon', case=False, na=False)]     = 'Devon and Cornwall'
    series.loc[series.str.contains('Metropolitan', case=False, na=False)] = 'Metropolitan Police'
    series.loc[series.str.contains('City of London', case=False, na=False)] = 'London, City of'
    return series

police_areas['PFA_Name'] = standardize_pfa_names(police_areas['PFA24NM'])

adjacency_dict = {}
police_areas['geometry'] = police_areas['geometry'].buffer(0.001)

for idx, row in police_areas.iterrows():
    neighbors = police_areas[police_areas.geometry.intersects(row['geometry'])]['PFA_Name'].tolist()
    neighbors = [n for n in neighbors if n != row['PFA_Name']]
    adjacency_dict[row['PFA_Name']] = neighbors

decile_col = [col for col in imd_df.columns if 'Decile' in col][0]
lad_col    = [col for col in imd_df.columns if 'Local Authority District name' in col][0]
imd_df['IMD_Decile'] = pd.to_numeric(imd_df[decile_col], errors='coerce')
imd_df['IMD_Score']  = (11 - imd_df['IMD_Decile']) * 10
lad_imd = (imd_df.groupby(lad_col)['IMD_Score'].mean().reset_index().rename(columns={lad_col: 'LAD_Name'}))

lookup_df = lookup_df[['LAD24NM', 'PFA24NM']].rename(columns={'LAD24NM': 'LAD_Name', 'PFA24NM': 'PFA_Name'})
lookup_df['PFA_Name'] = standardize_pfa_names(lookup_df['PFA_Name'])

merged_data = pd.merge(lookup_df, lad_imd, on='LAD_Name', how='left')
pfa_agg = merged_data.groupby('PFA_Name')['IMD_Score'].mean().reset_index()

years = [2021, 2022, 2023, 2024, 2025]
ts_data = pd.MultiIndex.from_product([pfa_agg['PFA_Name'], years], names=['PFA_Name', 'Year']).to_frame(index=False)
ts_data = pd.merge(ts_data, pfa_agg, on='PFA_Name', how='left')

police_df['PFA_Name'] = standardize_pfa_names(police_df['PFA_Name'])
police_df['Police_Count'] = pd.to_numeric(police_df['Police_Count'].astype(str).str.replace(',', '').str.replace('"', ''), errors='coerce')
ts_data = pd.merge(ts_data, police_df, on=['PFA_Name', 'Year'], how='left')
ts_data['IMD_Score'] = ts_data['IMD_Score'].fillna(ts_data['IMD_Score'].mean())

# ==========================================
# 2. CRIME INGESTION
# ==========================================
use_cols = ['Month', 'Reported by', 'Crime type']
target_crimes = ['Anti-social behaviour', 'Violence and sexual offences']
crime_counts_list = []
csv_files = glob.glob(os.path.join('csv/crime_data', '**', '*street.csv'), recursive=True)

for file in csv_files:
    try:
        df_temp = pd.read_csv(file, usecols=use_cols)
        df_filtered = df_temp[df_temp['Crime type'].isin(target_crimes)]
        agg_df = df_filtered.groupby(['Month', 'Reported by']).size().reset_index(name='Crime_Count')
        if not agg_df.empty: crime_counts_list.append(agg_df)
    except Exception as e: continue

raw_crime_df = pd.concat(crime_counts_list, ignore_index=True)
raw_crime_df['Year'] = raw_crime_df['Month'].str.split('-').str[0].astype(int)
raw_crime_df['PFA_Name'] = standardize_pfa_names(raw_crime_df['Reported by'])

yearly_crimes = raw_crime_df.groupby(['PFA_Name', 'Year'])['Crime_Count'].sum().reset_index()
ts_data = pd.merge(ts_data, yearly_crimes, on=['PFA_Name', 'Year'], how='left')
ts_data['Crime_Count'] = ts_data['Crime_Count'].fillna(0)

ts_data = ts_data.sort_values(by=['PFA_Name', 'Year'])
ts_data['Police_Count'] = ts_data.groupby('PFA_Name')['Police_Count'].transform(lambda g: g.interpolate(method='linear', limit_direction='both'))
ts_data['Police_Count'] = ts_data['Police_Count'].fillna(ts_data['Police_Count'].mean())

# ==========================================
# 3. COEFFICIENT FITTING
# ==========================================
E_raw, months, e_pfa_names = build_E_data()
master_pfas = get_master_pfa_list_from_lookup()
E_data = align_to_master_pfa_list(E_raw, e_pfa_names, master_pfas)
P_data = build_P_monthly(police_df, months, master_pfas)
ADJ = build_adj_matrix(master_pfas, adjacency_dict)

alpha_fit = (ts_data.groupby('PFA_Name')['IMD_Score'].first().reindex(master_pfas).fillna(ts_data['IMD_Score'].mean()).values * 0.0001)

# Fit B (Amplitude) and Beta (Suppression)
B_fit, beta_fit = fit_goldilocks_coefficients(E_data, P_data, ADJ, alpha_fit, d=0.3)

# Clamp negative police elasticity to median
beta_median = float(np.median(beta_fit[beta_fit > 0]))
negative_mask = beta_fit < 0
if negative_mask.any():
    beta_fit[negative_mask] = beta_median

# Cap all extreme beta outliers
beta_fit = np.clip(beta_fit, 0, 0.5)

print_diagnostics(alpha_fit, B_fit, beta_fit, master_pfas)
ts_data = apply_fitted_coefficients(ts_data, alpha_fit, B_fit, beta_fit, master_pfas)

# ==========================================
# 4. SDE CALCULATION
# ==========================================
ts_data['Sigma_i'] = ts_data.groupby('PFA_Name')['Crime_Count'].transform(lambda x: x.std() / (x.mean() + 1e-5))
ts_data['Sigma_i'] = ts_data['Sigma_i'].replace(0, ts_data['Sigma_i'].mean()).fillna(ts_data['Sigma_i'].mean())

alpha_dict = dict(zip(master_pfas, alpha_fit))

def calculate_additive_spillover(row, df, adjacency, alpha_map):
    neighbors = adjacency.get(row['PFA_Name'], [])
    if not neighbors: return 0.0
    current_year_df = df[df['Year'] == row['Year']]
    n_nb = len(neighbors)
    spill = 0.0
    for nb in neighbors:
        nb_data = current_year_df[current_year_df['PFA_Name'] == nb]
        if not nb_data.empty:
            E_j = nb_data['Crime_Count'].values[0]
            spill += (alpha_map.get(nb, 0.0) / n_nb) * E_j
    return spill

ts_data['Spillover_Force'] = ts_data.apply(lambda row: calculate_additive_spillover(row, ts_data, adjacency_dict, alpha_dict), axis=1)

def run_goldilocks_sde(row):
    dt = 1.0
    E_i = row['Crime_Count']
    P_i = row['Police_Count']
    dW = np.random.normal(0, np.sqrt(dt))

    # Deterministic components
    growth = row['Alpha_i'] * E_i
    spillover = row['Spillover_Force']
    phase_shift = 6 * (2 * np.pi / 12)
    seasonal = 1.7 * row['B_i'] * np.cos((2 * np.pi / 12) * 6 - phase_shift)
    suppression = min(row['Beta_i'], 1.0) * E_i * (P_i ** -0.3)

    # Drift = 1.40 * (sum of deterministic forces)
    drift = 1.40 * (growth + spillover - suppression + seasonal)

    # Noise = (P^-0.3) * E * sigma * dW
    noise = (P_i ** -0.3) * row['Sigma_i'] * E_i * dW

    return drift + noise

ts_data['E_Prime_Monthly_Snapshot'] = ts_data.apply(run_goldilocks_sde, axis=1)
ts_data.to_csv('time_series_master_goldilocks.csv', index=False)

# ==========================================
# 5. SOLVE ODE FOR PLOTTING
# ==========================================
print("\nGenerating Goldilocks ODE solution plots...")
t_idx = np.arange(len(months))
t_span = (t_idx[0], t_idx[-1])

P_interp = []
for j in range(len(master_pfas)):
    P_interp.append(interp1d(t_idx, P_data[:, j], kind='linear', fill_value='extrapolate'))

# Build empirical total for plotting
E_emp_total = E_data.sum(axis=1)
# sigma_i = stdev/mean for all pfa's so the sum of that
# Build sigma vector from ts_data (one value per PFA)
sigma_vector = (
    ts_data.groupby('PFA_Name')['Sigma_i']
    .first()
    .reindex(master_pfas)
    .fillna(ts_data['Sigma_i'].mean())
    .values
)

# Set random seed for reproducible ensemble
np.random.seed(42)

# Run multiple stochastic ODE realizations
num_realizations = 50
# SDE, args are here
def ode_system_stochastic(t, E_vec, alpha_perturb, B_perturb, beta_perturb):
    dE = np.zeros(len(master_pfas))
    omega = 2 * np.pi / 12
    phase_shift = 3.0 * (2 * np.pi / 12)

    for i in range(len(master_pfas)):
        neighbours = np.where(ADJ[i])[0]
        n_nb = len(neighbours)

        # Interpolate police count for area i at time t; clamp to >= 1 to avoid P^-0.3 issues
        P_i_t = max(float(P_interp[i](t)), 1.0)

        growth = (alpha_fit[i] + alpha_perturb[i]) * E_vec[i]

        nb_term = 0.0
        if n_nb > 0:
            for j in neighbours:
                nb_term += ((alpha_fit[j] + alpha_perturb[j]) / n_nb) * E_vec[j]

        seasonal = (1.7 * (B_fit[i] + B_perturb[i])) * np.cos(omega * t - phase_shift)
        suppression = min(beta_fit[i] + beta_perturb[i], 1.0) * E_vec[i] * (P_i_t ** -0.3)

        # amplitude 
        drift = 3 * (growth + nb_term - suppression  + seasonal)  # 
        noise = (P_i_t ** -0.3) * 10.0 * sigma_vector[i] * np.random.normal(0, np.sqrt(1 / 12))

        dE[i] = drift + noise

    return dE  # outside the loop

# Store solutions from multiple realizations
all_solutions = []

for realization in range(num_realizations):
    # Only perturb seasonal amplitude — alpha/beta multiply E_vec and cause CI fan-out
    alpha_perturb = np.zeros_like(alpha_fit)
    B_perturb = np.random.normal(0, 0.05 * B_fit)
    beta_perturb = np.zeros_like(beta_fit)

    ode_closure = lambda t, E_vec: ode_system_stochastic(t, E_vec, alpha_perturb, B_perturb, beta_perturb)

    E0 = E_data[0, :].copy()
    sol = solve_ivp(ode_closure, t_span, E0, method='RK45', t_eval=t_idx, max_step=1/12, rtol=1e-4, atol=1e-2)

    if sol.success:
        all_solutions.append(sol.y.T)

if all_solutions:
    # Calculate ensemble statistics
    all_solutions = np.array(all_solutions)
    E_mod_mean = all_solutions.mean(axis=0).sum(axis=1)
    E_mod_std = all_solutions.std(axis=0).sum(axis=1)
    E_mod_upper = E_mod_mean + 1.96 * E_mod_std
    E_mod_lower = E_mod_mean - 1.96 * E_mod_std

    E_modelled = all_solutions[0]  # Keep first realization for reference
    E_mod_total = E_modelled.sum(axis=1)

    t_years = np.array([int(m[:4]) + (int(m[5:7]) - 1) / 12.0 for m in months])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_years, E_emp_total, color='steelblue', linewidth=1.5, label='Empirical E (observed)')
    ax.plot(t_years, E_mod_mean, color='tomato', linewidth=1.5, linestyle='--', label='Goldilocks ODE Model (ensemble mean)')
    ax.fill_between(t_years, E_mod_lower, E_mod_upper, color='tomato', alpha=0.15, label='95% CI (stochastic ensemble)')

    ax.set_xlabel('Year')
    ax.set_ylabel('Total Crime Count (England & Wales)')
    ax.set_title('Goldilocks Spatio-Temporal Model (with Stochastic Uncertainty)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('Goldilocks_ODE_Model.png', dpi=150, bbox_inches='tight')
    print("Saved: Goldilocks_ODE_Model.png")
else:
    print("ODE integration failed for all realizations")