import pandas as pd
import numpy as np
import geopandas as gpd
import warnings
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ── Coefficient fitting module (fit_sde_coefficients.py must be in same dir) ──
from fit_sde_coefficients import (
    build_adj_matrix,
    build_P_monthly,
    build_N_monthly,
    fit_per_region,
    fit_global,
    fit_gamma_only,
    print_diagnostics,
    apply_fitted_coefficients,
)

# ── Crime data builder (build-E_data-matrix.py must be in same dir) ───────────
# Note: Python module names can't have hyphens, so rename the file to
# build_E_data_matrix.py (underscore) if you haven't already.
from build_E_data_matrix import (
    build_E_data,
    align_to_master_pfa_list,
    get_master_pfa_list_from_lookup,
)

warnings.filterwarnings('ignore', category=UserWarning)

print("Starting the Spatial Network SDE Pipeline (Reaction-Diffusion + Regression)...")

# ==========================================
# 1. LOAD DATA & SANITIZE NAMES
# ==========================================
pop_df        = pd.read_csv('csv/population_density.csv', skiprows=3)
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

# ==========================================
# 2. BUILD THE SPATIAL ADJACENCY NETWORK
# ==========================================
print("Building the geographic neighbor network...")
adjacency_dict = {}
police_areas['geometry'] = police_areas['geometry'].buffer(0.001)

for idx, row in police_areas.iterrows():
    neighbors = police_areas[
        police_areas.geometry.intersects(row['geometry'])
    ]['PFA_Name'].tolist()
    neighbors = [n for n in neighbors if n != row['PFA_Name']]
    adjacency_dict[row['PFA_Name']] = neighbors

# ==========================================
# 3. PREPROCESS TIME-SERIES & MERGE
# ==========================================
# IMD
decile_col = [col for col in imd_df.columns if 'Decile' in col][0]
lad_col    = [col for col in imd_df.columns if 'Local Authority District name' in col][0]
imd_df['IMD_Decile'] = pd.to_numeric(imd_df[decile_col], errors='coerce')
imd_df['IMD_Score']  = (11 - imd_df['IMD_Decile']) * 10
lad_imd = (imd_df.groupby(lad_col)['IMD_Score']
               .mean().reset_index()
               .rename(columns={lad_col: 'LAD_Name'}))

# Population
pop_df.rename(columns={'LAD 2023 Name': 'LAD_Name', 'Area Sq Km': 'Area_Sq_Km'}, inplace=True)
pop_cols = ['Mid-2021: Population', 'Mid-2022: Population',
            'Mid-2023: Population', 'Mid-2024: Population']
for col in ['Area_Sq_Km'] + pop_cols:
    pop_df[col] = pop_df[col].astype(str).str.replace(',', '').astype(float)
pop_df['Mid-2025: Population'] = pop_df['Mid-2024: Population']
pop_cols.append('Mid-2025: Population')

lad_pop_long = pd.melt(
    pop_df.groupby('LAD_Name')[['Area_Sq_Km'] + pop_cols].sum().reset_index(),
    id_vars=['LAD_Name', 'Area_Sq_Km'], value_vars=pop_cols,
    var_name='Year_Str', value_name='Population'
)
lad_pop_long['Year'] = lad_pop_long['Year_Str'].str.extract(r'(\d{4})').astype(int)
lad_pop_long['LAD_Name'] = lad_pop_long['LAD_Name'].replace({
    'Bristol, City of': 'Bristol',
    'Kingston upon Hull, City of': 'Kingston upon Hull',
    'Herefordshire, County of': 'Herefordshire'
})

lad_summary = pd.merge(lad_pop_long, lad_imd, on='LAD_Name', how='left')
lookup_df   = lookup_df[['LAD24NM', 'PFA24NM']].rename(
    columns={'LAD24NM': 'LAD_Name', 'PFA24NM': 'PFA_Name'})
lookup_df['PFA_Name'] = standardize_pfa_names(lookup_df['PFA_Name'])

merged_data = pd.merge(lookup_df, lad_summary, on='LAD_Name', how='left')
pfa_agg = (merged_data.groupby(['PFA_Name', 'Year'])
               .agg({'Population': 'sum', 'Area_Sq_Km': 'sum', 'IMD_Score': 'mean'})
               .reset_index())

police_df['PFA_Name']     = standardize_pfa_names(police_df['PFA_Name'])
police_df['Police_Count'] = pd.to_numeric(
    police_df['Police_Count'].astype(str).str.replace(',', '').str.replace('"', ''),
    errors='coerce'
)
ts_data = pd.merge(pfa_agg, police_df, on=['PFA_Name', 'Year'], how='left')
ts_data['Area_Sq_Km'] = ts_data['Area_Sq_Km'].replace(0, np.nan).fillna(ts_data['Area_Sq_Km'].mean())
ts_data['IMD_Score']  = ts_data['IMD_Score'].fillna(ts_data['IMD_Score'].mean())

# ==========================================
# 4. REAL CRIME DATA INGESTION
# ==========================================
print("Ingesting real crime data from /csv/crime_data folder...")

use_cols       = ['Month', 'Reported by', 'Crime type']
target_crimes  = ['Anti-social behaviour', 'Violence and sexual offences']
crime_counts_list = []
csv_files = glob.glob(os.path.join('csv/crime_data', '**', '*street.csv'), recursive=True)
print(f"Found {len(csv_files)} CSV files. Beginning processing...")

if len(csv_files) == 0:
    raise FileNotFoundError("0 files found. Check that csv/crime_data is next to run_sde_pipeline.py.")

for file in csv_files:
    try:
        df_temp   = pd.read_csv(file, usecols=use_cols)
        df_filtered = df_temp[df_temp['Crime type'].isin(target_crimes)]
        agg_df    = df_filtered.groupby(['Month', 'Reported by']).size().reset_index(name='Crime_Count')
        if not agg_df.empty:
            crime_counts_list.append(agg_df)
    except Exception as e:
        print(f"FAILED on {file}: {e}")
        continue

if len(crime_counts_list) == 0:
    raise ValueError("Files found but no data extracted. See FAILED messages above.")

raw_crime_df          = pd.concat(crime_counts_list, ignore_index=True)
raw_crime_df['Year']  = raw_crime_df['Month'].str.split('-').str[0].astype(int)
raw_crime_df['PFA_Name'] = standardize_pfa_names(raw_crime_df['Reported by'])
yearly_crimes = raw_crime_df.groupby(['PFA_Name', 'Year'])['Crime_Count'].sum().reset_index()

ts_data = pd.merge(ts_data, yearly_crimes, on=['PFA_Name', 'Year'], how='left')
ts_data['Crime_Count'] = ts_data.groupby('PFA_Name')['Crime_Count'].transform(
    lambda x: x.fillna(x.mean()))
ts_data['Crime_Count'] = ts_data['Crime_Count'].fillna(0)
print("Real crime data merged.")

# ==========================================
# 5. REGRESSION & INTERPOLATION
# ==========================================
print("Applying linear regression and interpolation to N_i and P_i...")
ts_data = ts_data.sort_values(by=['PFA_Name', 'Year'])

ts_data['Police_Count'] = ts_data.groupby('PFA_Name')['Police_Count'].transform(
    lambda g: g.interpolate(method='linear', limit_direction='both'))
ts_data['Police_Count'] = ts_data['Police_Count'].fillna(ts_data['Police_Count'].mean())

for pfa in ts_data['PFA_Name'].unique():
    mask       = ts_data['PFA_Name'] == pfa
    valid_data = ts_data[mask].dropna(subset=['Year', 'Population'])
    if len(valid_data) > 1:
        slope, intercept = np.polyfit(valid_data['Year'], valid_data['Population'], 1)
        ts_data.loc[mask, 'Population'] = (slope * ts_data.loc[mask, 'Year']) + intercept

# ==========================================
# 6. COEFFICIENT FITTING  ← NEW SECTION
# ==========================================
print("\nFitting α_i and γ_i coefficients from monthly crime data...")

# 6a. Build monthly E_data from the raw crime CSVs
E_raw, months, e_pfa_names = build_E_data()
master_pfas = get_master_pfa_list_from_lookup()
E_data = align_to_master_pfa_list(E_raw, e_pfa_names, master_pfas)  # (K, 43)

# 6b. Build monthly P_data (linear interpolation of annual police counts)
P_data = build_P_monthly(police_df, months, master_pfas)             # (K, 43)

# 6c. Build monthly N_data (linear regression of annual population)
N_data = build_N_monthly(pfa_agg, months, master_pfas)               # (K, 43)

# 6d. Adjacency matrix (N, N) from the geographic network built in Section 2
ADJ = build_adj_matrix(master_pfas, adjacency_dict)                  # (43, 43)

# 6e. Area per PFA — constant, taken from ts_data
area = (
    ts_data.groupby('PFA_Name')['Area_Sq_Km']
           .first()
           .reindex(master_pfas)
           .fillna(ts_data['Area_Sq_Km'].mean())
           .values
)  # (43,)

# ── Choose fitting strategy ────────────────────────────────────────────────
#
# OPTION A — fix β=0.15, fit α_i and γ_i per region independently (fast)
#   Use when: you want to keep the original spatial weight assumption.
#   Risk: α and γ may be poorly identified if police data is near-constant.
#
# OPTION B — fit α_i, γ_i AND β simultaneously in one global system
#   Use when: you want β recovered from data too.
#   Risk: same identification issue as A, plus GMP outlier pollutes β.
#
# OPTION C (RECOMMENDED) — fix α_i from IMD deprivation index, fit only γ_i
#   Use when: police data is interpolated (near-constant ΔP per region),
#   making joint α/γ identification unreliable. Anchoring α in deprivation
#   data is physically meaningful AND gives a well-conditioned 1-unknown system.
#   This is Option 2 from the project PDF.

# OPTION C: α fixed from deprivation index (same formula as the original mock)
alpha_fit = (
    ts_data.groupby('PFA_Name')['IMD_Score']
           .first()
           .reindex(master_pfas)
           .fillna(ts_data['IMD_Score'].mean())
           .values
    * 0.0001          # same scaling as the original pipeline
)
beta_fit  = 0.15     # keep spatial weight fixed (or swap for fit_global result)



# # OPTION A: fit α_i and γ_i with β fixed at 0.15
# alpha_fit, gamma_fit = fit_per_region(E_data, P_data, area, ADJ, beta=0.15)
# beta_fit = 0.15

# # OPTION B: fit everything together (α_i, γ_i, β)
#alpha_fit, gamma_fit, beta_fit = fit_global(E_data, P_data, area, ADJ)
# ── Post-processing: clamp unphysical γ_i values which are <0, instead of with London outlier 2.1

# Problem with these 2 is that police is yearly interpolated -> delta_P is near constant and so delta_Pi ~ C_i => least squares can't distinguish
# -> MUST use Option C, with fixed beta, and alpha anchored in depravation index, getting gamma per region

# Option C, reasonable values with mean correction for - y's, only 4, which could be due to noise?

gamma_fit = fit_gamma_only(E_data, P_data, area, ADJ,
                         alpha=alpha_fit, beta=beta_fit)

# ── Post-processing: clamp unphysical γ_i values which are <0, instead of with London outlier 2.1
gamma_median = float(np.median(gamma_fit[gamma_fit > 0]))
negative_mask = gamma_fit < 0
if negative_mask.any():
    flagged = [master_pfas[i] for i in np.where(negative_mask)[0]]
    print(f"  Clamping {len(flagged)} negative γ_i to median ({gamma_median:.5f}): {flagged}")
    gamma_fit[negative_mask] = gamma_median





print_diagnostics(alpha_fit, gamma_fit, master_pfas, beta=beta_fit)

# graph of E

# ── Plot empirical E national total ───────────────────────────────────────
t_months = np.array([
    int(m[:4]) + (int(m[5:7]) - 1) / 12.0
    for m in months
])
E_emp_total = E_data.sum(axis=1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t_months, E_emp_total, color='steelblue', linewidth=1.5)
ax.set_xlabel('Year')
ax.set_ylabel('Total Crime Count (England & Wales)')
ax.set_title('Empirical E_i — National Total (Anti-social behaviour + Violence)')
plt.tight_layout()
plt.savefig('E_i_empirical.png', dpi=150, bbox_inches='tight')
plt.show()


# 6f. Slot fitted values into ts_data (creates Alpha_i and Gamma_i columns)
ts_data = apply_fitted_coefficients(ts_data, alpha_fit, gamma_fit, master_pfas, beta_fit)

# ==========================================
# 7. CALCULATING SPATIAL SDE VARIABLES
# ==========================================
print("\nCalculating spatial gradients and empirical coefficients...")

ts_data['Delta_Police'] = ts_data.groupby('PFA_Name')['Police_Count'].diff().fillna(0)
ts_data['Sigma_i']      = (ts_data.groupby('PFA_Name')['Crime_Count']
                               .transform('std') / ts_data['Area_Sq_Km'])

national_avg_sigma      = ts_data.loc[ts_data['Sigma_i'] > 0, 'Sigma_i'].mean()
ts_data['Sigma_i']      = ts_data['Sigma_i'].replace(0, national_avg_sigma).fillna(national_avg_sigma)

# Alpha_i and Gamma_i are now FITTED values (set in Section 6f above).
# Sigma_i remains empirical as before.

def calculate_spatial_spillover(row, df, adjacency, beta):   # ← beta now a parameter
    neighbors = adjacency.get(row['PFA_Name'], [])
    if not neighbors:
        return 0.0
    current_year_df = df[df['Year'] == row['Year']]
    neighbor_data   = current_year_df[current_year_df['PFA_Name'].isin(neighbors)]
    if neighbor_data.empty:
        return 0.0
    avg_neighbor_density = (neighbor_data['Crime_Count'] / neighbor_data['Area_Sq_Km']).mean()
    local_density        = row['Crime_Count'] / row['Area_Sq_Km']
    return beta * (avg_neighbor_density - local_density)

# Pass the fitted (or fixed) beta into the spillover calculation
ts_data['Spillover_Force'] = ts_data.apply(
    lambda row: calculate_spatial_spillover(row, ts_data, adjacency_dict, beta=beta_fit),
    axis=1
)

# ==========================================
# 8. RUNNING THE SPATIAL SDE
# ==========================================
print("Solving the Network-Coupled Stochastic Differential Equation...")

def run_spatial_sde(row):
    dt    = 1.0
    C_i   = row['Crime_Count'] / row['Area_Sq_Km']
    dP_i  = row['Delta_Police'] / row['Area_Sq_Km']
    dW    = np.random.normal(0, np.sqrt(dt))

    growth      = row['Alpha_i'] * C_i           # fitted α_i
    suppression = row['Gamma_i'] * dP_i          # fitted γ_i
    spillover   = row['Spillover_Force']         # fitted β baked in
    stochastic  = row['Sigma_i'] * C_i * dW     # empirical σ_i unchanged

    return growth - suppression + spillover + stochastic

ts_data['C_Prime_i'] = ts_data.apply(run_spatial_sde, axis=1)

ts_data = ts_data.sort_values(by=['Year', 'PFA_Name'])
ts_data.to_csv('time_series_master_calculated.csv', index=False)

print(f"\nSuccess! Analyzed {len(ts_data)} empirical spatial data points.")
print("Pipeline complete. Exported time_series_master_calculated.csv")



# SECTION 9

print("\nGenerating E_i ODE solution plots from fitted coefficients...")

# ── Time axis ─────────────────────────────────────────────────────────────
t_months = np.array([
    int(m[:4]) + (int(m[5:7]) - 1) / 12.0
    for m in months
])
t_span = (t_months[0], t_months[-1])
t_eval = t_months

N = len(master_pfas)
pfa_idx = {name: i for i, name in enumerate(master_pfas)}

# ── Build continuous P_i(t) interpolants ──────────────────────────────────
P_interp = []
for j in range(N):
    P_interp.append(
        interp1d(t_months, P_data[:, j],
                 kind='linear', fill_value='extrapolate')
    )

# ── Define the ODE system ──────────────────────────────────────────────────
def ode_system(t, E_vec):
    dE = np.zeros(N)
    C  = E_vec / area

    for i in range(N):
        neighbours = np.where(ADJ[i])[0]
        n_nb = len(neighbours)

        growth = alpha_fit[i] * E_vec[i]

        nb_term = 0.0
        if n_nb > 0:
            for j in neighbours:
                nb_term += (alpha_fit[j] / n_nb) * E_vec[j]

        dt_fd = 1 / 12
        dP_i  = (P_interp[i](t + dt_fd) - P_interp[i](t - dt_fd)) / (2 * dt_fd)
        suppression = gamma_fit[i] * dP_i

        spill = 0.0
        for j in neighbours:
            spill += beta_fit * (C[j] - C[i])

        dE[i] = growth + nb_term - suppression + spill

    return dE

# ── Solve ──────────────────────────────────────────────────────────────────
E0 = E_data[0, :].copy()
print("  Solving ODE system (this may take ~10-30 seconds)...")
sol = solve_ivp(
    ode_system,
    t_span,
    E0,
    method='RK45',
    t_eval=t_eval,
    max_step=1/12,
    rtol=1e-4,
    atol=1e-2,
)

if not sol.success:
    print(f"  WARNING: ODE solver did not converge: {sol.message}")
else:
    print("  ODE solved successfully.")

E_modelled = sol.y.T  # (K, N)

E_emp_total = E_data.sum(axis=1)        # sum across all 43 PFAs each month
E_mod_total = E_modelled.sum(axis=1)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(t_months, E_emp_total,
        color='steelblue', linewidth=1.5, label='Empirical E (observed)')
ax.plot(t_months, E_mod_total,
        color='tomato', linewidth=1.5, linestyle='--', label='ODE model E')
ax.fill_between(t_months,
                E_mod_total * 0.9, E_mod_total * 1.1,
                color='tomato', alpha=0.15, label='±10% band')

ax.set_xlabel('Year')
ax.set_ylabel('Total Crime Count (England & Wales)')
ax.set_title('Modelled vs Empirical E_i — National Total')
ax.legend()
plt.tight_layout()
plt.savefig('E_i_ode_vs_empirical.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: E_i_ode_vs_empirical.png")