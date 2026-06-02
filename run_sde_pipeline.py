import pandas as pd
import numpy as np
import geopandas as gpd
import warnings
import glob
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os

# ── Updated Imports matching the new fitting script ──
from fit_sde_coefficients import (
    build_adj_matrix,
    build_P_monthly,
    build_N_monthly,
    fit_elasticity_coefficient,
    print_diagnostics,
    apply_fitted_coefficients,
)

from build_E_data_matrix import (
    build_E_data,
    align_to_master_pfa_list,
    get_master_pfa_list_from_lookup,
)

warnings.filterwarnings('ignore', category=UserWarning)

print("Starting the Spatial Network SDE Pipeline (Econometric Elasticity Model)...")

# ==========================================
# 1-3. LOAD DATA, ADJACENCY & PREPROCESS (Unchanged mechanics)
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

print("Building the geographic neighbor network...")
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

pop_df.rename(columns={'LAD 2023 Name': 'LAD_Name', 'Area Sq Km': 'Area_Sq_Km'}, inplace=True)
pop_cols = ['Mid-2021: Population', 'Mid-2022: Population', 'Mid-2023: Population', 'Mid-2024: Population']
for col in ['Area_Sq_Km'] + pop_cols:
    pop_df[col] = pop_df[col].astype(str).str.replace(',', '').astype(float)
pop_df['Mid-2025: Population'] = pop_df['Mid-2024: Population']
pop_cols.append('Mid-2025: Population')

lad_pop_long = pd.melt(
    pop_df.groupby('LAD_Name')[['Area_Sq_Km'] + pop_cols].sum().reset_index(),
    id_vars=['LAD_Name', 'Area_Sq_Km'], value_vars=pop_cols, var_name='Year_Str', value_name='Population'
)
lad_pop_long['Year'] = lad_pop_long['Year_Str'].str.extract(r'(\d{4})').astype(int)
lad_pop_long['LAD_Name'] = lad_pop_long['LAD_Name'].replace({
    'Bristol, City of': 'Bristol', 'Kingston upon Hull, City of': 'Kingston upon Hull', 'Herefordshire, County of': 'Herefordshire'
})

lad_summary = pd.merge(lad_pop_long, lad_imd, on='LAD_Name', how='left')
lookup_df   = lookup_df[['LAD24NM', 'PFA24NM']].rename(columns={'LAD24NM': 'LAD_Name', 'PFA24NM': 'PFA_Name'})
lookup_df['PFA_Name'] = standardize_pfa_names(lookup_df['PFA_Name'])

merged_data = pd.merge(lookup_df, lad_summary, on='LAD_Name', how='left')
pfa_agg = (merged_data.groupby(['PFA_Name', 'Year']).agg({'Population': 'sum', 'Area_Sq_Km': 'sum', 'IMD_Score': 'mean'}).reset_index())

police_df['PFA_Name']     = standardize_pfa_names(police_df['PFA_Name'])
police_df['Police_Count'] = pd.to_numeric(police_df['Police_Count'].astype(str).str.replace(',', '').str.replace('"', ''), errors='coerce')
ts_data = pd.merge(pfa_agg, police_df, on=['PFA_Name', 'Year'], how='left')
ts_data['IMD_Score']  = ts_data['IMD_Score'].fillna(ts_data['IMD_Score'].mean())

# ==========================================
# 4-5. CRIME INGESTION & REGRESSION (Unchanged mechanics)
# ==========================================
print("Ingesting real crime data from /csv/crime_data folder...")
use_cols       = ['Month', 'Reported by', 'Crime type']
target_crimes  = ['Anti-social behaviour', 'Violence and sexual offences']
crime_counts_list = []
csv_files = glob.glob(os.path.join('csv/crime_data', '**', '*street.csv'), recursive=True)

for file in csv_files:
    try:
        df_temp   = pd.read_csv(file, usecols=use_cols)
        df_filtered = df_temp[df_temp['Crime type'].isin(target_crimes)]
        agg_df    = df_filtered.groupby(['Month', 'Reported by']).size().reset_index(name='Crime_Count')
        if not agg_df.empty: crime_counts_list.append(agg_df)
    except Exception as e: continue

raw_crime_df          = pd.concat(crime_counts_list, ignore_index=True)
raw_crime_df['Year']  = raw_crime_df['Month'].str.split('-').str[0].astype(int)
raw_crime_df['PFA_Name'] = standardize_pfa_names(raw_crime_df['Reported by'])
yearly_crimes = raw_crime_df.groupby(['PFA_Name', 'Year'])['Crime_Count'].sum().reset_index()

ts_data = pd.merge(ts_data, yearly_crimes, on=['PFA_Name', 'Year'], how='left')
ts_data['Crime_Count'] = ts_data.groupby('PFA_Name')['Crime_Count'].transform(lambda x: x.fillna(x.mean()))
ts_data['Crime_Count'] = ts_data['Crime_Count'].fillna(0)

ts_data = ts_data.sort_values(by=['PFA_Name', 'Year'])
ts_data['Police_Count'] = ts_data.groupby('PFA_Name')['Police_Count'].transform(lambda g: g.interpolate(method='linear', limit_direction='both'))
ts_data['Police_Count'] = ts_data['Police_Count'].fillna(ts_data['Police_Count'].mean())

for pfa in ts_data['PFA_Name'].unique():
    mask       = ts_data['PFA_Name'] == pfa
    valid_data = ts_data[mask].dropna(subset=['Year', 'Population'])
    if len(valid_data) > 1:
        slope, intercept = np.polyfit(valid_data['Year'], valid_data['Population'], 1)
        ts_data.loc[mask, 'Population'] = (slope * ts_data.loc[mask, 'Year']) + intercept

# ==========================================
# 6. COEFFICIENT FITTING (Econometric Integration)
# ==========================================
print("\nFitting α_i and β_i coefficients from monthly crime data using Elasticity Logic...")

E_raw, months, e_pfa_names = build_E_data()
master_pfas = get_master_pfa_list_from_lookup()
E_data = align_to_master_pfa_list(E_raw, e_pfa_names, master_pfas) 
P_data = build_P_monthly(police_df, months, master_pfas)             
ADJ = build_adj_matrix(master_pfas, adjacency_dict)                  

# α fixed from deprivation index
alpha_fit = (ts_data.groupby('PFA_Name')['IMD_Score'].first().reindex(master_pfas).fillna(ts_data['IMD_Score'].mean()).values * 0.0001)

# Fit β (Police Suppression) using the integral equation solver
beta_fit = fit_elasticity_coefficient(E_data, P_data, ADJ, alpha_fit, d=0.3)

# Clamp unphysical negative police elasticity to the median
beta_median = float(np.median(beta_fit[beta_fit > 0]))
negative_mask = beta_fit < 0
if negative_mask.any():
    flagged = [master_pfas[i] for i in np.where(negative_mask)[0]]
    print(f"  Clamping {len(flagged)} negative β_i to median ({beta_median:.5f}): {flagged}")
    beta_fit[negative_mask] = beta_median

print_diagnostics(alpha_fit, beta_fit, master_pfas)

# Apply to dataframe
ts_data = apply_fitted_coefficients(ts_data, alpha_fit, beta_fit, master_pfas)

# ==========================================
# 7. CALCULATING SPATIAL SDE VARIABLES
# ==========================================
print("\nCalculating spatial gradients and empirical coefficients...")

# Sigma_i is now fractional volatility to match the PDF: + E_i * sigma * dW
# We calculate standard deviation divided by the mean for each region
ts_data['Sigma_i'] = ts_data.groupby('PFA_Name')['Crime_Count'].transform(lambda x: x.std() / (x.mean() + 1e-5))
national_avg_sigma = ts_data.loc[ts_data['Sigma_i'] > 0, 'Sigma_i'].mean()
ts_data['Sigma_i'] = ts_data['Sigma_i'].replace(0, national_avg_sigma).fillna(national_avg_sigma)

# Dictionary map of alphas to calculate neighbor additive spillover
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
            a_j = alpha_map.get(nb, 0.0)
            spill += (a_j / n_nb) * E_j  # Additive spillover per PDF
            
    return spill

ts_data['Spillover_Force'] = ts_data.apply(lambda row: calculate_additive_spillover(row, ts_data, adjacency_dict, alpha_dict), axis=1)

# ==========================================
# 8. RUNNING THE SPATIAL SDE
# ==========================================
print("Solving the Network-Coupled Stochastic Differential Equation...")

def run_spatial_sde(row):
    dt    = 1.0
    E_i   = row['Crime_Count']
    P_i   = row['Police_Count']
    dW    = np.random.normal(0, np.sqrt(dt))

    growth      = row['Alpha_i'] * E_i
    spillover   = row['Spillover_Force']
    suppression = row['Beta_i'] * E_i * (P_i ** -0.3)  # Elasticity mechanics
    stochastic  = row['Sigma_i'] * E_i * dW            # Multiplicative noise

    return growth + spillover - suppression + stochastic

ts_data['E_Prime_i'] = ts_data.apply(run_spatial_sde, axis=1)

ts_data = ts_data.sort_values(by=['Year', 'PFA_Name'])
ts_data.to_csv('time_series_master_calculated.csv', index=False)

# ==========================================
# 9. SOLVE ODE FOR PLOTTING
# ==========================================
print("\nGenerating E_i ODE solution plots from fitted coefficients...")
t_months = np.array([int(m[:4]) + (int(m[5:7]) - 1) / 12.0 for m in months])

P_interp = []
for j in range(len(master_pfas)):
    P_interp.append(interp1d(t_months, P_data[:, j], kind='linear', fill_value='extrapolate'))

def ode_system(t, E_vec):
    dE = np.zeros(len(master_pfas))
    for i in range(len(master_pfas)):
        neighbours = np.where(ADJ[i])[0]
        n_nb = len(neighbours)

        growth = alpha_fit[i] * E_vec[i]
        
        nb_term = 0.0
        if n_nb > 0:
            for j in neighbours:
                nb_term += (alpha_fit[j] / n_nb) * E_vec[j]

        P_i_t = P_interp[i](t)
        suppression = beta_fit[i] * E_vec[i] * (P_i_t ** -0.3)

        dE[i] = growth + nb_term - suppression

    return dE

E0 = E_data[0, :].copy()
sol = solve_ivp(ode_system, (t_months[0], t_months[-1]), E0, method='RK45', t_eval=t_months, max_step=1/12, rtol=1e-4, atol=1e-2)

if sol.success:
    E_modelled = sol.y.T 
    E_emp_total = E_data.sum(axis=1)        
    E_mod_total = E_modelled.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_months, E_emp_total, color='steelblue', linewidth=1.5, label='Empirical E (observed)')
    ax.plot(t_months, E_mod_total, color='tomato', linewidth=1.5, linestyle='--', label='ODE model E')
    ax.fill_between(t_months, E_mod_total * 0.9, E_mod_total * 1.1, color='tomato', alpha=0.15, label='±10% band')

    ax.set_xlabel('Year')
    ax.set_ylabel('Total Crime Count (England & Wales)')
    ax.set_title('Modelled vs Empirical E_i — National Total (Elasticity Model)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('E_i_ode_vs_empirical_elastic.png', dpi=150, bbox_inches='tight')
    print("Saved: E_i_ode_vs_empirical_elastic.png")