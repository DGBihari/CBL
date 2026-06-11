import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
import warnings

warnings.filterwarnings('ignore')

TARGET_CRIMES = [
    'anti_social_behaviour',
    'violence_and_sexual_offences'
]

# booleans for toggling policies on/off
POLICY_HOUSING_FIRST = True
POLICY_HOTSPOT_POLICING = False
POLICY_DV_MENTORING = False

num_realizations = 1000
t_span = (0, 36)
t_forecast = np.linspace(0, 36, 37)

# load SDE data & coefficients
ts_data = pd.read_csv('../time_series_master_goldilocks.csv')
ts_2025 = ts_data[ts_data['Year'] == 2025].copy()
pfa_names = ts_2025['PFA_Name'].values

E0 = ts_2025['Crime_Count'].values / 12.0

alpha_vec = ts_2025['Alpha_i'].values
beta_vec = ts_2025['Beta_i'].values
sigma_vec = ts_2025['Sigma_i'].values
B_vec = ts_2025['B_i'].values


# dynamic police extrapolation
def make_extrapolator(poly_coeffs):
    return lambda t: max(np.polyval(poly_coeffs, 2025.0 + t / 12.0), 1.0)


police_extrapolators = {}
for pfa in pfa_names:
    pfa_history = ts_data[ts_data['PFA_Name'] == pfa].dropna(subset=['Year', 'Police_Count'])
    if len(pfa_history) > 1:
        poly = np.polyfit(pfa_history['Year'], pfa_history['Police_Count'], 1)
        police_extrapolators[pfa] = make_extrapolator(poly)
    else:
        static_val = max(ts_2025[ts_2025['PFA_Name'] == pfa]['Police_Count'].values[0], 1.0)
        police_extrapolators[pfa] = lambda t, v=static_val: v

# build Spatial Adjacency Matrix
police_areas = gpd.read_file('../police_areas.geojson')
police_areas['PFA24NM'] = police_areas['PFA24NM'].astype(str).str.strip()
police_areas.loc[police_areas['PFA24NM'].str.contains('Devon', case=False, na=False), 'PFA24NM'] = 'Devon and Cornwall'
police_areas.loc[
    police_areas['PFA24NM'].str.contains('Hampshire', case=False, na=False), 'PFA24NM'] = 'Hampshire and Isle of Wight'
police_areas.loc[
    police_areas['PFA24NM'].str.contains('Metropolitan', case=False, na=False), 'PFA24NM'] = 'Metropolitan Police'
police_areas.loc[
    police_areas['PFA24NM'].str.contains('City of London', case=False, na=False), 'PFA24NM'] = 'London, City of'

adjacency_dict = {}
police_areas['geometry'] = police_areas['geometry'].buffer(0.001)
for idx, row in police_areas.iterrows():
    neighbors = police_areas[police_areas.geometry.intersects(row['geometry'])]['PFA24NM'].tolist()
    neighbors = [n for n in neighbors if n != row['PFA24NM']]
    adjacency_dict[row['PFA24NM']] = neighbors

n_pfas = len(pfa_names)
ADJ = np.zeros((n_pfas, n_pfas), dtype=bool)
pfa_to_idx = {name: i for i, name in enumerate(pfa_names)}

for i, pfa in enumerate(pfa_names):
    if pfa in adjacency_dict:
        for nb in adjacency_dict[pfa]:
            if nb in pfa_to_idx:
                ADJ[i, pfa_to_idx[nb]] = True

# ingest Prophet Forecasts
cluster_mapping = {
    'Cluster_A': ['Greater Manchester', 'Merseyside', 'West Midlands', 'Metropolitan Police', 'West Yorkshire'],
    'Cluster_B': ['Cleveland', 'Durham', 'Humberside', 'Northumbria', 'South Yorkshire'],
    'Cluster_C': ['Cheshire', 'Lancashire', 'Nottinghamshire'],
    'Cluster_D': ['Bedfordshire', 'Cambridgeshire', 'Essex', 'Hertfordshire', 'Kent', 'Leicestershire',
                  'London, City of', 'Northamptonshire', 'Staffordshire'],
    'Cluster_E': ['Avon and Somerset', 'Cumbria', 'Derbyshire', 'Devon and Cornwall', 'Dorset', 'Dyfed-Powys',
                  'Gloucestershire', 'Gwent', 'Hampshire and Isle of Wight', 'Lincolnshire', 'Norfolk', 'North Wales',
                  'North Yorkshire', 'South Wales', 'Suffolk', 'Surrey', 'Sussex', 'Thames Valley', 'Warwickshire',
                  'West Mercia', 'Wiltshire']
}

prophet_derivatives = {}
script_dir = os.path.dirname(os.path.abspath(__file__))
base_cbl_dir = os.path.dirname(script_dir)

for cluster, pfas in cluster_mapping.items():
    combined_forecast = np.zeros(37)
    for crime in TARGET_CRIMES:
        file_crime_str = 'anti-social_behaviour' if crime == 'anti_social_behaviour' else crime
        file_path = os.path.join(base_cbl_dir, "csv", crime,
                                 f"{cluster.replace('Cluster', 'cluster')}_{file_crime_str}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            target_col = 'yhat' if 'yhat' in df.columns else df.columns[1]
            forecast_vals = df[target_col].values[:37]
            if len(forecast_vals) < 37:
                forecast_vals = np.pad(forecast_vals, (0, 37 - len(forecast_vals)), mode='edge')
            combined_forecast += forecast_vals
        else:
            print(f"Warning: Could not find {file_path}")

    raw_dF_dt = np.gradient(combined_forecast)
    percentage_dF_dt = raw_dF_dt / combined_forecast
    prophet_derivatives[cluster] = interp1d(t_forecast, percentage_dF_dt, kind='cubic', fill_value="extrapolate")

pfa_dF_dt = []
for i, pfa in enumerate(pfa_names):
    assigned = False
    for cluster, pfas in cluster_mapping.items():
        if pfa in pfas:
            def make_pfa_drift(clust_pct_func, baseline):
                return lambda t: clust_pct_func(t) * baseline


            pfa_dF_dt.append(make_pfa_drift(prophet_derivatives[cluster], E0[i]))
            assigned = True
            break
    if not assigned:
        pfa_dF_dt.append(lambda t: 0.0)


# sde engine factory
def create_sde_engine(alpha_arr, beta_arr, B_arr, P_funcs, spill_mult):
    def hybrid_ode(t, E_vec, alpha_perturb, B_perturb, beta_perturb):
        E_vec_safe = np.maximum(E_vec, 1.0)
        dE = np.zeros(n_pfas)

        for i in range(n_pfas):
            P_i_t = P_funcs[pfa_names[i]](t)
            prophet_drift = pfa_dF_dt[i](t)

            growth = (alpha_arr[i] + alpha_perturb[i]) * E_vec_safe[i]

            neighbors = np.where(ADJ[i])[0]
            n_nb = len(neighbors)
            spillover = 0.0
            if n_nb > 0:
                for j in neighbors:
                    spillover += ((alpha_arr[j] + alpha_perturb[j]) / n_nb) * E_vec_safe[j]

            suppression = min(beta_arr[i] + beta_perturb[i], 1.0) * E_vec_safe[i] * (P_i_t ** -0.3)

            seasonal = 4.5 * (B_arr[i] + B_perturb[i]) * np.cos((np.pi / 6.0) * t - (np.pi / 2.0))

            noise = (P_i_t ** -0.3) * 30.0 * sigma_vec[i] * np.random.normal(0, np.sqrt(1 / 12))

            dE[i] = growth + prophet_drift + spillover - suppression + seasonal + noise

        return dE

    return hybrid_ode


def run_ensemble(ode_func, alpha_arr, beta_arr, B_arr, label):
    print(f"\nRunning {label} Ensemble ({num_realizations} realizations)...")
    all_sols = []
    for i in range(num_realizations):
        a_p = np.random.normal(0, 0.15 * alpha_arr)
        B_p = np.random.normal(0, 0.10 * B_arr)
        b_p = np.random.normal(0, 0.15 * beta_arr)

        ode_closure = lambda t, E_vec: ode_func(t, E_vec, a_p, B_p, b_p)
        sol = solve_ivp(ode_closure, t_span, E0, method='RK45', t_eval=t_forecast)
        all_sols.append(sol.y.T.sum(axis=1))

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_realizations} runs...")

    arr = np.array(all_sols)
    return arr.mean(axis=0), arr.std(axis=0)


# Create Policy Variants
alpha_pol = alpha_vec.copy()
beta_pol = beta_vec.copy()
P_pol_funcs = police_extrapolators.copy()
spill_pol = np.ones(n_pfas)
spill_control = np.ones(n_pfas)

active_policies = []

if POLICY_HOUSING_FIRST:
    alpha_pol = alpha_pol * (1 - 0.34)
    active_policies.append("Housing First (-34% Growth)")

if POLICY_HOTSPOT_POLICING:
    top_10_indices = np.argsort(E0)[-10:]
    top_10_pfas = pfa_names[top_10_indices]
    for i in top_10_indices:
        pfa = pfa_names[i]

        beta_pol[i] = beta_pol[i] * 1.60
        alpha_pol[i] = alpha_pol[i] * (1 - 0.60)
        spill_pol[i] = (1 - 0.60)

        orig_func = police_extrapolators[pfa]
        P_pol_funcs[pfa] = lambda t, f=orig_func: f(t) * 1.5

    active_policies.append("Hotspot Policing (Multi-factor 60% Effect in Top 10)")

if POLICY_DV_MENTORING:
    alpha_pol = alpha_pol * (1 - 0.77)
    active_policies.append("DV Mentoring (-77% Growth)")

policy_title = " + ".join(active_policies) if active_policies else "No Policy Active"

# Create SDE Engines
control_ode = create_sde_engine(alpha_vec, beta_vec, B_vec, police_extrapolators, spill_control)
policy_ode = create_sde_engine(alpha_pol, beta_pol, B_vec, P_pol_funcs, spill_pol)

# Run Simulations
c_mean, c_std = run_ensemble(control_ode, alpha_vec, beta_vec, B_vec, "Status Quo")
p_mean, p_std = run_ensemble(policy_ode, alpha_pol, beta_pol, B_vec, "Intervention")

# Visualization
t_years = 2025 + t_forecast / 12.0
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(t_years, c_mean, color='#555555', linewidth=2.5, linestyle='--', label='Status Quo (Control)')
ax.fill_between(t_years, c_mean - 1.96 * c_std, c_mean + 1.96 * c_std, color='#555555', alpha=0.15)

ax.plot(t_years, p_mean, color='#009E73', linewidth=2.5, label=f'Intervention: {policy_title}')
ax.fill_between(t_years, p_mean - 1.96 * p_std, p_mean + 1.96 * p_std, color='#009E73', alpha=0.25)

ax.set_xlim(2026, 2028)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Combined Target Crimes', fontsize=12)
ax.set_title('Comparative Policy Laboratory (Spatio-Temporal SDE)', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparative_policy_results.png', dpi=150, bbox_inches='tight')