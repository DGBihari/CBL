import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import warnings
import geopandas as gpd

warnings.filterwarnings('ignore')

TARGET_CRIMES = [
    'anti_social_behaviour',
    'violence_and_sexual_offences'
]

POLICY_HOUSING_FIRST = True
POLICY_HOTSPOT_POLICING = False
POLICY_DV_MENTORING = False

num_realizations = 10000
t_span = (0, 36)  
t_forecast = np.linspace(0, 36, 37)

# load sde and police data & coefficients
ts_data = pd.read_csv('../time_series_master_goldilocks.csv')
ts_2025 = ts_data[ts_data['Year'] == 2025].copy()
pfa_names = ts_2025['PFA_Name'].values

E0 = ts_2025['Crime_Count'].values
alpha_vec = ts_2025['Alpha_i'].values
beta_vec = ts_2025['Beta_i'].values
sigma_vec = ts_2025['Sigma_i'].values


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

# adj mapping & vectorized spillover matrix build

police_areas = gpd.read_file('../police_areas.geojson')
police_areas['PFA_Name'] = police_areas['PFA24NM'].astype(str).str.strip()
police_areas['geometry'] = police_areas['geometry'].buffer(0.001)

n_pfas = len(pfa_names)
ADJ = np.zeros((n_pfas, n_pfas))
SPILLOVER_WEIGHTS = np.zeros((n_pfas, n_pfas))

for i, pfa in enumerate(pfa_names):
    geom_i = police_areas[police_areas['PFA_Name'] == pfa]['geometry'].values
    if len(geom_i) > 0:
        neighbors = police_areas[police_areas.geometry.intersects(geom_i[0])]['PFA_Name'].tolist()
        for j, pfa_j in enumerate(pfa_names):
            if pfa_j in neighbors and pfa_j != pfa:
                ADJ[i, j] = 1

# precompute the exact spillover fractions so we don't use slow loops during the simulation
for i in range(n_pfas):
    neighbors_idx = np.where(ADJ[i])[0]
    n_nb = len(neighbors_idx)
    if n_nb > 0:
        for j in neighbors_idx:
            SPILLOVER_WEIGHTS[i, j] = alpha_vec[j] / n_nb

# 2. prophet forecasts
print("Fusing Prophet Machine Learning Forecasts...")
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

    dF_dt = np.gradient(combined_forecast)
    prophet_derivatives[cluster] = interp1d(t_forecast, dF_dt, kind='cubic', fill_value="extrapolate")

pfa_dF_dt = []
for pfa in pfa_names:
    assigned = False
    for cluster, pfas in cluster_mapping.items():
        if pfa in pfas:
            pfa_dF_dt.append(prophet_derivatives[cluster])
            assigned = True
            break
    if not assigned:
        pfa_dF_dt.append(lambda t: 0.0)

# vectorized ODE simulation of parallel realities
policy_name = "Housing First" if POLICY_HOUSING_FIRST else "Hotspot Policing" if POLICY_HOTSPOT_POLICING else "DV Mentoring"

n_months = len(t_forecast)
sub_steps = 10
dt = 1.0 / sub_steps

# result trackers
monthly_control = np.zeros((n_months, num_realizations))
monthly_policy = np.zeros((n_months, num_realizations))

# set initial conditions
E_ctrl = np.tile(E0, (num_realizations, 1))
E_pol = np.tile(E0, (num_realizations, 1))
monthly_control[0, :] = E_ctrl.sum(axis=1)
monthly_policy[0, :] = E_pol.sum(axis=1)

for month_idx in range(1, n_months):
    for step in range(sub_steps):
        t_curr = (month_idx - 1) + step * dt

        # base environment (Drift & Police)
        drift = np.array([pfa_dF_dt[i](t_curr) for i in range(n_pfas)])
        P_curr = np.array([police_extrapolators[pfa_names[i]](t_curr) for i in range(n_pfas)])
        noise_scale = (P_curr ** -0.3) * 10.0 * sigma_vec

        # generate independent random shocks for control and policy universes
        noise_ctrl = noise_scale * np.random.normal(0, np.sqrt(1 / 12), size=(num_realizations, n_pfas))
        noise_pol = noise_scale * np.random.normal(0, np.sqrt(1 / 12), size=(num_realizations, n_pfas))

        # control update (Status Quo)
        E_ctrl += (drift + noise_ctrl) * dt

        # policy update (Intervention)
        alpha_prev = np.zeros_like(E_pol)
        beta_prev = np.zeros_like(E_pol)
        spillover_prev = np.zeros_like(E_pol)

        if POLICY_HOUSING_FIRST:
            alpha_prev = 0.34 * alpha_vec * E_pol
        if POLICY_DV_MENTORING:
            alpha_prev = 0.77 * alpha_vec * E_pol
        if POLICY_HOTSPOT_POLICING:
            beta_prev = 0.60 * beta_vec * E_pol * (P_curr ** -0.3)
            # Matrix dot-product for performance instead of slow loops 
            spillover_prev = 0.50 * (E_pol @ SPILLOVER_WEIGHTS.T)

        E_pol += (drift - alpha_prev - beta_prev - spillover_prev + noise_pol) * dt

    # log the national snapshot at the end of the month
    monthly_control[month_idx, :] = E_ctrl.sum(axis=1)
    monthly_policy[month_idx, :] = E_pol.sum(axis=1)

c_arr, p_arr = monthly_control.T, monthly_policy.T
c_mean, c_std = c_arr.mean(axis=0), c_arr.std(axis=0)
p_mean, p_std = p_arr.mean(axis=0), p_arr.std(axis=0)

# visualization 
print("Rendering visualizations...")
t_years = 2025 + t_forecast / 12.0
fig, ax = plt.subplots(figsize=(14, 7))

# status quo
ax.plot(t_years, c_mean, color='#555555', linewidth=2.5, linestyle='--', label='Status Quo (Forecast Only)')
ax.fill_between(t_years, c_mean - 1.96 * c_std, c_mean + 1.96 * c_std, color='#555555', alpha=0.15)

# intervention
ax.plot(t_years, p_mean, color='#0072B2', linewidth=2.5, label=f'Intervention ({policy_name})')
ax.fill_between(t_years, p_mean - 1.96 * p_std, p_mean + 1.96 * p_std, color='#0072B2', alpha=0.25,
                label='95% Confidence Interval')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Combined Target Crimes (England & Wales)', fontsize=12)
ax.set_title(f'Comparative Policy Simulation: Status Quo vs. {policy_name}', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(2025, 2028)

plt.tight_layout()
plt.savefig('comparative_policy_simulation_2028.png', dpi=150, bbox_inches='tight')
