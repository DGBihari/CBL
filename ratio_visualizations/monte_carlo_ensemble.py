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
num_realizations = 5000
t_span = (0, 36) 
t_forecast = np.linspace(0, 36, 37)

# load SDE data & coefficients
print("Loading baseline data...")
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
print("Building spatial adjacency matrix...")
police_areas = gpd.read_file('../police_areas.geojson')
police_areas['PFA24NM'] = police_areas['PFA24NM'].astype(str).str.strip()
police_areas.loc[police_areas['PFA24NM'].str.contains('Devon', case=False, na=False), 'PFA24NM'] = 'Devon and Cornwall'
police_areas.loc[police_areas['PFA24NM'].str.contains('Hampshire', case=False, na=False), 'PFA24NM'] = 'Hampshire and Isle of Wight'
police_areas.loc[police_areas['PFA24NM'].str.contains('Metropolitan', case=False, na=False), 'PFA24NM'] = 'Metropolitan Police'
police_areas.loc[police_areas['PFA24NM'].str.contains('City of London', case=False, na=False), 'PFA24NM'] = 'London, City of'

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
print("Ingesting Prophet ML clusters...")
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

# the Hybrid SDE / ODE System
def hybrid_ode(t, E_vec, alpha_perturb, B_perturb, beta_perturb):
    E_vec_safe = np.maximum(E_vec, 1.0)
    dE = np.zeros(n_pfas)
    
    for i in range(n_pfas):
        P_i_t = police_extrapolators[pfa_names[i]](t)
        
        prophet_drift = pfa_dF_dt[i](t)
        
        K_i = E0[i] * 1.20 
        
        growth = (alpha_vec[i] + alpha_perturb[i]) * E_vec_safe[i] * max(1.0 - (E_vec_safe[i] / K_i), 0.0)
        
        neighbors = np.where(ADJ[i])[0]
        n_nb = len(neighbors)
        spillover = 0.0
        if n_nb > 0:
            for j in neighbors:
                K_j = E0[j] * 1.20
                spillover += ((alpha_vec[j] + alpha_perturb[j]) / n_nb) * E_vec_safe[j] * max(1.0 - (E_vec_safe[j] / K_j), 0.0)

        suppression = min(beta_vec[i] + beta_perturb[i], 1.0) * E_vec_safe[i] * (P_i_t ** -0.3)
        
        seasonal = 4.5 * (B_vec[i] + B_perturb[i]) * np.cos((np.pi / 6.0) * t - (np.pi / 2.0))
        
        noise = (P_i_t ** -0.3) * 30.0 * sigma_vec[i] * np.random.normal(0, np.sqrt(1 / 12))
        
        dE[i] = growth + prophet_drift + spillover - suppression + seasonal + noise

    return dE

# run Ensemble of ODE Simulations
print(f"Running {num_realizations} realizations...")
all_solutions = []

for i in range(num_realizations):
    alpha_perturb = np.random.normal(0, 0.15 * alpha_vec)  # Increased from 8% to 15%
    B_perturb = np.random.normal(0, 0.10 * B_vec)          # Increased from 5% to 10%
    beta_perturb = np.random.normal(0, 0.15 * beta_vec)    # Increased from 8% to 15%

    ode_closure = lambda t, E_vec: hybrid_ode(t, E_vec, alpha_perturb, B_perturb, beta_perturb)
    
    sol = solve_ivp(ode_closure, t_span, E0, method='RK45', t_eval=t_forecast)
    all_solutions.append(sol.y.T.sum(axis=1))

    if (i + 1) % 500 == 0:
        print(f"  Completed {i + 1}/{num_realizations} runs...")

arr = np.array(all_solutions)
E_mean, E_std = arr.mean(axis=0), arr.std(axis=0)

# generate Plot
print("Generating visualization...")
t_years = 2026 + t_forecast / 12.0
fig, ax = plt.subplots(figsize=(14, 7))

# draw the background chaos threads
for i in range(min(len(all_solutions), 500)): 
    ax.plot(t_years, arr[i], color='gray', alpha=0.01, linewidth=0.5)

# plot the Central Status Quo Projection
ax.plot(t_years, E_mean, color='#0072B2', linewidth=2.5, label='Future Status Quo Mean')
ax.fill_between(t_years, E_mean - 1.96*E_std, E_mean + 1.96*E_std, color='#0072B2', alpha=0.25, label='95% Confidence Interval')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Combined Target Crimes', fontsize=12)
ax.set_title('Predictive Policy Laboratory: Status Quo (2026-2028)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(2026, 2028)

plt.tight_layout()
plt.savefig('monte_carlo_ensemble_2028.png', dpi=150, bbox_inches='tight')
print("Complete. Saved to monte_carlo_ensemble_2028.png")