import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
import warnings

warnings.filterwarnings('ignore')
print("Initializing Prophet-Backed Monte Carlo Ensemble (Status Quo)...")

TARGET_CRIMES = [
    'anti_social_behaviour',
    'violence_and_sexual_offences'
]
num_realizations = 10000
t_span = (0, 36) # 3 Years: Jan 2025 to Jan 2028, 36 months
t_forecast = np.linspace(0, 36, 37)

# load data & coefficients
ts_data = pd.read_csv('../time_series_master_goldilocks.csv')
ts_2025 = ts_data[ts_data['Year'] == 2025].copy()
pfa_names = ts_2025['PFA_Name'].values

E0 = ts_2025['Crime_Count'].values
alpha_vec = ts_2025['Alpha_i'].values
beta_vec = ts_2025['Beta_i'].values
sigma_vec = ts_2025['Sigma_i'].values

# police extrapolation
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

# include prophet forecasts
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

# Find the absolute path to the main CBL folder
script_dir = os.path.dirname(os.path.abspath(__file__))
base_cbl_dir = os.path.dirname(script_dir)

for cluster, pfas in cluster_mapping.items():
    combined_forecast = np.zeros(37)

    for crime in TARGET_CRIMES:

        # check for spelling mistake
        if crime == 'anti_social_behaviour':
            file_crime_str = 'anti-social_behaviour'  # File has hyphen
        else:
            file_crime_str = crime

        # fix path
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

    # Calculate the gradient (dF/dt) on the final COMBINED curve
    dF_dt = np.gradient(combined_forecast)
    prophet_derivatives[cluster] = interp1d(t_forecast, dF_dt, kind='cubic', fill_value="extrapolate")

# Map the combined cluster derivatives back to the individual PFAs
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

# ode system
n_pfas = len(pfa_names)
def status_quo_ode(t, E_vec):
    dE = np.zeros(n_pfas)
    for i in range(n_pfas):
        P_i_t = police_extrapolators[pfa_names[i]](t)
        prophet_drift = pfa_dF_dt[i](t)
        noise = (P_i_t ** -0.3) * 10.0 * sigma_vec[i] * np.random.normal(0, np.sqrt(1 / 12))
        dE[i] = prophet_drift + noise
    return dE


# run ensemble
print(f"Running {num_realizations} realizations...")
all_solutions = []
for i in range(num_realizations):
    sol = solve_ivp(status_quo_ode, t_span, E0, method='RK45', t_eval=t_forecast)
    all_solutions.append(sol.y.T.sum(axis=1))

    # Prints every single run
    print(f"  Completed {i + 1}/{num_realizations} runs...")

arr = np.array(all_solutions)
E_mean, E_std = arr.mean(axis=0), arr.std(axis=0)


# plot
t_years = 2025 + t_forecast / 12.0
fig, ax = plt.subplots(figsize=(14, 7))

for i in range(len(all_solutions)):
    ax.plot(t_years, arr[i], color='gray', alpha=0.01, linewidth=0.5)

ax.plot(t_years, E_mean, color='#D55E00', linewidth=2.5, label='Ensemble Mean (Status Quo)')
ax.fill_between(t_years, E_mean - 1.96*E_std, E_mean + 1.96*E_std, color='#D55E00', alpha=0.25, label='95% Confidence Interval')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Combined Target Crimes', fontsize=12)
ax.set_title(f'Prophet-Backed Status Quo Forecast: 2025-2028', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(2025, 2028)

plt.tight_layout()
plt.savefig('monte_carlo_ensemble_2028.png', dpi=150, bbox_inches='tight')
print("Saved: monte_carlo_ensemble_2028.png")