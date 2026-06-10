import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import os
import warnings
import geopandas as gpd

warnings.filterwarnings('ignore')
print("Initializing Refined Hybrid Shielded Spatial Optimizer...")

TARGET_CRIMES = ['anti_social_behaviour', 'violence_and_sexual_offences']

TOTAL_NEW_OFFICERS = 3000
KEEP_RATIO = 0.95
BATCH_SIZE = 100
num_realizations = 1000
t_span = (0, 36)
t_forecast = np.linspace(0, 36, 37)
POLICY_STRENGTH = 0.30  # stronger so blue zones emerge more clearly

# load sde & data
POLICY_STRENGTH = 0.30  

ts_data = pd.read_csv('time_series_master_goldilocks.csv')
ts_2025 = ts_data[ts_data['Year'] == 2025].copy()
pfa_names = ts_2025['PFA_Name'].values

E0 = ts_2025['Crime_Count'].values
alpha_vec = ts_2025['Alpha_i'].values
beta_vec = ts_2025['Beta_i'].values
sigma_vec = ts_2025['Sigma_i'].values
n_pfas = len(pfa_names)

police_areas = gpd.read_file('police_areas.geojson')
police_areas['PFA_Name'] = police_areas['PFA24NM'].astype(str).str.strip()
police_areas['geometry'] = police_areas['geometry'].buffer(0.001)

ADJ = np.zeros((n_pfas, n_pfas))
SPILLOVER_WEIGHTS = np.zeros((n_pfas, n_pfas))

for i, pfa in enumerate(pfa_names):
    geom_i = police_areas[police_areas['PFA_Name'] == pfa]['geometry'].values
    if len(geom_i) > 0:
        neighbors = police_areas[police_areas.geometry.intersects(geom_i[0])]['PFA_Name'].tolist()
        for j, pfa_j in enumerate(pfa_names):
            if pfa_j in neighbors and pfa_j != pfa:
                ADJ[i, j] = 1

for i in range(n_pfas):
    neighbors_idx = np.where(ADJ[i])[0]
    n_nb = len(neighbors_idx)
    if n_nb > 0:
        for j in neighbors_idx:
            SPILLOVER_WEIGHTS[i, j] = alpha_vec[j] / n_nb


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

# Prophet
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
if "csv" in os.listdir(script_dir):
    base_cbl_dir = script_dir
else:
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
            if len(forecast_vals) < 37: forecast_vals = np.pad(forecast_vals, (0, 37 - len(forecast_vals)), mode='edge')
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
    if not assigned: pfa_dF_dt.append(lambda t: 0.0)

# hybrid optimization engine
base_police_end_2025 = np.array([police_extrapolators[pfa_names[i]](11) for i in range(n_pfas)])
current_allocation = np.zeros(n_pfas)

# outliers like London + Manchester are fixed at 100%, no changes
for i, pfa in enumerate(pfa_names):
    if pfa in ['Metropolitan Police', 'London, City of', 'Greater Manchester']:
        current_allocation[i] = base_police_end_2025[i]
    else:
        current_allocation[i] = np.floor(base_police_end_2025[i] * KEEP_RATIO)

officers_stripped = np.sum(base_police_end_2025) - np.sum(current_allocation)
pool_to_allocate = TOTAL_NEW_OFFICERS + officers_stripped
batches = int(pool_to_allocate // BATCH_SIZE)

print(f"\nNational Budget Formulation:")
print(f" - New Expansion Officers: {TOTAL_NEW_OFFICERS}")
print(f" - Efficiency Reallocation (5% Cut): {int(officers_stripped)}")
print(f" - Total Floating Pool: {int(pool_to_allocate)} officers")


def fast_deterministic_cost(police_allocation):
    def ode(t, E_vec):
        drift = np.array([pfa_dF_dt[i](t) for i in range(n_pfas)])

        police_ratio = police_allocation / base_police_end_2025
        ratio_delta = police_ratio - 1.0

        loss_multiplier = np.where(ratio_delta < 0, 1.2, 1.0)

        intervention = POLICY_STRENGTH * beta_vec * E_vec * ratio_delta * loss_multiplier
        spillover = 0.05 * (E_vec @ SPILLOVER_WEIGHTS.T) * np.maximum(0, ratio_delta)

        return drift - intervention - spillover

    sol = solve_ivp(ode, t_span, E0, method='RK23', t_eval=[36])
    return np.sum(sol.y[:, -1])


for b in range(batches):
    best_pfa_idx = -1
    lowest_cost = float('inf')

    for i in range(n_pfas):
        # GM no new officers
        if pfa_names[i] == 'Greater Manchester':
            continue

        test_allocation = current_allocation.copy()
        test_allocation[i] += BATCH_SIZE

        cost = fast_deterministic_cost(test_allocation)
        if cost < lowest_cost:
            lowest_cost = cost
            best_pfa_idx = i

    if best_pfa_idx != -1:
        current_allocation[best_pfa_idx] += BATCH_SIZE

    if (b + 1) % 25 == 0:
        print(f"  Allocated {(b + 1) * BATCH_SIZE} / {int(pool_to_allocate)} floating officers...")

remainder = pool_to_allocate % BATCH_SIZE
if remainder > 0 and best_pfa_idx != -1:
    current_allocation[best_pfa_idx] += remainder

# Final optimized allocation is now in current_allocation
n_months = len(t_forecast)
sub_steps = 10
dt = 1.0 / sub_steps

monthly_control = np.zeros((n_months, num_realizations))
monthly_optim = np.zeros((n_months, num_realizations))

E_ctrl = np.tile(E0, (num_realizations, 1))
E_opt = np.tile(E0, (num_realizations, 1))
monthly_control[0, :] = E_ctrl.sum(axis=1)
monthly_optim[0, :] = E_opt.sum(axis=1)

for month_idx in range(1, n_months):
    for step in range(sub_steps):
        t_curr = (month_idx - 1) + step * dt
        drift = np.array([pfa_dF_dt[i](t_curr) for i in range(n_pfas)])
        P_curr = np.array([police_extrapolators[pfa_names[i]](t_curr) for i in range(n_pfas)])

        noise_ctrl = (P_curr ** -0.3) * 10.0 * sigma_vec * np.random.normal(0, np.sqrt(1 / 12),
                                                                            size=(num_realizations, n_pfas))
        noise_opt = (current_allocation ** -0.3) * 10.0 * sigma_vec * np.random.normal(0, np.sqrt(1 / 12),
                                                                                       size=(num_realizations, n_pfas))

        E_ctrl += (drift + noise_ctrl) * dt

        opt_ratio = current_allocation / base_police_end_2025
        ratio_delta = opt_ratio - 1.0
        loss_multiplier = np.where(ratio_delta < 0, 1.2, 1.0)

        intervention_opt = POLICY_STRENGTH * beta_vec * E_opt * ratio_delta * loss_multiplier
        spillover_opt = 0.05 * (E_opt @ SPILLOVER_WEIGHTS.T) * np.maximum(0, ratio_delta)

        E_opt += (drift - intervention_opt - spillover_opt + noise_opt) * dt

    monthly_control[month_idx, :] = E_ctrl.sum(axis=1)
    monthly_optim[month_idx, :] = E_opt.sum(axis=1)

c_mean, c_std = monthly_control.T.mean(axis=0), monthly_control.T.std(axis=0)
p_mean, p_std = monthly_optim.T.mean(axis=0), monthly_optim.T.std(axis=0)

final_crime_ctrl = E_ctrl.mean(axis=0)
final_crime_opt = E_opt.mean(axis=0)

police_delta = current_allocation - base_police_end_2025

delta_df = pd.DataFrame({
    'PFA_Name': pfa_names,
    'Police_Delta': police_delta,
    'Crime_Delta': final_crime_opt - final_crime_ctrl
})

map_df = police_areas.merge(delta_df, on='PFA_Name')

map_df.loc[map_df['PFA_Name'] == 'Greater Manchester', 'Police_Delta'] = np.nan
map_df.loc[map_df['PFA_Name'] == 'Greater Manchester', 'Crime_Delta'] = np.nan

# visualizations
print("Rendering visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(20, 18))
ax1, ax2 = axes[0, 0], axes[0, 1]
ax3, ax4 = axes[1, 0], axes[1, 1]

t_years = 2025 + t_forecast / 12.0
ax1.plot(t_years, c_mean, color='#555555', linewidth=2.5, linestyle='--', label='Status Quo')
ax1.fill_between(t_years, c_mean - 1.96*c_std, c_mean + 1.96*c_std, color='#555555', alpha=0.15)
ax1.plot(t_years, p_mean, color='#009E73', linewidth=2.5, label='Hybrid Shielded Optimization')
ax1.fill_between(t_years, p_mean - 1.96*p_std, p_mean + 1.96*p_std, color='#009E73', alpha=0.25)
ax1.set_title('National Target Crime Trajectory: 2025-2028', fontsize=14)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Total Combined Target Crimes', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

bar_df = delta_df[delta_df['PFA_Name'] != 'Greater Manchester'].copy()
bar_df = bar_df.sort_values('Police_Delta', ascending=False)
ax2.barh(bar_df['PFA_Name'][::-1], bar_df['Police_Delta'][::-1], color='#0072B2')
ax2.set_title(f'Net Staffing Change (3000 New + 5% Reallocation)', fontsize=14)
ax2.set_xlabel('Net Change in Officers (From 2025)', fontsize=12)
ax2.tick_params(axis='y', labelsize=8)


# Calculate symmetric bounds using the absolute max value (ignoring NaNs)
max_p_delta = map_df['Police_Delta'].abs().max()
max_c_delta = map_df['Crime_Delta'].abs().max()

# Map 1: vmin and vmax force the white center to equal 0
map_df.plot(column='Police_Delta', ax=ax3, cmap='PiYG', legend=True,
            vmin=-max_p_delta, vmax=max_p_delta,
            edgecolor='white', linewidth=0.3,
            missing_kwds={'color': 'black', 'label': 'Excluded Data'},
            legend_kwds={'label': 'Net Change in Police Officers', 'orientation': 'horizontal'})
ax3.set_title("Targeted Resource Shift (London Shielded)", fontsize=14)
ax3.set_axis_off()

# Map 2: vmin and vmax force the white center to equal 0
map_df.plot(column='Crime_Delta', ax=ax4, cmap='RdBu_r', legend=True,
            vmin=-max_c_delta, vmax=max_c_delta,
            edgecolor='white', linewidth=0.3,
            missing_kwds={'color': 'black', 'label': 'Excluded Data'},
            legend_kwds={'label': 'Net Change in Target Crimes (Optimized - Status Quo)', 'orientation': 'horizontal'})
ax4.set_title("Net Crime Impact per PFA (End of 2028)", fontsize=14)
ax4.set_axis_off()

plt.tight_layout(pad=3.0)
plt.savefig('spatial_optimization_dashboard.png', dpi=150, bbox_inches='tight')