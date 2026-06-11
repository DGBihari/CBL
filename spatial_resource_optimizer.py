import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
import warnings

warnings.filterwarnings('ignore')

TARGET_CRIMES = ['anti_social_behaviour', 'violence_and_sexual_offences']

TOTAL_NEW_OFFICERS = 3000
KEEP_RATIO = 0.95
BATCH_SIZE = 100
num_realizations = 1000
t_span = (0, 36)
t_forecast = np.linspace(0, 36, 37)

# load data
ts_data = pd.read_csv('time_series_master_goldilocks.csv')
ts_2025 = ts_data[ts_data['Year'] == 2025].copy()
pfa_names = ts_2025['PFA_Name'].values

E0 = ts_2025['Crime_Count'].values / 12.0

alpha_vec = ts_2025['Alpha_i'].values
beta_vec = ts_2025['Beta_i'].values
sigma_vec = ts_2025['Sigma_i'].values
B_vec = ts_2025['B_i'].values
n_pfas = len(pfa_names)

# build Spatial Adjacency Matrix
print("Building spatial adjacency matrix...")
police_areas = gpd.read_file('police_areas.geojson')
police_areas['PFA24NM'] = police_areas['PFA24NM'].astype(str).str.strip()
police_areas.loc[police_areas['PFA24NM'].str.contains('Devon', case=False, na=False), 'PFA24NM'] = 'Devon and Cornwall'
police_areas.loc[police_areas['PFA24NM'].str.contains('Hampshire', case=False, na=False), 'PFA24NM'] = 'Hampshire and Isle of Wight'
police_areas.loc[police_areas['PFA24NM'].str.contains('Metropolitan', case=False, na=False), 'PFA24NM'] = 'Metropolitan Police'
police_areas.loc[police_areas['PFA24NM'].str.contains('City of London', case=False, na=False), 'PFA24NM'] = 'London, City of'

police_areas['PFA_Name'] = police_areas['PFA24NM']

adjacency_dict = {}
police_areas['geometry'] = police_areas['geometry'].buffer(0.001)
for idx, row in police_areas.iterrows():
    neighbors = police_areas[police_areas.geometry.intersects(row['geometry'])]['PFA24NM'].tolist()
    neighbors = [n for n in neighbors if n != row['PFA24NM']]
    adjacency_dict[row['PFA24NM']] = neighbors

ADJ = np.zeros((n_pfas, n_pfas), dtype=bool)
pfa_to_idx = {name: i for i, name in enumerate(pfa_names)}

for i, pfa in enumerate(pfa_names):
    if pfa in adjacency_dict:
        for nb in adjacency_dict[pfa]:
            if nb in pfa_to_idx:
                ADJ[i, pfa_to_idx[nb]] = True

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

# prophet
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
if "csv" in os.listdir(script_dir):
    base_cbl_dir = script_dir
else:
    base_cbl_dir = os.path.dirname(script_dir)

for cluster, pfas in cluster_mapping.items():
    combined_forecast = np.zeros(37)
    for crime in TARGET_CRIMES:
        file_crime_str = 'anti-social_behaviour' if crime == 'anti_social_behaviour' else crime
        file_path = os.path.join(base_cbl_dir, "csv", crime, f"{cluster.replace('Cluster', 'cluster')}_{file_crime_str}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            target_col = 'yhat' if 'yhat' in df.columns else df.columns[1]
            forecast_vals = df[target_col].values[:37]
            if len(forecast_vals) < 37:
                forecast_vals = np.pad(forecast_vals, (0, 37 - len(forecast_vals)), mode='edge')
            combined_forecast += forecast_vals

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
    if not assigned: pfa_dF_dt.append(lambda t: 0.0)

# SDE Engine Generator 
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
                    spillover += (((alpha_arr[j] + alpha_perturb[j]) / n_nb) * E_vec_safe[j]) * spill_mult[j]
                    
            suppression = min(beta_arr[i] + beta_perturb[i], 1.0) * E_vec_safe[i] * (P_i_t ** -0.3)
            
            seasonal = 4.5 * (B_arr[i] + B_perturb[i]) * np.cos((np.pi / 6.0) * t - (np.pi / 2.0))
            
            noise = (P_i_t ** -0.3) * 30.0 * sigma_vec[i] * np.random.normal(0, np.sqrt(1 / 12))
            
            dE[i] = growth + prophet_drift + spillover - suppression + seasonal + noise
            
        return dE
    return hybrid_ode

def run_ensemble(ode_func, alpha_arr, beta_arr, B_arr, label):
    all_sols = []
    final_crime_per_pfa = []
    
    for i in range(num_realizations):
        a_p = np.random.normal(0, 0.15 * alpha_arr)
        B_p = np.random.normal(0, 0.10 * B_arr)
        b_p = np.random.normal(0, 0.15 * beta_arr)

        ode_closure = lambda t, E_vec: ode_func(t, E_vec, a_p, B_p, b_p)
        sol = solve_ivp(ode_closure, t_span, E0, method='RK45', t_eval=t_forecast)
        
        all_sols.append(sol.y.T.sum(axis=1))
        # Store the very last timestep (month 36) for geographical mapping
        final_crime_per_pfa.append(sol.y[:, -1])

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_realizations} runs")
            
    arr = np.array(all_sols)
    final_arr = np.array(final_crime_per_pfa)
    
    return arr.mean(axis=0), arr.std(axis=0), final_arr.mean(axis=0)

# allocation Optimizer Logic
base_police_end_2025 = np.array([police_extrapolators[pfa_names[i]](11) for i in range(n_pfas)])
current_allocation = np.zeros(n_pfas)

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

fast_dF_dt = np.array([pfa_dF_dt[i](36) for i in range(n_pfas)])
initial_baseline = current_allocation.copy()

def fast_deterministic_cost(police_allocation):
    police_ratio = police_allocation / base_police_end_2025
    ratio_delta = police_ratio - 1.0
    loss_multiplier = np.where(ratio_delta < 0, 1.2, 1.0)
    
    # activation of Hotspot Mechanics which guides the optimizer,aAllows the greedy algorithm to hunt for the operational rewards
    activation = np.clip(ratio_delta / 0.10, 0.0, 1.0)
    
    effective_alpha = alpha_vec * (1.0 - (0.60 * activation))
    effective_beta = beta_vec * (1.0 + (0.60 * activation))
    
    intervention = effective_beta * E0 * (police_allocation ** -0.3) * loss_multiplier
    
    # Equity Penalty to resolve ties if multiple PFAs cap out
    added_officers = np.maximum(police_allocation - initial_baseline, 0)
    equity_penalty = 10000.0 * np.sum((added_officers / pool_to_allocate) ** 2)
    
    return np.sum((effective_alpha * E0) + fast_dF_dt - intervention) + equity_penalty

for b in range(batches):
    best_pfa_idx = -1
    lowest_cost = float('inf')

    for i in range(n_pfas):
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
        print(f"  Allocated {(b + 1) * BATCH_SIZE} / {int(pool_to_allocate)} floating officers")

remainder = pool_to_allocate % BATCH_SIZE
if remainder > 0 and best_pfa_idx != -1:
    current_allocation[best_pfa_idx] += remainder

# apply Optimized Functions to the standard SDE Engine
P_opt_funcs = police_extrapolators.copy()
spill_opt = np.ones(n_pfas)
spill_control = np.ones(n_pfas)

alpha_opt = alpha_vec.copy()
beta_opt = beta_vec.copy()

for i, pfa in enumerate(pfa_names):
    orig_count = base_police_end_2025[i]
    new_count = current_allocation[i]
    
    ratio = new_count / orig_count if orig_count > 0 else 1.0
    orig_func = police_extrapolators[pfa]
    P_opt_funcs[pfa] = lambda t, f=orig_func, r=ratio: f(t) * r
    
    ratio_delta = ratio - 1.0
    if ratio_delta > 0:
        activation = min(ratio_delta / 0.10, 1.0)
        alpha_opt[i] = alpha_vec[i] * (1.0 - (0.60 * activation))
        beta_opt[i] = beta_vec[i] * (1.0 + (0.60 * activation))
        spill_opt[i] = 1.0 - (0.60 * activation)

# run the Unified SDEs 
control_ode = create_sde_engine(alpha_vec, beta_vec, B_vec, police_extrapolators, spill_control)
policy_ode = create_sde_engine(alpha_opt, beta_opt, B_vec, P_opt_funcs, spill_opt)

c_mean, c_std, final_crime_ctrl = run_ensemble(control_ode, alpha_vec, beta_vec, B_vec, "Status Quo")
p_mean, p_std, final_crime_opt = run_ensemble(policy_ode, alpha_vec, beta_vec, B_vec, "Optimized Allocation")

# dashboard visualizations
police_delta = current_allocation - base_police_end_2025

delta_df = pd.DataFrame({
    'PFA_Name': pfa_names,
    'Police_Delta': police_delta,
    'Crime_Delta': final_crime_opt - final_crime_ctrl
})

map_df = police_areas.merge(delta_df, on='PFA_Name')

# exclude Greater manchester
map_df.loc[map_df['PFA_Name'] == 'Greater Manchester', 'Police_Delta'] = np.nan
map_df.loc[map_df['PFA_Name'] == 'Greater Manchester', 'Crime_Delta'] = np.nan

fig, axes = plt.subplots(2, 2, figsize=(20, 18))
ax1, ax2 = axes[0, 0], axes[0, 1]
ax3, ax4 = axes[1, 0], axes[1, 1]

t_years = 2025 + t_forecast / 12.0
ax1.plot(t_years, c_mean, color='#555555', linewidth=2.5, linestyle='--', label='Status Quo')
ax1.fill_between(t_years, c_mean - 1.96*c_std, c_mean + 1.96*c_std, color='#555555', alpha=0.15)
ax1.plot(t_years, p_mean, color='#009E73', linewidth=2.5, label='Hybrid Shielded Optimization')
ax1.fill_between(t_years, p_mean - 1.96*p_std, p_mean + 1.96*p_std, color='#009E73', alpha=0.25)
ax1.set_title('National Target Crime Trajectory: 2026-2028', fontsize=14)
ax1.set_xlim([2026, 2028])
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

max_p_delta = map_df['Police_Delta'].abs().max()
max_c_delta = map_df['Crime_Delta'].abs().max()

map_df.plot(column='Police_Delta', ax=ax3, cmap='PiYG', legend=True,
            vmin=-max_p_delta, vmax=max_p_delta,
            edgecolor='white', linewidth=0.3,
            missing_kwds={'color': '#d3d3d3', 'label': 'Excluded Data'},
            legend_kwds={'label': 'Net Change in Police Officers', 'orientation': 'horizontal'})
ax3.set_title("Targeted Resource Shift (London Shielded)", fontsize=14)
ax3.set_axis_off()

map_df.plot(column='Crime_Delta', ax=ax4, cmap='RdBu_r', legend=True,
            vmin=-max_c_delta, vmax=max_c_delta,
            edgecolor='white', linewidth=0.3,
            missing_kwds={'color': '#d3d3d3', 'label': 'Excluded Data'},
            legend_kwds={'label': 'Net Change in Target Crimes (Optimized - Status Quo)', 'orientation': 'horizontal'})
ax4.set_title("Net Crime Impact per PFA (End of 2028)", fontsize=14)
ax4.set_axis_off()

plt.tight_layout(pad=3.0)
plt.savefig('spatial_optimization_dashboard.png', dpi=150, bbox_inches='tight')