import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import sys
import warnings

warnings.filterwarnings('ignore')

print("Running Monte Carlo Ensemble with Real Data & Goldilocks ODE...")

# ==========================================
# 1. LOAD REAL DATA & COEFFICIENTS
# ==========================================
ts_data = pd.read_csv('../time_series_master_goldilocks.csv')

# Extract 2025 data for initial conditions
ts_2025 = ts_data[ts_data['Year'] == 2025].copy()
pfa_names = ts_2025['PFA_Name'].values
E0_dict = dict(zip(ts_2025['PFA_Name'], ts_2025['Crime_Count']))
P0_dict = dict(zip(ts_2025['PFA_Name'], ts_2025['Police_Count']))
alpha_dict = dict(zip(ts_2025['PFA_Name'], ts_2025['Alpha_i']))
B_dict = dict(zip(ts_2025['PFA_Name'], ts_2025['B_i']))
beta_dict = dict(zip(ts_2025['PFA_Name'], ts_2025['Beta_i']))
sigma_dict = dict(zip(ts_2025['PFA_Name'], ts_2025['Sigma_i']))

# Build initial condition vector
E0 = np.array([E0_dict.get(pfa, 1.0) for pfa in pfa_names])
alpha_vec = np.array([alpha_dict.get(pfa, 0.0) for pfa in pfa_names])
B_vec = np.array([B_dict.get(pfa, 100.0) for pfa in pfa_names])
beta_vec = np.array([beta_dict.get(pfa, 0.01) for pfa in pfa_names])
sigma_vec = np.array([sigma_dict.get(pfa, 0.1) for pfa in pfa_names])

# Load adjacency for spillover
import geopandas as gpd
police_areas = gpd.read_file('../police_areas.geojson')

def standardize_pfa_names(series):
    series = series.astype(str).str.strip().str.replace('\n', '', regex=True)
    series = series.str.replace(' Police', '', regex=False)
    series = series.str.replace(' Constabulary', '', regex=False)
    series = series.str.replace(' Service', '', regex=False)
    series.loc[series.str.contains('Hampshire', case=False, na=False)] = 'Hampshire and Isle of Wight'
    series.loc[series.str.contains('Devon', case=False, na=False)] = 'Devon and Cornwall'
    series.loc[series.str.contains('Metropolitan', case=False, na=False)] = 'Metropolitan Police'
    series.loc[series.str.contains('City of London', case=False, na=False)] = 'London, City of'
    return series

police_areas['PFA_Name'] = standardize_pfa_names(police_areas['PFA24NM'])
police_areas['geometry'] = police_areas['geometry'].buffer(0.001)

adjacency_dict = {}
for idx, row in police_areas.iterrows():
    neighbors = police_areas[police_areas.geometry.intersects(row['geometry'])]['PFA_Name'].tolist()
    neighbors = [n for n in neighbors if n != row['PFA_Name']]
    adjacency_dict[row['PFA_Name']] = neighbors

# Build adjacency matrix
n_pfas = len(pfa_names)
ADJ = np.zeros((n_pfas, n_pfas))
for i, pfa in enumerate(pfa_names):
    neighbors = adjacency_dict.get(pfa, [])
    for j, pfa_j in enumerate(pfa_names):
        if pfa_j in neighbors:
            ADJ[i, j] = 1

# Static police count interpolator (constant for 2025)
P_static = np.array([P0_dict.get(pfa, 1000.0) for pfa in pfa_names])

# ==========================================
# 2. GOLDILOCKS ODE SYSTEM WITH STOCHASTIC NOISE
# ==========================================
num_realizations = 1000
t_span = (0, 48)  # 4 years into future in months
t_eval = np.linspace(0, 48, 49)

def ode_system_stochastic(t, E_vec, B_perturb):
    """Goldilocks ODE with stochastic noise term."""
    dE = np.zeros(len(pfa_names))
    omega = 2 * np.pi / 12
    phase_shift = 3.0 * (2 * np.pi / 12)

    for i in range(len(pfa_names)):
        neighbours = np.where(ADJ[i])[0]
        n_nb = len(neighbours)

        P_i_t = max(P_static[i], 1.0)

        growth = alpha_vec[i] * E_vec[i]

        nb_term = 0.0
        if n_nb > 0:
            for j in neighbours:
                nb_term += (alpha_vec[j] / n_nb) * E_vec[j]

        seasonal = (1.7 * (B_vec[i] + B_perturb[i])) * np.cos(omega * t - phase_shift)
        suppression = min(beta_vec[i], 1.0) * E_vec[i] * (P_i_t ** -0.3)

        drift = 3 * (growth + nb_term - suppression + seasonal)
        noise = (P_i_t ** -0.3) * 10.0 * sigma_vec[i] * np.random.normal(0, np.sqrt(1 / 12))

        dE[i] = drift + noise

    return dE

# ==========================================
# 3. RUN 1000 MONTE CARLO REALIZATIONS
# ==========================================
np.random.seed(42)
all_solutions = []

for realization in range(num_realizations):
    B_perturb = np.random.normal(0, 0.05 * B_vec)

    ode_closure = lambda t, E_vec: ode_system_stochastic(t, E_vec, B_perturb)

    try:
        sol = solve_ivp(ode_closure, t_span, E0, method='RK45', t_eval=t_eval, max_step=1, rtol=1e-4, atol=1e-2)
        if sol.success:
            all_solutions.append(sol.y.T)
    except:
        continue

    if (realization + 1) % 100 == 0:
        print(f"  Completed {realization + 1}/{num_realizations} realizations...")

all_solutions = np.array(all_solutions)
print(f"Successfully completed {len(all_solutions)} realizations")

# ==========================================
# 4. CALCULATE ENSEMBLE STATISTICS
# ==========================================
E_total_ensemble = all_solutions.sum(axis=2)  # Sum across all PFAs
E_mean = E_total_ensemble.mean(axis=0)
E_std = E_total_ensemble.std(axis=0)
E_upper = E_mean + 1.96 * E_std
E_lower = np.maximum(E_mean - 1.96 * E_std, 0)

# ==========================================
# 5. PLOT ENSEMBLE
# ==========================================
t_months = t_eval
t_years = 2025 + t_months / 12.0

fig, ax = plt.subplots(figsize=(14, 7))

# Plot all individual realizations lightly
for i in range(len(all_solutions)):
    ax.plot(t_years, E_total_ensemble[i], color='gray', alpha=0.05, linewidth=0.5)

# Plot ensemble mean
ax.plot(t_years, E_mean, color='red', linewidth=2.5, label='Ensemble Mean (1000 realizations)')

# Plot 95% confidence interval
ax.fill_between(t_years, E_lower, E_upper, color='red', alpha=0.25, label='95% Confidence Interval')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Crime Count (England & Wales)', fontsize=12)
ax.set_title('Goldilocks Monte Carlo Ensemble: 2025-2029 Crime Forecast (1000 Realizations)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(2025, 2029)

plt.tight_layout()
plt.savefig('monte_carlo_ensemble.png', dpi=150, bbox_inches='tight')
print("Saved: monte_carlo_ensemble.png")
plt.show()