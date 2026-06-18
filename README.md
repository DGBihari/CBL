# Predictive Policing Research: UK Spatial-Temporal SDE Simulator

## Project Overview
This repository contains the core simulation engine for advanced predictive policing research. Designed to evaluate and optimize police force distribution, this framework forecasts the future trajectory of **Anti-Social Behaviour** and **Violence & Sexual Offences** across 43 Police Force Areas in England and Wales.

By combining Machine Learning, Stochastic Differential Equations (SDEs), and geospatial mapping, the model transitions from simply predicting the "Status Quo" to acting as a **Comparative Policy Laboratory**. It runs thousands of parallel temporal realities to measure the exact net benefit of deploying targeted interventions.

---

## Core Simulation Architecture

### 1. Prophet ML Multi-Crime Fusion
Instead of relying on rigid mathematical extrapolation, the underlying macro-trends ($\alpha$) are driven by pre-computed Prophet machine learning forecasts. The engine dynamically fuses the trajectories of multiple crime categories into a single, unified predictive curve per geographic cluster.

### 2. The Goldilocks SDE Physics Engine
The simulation runs on a highly calibrated Stochastic Differential Equation that models crime through four distinct forces:
* **Deterministic Drift ($\alpha$):** The baseline growth or decline.
* **Seasonal Amplitude ($B$):** The natural summer peaks and winter troughs.
* **Police Suppression ($\beta$):** An inverse power-law elasticity model ($P^{-0.3}$) where active police presence dampens random crime volatility.
* **Stochastic Noise ($\sigma$):** A Wiener process ($dW$) simulating unpredictable, real-world shocks.

### 3. Geospatial Spillover & Displacement
Crimes do not exist in a vacuum. Using `geopandas` and adjacency matrices, the model physically maps the borders of England and Wales. When hyper-local interventions (like Hotspot Policing) are activated, the engine explicitly calculates spatial spillover, pushing displaced crime into neighboring jurisdictions.

### 4. Vectorized Difference-in-Differences Engine
To circumvent the bottleneck of slow iterative loops, the policy simulator utilizes **High-Fidelity Matrix Vectorization**. It simultaneously steps 5,000+ parallel universes forward in time, calculating state-dependent interventions (where the policy effectiveness depends on the current month's crime count) in a fraction of a second.

---

## Repo Structure

```
CBL/
├── run_sde_pipeline.py                        # Main pipeline entry point
├── build_E_data_matrix.py                     # Builds the crime count matrix from raw CSV data
├── fit_sde_coefficients.py                    # Fits seasonal and suppression SDE coefficients
├── spatial_resource_optimizer.py              # Optimizes allocation of new officers across PFAs
│
├── ratio_visualizations/
│   ├── crime_derivative_snapshots.py          # Interactive maps of crime rate-of-change
│   ├── scenario_optimization_maps.py          # Interactive maps of police deployment leverage
│   ├── dynamic_heatmap_snapshots.py           # Animated time-slider choropleth heatmaps
│   └── monte_carlo_ensemble.py               # 1,000-realization Status Quo baseline forecast
│
├── policy_examination/
│   └── comparative_policy_simulations.py      # Control vs. Intervention policy simulator
│
└── extras/                                    # Legacy and experimental scripts (not part of pipeline)
    ├── run_sde_pipeline_old.py
    ├── old_ode.py
    ├── old_linear_solution.py
    └── fit_sde_coefficients_extras.py
```

### Core Execution Scripts

* **`run_sde_pipeline.py`**
  The main pipeline entry point. Ingests historical crime, police staffing, and Index of Multiple Deprivation (IMD) data. Delegates matrix construction to `build_E_data_matrix.py` and coefficient fitting to `fit_sde_coefficients.py`, then validates the calibrated model against empirical history. Outputs `time_series_master_goldilocks.csv`.

* **`build_E_data_matrix.py`**
  Constructs the crime count matrix (`E_data`) from raw monthly street-level crime CSVs. Handles PFA name standardization and aligns all data to the master 43-area list.

* **`fit_sde_coefficients.py`**
  Solves for the per-region Seasonal Amplitude ($B_i$) and Police Suppression ($\beta_i$) coefficients. Also builds the spatial adjacency matrix and monthly police headcount matrix used throughout the pipeline.

* **`spatial_resource_optimizer.py`**
  Simulates the optimal redeployment of 3,000 new officers across all 43 PFAs. Runs 1,000 Monte Carlo realizations per candidate allocation to identify which deployments produce the greatest crime reduction.

### Visualization Scripts (`ratio_visualizations/`)

* **`crime_derivative_snapshots.py`**
  Generates interactive Folium choropleth maps showing the monthly rate-of-change ($E'$) of crime across England and Wales, colour-coded by acceleration or deceleration.

* **`scenario_optimization_maps.py`**
  Produces interactive Folium maps colour-coded by each PFA's optimization leverage score — a measure of how sensitive crime counts are to additional officer deployment.

* **`dynamic_heatmap_snapshots.py`**
  Renders an animated time-slider choropleth map using `folium.TimeSliderChoropleth`, allowing exploration of crime trajectory changes across the full forecast horizon.

* **`monte_carlo_ensemble.py`**
  Runs 1,000 SDE realizations under the Status Quo (no new policy) scenario, producing confidence-band forecasts for 2025–2028.

### Policy Simulator (`policy_examination/`)

* **`comparative_policy_simulations.py`**
  The vectorized policy laboratory. Runs a Control universe alongside an Intervention universe. Toggle the three policy booleans at the top of the file to activate strategies individually or in combination.

---

## Required Datasets

To execute the pipeline, the following files must be present:

| File | Description |
|---|---|
| `csv/imd.csv` | Index of Multiple Deprivation data |
| `csv/lad_to_pfa_lookup.csv` | LAD to PFA name lookup table |
| `csv/police_data.csv` | Annual police headcount per PFA |
| `csv/crime_data/**/*-street.csv` | Raw monthly street-level crime CSVs |
| `police_areas.geojson` | PFA boundary geometries for spatial mapping |
| `csv/<crime_type>/<cluster>_<crime_type>.csv` | Pre-computed Prophet ML forecast outputs |
| `time_series_master_goldilocks.csv` | Generated by `run_sde_pipeline.py` |

---

## Modeled Policy Interventions

The `comparative_policy_simulations.py` script allows researchers to test distinct strategic interventions:

1. **Hotspot Policing:** Targets volatile crime spikes. Suppresses localized spikes by 60%, but physically displaces 50% of the prevented crimes into adjacent neighborhoods.
2. **Housing First:** A systemic, non-punitive intervention targeting the baseline drift ($\alpha$). Structurally reduces baseline crime by 34% with zero spatial spillover.
3. **Domestic Violence (DV) Mentoring:** A highly targeted systemic intervention that cuts the underlying deterministic drift for violent offenses by 77%.

To activate a policy, set the corresponding boolean at the top of `comparative_policy_simulations.py`:
```python
POLICY_HOUSING_FIRST     = True
POLICY_HOTSPOT_POLICING  = False
POLICY_DV_MENTORING      = False
```

---

## Sequence of Running

1. `run_sde_pipeline.py`
2. `ratio_visualizations/crime_derivative_snapshots.py`
3. `ratio_visualizations/scenario_optimization_maps.py`
4. `ratio_visualizations/dynamic_heatmap_snapshots.py`
5. `spatial_resource_optimizer.py`
6. `ratio_visualizations/monte_carlo_ensemble.py`
7. `policy_examination/comparative_policy_simulations.py`

---

## Contributions
1. **Daniel Gergo Bihari:** SDE Pipeline Implementation, Report, Final Presentation
2. **Salih Bosna:** SDE Theoretical Aspect, Report, Final Presentation
3. **Sven van den Broek:** Prophet Model Implementation, SARIMA Model Implementation, Report, Midterm Presentation
4. **Alexandros Christou:** SDE Pipeline Implementation, Simulations & Predictions, Report, Final Presentation
5. **Kieran van Eijk:** R&D, Policies & Final Advice, Report, Midterm Presentation

---

## Installation & Usage

### Prerequisites
Requires Python 3.8+.

```bash
pip install numpy pandas scipy matplotlib geopandas folium branca
```
