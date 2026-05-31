import pandas as pd
import numpy as np
import geopandas as gpd
import warnings
import glob 
import os    

# Suppress geometry warnings for clean terminal output
warnings.filterwarnings('ignore', category=UserWarning)

print("Starting the Spatial Network SDE Pipeline (Reaction-Diffusion + Regression)...")

# ==========================================
# 1. LOAD DATA & SANITIZE NAMES
# ==========================================
pop_df = pd.read_csv('csv/population_density.csv', skiprows=3)
imd_df = pd.read_csv('csv/imd.csv')
lookup_df = pd.read_csv('csv/lad_to_pfa_lookup.csv')
police_df = pd.read_csv('csv/police_data.csv')
police_areas = gpd.read_file('police_areas.geojson')

# Aggressive Sanitization Function
def standardize_pfa_names(series):
    series = series.astype(str).str.strip().str.replace('\n', '', regex=True)
    
    # Strip out the formal suffixes from the raw UK Police data
    series = series.str.replace(' Police', '', regex=False)
    series = series.str.replace(' Constabulary', '', regex=False)
    series = series.str.replace(' Service', '', regex=False)
    
    # Fix minor mismatches across government datasets
    series.loc[series.str.contains('Hampshire', case=False, na=False)] = 'Hampshire and Isle of Wight'
    series.loc[series.str.contains('Devon', case=False, na=False)] = 'Devon and Cornwall'
    
    # The Two Londons Fix:
    series.loc[series.str.contains('Metropolitan', case=False, na=False)] = 'Metropolitan Police'
    
    # Force the crime data's "City of London" to match the map's "London, City of"
    series.loc[series.str.contains('City of London', case=False, na=False)] = 'London, City of'
    
    return series

# Clean the Map Data
police_areas['PFA_Name'] = standardize_pfa_names(police_areas['PFA24NM'])

# ==========================================
# 2. BUILD THE SPATIAL ADJACENCY NETWORK
# ==========================================
print("Building the geographic neighbor network...")
adjacency_dict = {}
# Add a microscopic buffer to borders to ensure regions separated by rivers/roads still "touch"
police_areas['geometry'] = police_areas['geometry'].buffer(0.001)

for idx, row in police_areas.iterrows():
    # Find all regions that intersect with this region's border
    neighbors = police_areas[police_areas.geometry.intersects(row['geometry'])]['PFA_Name'].tolist()
    # Remove the region itself from its own neighbor list
    neighbors = [n for n in neighbors if n != row['PFA_Name']]
    adjacency_dict[row['PFA_Name']] = neighbors

# ==========================================
# 3. PREPROCESS TIME-SERIES & MERGE
# ==========================================
# IMD
decile_col = [col for col in imd_df.columns if 'Decile' in col][0]
lad_col = [col for col in imd_df.columns if 'Local Authority District name' in col][0]
imd_df['IMD_Decile'] = pd.to_numeric(imd_df[decile_col], errors='coerce')
imd_df['IMD_Score'] = (11 - imd_df['IMD_Decile']) * 10
lad_imd = imd_df.groupby(lad_col)['IMD_Score'].mean().reset_index().rename(columns={lad_col: 'LAD_Name'})

# Population
pop_df.rename(columns={'LAD 2023 Name': 'LAD_Name', 'Area Sq Km': 'Area_Sq_Km'}, inplace=True)
pop_cols = ['Mid-2021: Population', 'Mid-2022: Population', 'Mid-2023: Population', 'Mid-2024: Population']
for col in ['Area_Sq_Km'] + pop_cols:
    pop_df[col] = pop_df[col].astype(str).str.replace(',', '').astype(float)
# Create a placeholder for 2025; the regression loop will overwrite this with mathematical accuracy
pop_df['Mid-2025: Population'] = pop_df['Mid-2024: Population'] 
pop_cols.append('Mid-2025: Population')

lad_pop_long = pd.melt(pop_df.groupby('LAD_Name')[['Area_Sq_Km'] + pop_cols].sum().reset_index(), 
                       id_vars=['LAD_Name', 'Area_Sq_Km'], value_vars=pop_cols, var_name='Year_Str', value_name='Population')
lad_pop_long['Year'] = lad_pop_long['Year_Str'].str.extract(r'(\d{4})').astype(int)
lad_pop_long['LAD_Name'] = lad_pop_long['LAD_Name'].replace({'Bristol, City of': 'Bristol', 'Kingston upon Hull, City of': 'Kingston upon Hull', 'Herefordshire, County of': 'Herefordshire'})

# Merge Geography and Demographics
lad_summary = pd.merge(lad_pop_long, lad_imd, on='LAD_Name', how='left')
lookup_df = lookup_df[['LAD24NM', 'PFA24NM']].rename(columns={'LAD24NM': 'LAD_Name', 'PFA24NM': 'PFA_Name'})
lookup_df['PFA_Name'] = standardize_pfa_names(lookup_df['PFA_Name'])

merged_data = pd.merge(lookup_df, lad_summary, on='LAD_Name', how='left')
pfa_agg = merged_data.groupby(['PFA_Name', 'Year']).agg({'Population': 'sum', 'Area_Sq_Km': 'sum', 'IMD_Score': 'mean'}).reset_index()

# Clean and Merge Police Data
police_df['PFA_Name'] = standardize_pfa_names(police_df['PFA_Name'])
police_df['Police_Count'] = pd.to_numeric(police_df['Police_Count'].astype(str).str.replace(',', '').str.replace('"', ''), errors='coerce')
ts_data = pd.merge(pfa_agg, police_df, on=['PFA_Name', 'Year'], how='left')

# Failsafes for static map data 
ts_data['Area_Sq_Km'] = ts_data['Area_Sq_Km'].replace(0, np.nan).fillna(ts_data['Area_Sq_Km'].mean())
ts_data['IMD_Score'] = ts_data['IMD_Score'].fillna(ts_data['IMD_Score'].mean())

# ==========================================
# NEW: REAL CRIME DATA INGESTION
# ==========================================
print("Ingesting real crime data from /csv/crime_data folder...")

use_cols = ['Month', 'Reported by', 'Crime type']
target_crimes = ['Anti-social behaviour', 'Violence and sexual offences']
crime_counts_list = []

# STRICT FILTER: Only target the files ending in 'street.csv' to avoid court outcomes
search_pattern = os.path.join('csv/crime_data', '**', '*street.csv')
csv_files = glob.glob(search_pattern, recursive=True)

print(f"Found {len(csv_files)} CSV files. Beginning processing...")

if len(csv_files) == 0:
    raise FileNotFoundError("Python found 0 files. Check if the 'csv/crime_data' folder is in the exact same directory as run_sde_pipeline.py.")

for file in csv_files:
    try:
        df_temp = pd.read_csv(file, usecols=use_cols)
        df_filtered = df_temp[df_temp['Crime type'].isin(target_crimes)]
        agg_df = df_filtered.groupby(['Month', 'Reported by']).size().reset_index(name='Crime_Count')
        
        # Only append if the file actually contained relevant crimes
        if not agg_df.empty:
            crime_counts_list.append(agg_df)
            
    except Exception as e:
        # Instead of failing silently, print the exact error for the first few files
        print(f"FAILED on file {file} - Error: {e}")
        continue

# Failsafe check before concatenating
if len(crime_counts_list) == 0:
    raise ValueError("Files were found, but no data could be extracted. Read the FAILED error messages above to see why.")

# Combine all the lightweight summary dataframes into one master dataframe
raw_crime_df = pd.concat(crime_counts_list, ignore_index=True)

# Extract the Year from the 'YYYY-MM' format
raw_crime_df['Year'] = raw_crime_df['Month'].str.split('-').str[0].astype(int)

# Standardize the internal PFA names to match our network map exactly
raw_crime_df['PFA_Name'] = standardize_pfa_names(raw_crime_df['Reported by'])

# Final Aggregation: Group by Year and PFA_Name to get total yearly counts per region
yearly_crimes = raw_crime_df.groupby(['PFA_Name', 'Year'])['Crime_Count'].sum().reset_index()

# Merge the real crime data into our main ts_data dataframe
ts_data = pd.merge(ts_data, yearly_crimes, on=['PFA_Name', 'Year'], how='left')

ts_data['Crime_Count'] = ts_data.groupby('PFA_Name')['Crime_Count'].transform(lambda x: x.fillna(x.mean()))
ts_data['Crime_Count'] = ts_data['Crime_Count'].fillna(0)

print("Real crime data successfully merged. Calculating SDE...")

# ==========================================
# 4. REGRESSION & INTERPOLATION METHODOLOGY
# ==========================================
print("Applying linear regression and interpolation to N_i and P_i...")
ts_data = ts_data.sort_values(by=['PFA_Name', 'Year'])

# Police Interpolation (Connecting the dots for missing hiring cycles)
ts_data['Police_Count'] = ts_data.groupby('PFA_Name')['Police_Count'].transform(
    lambda group: group.interpolate(method='linear', limit_direction='both')
)
ts_data['Police_Count'] = ts_data['Police_Count'].fillna(ts_data['Police_Count'].mean()) # Absolute failsafe

# Population Regression (Finding the continuous line of best fit to prevent Pandas apply/groupby crashes)
for pfa in ts_data['PFA_Name'].unique():
    mask = ts_data['PFA_Name'] == pfa
    valid_data = ts_data[mask].dropna(subset=['Year', 'Population'])
    
    if len(valid_data) > 1:
        # np.polyfit calculates the slope (m) and intercept (c) of the line
        slope, intercept = np.polyfit(valid_data['Year'], valid_data['Population'], 1)
        
        # Overwrite the bumpy population data with the perfect mathematical line
        ts_data.loc[mask, 'Population'] = (slope * ts_data.loc[mask, 'Year']) + intercept

# ==========================================
# 5. CALCULATING SPATIAL SDE VARIABLES
# ==========================================
print("Calculating spatial gradients and empirical coefficients...")

# Delta Police (Year-over-Year change)
ts_data['Delta_Police'] = ts_data.groupby('PFA_Name')['Police_Count'].diff().fillna(0)

# Empirical Constants
ts_data['Sigma_i'] = ts_data.groupby('PFA_Name')['Crime_Count'].transform('std') / ts_data['Area_Sq_Km']

# NEW: The GMP Failsafe. Calculate the average volatility of the rest of the UK.
# Then replace any corrupted 0s (like Greater Manchester) or NaNs with that national average.
national_avg_sigma = ts_data.loc[ts_data['Sigma_i'] > 0, 'Sigma_i'].mean()
ts_data['Sigma_i'] = ts_data['Sigma_i'].replace(0, national_avg_sigma).fillna(national_avg_sigma)

ts_data['Gamma_i'] = 0.05 / (ts_data['Population'] / ts_data['Area_Sq_Km']) 
ts_data['Alpha_i'] = ts_data['IMD_Score'] * 0.005

# Spatial Spillover Calculation
def calculate_spatial_spillover(row, df, adjacency, beta=0.15):
    neighbors = adjacency.get(row['PFA_Name'], [])
    if not neighbors: return 0.0
    
    # Isolate the exact same year
    current_year_df = df[df['Year'] == row['Year']]
    neighbor_data = current_year_df[current_year_df['PFA_Name'].isin(neighbors)]
    
    if neighbor_data.empty: return 0.0
    
    # Calculate gradient (Average Neighbor Crime Density - Local Crime Density)
    avg_neighbor_density = (neighbor_data['Crime_Count'] / neighbor_data['Area_Sq_Km']).mean()
    local_density = row['Crime_Count'] / row['Area_Sq_Km']
    
    return beta * (avg_neighbor_density - local_density)

ts_data['Spillover_Force'] = ts_data.apply(lambda row: calculate_spatial_spillover(row, ts_data, adjacency_dict), axis=1)

# ==========================================
# 6. RUNNING THE SPATIAL SDE
# ==========================================
print("Solving the Network-Coupled Stochastic Differential Equation...")

def run_spatial_sde(row):
    dt = 1.0
    C_i = row['Crime_Count'] / row['Area_Sq_Km']
    Delta_P_i = row['Delta_Police'] / row['Area_Sq_Km']
    
    growth = row['Alpha_i'] * C_i
    suppression = row['Gamma_i'] * Delta_P_i
    spillover = row['Spillover_Force']
    
    dW = np.random.normal(0, np.sqrt(dt))
    stochastic = row['Sigma_i'] * C_i * dW
    
    return growth - suppression + spillover + stochastic

ts_data['C_Prime_i'] = ts_data.apply(run_spatial_sde, axis=1)

# Sort chronologically and export
ts_data = ts_data.sort_values(by=['Year', 'PFA_Name'])
ts_data.to_csv('time_series_master_calculated.csv', index=False)

print(f"Success! Analyzed {len(ts_data)} empirical spatial data points.")
print("Pipeline complete! Exported continuous, regression-smoothed dataset.")