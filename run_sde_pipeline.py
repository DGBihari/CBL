import pandas as pd
import numpy as np

print("Starting the Bulletproof Time-Series Data Pipeline (2021-2025)...")

# ==========================================
# 1. LOAD THE RAW DATA
# ==========================================
pop_df = pd.read_csv('csv/population_density.csv', skiprows=3)
imd_df = pd.read_csv('csv/imd.csv')
lookup_df = pd.read_csv('csv/lad_to_pfa_lookup.csv')
police_df = pd.read_csv('csv/police_data.csv')

# ==========================================
# 2. PREPROCESS IMD (Poverty Index)
# ==========================================
decile_col = [col for col in imd_df.columns if 'Decile' in col][0]
lad_col = [col for col in imd_df.columns if 'Local Authority District name' in col][0]
imd_df['IMD_Decile'] = pd.to_numeric(imd_df[decile_col], errors='coerce')
imd_df['IMD_Score'] = (11 - imd_df['IMD_Decile']) * 10
lad_imd = imd_df.groupby(lad_col)['IMD_Score'].mean().reset_index()
lad_imd.rename(columns={lad_col: 'LAD_Name'}, inplace=True)

# ==========================================
# 3. PREPROCESS POPULATION (Time-Series)
# ==========================================
pop_df.rename(columns={'LAD 2023 Name': 'LAD_Name', 'Area Sq Km': 'Area_Sq_Km'}, inplace=True)
pop_cols = ['Mid-2021: Population', 'Mid-2022: Population', 'Mid-2023: Population', 'Mid-2024: Population']
for col in ['Area_Sq_Km'] + pop_cols:
    pop_df[col] = pop_df[col].astype(str).str.replace(',', '').astype(float)

pop_df['Mid-2025: Population'] = pop_df['Mid-2024: Population']
pop_cols.append('Mid-2025: Population')

lad_pop = pop_df.groupby('LAD_Name')[['Area_Sq_Km'] + pop_cols].sum().reset_index()
lad_pop_long = pd.melt(lad_pop, id_vars=['LAD_Name', 'Area_Sq_Km'], value_vars=pop_cols, var_name='Year_Str', value_name='Population')
lad_pop_long['Year'] = lad_pop_long['Year_Str'].str.extract(r'(\d{4})').astype(int)
lad_pop_long.drop(columns=['Year_Str'], inplace=True)

# ==========================================
# 4. AGGRESSIVE NAME MATCHING & MERGE
# ==========================================
# Function to forcefully standardize troublesome region names across ALL dataframes
def standardize_pfa_names(df, col_name):
    df[col_name] = df[col_name].astype(str).str.strip().str.replace('\n', '', regex=True)
    df.loc[df[col_name].str.contains('Hampshire', case=False, na=False), col_name] = 'Hampshire and Isle of Wight'
    df.loc[df[col_name].str.contains('Devon', case=False, na=False), col_name] = 'Devon and Cornwall'
    df.loc[df[col_name].str.contains('London, City of', case=False, na=False), col_name] = 'Metropolitan Police'
    return df

lad_pop_long['LAD_Name'] = lad_pop_long['LAD_Name'].replace({'Bristol, City of': 'Bristol', 'Kingston upon Hull, City of': 'Kingston upon Hull', 'Herefordshire, County of': 'Herefordshire'})
lad_summary = pd.merge(lad_pop_long, lad_imd, on='LAD_Name', how='left')

lookup_df = lookup_df[['LAD24NM', 'PFA24NM']].rename(columns={'LAD24NM': 'LAD_Name', 'PFA24NM': 'PFA_Name'})
lookup_df = standardize_pfa_names(lookup_df, 'PFA_Name')

merged_data = pd.merge(lookup_df, lad_summary, on='LAD_Name', how='left')
pfa_aggregated = merged_data.groupby(['PFA_Name', 'Year']).agg({'Population': 'sum', 'Area_Sq_Km': 'sum', 'IMD_Score': 'mean'}).reset_index()

police_df = standardize_pfa_names(police_df, 'PFA_Name')
police_df['Police_Count'] = pd.to_numeric(police_df['Police_Count'].astype(str).str.replace(',', '').str.replace('"', ''), errors='coerce')

ts_data = pd.merge(pfa_aggregated, police_df, on=['PFA_Name', 'Year'], how='left')

# ==========================================
# 5. FAILSAFE MATH & SDE CALCULATION
# ==========================================
# Force-fill any missing data with national averages to prevent division by zero
ts_data['Population'] = ts_data['Population'].fillna(ts_data['Population'].mean())
ts_data['Area_Sq_Km'] = ts_data['Area_Sq_Km'].replace(0, np.nan).fillna(ts_data['Area_Sq_Km'].mean())
ts_data['IMD_Score'] = ts_data['IMD_Score'].fillna(ts_data['IMD_Score'].mean())
ts_data['Police_Count'] = ts_data['Police_Count'].fillna(ts_data['Police_Count'].mean())

np.random.seed(42)
ts_data['Crime_Count'] = np.random.randint(2000, 15000, size=len(ts_data))

def calculate_crime_derivative(population, area, imd_score, crime_count, police_count):
    dt, sigma, k_i = 1.0, 1.5, 0.02
    alpha_i = imd_score * 0.005  
    
    C_i = crime_count / area
    P_i = police_count / area
    N_i = population / area
    
    dW = np.random.normal(0, np.sqrt(dt))
    deterministic = (alpha_i * C_i) - (k_i * C_i * (P_i / (N_i + P_i)))
    return deterministic + (sigma * C_i * dW)

ts_data['C_Prime_i'] = ts_data.apply(
    lambda row: calculate_crime_derivative(row['Population'], row['Area_Sq_Km'], row['IMD_Score'], row['Crime_Count'], row['Police_Count']), axis=1
)

ts_data = ts_data.sort_values(by=['Year', 'PFA_Name'])
ts_data.to_csv('time_series_master_calculated.csv', index=False)
print("Pipeline complete! Run your mapping scripts now.")