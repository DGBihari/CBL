import pandas as pd
import numpy as np

print("Starting the Data Pipeline...")

# ==========================================
# 1. LOAD THE DATA
# ==========================================
deprivation = pd.read_csv('csv/deprivation.csv')
pop_density = pd.read_csv('csv/population_density.csv', skiprows=3)
lookup_table = pd.read_csv('csv/lad_to_pfa_lookup.csv')

# 🚨 NEW: Load your freshly extracted real police data
police_data = pd.read_csv('csv/police_data.csv')

# ==========================================
# 2. CLEAN AND MERGE TO LAD LEVEL
# ==========================================
deprivation = deprivation.rename(columns={
    'Local Authority District name (2019)': 'LAD_Name',
    'Index of Multiple Deprivation (IMD) Score': 'IMD_Score'
})

pop_density = pop_density.rename(columns={
    'LAD 2023 Name': 'LAD_Name',
    'Area Sq Km': 'Area_Sq_Km',
    'Mid-2024: Population': 'Population'
})

# Fix spelling differences
pop_density['LAD_Name'] = pop_density['LAD_Name'].replace({
    'Bristol, City of': 'Bristol',
    'Kingston upon Hull, City of': 'Kingston upon Hull',
    'Herefordshire, County of': 'Herefordshire'
})

pop_density['Population'] = pop_density['Population'].astype(str).str.replace(',', '').astype(float)
pop_density['Area_Sq_Km'] = pop_density['Area_Sq_Km'].astype(str).str.replace(',', '').astype(float)

lad_pop = pop_density.groupby('LAD_Name')[['Population', 'Area_Sq_Km']].sum().reset_index()
lad_imd = deprivation.groupby('LAD_Name')['IMD_Score'].mean().reset_index()

# Left merge to preserve Welsh and restructured districts
lad_summary = pd.merge(lad_pop, lad_imd, on='LAD_Name', how='left')

# Impute missing poverty scores with the national average
national_avg_imd = lad_summary['IMD_Score'].mean()
lad_summary['IMD_Score'] = lad_summary['IMD_Score'].fillna(national_avg_imd)

# ==========================================
# 3. UPGRADE TO POLICE FORCE LEVEL
# ==========================================
lookup_table = lookup_table[['LAD24NM', 'PFA24NM']].rename(columns={'LAD24NM': 'LAD_Name', 'PFA24NM': 'PFA_Name'})

# Map City of London to Met Police to prevent density anomalies
lookup_table['PFA_Name'] = lookup_table['PFA_Name'].replace({'London, City of': 'Metropolitan Police'})

merged_data = pd.merge(lookup_table, lad_summary, on='LAD_Name', how='left')

pfa_aggregated = merged_data.groupby('PFA_Name').agg({
    'Population': 'sum',
    'Area_Sq_Km': 'sum',
    'IMD_Score': 'mean' 
}).reset_index()

pfa_aggregated = pfa_aggregated.dropna(subset=['Population'])

# ==========================================
# 4. SOLVE THE SDE (Now with REAL Police Data)
# ==========================================
# 🚨 Merge the real police numbers into our aggregated data
pfa_aggregated = pd.merge(pfa_aggregated, police_data, on='PFA_Name', how='left')

# CLEANUP: Strip out any sneaky commas and force the text into math numbers
pfa_aggregated['Police_Count'] = pfa_aggregated['Police_Count'].astype(str).str.replace(',', '')
pfa_aggregated['Police_Count'] = pd.to_numeric(pfa_aggregated['Police_Count'], errors='coerce')

# Handle any missing police data safely so the math doesn't crash
pfa_aggregated['Police_Count'] = pfa_aggregated['Police_Count'].fillna(pfa_aggregated['Police_Count'].mean())

# MOCK DATA: We still simulate crimes until you have the final crime CSV
np.random.seed(42)
pfa_aggregated['Crime_Count'] = np.random.randint(2000, 15000, size=len(pfa_aggregated))

def calculate_crime_derivative(population, area, imd_score, crime_count, police_count):
    dt = 1.0 
    sigma = 1.5
    alpha_i = imd_score * 0.005  
    k_i = 0.02                   
    
    C_i = crime_count / area
    P_i = police_count / area
    N_i = population / area
    
    dW = np.random.normal(0, np.sqrt(dt))
    deterministic = (alpha_i * C_i) - (k_i * C_i * (P_i / (N_i + P_i)))
    stochastic = sigma * C_i * dW
    
    return deterministic + stochastic

pfa_aggregated['C_Prime_i'] = pfa_aggregated.apply(
    lambda row: calculate_crime_derivative(
        row['Population'], row['Area_Sq_Km'], row['IMD_Score'], 
        row['Crime_Count'], row['Police_Count']
    ), axis=1
)

# ==========================================
# 5. EXPORT
# ==========================================
pfa_aggregated.to_csv('final_pfa_derivatives.csv', index=False)
print("Pipeline complete! Math solved using real Police counts.")