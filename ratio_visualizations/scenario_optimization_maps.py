import geopandas as gpd
import folium
import pandas as pd

print("Loading data and cleaning map boundaries for optimization map...")

# 1. Load Data
police_areas = gpd.read_file('../police_areas.geojson')
ts_data = pd.read_csv('../time_series_master_goldilocks.csv')

# ==========================================
# 🚨 BRUTE-FORCE GEOJSON NAMES 🚨
# Overwrite any invisible characters directly in the map boundaries
# ==========================================
police_areas['PFA24NM'] = police_areas['PFA24NM'].astype(str).str.strip()
police_areas.loc[police_areas['PFA24NM'].str.contains('Devon', case=False, na=False), 'PFA24NM'] = 'Devon and Cornwall'
police_areas.loc[police_areas['PFA24NM'].str.contains('Hampshire', case=False, na=False), 'PFA24NM'] = 'Hampshire and Isle of Wight'

# 2. Prepare 2025 CSV Data
opt_data = ts_data[ts_data['Year'] == 2025].copy()
opt_data['PFA_Name'] = opt_data['PFA_Name'].astype(str).str.strip()

# Fix City of London Map Hole
met_data = opt_data[opt_data['PFA_Name'] == 'Metropolitan Police'].copy()
met_data['PFA_Name'] = 'London, City of'
opt_data = pd.concat([opt_data, met_data], ignore_index=True)

# ==========================================
# 3. CALCULATE OPTIMIZATION LEVERAGE
# ==========================================
# Leverage Ratio: (Poverty Multiplier / Current Police Allocation)
opt_data['Optimization_Leverage'] = (opt_data['IMD_Score'] * 0.005) / opt_data['Police_Count']

# Failsafe for NaN math to prevent black regions
opt_data['Optimization_Leverage'] = opt_data['Optimization_Leverage'].fillna(0)
opt_data = opt_data.drop_duplicates(subset=['PFA_Name'])

# ==========================================
# 4. RENDER MAP
# ==========================================
uk_map = folium.Map(location=[54.5, -3.0], zoom_start=6, tiles="cartodb dark_matter")

folium.Choropleth(
    geo_data=police_areas,
    name="Intervention Leverage",
    data=opt_data,
    columns=["PFA_Name", "Optimization_Leverage"],
    key_on="feature.properties.PFA24NM",
    fill_color="YlGnBu",     # <--- Colorblind safe sequential scale (Dark Blue = High Priority)
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="Resource Optimization Priority (Dark Blue = Highest Leverage)",
    nan_fill_color="black"
).add_to(uk_map)

uk_map.save('optimized_leverage_map_2025.html')
print("Saved Intervention Target Map to optimized_leverage_map_2025.html")