import geopandas as gpd
import folium
import pandas as pd

print("Loading data and cleaning map boundaries...")

# 1. Load Data
police_areas = gpd.read_file('../police_areas.geojson')
ts_data = pd.read_csv('../time_series_master_calculated.csv')

# ==========================================
# 🚨 BRUTE-FORCE GEOJSON NAMES 🚨
# Overwrite any invisible characters directly in the map boundaries
# ==========================================
police_areas['PFA24NM'] = police_areas['PFA24NM'].astype(str).str.strip()
police_areas.loc[police_areas['PFA24NM'].str.contains('Devon', case=False, na=False), 'PFA24NM'] = 'Devon and Cornwall'
police_areas.loc[police_areas['PFA24NM'].str.contains('Hampshire', case=False, na=False), 'PFA24NM'] = 'Hampshire and Isle of Wight'

# 2. Prepare 2025 CSV Data
current_data = ts_data[ts_data['Year'] == 2025].copy()
current_data['PFA_Name'] = current_data['PFA_Name'].astype(str).str.strip()

# Fix City of London Map Hole
met_data = current_data[current_data['PFA_Name'] == 'Metropolitan Police'].copy()
met_data['PFA_Name'] = 'London, City of'
current_data = pd.concat([current_data, met_data], ignore_index=True)

# ==========================================
# 🚨 PREVENT BLACK REGIONS FROM NaN MATH 🚨
# If the SDE failed (NaN), force it to 0 so the region renders neutrally
# ==========================================
current_data['C_Prime_i'] = current_data['C_Prime_i'].fillna(0)
current_data = current_data.drop_duplicates(subset=['PFA_Name']) # Remove any accidental duplicates

# 3. Render Map
uk_map = folium.Map(location=[54.5, -3.0], zoom_start=6, tiles="cartodb positron")

# Calculate color bins safely
max_val = max(abs(current_data['C_Prime_i'].min()), abs(current_data['C_Prime_i'].max()))
limit = max_val + 1
custom_bins = [-limit, -limit*0.66, -limit*0.33, 0, limit*0.33, limit*0.66, limit]

folium.Choropleth(
    geo_data=police_areas,
    name="2025 Real Derivative Map",
    data=current_data,
    columns=["PFA_Name", "C_Prime_i"],
    key_on="feature.properties.PFA24NM",
    fill_color="RdYlGn_r",
    bins=custom_bins,
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="2025 Crime Growth Rate (C'_i)",
    nan_fill_color="black" # Explicitly setting this so we know if the join fails
).add_to(uk_map)

uk_map.save('real_crime_derivative_map_2025.html')
print("Map generated! Open real_crime_derivative_map_2025.html to view.")