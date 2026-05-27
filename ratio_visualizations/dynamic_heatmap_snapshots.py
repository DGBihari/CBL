import os
import geopandas as gpd
import folium
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
GEOJSON_PATH = '../police_areas.geojson'
POLICE_FORCE_COLUMN = 'PFA24NM'  # Updated to match your exact file

# ==========================================
# 1. Load Boundaries
# ==========================================
if not os.path.exists(GEOJSON_PATH):
    print(f"Error: Could not find '{GEOJSON_PATH}'.")
    exit()

police_areas = gpd.read_file(GEOJSON_PATH)
police_force_names = police_areas[POLICE_FORCE_COLUMN].unique()

# ==========================================
# 2. Simulate SDE Data Over Time (Mock Data)
# ==========================================
time_steps = [0, 15, 30] # Specific days to map
simulated_ratios = {}
np.random.seed(42)

for t in time_steps:
    if t == 0:
        # Initial state: higher ratios, more variance
        ratios = np.random.uniform(5, 12, len(police_force_names))
    else:
        # Simulating random growth/reduction
        ratios = simulated_ratios[t-15] + np.random.normal(0, 1.5, len(police_force_names))
    
    simulated_ratios[t] = np.clip(ratios, 0, 15) # Cap boundaries for coloring

# ==========================================
# 3. Generate a Map for Each Time Step
# ==========================================
for t in time_steps:
    time_data = pd.DataFrame({
        'Police_Force_Name': police_force_names,
        'Ratio': simulated_ratios[t]
    })
    
    uk_map = folium.Map(location=[54.5, -3.0], zoom_start=6, tiles="cartodb positron")
    
    folium.Choropleth(
        geo_data=police_areas,
        name=f"Heatmap Day {t}",
        data=time_data,
        columns=["Police_Force_Name", "Ratio"],
        key_on=f"feature.properties.{POLICE_FORCE_COLUMN}",
        fill_color="YlOrRd",
        fill_opacity=0.8,
        line_opacity=0.3,
        legend_name=f"Crime to Police Ratio (Day {t})"
    ).add_to(uk_map)
    
    folium.LayerControl().add_to(uk_map)
    
    output_file = f'simulation_day_{t}.html'
    uk_map.save(output_file)
    print(f"Generated {output_file}")