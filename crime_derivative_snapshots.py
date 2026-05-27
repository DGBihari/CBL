import os
import geopandas as gpd
import folium
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
GEOJSON_PATH = 'police_areas.geojson' # Use '../police_areas.geojson' if your file is one folder up
POLICE_FORCE_COLUMN = 'PFA24NM' 
PIPELINE_DATA_PATH = 'final_pfa_derivatives.csv'

# ==========================================
# 1. Load Boundaries & Pipeline Data
# ==========================================
if not os.path.exists(GEOJSON_PATH):
    print(f"Error: Could not find '{GEOJSON_PATH}'.")
    exit()

if not os.path.exists(PIPELINE_DATA_PATH):
    print(f"Error: Could not find '{PIPELINE_DATA_PATH}'. Run the pipeline script first!")
    exit()

police_areas = gpd.read_file(GEOJSON_PATH)
pipeline_data = pd.read_csv(PIPELINE_DATA_PATH)

# ==========================================
# Map Shape Fix: Color the City of London 
# ==========================================
# Duplicate the Metropolitan Police row and rename it to 'London, City of' 
# so the map knows to paint the geographical hole the exact same color!
met_data = pipeline_data[pipeline_data['PFA_Name'] == 'Metropolitan Police'].copy()
met_data['PFA_Name'] = 'London, City of'
pipeline_data = pd.concat([pipeline_data, met_data], ignore_index=True)

# Format the data for the map
time_data = pd.DataFrame({
    'Police_Force_Name': pipeline_data['PFA_Name'],
    'Crime_Growth_Rate': pipeline_data['C_Prime_i']
})

# ==========================================
# 2. Generate the Real Data Map
# ==========================================
print("Generating the complete, zero-anchored diverging map...")

uk_map = folium.Map(location=[54.5, -3.0], zoom_start=6, tiles="cartodb positron")

# 🚨 THE FIX: Create dynamic bins so the map never crashes!
# 1. Find the highest absolute number to make the scale completely symmetrical around 0
max_val = max(abs(time_data['Crime_Growth_Rate'].min()), abs(time_data['Crime_Growth_Rate'].max()))

# 2. Add a tiny buffer (+1) so no data point sits perfectly on the mathematical edge
limit = max_val + 1

# 3. Create 7 perfectly spaced bins from negative to positive, guaranteed to pass through exactly 0
custom_bins = [-limit, -limit*0.66, -limit*0.33, 0, limit*0.33, limit*0.66, limit]

folium.Choropleth(
    geo_data=police_areas,
    name="Real Derivative Map",
    data=time_data,
    columns=["Police_Force_Name", "Crime_Growth_Rate"],
    key_on=f"feature.properties.{POLICE_FORCE_COLUMN}",
    fill_color="RdYlGn_r",  # Red-Yellow-Green (reversed)
    bins=custom_bins,       # Use our newly calculated dynamic bins!
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="Calculated Crime Growth Rate (C'_i)"
).add_to(uk_map)

folium.LayerControl().add_to(uk_map)

output_file = 'real_crime_derivative_map.html'
uk_map.save(output_file)
print(f"Success! Perfect map saved to {output_file}")