import os
import geopandas as gpd
import folium
import pandas as pd
import numpy as np

# 1. Load the Police Boundaries
geojson_path = '../police_areas.geojson'
if not os.path.exists(geojson_path):
    print(f"Error: Could not find '{geojson_path}'.")
    exit()

police_areas = gpd.read_file(geojson_path)

# Extract the actual police force names from the geojson to mock our simulation data
# Note: 'pfa23nm' is the property key you identified in your heatmap.py
police_force_names = police_areas['PFA24NM'].unique()

# 2. Load/Mock the Optimized Simulation Data
# In reality, this data comes from solving your SDEs and finding the best initial police distributions
rng = np.random.default_rng(42)
optimized_data = pd.DataFrame({
    'Police_Force_Name': police_force_names,
    # Mocking a low crime/police ratio for the "Coolest" scenario
    'Optimized_Crime_Ratio': rng.uniform(2.0, 5.0, len(police_force_names)) 
})

# 3. Initialize the Base Map
uk_map = folium.Map(location=[54.5, -3.0], zoom_start=6, tiles="cartodb positron")

# 4. Add the Choropleth layer based on your heatmap.py
folium.Choropleth(
    geo_data=police_areas,
    name="Optimized Crime Heatmap",
    data=optimized_data,
    columns=["Police_Force_Name", "Optimized_Crime_Ratio"], 
    key_on="feature.properties.PFA24NM", 
    fill_color="YlOrRd", # Using the Yellow-Orange-Red palette
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="Optimized Crime to Police Ratio"
).add_to(uk_map)

# Add layer control menu
folium.LayerControl().add_to(uk_map)

# Save the map
output_file = 'optimized_coolest_map.html'
uk_map.save(output_file)
print(f"Optimization map saved to {output_file}")