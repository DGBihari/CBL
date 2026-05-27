import os
import geopandas as gpd
import folium
import pandas as pd 

# 1. Load the Police Boundaries
geojson_path = 'police_areas.geojson'

if not os.path.exists(geojson_path):
    print(f" Error: Could not find '{geojson_path}'. Please ensure it is in the same folder.")
    exit()

police_areas = gpd.read_file(geojson_path)

# 2. Initialize the Base Map
uk_map = folium.Map(location=[54.5, -3.0], zoom_start=6, tiles="cartodb positron")

# 3. Create the Empty Base (The Blank Canvas)
# This draws the police departments but leaves them transparent/uncolored
folium.GeoJson(
    police_areas,
    name="Police Departments (Empty Base)",
    style_function=lambda feature: {
        'fillColor': '#ffffff',  # White fill
        'fillOpacity': 0.1,      # Nearly transparent so you just see the base map
        'color': '#333333',      # Dark grey borders
        'weight': 1.5,           # Border thickness
    }
).add_to(uk_map)

# Add a layer control menu
folium.LayerControl().add_to(uk_map)

# Save the map
output_file = 'uk_empty_base.html'
uk_map.save(output_file)