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



# =====================================================================
# THE FUTURE UPGRADE: HOW TO COLOR IT LATER
# =====================================================================
# When you have your real crime data and want to make it look exactly 
# like the photo you sent, you will delete Step 3 above and replace 
# it with the code below. 

"""
# A. Load your real data (e.g., from a CSV file you create)
# It needs at least two columns: The police force name, and the crime count.
crime_data = pd.read_csv('my_real_crime_data.csv')

# B. Add the Choropleth layer
folium.Choropleth(
    geo_data=police_areas,
    name="Crime Heatmap",
    data=crime_data,
    columns=["Name_of_Police_Force_Column", "Name_of_Crime_Count_Column"], # Update these names!
    key_on="feature.properties.pfa23nm", # This links your CSV to the map shapes
    fill_color="YlOrRd",                 # THIS is the Yellow-Orange-Red from your photo
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="Crime Density by Police Force"
).add_to(uk_map)
"""