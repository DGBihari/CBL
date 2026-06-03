import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import TimeSliderChoropleth
import branca.colormap as cm
import re

print("Implementing Numeric-ID failsafe for animated map...")

# 1. Load Data
gdf = gpd.read_file('../police_areas.geojson')
ts_data = pd.read_csv('../time_series_master_goldilocks.csv')

# ==========================================
# 2. BULLETPROOF NAME MATCHING (Fuzzy Match)
# ==========================================
# This function removes ALL spaces, punctuation, and capitalization.
# Example: "Avon & Somerset" and "Avon and Somerset " both become "avonandsomerset"
def simplify_name(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = text.replace('&', 'and')
    text = re.sub(r'[^a-z]', '', text) 
    return text

gdf['Simplified_Name'] = gdf['PFA24NM'].apply(simplify_name)
ts_data['Simplified_Name'] = ts_data['PFA_Name'].apply(simplify_name)

# Manual overrides for known tricky regions
gdf.loc[gdf['Simplified_Name'].str.contains('devon'), 'Simplified_Name'] = 'devonandcornwall'
gdf.loc[gdf['Simplified_Name'].str.contains('hampshire'), 'Simplified_Name'] = 'hampshireandisleofwight'

# Fix the "City of London" hole by duplicating the Met Police data
met_data = ts_data[ts_data['PFA_Name'] == 'Metropolitan Police'].copy()
met_data['Simplified_Name'] = 'londoncityof'
ts_data = pd.concat([ts_data, met_data], ignore_index=True)

# ==========================================
# 3. THE MAGIC FIX: ASSIGN NUMERIC IDs
# ==========================================
# By changing the GeoJSON IDs to pure numbers, we prevent the JavaScript crash.
gdf['numeric_id'] = [str(i) for i in range(len(gdf))]
gdf = gdf.set_index('numeric_id')

# Create a dictionary to translate our fuzzy names into these safe numeric IDs
name_to_id = dict(zip(gdf['Simplified_Name'], gdf.index))


# ==========================================
# 4. PREPARE THE TIMELINE
# ==========================================
# Ensure no NaN math values crash the color scale
ts_data['E_Prime_Monthly_Snapshot'] = ts_data['E_Prime_Monthly_Snapshot'].fillna(0)

# Calculate the exact same limit and bins as the static map
max_val = max(abs(ts_data['E_Prime_Monthly_Snapshot'].min()), abs(ts_data['E_Prime_Monthly_Snapshot'].max()))
limit = max_val + 1
custom_bins = [-limit, -limit*0.66, -limit*0.33, 0, limit*0.33, limit*0.66, limit]

# Use a StepColormap with the exact Hex codes of Folium's "RdYlGn_r"
cmap = cm.StepColormap(
    colors=['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027'], 
    vmin=-limit, 
    vmax=limit,
    index=custom_bins,
    caption="Crime Growth Rate (E'_i) [Green = Decrease, Red = Increase]"
)

style_dict = {}
missing_regions = []

for _, row in ts_data.iterrows():
    sim_name = row['Simplified_Name']
    
    # Only process if we found a matching numeric ID
    if sim_name in name_to_id:
        region_id = name_to_id[sim_name]
        
        if region_id not in style_dict:
            style_dict[region_id] = {}
            
        time_sec = pd.to_datetime(f"{int(row['Year'])}-01-01").timestamp()
        hex_color = cmap(row['E_Prime_Monthly_Snapshot'])
        
        # Keep it simple for the animation plugin
        style_dict[region_id][str(int(time_sec))] = {
            'color': hex_color, 
            'opacity': 0.8
        }
    else:
        if row['PFA_Name'] not in missing_regions:
            missing_regions.append(row['PFA_Name'])

# Print a diagnostic report
if missing_regions:
    print("\n⚠️ WARNING: The following regions failed to match:")
    for r in missing_regions: print(f" - {r}")
else:
    print("\n✅ Perfect Match! All regions successfully linked to numeric IDs.")

# ==========================================
# 5. RENDER THE ANIMATED MAP
# ==========================================
uk_map = folium.Map(location=[54.5, -3.0], zoom_start=6, tiles="cartodb positron")

# Layer 1: The dynamic animated colors
TimeSliderChoropleth(
    data=gdf.to_json(),
    styledict=style_dict,
).add_to(uk_map)

# Layer 2: The crisp black borders (completely transparent inside)
folium.GeoJson(
    gdf,
    style_function=lambda feature: {
        'color': '#000000',   # Black borders
        'weight': 1,          # Border thickness
        'fillOpacity': 0      # 100% transparent fill so the animation shows through!
    },
    name="PFA Boundaries"
).add_to(uk_map)

# Add Legend
uk_map.add_child(cmap)

uk_map.save('animated_timeline_2021_2025.html')
print("\nSaved bulletproof animated map to animated_timeline_2021_2025.html")