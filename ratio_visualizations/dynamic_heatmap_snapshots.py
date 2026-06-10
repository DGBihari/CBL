import geopandas as gpd
import pandas as pd
import folium
import branca.colormap as cm
from folium.plugins import TimeSliderChoropleth
import warnings

warnings.filterwarnings('ignore')

# load data
gdf = gpd.read_file('../police_areas.geojson')
ts_data = pd.read_csv('../time_series_master_goldilocks.csv')

# clean and standardize names for mapping
gdf['PFA24NM'] = gdf['PFA24NM'].astype(str).str.strip()
gdf.loc[gdf['PFA24NM'].str.contains('Devon', case=False, na=False), 'PFA24NM'] = 'Devon and Cornwall'
gdf.loc[gdf['PFA24NM'].str.contains('Hampshire', case=False, na=False), 'PFA24NM'] = 'Hampshire and Isle of Wight'

# add simplified names for guaranteed mapping
gdf['Simplified_Name'] = gdf['PFA24NM'].str.lower().str.replace('[^a-z]', '', regex=True)
ts_data['Simplified_Name'] = ts_data['PFA_Name'].str.lower().str.replace('[^a-z]', '', regex=True)

# convert map index to string IDs
gdf['id'] = gdf.index.astype(str)
name_to_id = dict(zip(gdf['Simplified_Name'], gdf['id']))

# prepare timeline and scales
ts_data['E_Prime_Monthly_Snapshot'] = ts_data['E_Prime_Monthly_Snapshot'].fillna(0)

# calculate global limits across all 5 years
max_val = max(abs(ts_data['E_Prime_Monthly_Snapshot'].min()), abs(ts_data['E_Prime_Monthly_Snapshot'].max()))
limit = max_val + 1
custom_bins = [-limit, -limit*0.66, -limit*0.33, 0, limit*0.33, limit*0.66, limit]

cmap = cm.StepColormap(
    colors=['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027'], 
    vmin=-limit, 
    vmax=limit,
    index=custom_bins,
    caption="Crime Growth Rate (E'_i) [Blue = Decrease, Orange/Red = Increase]"
)

# create style dict for TimeSliderChoropleth
style_dict = {}
for _, row in ts_data.iterrows():
    sim_name = row['Simplified_Name']
    
    if sim_name in name_to_id:
        region_id = name_to_id[sim_name]
        
        if region_id not in style_dict:
            style_dict[region_id] = {}
            
        time_sec = pd.to_datetime(f"{int(row['Year'])}-01-01").timestamp()
        
        # color manchester with black due to insufficient data
        if sim_name == 'greatermanchester':
            hex_color = "#535353"
        else:
            hex_color = cmap(row['E_Prime_Monthly_Snapshot'])
        
        style_dict[region_id][str(int(time_sec))] = {
            'color': hex_color, 
            'opacity': 0.8
        }

# render map
uk_map = folium.Map(location=[54.5, -3.0], zoom_start=6, tiles="cartodb positron")

TimeSliderChoropleth(
    data=gdf.to_json(),
    styledict=style_dict,
).add_to(uk_map)

folium.GeoJson(
    gdf,
    style_function=lambda feature: {
        'color': '#888888',   
        'weight': 0.8,        
        'opacity': 0.5,       
        'fillOpacity': 0      
    },
    name="PFA Boundaries"
).add_to(uk_map)

uk_map.add_child(cmap)
uk_map.save('animated_timeline_2021_2025.html')
