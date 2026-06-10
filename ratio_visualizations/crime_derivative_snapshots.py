import geopandas as gpd
import folium
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
print("Loading data and cleaning map boundaries...")


police_areas = gpd.read_file('../police_areas.geojson')
ts_data = pd.read_csv('../time_series_master_goldilocks.csv')

police_areas['PFA24NM'] = police_areas['PFA24NM'].astype(str).str.strip()
police_areas.loc[police_areas['PFA24NM'].str.contains('Devon', case=False, na=False), 'PFA24NM'] = 'Devon and Cornwall'
police_areas.loc[police_areas['PFA24NM'].str.contains('Hampshire', case=False, na=False), 'PFA24NM'] = 'Hampshire and Isle of Wight'

# calc global 5 year color scale
ts_data['E_Prime_Monthly_Snapshot'] = ts_data['E_Prime_Monthly_Snapshot'].fillna(0)

global_max = max(abs(ts_data['E_Prime_Monthly_Snapshot'].min()), abs(ts_data['E_Prime_Monthly_Snapshot'].max()))
limit = global_max + 1
custom_bins = [-limit, -limit*0.66, -limit*0.33, 0, limit*0.33, limit*0.66, limit]

# prepare 2025 csv data
current_data = ts_data[ts_data['Year'] == 2025].copy()
current_data['PFA_Name'] = current_data['PFA_Name'].astype(str).str.strip()

# fix City of London Map
met_data = current_data[current_data['PFA_Name'] == 'Metropolitan Police'].copy()
met_data['PFA_Name'] = 'London, City of'
current_data = pd.concat([current_data, met_data], ignore_index=True)

# Force Greater Manchester to black
current_data = current_data[current_data['PFA_Name'] != 'Greater Manchester']
current_data = current_data.drop_duplicates(subset=['PFA_Name']) 

# render map
uk_map = folium.Map(location=[54.5, -3.0], zoom_start=6, tiles="cartodb positron")

folium.Choropleth(
    geo_data=police_areas,
    name="2025 Real Derivative Map",
    data=current_data,
    columns=["PFA_Name", "E_Prime_Monthly_Snapshot"],
    key_on="feature.properties.PFA24NM",
    fill_color="RdBu_r",
    bins=custom_bins,
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="2025 Crime Growth Rate (E'_i)",
    nan_fill_color="#000000" 
).add_to(uk_map)

uk_map.save('real_crime_derivative_map_2025.html')
print("Map generated! Open real_crime_derivative_map_2025.html to view.")