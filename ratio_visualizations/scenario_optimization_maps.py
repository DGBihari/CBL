import geopandas as gpd
import pandas as pd
import folium
import numpy as np
import warnings

warnings.filterwarnings('ignore')
print("Generating Scenario Optimization Leverage Map...")

# load data
police_areas = gpd.read_file('../police_areas.geojson')
ts_data = pd.read_csv('../time_series_master_goldilocks.csv')

police_areas['PFA24NM'] = police_areas['PFA24NM'].astype(str).str.strip()
police_areas.loc[police_areas['PFA24NM'].str.contains('Devon', case=False, na=False), 'PFA24NM'] = 'Devon and Cornwall'
police_areas.loc[police_areas['PFA24NM'].str.contains('Hampshire', case=False, na=False), 'PFA24NM'] = 'Hampshire and Isle of Wight'

# calc optimization leverage
# Extract latest year for baseline
opt_data = ts_data[ts_data['Year'] == 2025].copy()
opt_data['PFA_Name'] = opt_data['PFA_Name'].astype(str).str.strip()

# calc leverage, how much does 1 extra officer reduce crime?
# Derivative of (-Beta * E * P^-0.3) with respect to P, diminishing returns on police count
opt_data['Optimization_Leverage'] = opt_data['Beta_i'] * opt_data['Crime_Count'] * 0.3 * (opt_data['Police_Count'] ** -1.3)

# clean data & apply overrides
# Fix City of London Map Hole
met_data = opt_data[opt_data['PFA_Name'] == 'Metropolitan Police'].copy()
met_data['PFA_Name'] = 'London, City of'
opt_data = pd.concat([opt_data, met_data], ignore_index=True)

# black out Greater Manchester
opt_data = opt_data[opt_data['PFA_Name'] != 'Greater Manchester']
opt_data = opt_data.drop_duplicates(subset=['PFA_Name'])

# render
uk_map = folium.Map(location=[54.5, -3.0], zoom_start=6, tiles="cartodb positron")

folium.Choropleth(
    geo_data=police_areas,
    name="Intervention Leverage",
    data=opt_data,
    columns=["PFA_Name", "Optimization_Leverage"],
    key_on="feature.properties.PFA24NM",
    fill_color="YlGnBu",     
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="Resource Optimization Priority (Dark Blue = Highest Leverage)",
    nan_fill_color="#000000"
).add_to(uk_map)

uk_map.save('scenario_optimization_map.html')
print("Map successfully generated!")