import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Right now this is just a mockup to show how we can compare two different policy simulations on the same graph. The data is randomly generated to illustrate the concept, but in practice, you would replace the random generation with actual outputs from your SDE simulations under different policy scenarios.
#We first need to make up the policies 

ts_data = pd.read_csv('../time_series_master_calculated.csv')

# Pick three highly different regions to stress test from 2025
regions_to_test = ['Metropolitan Police', 'Avon and Somerset', 'Cumbria']
base_2025 = ts_data[(ts_data['Year'] == 2025) & (ts_data['PFA_Name'].isin(regions_to_test))]

def simulate_derivative(row, officer_adjustment):
    dt, sigma, k_i = 1.0, 0.0, 0.02 # Set sigma to 0 to see pure policy effect
    alpha_i = row['IMD_Score'] * 0.005
    C_i = row['Crime_Count'] / row['Area_Sq_Km']
    P_i = (row['Police_Count'] + officer_adjustment) / row['Area_Sq_Km']
    N_i = row['Population'] / row['Area_Sq_Km']
    
    if P_i < 0: P_i = 0.001
    return (alpha_i * C_i) - (k_i * C_i * (P_i / (N_i + P_i)))

adjustments = np.linspace(-1000, 2000, 50)
plt.figure(figsize=(10, 6))

for _, row in base_2025.iterrows():
    derivatives = [simulate_derivative(row, adj) for adj in adjustments]
    plt.plot(adjustments, derivatives, label=row['PFA_Name'], linewidth=2)

plt.axhline(0, color='red', linestyle='--', label='Zero Growth Threshold')
plt.title("SDE Policy Simulation: Impact of Officer Adjustments (2025)")
plt.xlabel("Change in Number of Police Officers")
plt.ylabel("Resulting Crime Growth Rate (C'_i)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparative_policy_simulations_2025.png')
print("Saved Simulation Graph to comparative_policy_simulations_2025.png")