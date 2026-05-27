import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Setup Simulation Parameters
# ==========================================
num_simulations = 100
time_steps = 30
dt = 0.01

# Assuming we are looking at Region 1
initial_crime_density = 50.0
initial_police_density = 10.0
initial_ratio = initial_crime_density / initial_police_density

# SDE parameters (Mocked)
alpha, beta, kappa, sigma = 0.1, 0.02, 0.05, 1.5

# Store ratios for all simulations
all_ratios = np.zeros((num_simulations, time_steps))
rng = np.random.default_rng(42)

# ==========================================
# 2. Run Euler-Maruyama Monte Carlo SDE
# ==========================================
for sim in range(num_simulations):
    x_c = initial_crime_density
    x_p = initial_police_density
    
    ratios = [x_c / x_p]
    
    for t in range(1, time_steps):
        dW = rng.normal(0, np.sqrt(dt)) # Randomness
        
        # Simplified Mock SDE updates
        dx_c = (alpha * x_c - beta * x_c * x_p) * dt + sigma * x_c * dW
        dx_p = (-kappa * x_p + beta * x_c * x_p) * dt 
        
        x_c += dx_c
        x_p += dx_p
        
        ratios.append(max(x_c / max(x_p, 1e-5), 0)) # Avoid division by zero
        
    all_ratios[sim, :] = ratios

# ==========================================
# 3. Plotting the Ensemble
# ==========================================
time_axis = np.arange(time_steps)
mean_ratio = np.mean(all_ratios, axis=0)
percentile_85 = np.percentile(all_ratios, 85, axis=0)
percentile_15 = np.percentile(all_ratios, 15, axis=0) # Captures middle 70%

plt.figure(figsize=(10, 6))

# Plot all paths lightly
for sim in range(num_simulations):
    plt.plot(time_axis, all_ratios[sim], color='gray', alpha=0.1)

# Plot Mean and 70% Confidence Interval
plt.plot(time_axis, mean_ratio, color='red', linewidth=2, label='Mean Ratio')
plt.fill_between(time_axis, percentile_15, percentile_85, color='red', alpha=0.3, label='70% Probability Band')

plt.title('Monte Carlo Ensemble: Crime-to-Police Ratio (Region 1)')
plt.xlabel('Days (t)')
plt.ylabel('Ratio Density')
plt.legend()
plt.grid(True)
plt.savefig('monte_carlo_ensemble.png')
plt.show()