import matplotlib.pyplot as plt
import numpy as np

time_steps = 30
time_axis = np.arange(time_steps)

# Mock Data Generation for total UK Crime Count over 30 days
rng = np.random.default_rng(10)
# Simulation 1: Baseline
baseline_crimes = np.zeros(time_steps)
baseline_crimes[0] = 50000
for t in range(1, time_steps):
    baseline_crimes[t] = baseline_crimes[t-1] + rng.normal(500, 2000)

# Simulation 2: Policy implemented (e.g., increased catching efficiency beta)
# Assuming a 60% impact as mentioned in your documents
policy_crimes = np.zeros(time_steps)
policy_crimes[0] = 50000
for t in range(1, time_steps):
    # Policy reduces the positive drift
    policy_crimes[t] = policy_crimes[t-1] + rng.normal(-200, 1500)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(time_axis, baseline_crimes, color='crimson', linewidth=2, linestyle='--', label='Simulation 1: Baseline')
plt.plot(time_axis, policy_crimes, color='teal', linewidth=2, label='Simulation 2: Targeted Policy')

# Annotations to highlight the 60% assumption
plt.annotate('Policy impact starts here', xy=(0, 50000), xytext=(5, 52000),
             arrowprops={'facecolor': 'black', 'shrink': 0.05})

plt.title('Comparative Policy Simulation: Total Expected UK Crimes')
plt.xlabel('Days (t)')
plt.ylabel('Total Crimes Committed')
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig('comparative_policy.png')
plt.show()