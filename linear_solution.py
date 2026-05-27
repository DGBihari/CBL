import matplotlib.pyplot as plt

# B_ji vs alpha_j? pairwise? neighbor_influence = B @ xE
# currently it's biased, and + terms dominate so criminal pop is increasing each month, change police multiplier if you want it to decrease
"""

    dE_i/dt = alpha_i * E_i
              + sum_{j bordering i} alpha_j * E_j
              - k_i * E_i * P_i(t) / (N_i(t) + P_i(t))

Heat for the heatmap is defined as dE_i/dt reshaped to (ROWS, COLS):
    positive  → crime rising   (red)
    negative  → crime falling  (blue)
    near-zero → stable         (white/neutral)
"""

import numpy as np

# Grid for now
ROWS, COLS = 5, 9
N = ROWS * COLS
dt = 0.1  # time step ( frac of months) ~ 3 days,

positions = [(i // COLS, i % COLS) for i in range(N)]

# ── Adjacency: bordering = Manhattan distance 1 -> determine adj matrix
ADJ = np.zeros((N, N), dtype=bool)
for i in range(N):
    r1, c1 = positions[i]
    for j in range(N):
        r2, c2 = positions[j]
        if i != j and abs(r1 - r2) + abs(c1 - c2) == 1:
            ADJ[i, j] = True

# ── Parameters per region ─────────────────────────────────────────────────────
rng = np.random.default_rng(42)

alpha_base = rng.uniform(0.005, 0.04, N)   # local contagion, bounds 0.001–0.5
k_base     = rng.uniform(0.3,   0.8,  N)   # police catching efficiency

# find these from lin regr etc
N_civ0 = rng.uniform(50_000, 500_000, N) # initial civilian pop function -> this is currently linear in time not exp
P_pol0 = rng.uniform(500,    5_000,   N)
N_rate = rng.uniform(0.001,  0.003,   N)   # civilian monthly growth rate
P_rate = rng.uniform(0.002,  0.005,   N)   # police monthly growth rate

# civilian pop in region i at time t
def get_ni(t: float) -> np.ndarray:
    return N_civ0 * (1.0 + N_rate * t)

# police pop in each region i at time t, scaled by police_mult 
def get_pi(t: float, police_mult: float = 1.0) -> np.ndarray:
    return P_pol0 * (1.0 + P_rate * t) * police_mult

# ── ODE step ──────────────────────────────────────────────────────────────────
def ode_step(
    x_e: np.ndarray,
    t: float,
    alpha: np.ndarray = alpha_base,
    k: np.ndarray     = k_base,
    police_mult: float = 1.0, # slider idea to incr police and test?, can be removed in the future if we just want to test the base case
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One Euler step of the child model.

    Parameters
    
    x_e         : (N,) current monthly crime counts E_i
    t           : current time in months
    alpha       : (N,) contagion rates (pass alpha_base * slider_val from UI) -> change UI values later or idk if we even need this slider
    k           : (N,) police efficiency (pass k_base * slider_val from UI)
    police_mult : scalar multiplier on P_i (police allocation slider)

    Ret
    
    new_e : (N,)      updated crime counts after one dt step
    d_e   : (N,)      raw derivative dE_i/dt  — use this for the heatmap
    heat : (ROWS, COLS)  d_e reshaped; positive = rising crime, negative = falling
    """
    n_i = get_ni(t)
    p_i = get_pi(t, police_mult)

    neighbor_influence = ADJ @ (alpha * x_e)   # sum alpha_j*E_j over bordering j; adj masking
    police_ratio       = p_i / (n_i + p_i)        # in (0,1), diminishing returns

    d_e   = alpha * x_e + neighbor_influence - k * x_e * police_ratio
    new_e = np.maximum(x_e + d_e * dt, 0.0)
    heat = d_e.reshape(ROWS, COLS)             # hand this to the heatmap renderer

    return new_e, d_e, heat


# eg
if __name__ == "__main__":
    x_e = rng.uniform(50, 200, N)
    t  = 0.0

    for step in range(10):
        x_e, d_e, heat = ode_step(x_e, t)
        t += dt
        print(f"t={t:.1f}  E_mean={x_e.mean():.2f}  dE_mean={d_e.mean():.4f}") # x_e[i] current crime count in region i
        # heat is ready to pass straight to ax.imshow() or equivalent

        # civilian
        n_i = get_ni(t)

        # police
        p_i = get_pi(t)

        # compact table per region
        print("\nRegion Stats")

        for i in range(N):

            r, c = positions[i]

            print(
                f"Region ({r},{c}) | "
                f"Crime={xE[i]:7.2f} | "
                f"Population={Ni[i]:10.0f} | "
                f"Police={Pi[i]:7.0f}"
            )

        # optional heatmap-style grids
        crime_grid = xE.reshape(ROWS, COLS)

        print("\nCrime Grid")
        print(np.round(crime_grid, 1))