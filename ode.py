import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

ROWS = 5
COLS = 9
N = ROWS * COLS # 45 ffields in grid

# alpha_i - hdi and crime rate correlation factor
alpha = HDI.copy()

# beta_i, np array size of N with 0.0002 value
beta = np.full(N, 0.002)

# To do: pass heat to update the graph

rng = np.random.default_rng(42)

HDI = rng.random(N)


# delta_i
delta = np.full(N, 0.03)
# Smaller timestep for stability
#DT = 1e-5

#noise_scale = 1.0 # X_i randomness


dt = 0.01

sigma = 0.00


positions = []

# make grid 5x9
for i in range(N):

    r = i // COLS
    c = i % COLS

    positions.append((r, c))

def ode_step(xC, xP):

    dC = np.zeros(N)

    dP = np.zeros(N)

    # RANDOM IMPULSE DRIVEN CRIME TERM - X_i, we just assign random for now -> replaced wotj randomness_term

    #X = rng.normal(
    #    0,
    #    noise_scale,
    #    N
    #)

    # MAIN LOOP

    for i in range(N):
        # make sure coefficients regional

        randomness_term = sigma * np.sqrt(dt) * np.random.normal(0, 1) # regenerated in every step

        Ri = 0.0 # police recruitment in region i

        criminal_in = 0.0

        criminal_out = 0.0

        police_in = 0.0

        police_out = 0.0

        for j in range(N):

            if i == j:
                continue

            # gamma_i,j
            # gamma_ij = gamma_base / D[i, j]  # e.g. gamma_base = 0.001
            # theta_ij = theta_base / D[i, j]
            gamma_ij = (
                xP[i] - xP[j]
            ) / D[i, j]

            gamma_ji = (
                xP[j] - xP[i]
            ) / D[j, i]

            # theta_i,j

            theta_ij = (
                xC[i] - xC[j]
            ) / D[i, j]

            theta_ji = (
                xC[j] - xC[i]
            ) / D[j, i]

            # coefficients from doc main equations, already treated as sums

            criminal_in += (
                gamma_ji * xC[j]
            )

            criminal_out += (
                gamma_ij * xC[i]
            )

            police_in += (
                theta_ij * xP[j]
            )

            police_out += (
                theta_ji * xP[i]
            )

        # x'C,i - criminals equation

        dC[i] = (

            (alpha[i] * xC[i]

            - beta[i] * xC[i] * xP[i]

            + criminal_in

            - criminal_out) * dt 
            +  randomness_term
        )

        # x'P,i - police equation

        dP[i] = (
            
            Ri

            - delta[i] * xP[i]

            + beta[i] * xC[i] * xP[i]

            + police_in

            - police_out
        )

    # EULER INTEGRATION, small step DT -> next sec in time kinda

    newC = xC + dC

    newP = xP + dt*dP

    # non-neg densities -> can't have -10 criminals for example

    newC = np.maximum(newC, 0)

    newP = np.maximum(newP, 0)

    heat = np.where(dP != 0, dC / dP, 0).reshape(ROWS, COLS) # dx Ci / dx Pi 
    # determine heat based on this in the future

    return newC, newP, heat # new equations for criminal and police


# distance matrix/vector - d(i,j)
D = np.zeros((N, N))

for i in range(N):

    for j in range(N):

        if i == j:

            D[i, j] = 1.0

        else:

            r1, c1 = positions[i]
            r2, c2 = positions[j]
            # Euclidian distance
            D[i, j] = np.sqrt(
                (r1 - r2) ** 2 +
                (c1 - c2) ** 2
            )