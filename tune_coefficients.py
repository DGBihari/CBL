"""
Interactive coefficient tuner for the Goldilocks SDE model.

Workflow:
  1. Run run_sde_pipeline.py at least once to generate pipeline_cache.npz
  2. Run this file: python tune_coefficients.py
  3. Click any of the 43 PFA plots to select it
  4. Use the controls in the second window to adjust alpha / B / beta
  5. Click Apply — the selected plot updates immediately
  6. Click Save JSON — writes coeff_adjustments.json
  7. Re-run run_sde_pipeline.py to get the final plot with your adjustments applied

ODE constants below must match run_sde_pipeline.py.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ── Must match run_sde_pipeline.py ───────────────────────────────────────────
DRIFT_SCALE    = 3.0
SEASONAL_SCALE = 1.7
PHASE_SHIFT    = 3.0 * (2 * np.pi / 12)

CACHE_FILE       = 'pipeline_cache.npz'
ADJUSTMENTS_FILE = 'coeff_adjustments.json'

# ── Load cache ────────────────────────────────────────────────────────────────
if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(
        f"'{CACHE_FILE}' not found — run run_sde_pipeline.py first."
    )

cache        = np.load(CACHE_FILE, allow_pickle=True)
alpha_fit    = cache['alpha_fit']
B_fit        = cache['B_fit']
beta_fit     = cache['beta_fit']
E_data       = cache['E_data']
P_data       = cache['P_data']
ADJ          = cache['ADJ']
sigma_vector = cache['sigma_vector']
master_pfas  = list(cache['master_pfas'])
months       = list(cache['months'])

n_pfas  = len(master_pfas)
t_idx   = np.arange(len(months))
t_span  = (t_idx[0], t_idx[-1])
t_years = np.array([int(m[:4]) + (int(m[5:7]) - 1) / 12.0 for m in months])

P_interp = [
    interp1d(t_idx, P_data[:, j], kind='linear', fill_value='extrapolate')
    for j in range(n_pfas)
]

# ── Adjustments ───────────────────────────────────────────────────────────────
def _default():
    return {'alpha_add': 0.0, 'alpha_mul': 1.0,
            'B_add':     0.0, 'B_mul':     1.0,
            'beta_add':  0.0, 'beta_mul':  1.0}

def load_adj():
    return json.load(open(ADJUSTMENTS_FILE)) if os.path.exists(ADJUSTMENTS_FILE) else {}

def save_adj(adj):
    with open(ADJUSTMENTS_FILE, 'w') as f:
        json.dump(adj, f, indent=2)
    print(f"Saved {ADJUSTMENTS_FILE}")

def get_entry(adj, pfa):
    return adj.get(pfa, _default())

def apply_entry(base, entry, key):
    return (base + entry[f'{key}_add']) * entry[f'{key}_mul']

def has_changes(entry):
    d = _default()
    return any(entry.get(k, d[k]) != d[k] for k in d)

# ── Deterministic ODE (fast — no noise, no ensemble) ─────────────────────────
def run_ode(adj):
    omega = 2 * np.pi / 12

    def system(t, E_vec):
        dE = np.zeros(n_pfas)
        for i in range(n_pfas):
            e       = get_entry(adj, master_pfas[i])
            alpha_i = apply_entry(alpha_fit[i], e, 'alpha')
            B_i     = apply_entry(B_fit[i],     e, 'B')
            beta_i  = apply_entry(beta_fit[i],  e, 'beta')

            P_i  = max(float(P_interp[i](t)), 1.0)
            growth = alpha_i * E_vec[i]

            nb_term    = 0.0
            neighbours = np.where(ADJ[i])[0]
            if len(neighbours):
                for j in neighbours:
                    ej      = get_entry(adj, master_pfas[j])
                    alpha_j = apply_entry(alpha_fit[j], ej, 'alpha')
                    nb_term += (alpha_j / len(neighbours)) * E_vec[j]

            seasonal    = SEASONAL_SCALE * B_i * np.cos(omega * t - PHASE_SHIFT)
            suppression = min(beta_i, 1.0) * E_vec[i] * (P_i ** -0.3)

            dE[i] = DRIFT_SCALE * (growth + nb_term - suppression + seasonal)
        return dE

    sol = solve_ivp(
        system, t_span, E_data[0].copy(),
        method='RK45', t_eval=t_idx,
        max_step=1.0, rtol=1e-3, atol=1e-1
    )
    return sol.y.T if sol.success else None

# ── App state ─────────────────────────────────────────────────────────────────
adjustments = load_adj()
solution    = run_ode(adjustments)
selected    = [0]

# ── Overview figure: 43 mini-plots ────────────────────────────────────────────
COLS = 7
ROWS = (n_pfas + COLS - 1) // COLS

fig_ov, axes_ov = plt.subplots(ROWS, COLS, figsize=(22, ROWS * 2.8))
fig_ov.suptitle('All PFA Trajectories  —  click a plot to select it', fontsize=12)
flat = axes_ov.flatten()

def draw_mini(i, highlight=False):
    ax = flat[i]
    ax.clear()
    ax.plot(t_years, E_data[:, i], color='steelblue', lw=0.9, label='Empirical')
    if solution is not None:
        ax.plot(t_years, solution[:, i], color='tomato', lw=0.9, ls='--', label='Model')
    ax.set_title(master_pfas[i], fontsize=5.5, pad=1)
    ax.tick_params(labelsize=4.5)
    ax.set_facecolor('#fff9e6' if has_changes(get_entry(adjustments, master_pfas[i])) else 'white')
    for sp in ax.spines.values():
        sp.set_edgecolor('gold' if highlight else '#cccccc')
        sp.set_linewidth(2.5 if highlight else 0.5)

def refresh_ov():
    for i in range(n_pfas):
        draw_mini(i, highlight=(i == selected[0]))
    for j in range(n_pfas, len(flat)):
        flat[j].set_visible(False)
    fig_ov.tight_layout(rect=[0, 0, 1, 0.97])
    fig_ov.canvas.draw_idle()

# ── Detail + controls figure ──────────────────────────────────────────────────
fig_dt = plt.figure(figsize=(11, 7))
fig_dt.suptitle('Coefficient Tuner', fontsize=12)

ax_plot = fig_dt.add_axes([0.07, 0.42, 0.89, 0.50])

# --- Coefficient selector ---
ax_rc = fig_dt.add_axes([0.05, 0.04, 0.13, 0.30])
radio_coeff = RadioButtons(ax_rc, ['alpha', 'B', 'beta'], active=0)
ax_rc.set_title('Coefficient', fontsize=8)

# --- Operation selector ---
ax_ro = fig_dt.add_axes([0.22, 0.04, 0.10, 0.30])
radio_op = RadioButtons(ax_ro, ['+', '−', '×', '÷'], active=0)
ax_ro.set_title('Operation', fontsize=8)

# --- Value input ---
ax_tb = fig_dt.add_axes([0.38, 0.20, 0.22, 0.07])
textbox = TextBox(ax_tb, 'Value ', initial='0.0')

# --- Buttons ---
ax_ba = fig_dt.add_axes([0.38, 0.07, 0.10, 0.09])
btn_apply = Button(ax_ba, 'Apply')

ax_br = fig_dt.add_axes([0.51, 0.07, 0.12, 0.09])
btn_reset = Button(ax_br, 'Reset PFA')

ax_bs = fig_dt.add_axes([0.66, 0.07, 0.12, 0.09])
btn_save = Button(ax_bs, 'Save JSON')

# --- Current adjustments readout ---
info_text = fig_dt.text(0.05, 0.385, '', fontsize=7.5, color='#444',
                        transform=fig_dt.transFigure)

def refresh_dt():
    i  = selected[0]
    ax_plot.clear()
    ax_plot.plot(t_years, E_data[:, i], color='steelblue', lw=1.5, label='Empirical')
    if solution is not None:
        ax_plot.plot(t_years, solution[:, i], color='tomato', lw=1.5, ls='--', label='Model')
    ax_plot.set_title(master_pfas[i], fontsize=10)
    ax_plot.set_xlabel('Year', fontsize=8)
    ax_plot.set_ylabel('Crime Count', fontsize=8)
    ax_plot.legend(fontsize=8)

    e = get_entry(adjustments, master_pfas[i])
    info_text.set_text(
        f"Current adjustments —  "
        f"alpha: add={e['alpha_add']:+.5f}  ×{e['alpha_mul']:.3f}    "
        f"B: add={e['B_add']:+.5f}  ×{e['B_mul']:.3f}    "
        f"beta: add={e['beta_add']:+.5f}  ×{e['beta_mul']:.3f}"
    )
    fig_dt.canvas.draw_idle()

# ── Callbacks ─────────────────────────────────────────────────────────────────
def on_click_ov(event):
    for i, ax in enumerate(flat[:n_pfas]):
        if event.inaxes is ax:
            selected[0] = i
            refresh_ov()
            refresh_dt()
            break

def on_apply(_):
    global solution
    pfa   = master_pfas[selected[0]]
    coeff = radio_coeff.value_selected
    op    = radio_op.value_selected
    try:
        val = float(textbox.text)
    except ValueError:
        return

    entry = dict(get_entry(adjustments, pfa))
    if   op == '+':               entry[f'{coeff}_add'] += val
    elif op == '−':               entry[f'{coeff}_add'] -= val
    elif op == '×':               entry[f'{coeff}_mul'] *= val
    elif op == '÷' and val != 0:  entry[f'{coeff}_mul'] /= val

    adjustments[pfa] = entry
    print(f"Updating ODE for {pfa}…", end=' ', flush=True)
    solution = run_ode(adjustments)
    print("done")
    refresh_dt()
    refresh_ov()

def on_reset(_):
    global solution
    pfa = master_pfas[selected[0]]
    adjustments.pop(pfa, None)
    print(f"Reset {pfa}, updating ODE…", end=' ', flush=True)
    solution = run_ode(adjustments)
    print("done")
    refresh_dt()
    refresh_ov()

def on_save(_):
    save_adj(adjustments)

fig_ov.canvas.mpl_connect('button_press_event', on_click_ov)
btn_apply.on_clicked(on_apply)
btn_reset.on_clicked(on_reset)
btn_save.on_clicked(on_save)

refresh_ov()
refresh_dt()
plt.show()
