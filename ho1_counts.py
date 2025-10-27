from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

K = 4.0
m = 2.0
omega = np.sqrt(K/m)

posicao_inicial = 1.0
velocidade_inicial = 0.0

shots = 10_000




def harmonic_oscillator_sim(t):
    entrada1_matrix = velocidade_inicial * np.sqrt(m)
    entrada2_matrix = posicao_inicial * 1j * np.sqrt(K)
    cond_iniciais = np.array([entrada1_matrix, entrada2_matrix], dtype=complex)

    norm_ci = np.linalg.norm(cond_iniciais)
    cond_iniciais_norm = cond_iniciais / norm_ci

    qc = QuantumCircuit(1)
    qc.initialize(cond_iniciais_norm, 0)
    qc.rx(-2 * omega * t, 0)
    qc.measure_all()

    backend = Aer.get_backend("aer_simulator")
    qc_t = transpile(qc, backend)
    result = backend.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    # Safe access (if a key is missing, its probability is 0)
    p0 = counts.get('0', 0) / shots
    p1 = counts.get('1', 0) / shots
    a0 = np.sqrt(p0)
    a1 = np.sqrt(p1)



    velocidade = (a0 * norm_ci) / np.sqrt(m)
    posicao    = (a1 * norm_ci) / np.sqrt(K)
        

    """
    print("Counts:", counts)
    print("Probabilities:", probs)
    print("Abslute values: ", abs_values)
    print("Velocidade:", velocidade)
    print("Posição:", posicao)
    """
    return velocidade, posicao














# Time grid
dt = 0.05
t_grid = np.arange(0.0, 10.0 + 1e-9, dt)  # [0, 10] inclusive

# Collect data
v_series = []
x_series = []
for t in t_grid:
    v, x = harmonic_oscillator_sim(t)
    v_series.append(v)
    x_series.append(x)

v_series = np.array(v_series)
x_series = np.array(x_series)

# --------- MAKE POSITION SIGNED VIA ZERO-CROSSING DETECTION ----------
# We only observe |x| from the quantum readout. |x| has local minima at the true zero crossings.
# Detect local minima near zero and flip the running sign there so x alternates between + and -.
n = len(x_series)
thresh = 0.1 * np.max(x_series)  # "near zero" threshold (10% of peak)

# candidate minima where |x| is small and locally minimal
candidates = []
for i in range(1, n - 1):
    if x_series[i] < x_series[i - 1] and x_series[i] <= x_series[i + 1] and x_series[i] <= thresh:
        candidates.append(i)

# filter out minima that are too close to each other (robustness to noise)
min_sep = max(1, int(0.05 / dt))  # at least 0.05s apart
flip_idxs = []
last = -10**9
for idx in candidates:
    if idx - last >= min_sep:
        flip_idxs.append(idx)
        last = idx

# build running sign that flips at each detected zero crossing
sign_x = np.ones(n)
cur = 1.0  # start positive (x(0) ≈ +1 by your initial state)
j = 0
next_flip = flip_idxs[j] if flip_idxs else None
for i in range(n):
    sign_x[i] = cur
    if next_flip is not None and i == next_flip:
        cur *= -1.0
        j += 1
        next_flip = flip_idxs[j] if j < len(flip_idxs) else None

x_signed = sign_x * x_series

# --------- SIGN VELOCITY TO MATCH dx/dt OF SIGNED POSITION ----------
dx_dt = np.gradient(x_signed, dt)
sign_v = np.sign(dx_dt)
# replace zeros (flat points) with previous non-zero sign (default +1 at start)
for i in range(n):
    if sign_v[i] == 0:
        sign_v[i] = sign_v[i - 1] if i > 0 else 1.0

v_signed = np.abs(v_series) * sign_v

# (Optional) save results
# import pandas as pd
# pd.DataFrame({'t': t_grid, 'position': x_signed, 'velocity': v_signed}).to_csv('sho_results_signed.csv', index=False)

# --------- ANIMATE ----------
fig, ax = plt.subplots()
ax.set_title("Harmonic Oscillator: signed position & velocity vs time")
ax.set_xlabel("time")
ax.set_ylabel("value (arb. units)")
ax.set_xlim(t_grid[0], t_grid[-1])

ymin = min(np.min(v_signed), np.min(x_signed)) * 1.1
ymax = max(np.max(v_signed), np.max(x_signed)) * 1.1
if ymin == ymax:  # fallback if flat
    ymin, ymax = -1, 1
ax.set_ylim(ymin, ymax)

(line_v,) = ax.plot([], [], label="velocity (signed)")
(line_x,) = ax.plot([], [], label="position (signed)")
ax.legend(loc="upper right")

def init():
    line_v.set_data([], [])
    line_x.set_data([], [])
    return (line_v, line_x)

def update(i):
    line_v.set_data(t_grid[:i+1], v_signed[:i+1])
    line_x.set_data(t_grid[:i+1], x_signed[:i+1])
    return (line_v, line_x)

# Stop when reaching the last frame (no looping)
ani = FuncAnimation(
    fig, update, frames=len(t_grid), init_func=init, interval=50, blit=True, repeat=False
)






# --------- MAKE POSITION SIGNED USING CLASSICAL PHASE (ensures correct ICs) ----------
K_ref = K
m_ref = m
omega_ref = np.sqrt(K_ref / m_ref)
x0 = posicao_inicial
v0 = velocidade_inicial

# Phase and amplitude from ICs:
phi_ref = np.arctan2(-(v0 / omega_ref), x0)   # works for any (x0, v0)
A_ref   = np.hypot(x0, v0 / omega_ref)        # sqrt(x0**2 + (v0/omega)**2)

# Sign model from classical cosine phase
sign_x = np.sign(np.cos(omega_ref * t_grid + phi_ref))
# Replace exact zeros with previous sign for stability
for i in range(len(sign_x)):
    if sign_x[i] == 0:
        sign_x[i] = sign_x[i-1] if i > 0 else 1.0

x_signed = x_series * sign_x

# Velocity sign from the derivative of signed position
dt = t_grid[1] - t_grid[0]
dx_dt = np.gradient(x_signed, dt)
sign_v = np.sign(dx_dt)
for i in range(len(sign_v)):
    if sign_v[i] == 0:
        sign_v[i] = sign_v[i-1] if i > 0 else 1.0

v_signed = np.abs(v_series) * sign_v

# ---- Classical reference curves (same ICs & omega) ----
fx = lambda A, omega, phi, t: A * np.cos(omega * t + phi)
fv = lambda A, omega, phi, t: -omega * A * np.sin(omega * t + phi)

x_classic = fx(A_ref, omega_ref, phi_ref, t_grid)
v_classic = fv(A_ref, omega_ref, phi_ref, t_grid)

# ---- Positions: simulated vs classical ----
fig_pos, ax_pos = plt.subplots()
ax_pos.set_title("Position vs Time: simulated (signed) vs classical")
ax_pos.set_xlabel("time")
ax_pos.set_ylabel("position")
ax_pos.plot(t_grid, x_signed, label="position (simulated, signed)")
ax_pos.plot(t_grid, x_classic, label="position (classical)")
ax_pos.legend(loc="best")

# ---- Velocities: simulated vs classical ----
fig_vel, ax_vel = plt.subplots()
ax_vel.set_title("Velocity vs Time: simulated (signed) vs classical")
ax_vel.set_xlabel("time")
ax_vel.set_ylabel("velocity")
ax_vel.plot(t_grid, v_signed, label="velocity (simulated, signed)")
ax_vel.plot(t_grid, v_classic, label="velocity (classical)")
ax_vel.legend(loc="best")

# ---- Combined plot: everything ----
fig_all, ax_all = plt.subplots()
ax_all.set_title("Signed position & velocity vs classical reference")
ax_all.set_xlabel("time")
ax_all.set_ylabel("value")
ax_all.plot(t_grid, x_signed, label="position (simulated, signed)")
ax_all.plot(t_grid, v_signed, label="velocity (simulated, signed)")
ax_all.plot(t_grid, x_classic, label="position (classical)")
ax_all.plot(t_grid, v_classic, label="velocity (classical)")
ax_all.legend(loc="best")


# ---- Error plots: simulated - classical ----
err_x = x_signed - x_classic
err_v = v_signed - v_classic

rmse_x = np.sqrt(np.mean(err_x**2))
mae_x  = np.mean(np.abs(err_x))

rmse_v = np.sqrt(np.mean(err_v**2))
mae_v  = np.mean(np.abs(err_v))

# Position error
fig_ex, ax_ex = plt.subplots()
ax_ex.set_title("Position error (simulated - classical)")
ax_ex.set_xlabel("time")
ax_ex.set_ylabel("Δx")
ax_ex.plot(t_grid, err_x, label="Δx")
ax_ex.axhline(0, linestyle="--", linewidth=1)
ax_ex.legend(loc="best")
ax_ex.text(0.02, 0.95, f"RMSE={rmse_x:.3g}\nMAE={mae_x:.3g}",
           transform=ax_ex.transAxes, va="top")

# Velocity error
fig_ev, ax_ev = plt.subplots()
ax_ev.set_title("Velocity error (simulated - classical)")
ax_ev.set_xlabel("time")
ax_ev.set_ylabel("Δv")
ax_ev.plot(t_grid, err_v, label="Δv")
ax_ev.axhline(0, linestyle="--", linewidth=1)
ax_ev.legend(loc="best")
ax_ev.text(0.02, 0.95, f"RMSE={rmse_v:.3g}\nMAE={mae_v:.3g}",
           transform=ax_ev.transAxes, va="top")




"""
# ---- Save quantum simulation series to CSV (signed position & velocity) ----
df = pd.DataFrame({
    't': t_grid,
    'position_quantum': x_signed,
    'velocity_quantum': v_signed
})
df.to_csv('quantum_timeseries.csv', index=False)
print("✅ Saved: quantum_timeseries.csv")
"""



plt.show()