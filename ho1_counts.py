from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation





def ho_evolution_operator(t:float, K:float, m:float, x0:float, v0:float, shots:int):
    omega = np.sqrt(K/m)
    entrada1_matrix = v0 * np.sqrt(m)
    entrada2_matrix = x0 * 1j * np.sqrt(K)
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
def ho_qsim(t_grid: np.ndarray, K: float, m: float, x0: float, v0: float, shots: int, dt: float):
    """
    Quantum simulation of a 1D harmonic oscillator using a single qubit approximation.
    Returns signed position and velocity time series.
    """
    omega = np.sqrt(K / m)

    # ---- Collect quantum results ----
    x_series = []
    v_series = []
    for t in t_grid:
        # Use ho_evolution_operator
        v, x = ho_evolution_operator(t, K, m, x0, v0, shots)
        x_series.append(x)
        v_series.append(v)

    x_series = np.array(x_series)
    v_series = np.array(v_series)

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

    x_qsim = sign_x * x_series

# --------- SIGN VELOCITY TO MATCH dx/dt OF SIGNED POSITION ----------
    dx_dt = np.gradient(x_qsim, dt)
    sign_v = np.sign(dx_dt)
# replace zeros (flat points) with previous non-zero sign (default +1 at start)
    for i in range(n):
        if sign_v[i] == 0:
            sign_v[i] = sign_v[i - 1] if i > 0 else 1.0

    v_qsim = np.abs(v_series) * sign_v
    
    return x_qsim, v_qsim
def ho_csim(t_grid:np.ndarray, K:float, m:float, x0:float, v0:float):
    omega_ref = np.sqrt(K / m)
    A_ref   = np.hypot(x0, v0 / omega_ref)        # sqrt(x0**2 + (v0/omega)**2)
    phi_ref = np.arctan2(-(v0 / omega_ref), x0)   # works for any (x0, v0)


    fx = lambda A, omega, phi, t: A * np.cos(omega * t + phi)
    fv = lambda A, omega, phi, t: -omega * A * np.sin(omega * t + phi)

    x_csim = fx(A_ref, omega_ref, phi_ref, t_grid)
    v_csim = fv(A_ref, omega_ref, phi_ref, t_grid)
    return x_csim, v_csim
"""
def do_plots(do_dict: dict, data_dict: dict):
    t_grid = data_dict["t_grid"]

    # ---- Animated position & velocity ----
    fig_anim, ax_anim = plt.subplots()
    ax_anim.set_title("Harmonic Oscillator: signed position & velocity vs time")
    ax_anim.set_xlabel("time")
    ax_anim.set_ylabel("value (arb. units)")
    ax_anim.set_xlim(t_grid[0], t_grid[-1])

    ymin = min(np.min(data_dict["v_qsim"]), np.min(data_dict["x_qsim"])) * 1.1
    ymax = max(np.max(data_dict["v_qsim"]), np.max(data_dict["x_qsim"])) * 1.1
    if ymin == ymax:  # fallback
        ymin, ymax = -1, 1
    ax_anim.set_ylim(ymin, ymax)

    line_v, = ax_anim.plot([], [], label="velocity (signed)")
    line_x, = ax_anim.plot([], [], label="position (signed)")
    ax_anim.legend(loc="upper right")

    def init():
        line_v.set_data([], [])
        line_x.set_data([], [])
        return line_v, line_x

    def update(i):
        line_v.set_data(t_grid[:i+1], data_dict["v_qsim"][:i+1])
        line_x.set_data(t_grid[:i+1], data_dict["x_qsim"][:i+1])
        return line_v, line_x

    ani = FuncAnimation(
        fig_anim, update, frames=len(t_grid),
        init_func=init, interval=50, blit=True, repeat=False
    )

    # ---- Static plots ----
    if do_dict["do_position_plot"]:
        fig_pos, ax_pos = plt.subplots()
        ax_pos.set_title("Position vs Time")
        ax_pos.set_xlabel("time")
        ax_pos.set_ylabel("position")
        ax_pos.plot(t_grid, data_dict["x_qsim"], label="simulated (signed)")
        ax_pos.plot(t_grid, data_dict["x_csim"], label="classical")
        ax_pos.legend(loc="best")

    if do_dict["do_velocity_plot"]:
        fig_vel, ax_vel = plt.subplots()
        ax_vel.set_title("Velocity vs Time")
        ax_vel.set_xlabel("time")
        ax_vel.set_ylabel("velocity")
        ax_vel.plot(t_grid, data_dict["v_qsim"], label="simulated (signed)")
        ax_vel.plot(t_grid, data_dict["v_csim"], label="classical")
        ax_vel.legend(loc="best")

    if do_dict["do_position_velocity_plot"]:
        fig_all, ax_all = plt.subplots()
        ax_all.set_title("Position & Velocity vs Classical")
        ax_all.set_xlabel("time")
        ax_all.set_ylabel("value")
        ax_all.plot(t_grid, data_dict["x_qsim"], label="position (simulated)")
        ax_all.plot(t_grid, data_dict["v_qsim"], label="velocity (simulated)")
        ax_all.plot(t_grid, data_dict["x_csim"], label="position (classical)")
        ax_all.plot(t_grid, data_dict["v_csim"], label="velocity (classical)")
        ax_all.legend(loc="best")

    if do_dict["do_x_abs_err_plot"]:
        fig_ex, ax_ex = plt.subplots()
        ax_ex.set_title("Position error")
        ax_ex.set_xlabel("time")
        ax_ex.set_ylabel("Δx")
        ax_ex.plot(t_grid, data_dict["abs_err_x"], label="Δx")
        ax_ex.axhline(0, linestyle="--", linewidth=1)
        ax_ex.legend(loc="best")
        ax_ex.text(
            0.02, 0.95,
            f"RMSE={data_dict['rmse_x']:.4f}\nMAE={data_dict['mae_x']:.4f}",
            transform=ax_ex.transAxes, va="top"
        )

    if do_dict["do_v_abs_err_plot"]:
        fig_ev, ax_ev = plt.subplots()
        ax_ev.set_title("Velocity error")
        ax_ev.set_xlabel("time")
        ax_ev.set_ylabel("Δv")
        ax_ev.plot(t_grid, data_dict["abs_err_v"], label="Δv")
        ax_ev.axhline(0, linestyle="--", linewidth=1)
        ax_ev.legend(loc="best")
        ax_ev.text(
            0.02, 0.95,
            f"RMSE={data_dict['rmse_v']:.4f}\nMAE={data_dict['mae_v']:.4f}",
            transform=ax_ev.transAxes, va="top"
        )

    plt.show(block=True)
    return ani  # keep animation alive
"""
def do_plots(do_dict: dict, data_dict: dict):
    t_grid = data_dict["t_grid"]

    # ---- Position plot ----
    if do_dict.get("do_position_plot", True):
        plt.figure()
        plt.title("Position vs Time")
        plt.plot(t_grid, data_dict["x_qsim"], label="simulated (signed)")
        plt.plot(t_grid, data_dict["x_csim"], label="classical")
        plt.xlabel("time")
        plt.ylabel("position")
        plt.legend()

    # ---- Velocity plot ----
    if do_dict.get("do_velocity_plot", True):
        plt.figure()
        plt.title("Velocity vs Time")
        plt.plot(t_grid, data_dict["v_qsim"], label="simulated (signed)")
        plt.plot(t_grid, data_dict["v_csim"], label="classical")
        plt.xlabel("time")
        plt.ylabel("velocity")
        plt.legend()

    # ---- Position error plot ----
    if do_dict.get("do_x_abs_err_plot", False):
        plt.figure()
        plt.title("Position error")
        plt.plot(t_grid, data_dict["abs_err_x"], label="Δx")
        plt.axhline(0, linestyle="--")
        plt.xlabel("time")
        plt.ylabel("Δx")
        plt.legend()
        plt.text(0.02, 0.95, f"RMSE={data_dict['rmse_x']:.4f}\nMAE={data_dict['mae_x']:.4f}",
                 transform=plt.gca().transAxes, va="top")

    # ---- Velocity error plot ----
    if do_dict.get("do_v_abs_err_plot", False):
        plt.figure()
        plt.title("Velocity error")
        plt.plot(t_grid, data_dict["abs_err_v"], label="Δv")
        plt.axhline(0, linestyle="--")
        plt.xlabel("time")
        plt.ylabel("Δv")
        plt.legend()
        plt.text(0.02, 0.95, f"RMSE={data_dict['rmse_v']:.4f}\nMAE={data_dict['mae_v']:.4f}",
                 transform=plt.gca().transAxes, va="top")

    plt.show()  # display all static plots

if __name__ == "__main__":
    # Settings
    do_dict:dict = {
        "do_position_plot"  : False,
        "do_velocity_plot" : False,
        "do_position_velocity_plot" : False, 
        "do_x_abs_err_plot" : False,
        "do_v_abs_err_plot" : False
    }
    do_plotting:bool = False
    do_save_data:bool = False
    
    #System Definition
    K = 4.0
    m = 2.0
    omega = np.sqrt(K/m)

    x0 = 1.0
    v0 = 0.0

    t = 1
    dt = 0.05
    t_grid = np.arange(0.0, t + 1e-9, dt)  

    # --------- CLASSIC SIMULATION ----------
    x_csim, v_csim = ho_csim(t_grid, K, m, x0, v0)
    # print("x_csim\n", x_csim); print("v_csim\n", v_csim)
    print("PROGRAM INFO: Concluded Classical Simuation.")

    # --------- QUANTUM HAMILTONIAN SIMULATION --------- 
    x_qsim = []; v_qsim = []
    abs_err_x = []; abs_err_v = []
    rmse_x = []; rmse_v = []; mae_x = []; mae_v = []; max_abs_err_x = []; max_abs_err_v = []
    rmse_rel_x = [];  rmse_rel_v = []; mae_rel_x = []; mae_rel_v = []; max_abs_rel_err_x = []; max_abs_rel_err_v = []
    shots_i = 1000; shots_f = 20000; shots_step=500
    shots_list = range(shots_i, shots_f, shots_step)
    for shots in shots_list:
        x_qsim, v_qsim = ho_qsim(t_grid, K, m, x0, v0, shots, dt)
        print(f"PROGRAM INFO: Concluded Hamiltonian Simuation with {shots} shots.")
        # print("x_qsim\n", x_qsim); print("v_qsim\n", v_qsim)
  
        # --------- CALCULATE ERROR METRICS ----------
        print("PROGRAM INFO: Calculating error metrics.")
        # Absolute Errors
        abs_err_x = x_qsim - x_csim
        abs_err_v = v_qsim - v_csim
        # print("abs_err_x\n", abs_err_x); print("abs_err_v\n", abs_err_v)

        # Root Mean Squared Error
        rmse_x.append(np.sqrt(np.mean(abs_err_x**2)))
        rmse_v.append(np.sqrt(np.mean(abs_err_v**2)))
        print("rmse_x\n", rmse_x[-1]); print("rmse_v\n", rmse_v[-1])

        # Mean Squared Error
        mae_x.append(np.mean(np.abs(abs_err_x)))
        mae_v.append(np.mean(np.abs(abs_err_v)))
        print("mae_x\n", mae_x[-1]); print("mae_v\n", mae_v[-1])
        
        # Maximum Absolute Error
        max_abs_err_x.append(np.max(np.abs(abs_err_x)))
        max_abs_err_v.append(np.max(np.abs(abs_err_v)))
        print("max_abs_err_x\n", max_abs_err_x[-1]); print("max_abs_err_v\n", max_abs_err_v[-1])

        # Relative Errors
        rel_err_x = abs_err_x / x_csim
        rel_err_v = abs_err_v / v_csim

        # Root Mean Squared Relative Error
        rmse_rel_x.append(np.sqrt(np.mean(rel_err_x**2)))
        rmse_rel_v.append(np.sqrt(np.mean(rel_err_v**2)))
        print("rmse_rel_x\n", rmse_rel_x[-1]); print("rmse_rel_v\n", rmse_rel_v[-1])

        # Mean Absolute Relative Error
        mae_rel_x.append(np.mean(np.abs(rel_err_x)))
        mae_rel_v.append(np.mean(np.abs(rel_err_v)))
        print("mae_rel_x\n", mae_rel_x[-1]); print("mae_rel_v\n", mae_rel_v[-1])

        # Maximum Absolute Relative Error
        max_abs_rel_err_x.append(np.max(np.abs(rel_err_x)))
        max_abs_rel_err_v.append(np.max(np.abs(rel_err_v)))
        print("max_abs_rel_err_x\n", max_abs_rel_err_x[-1]); print("max_abs_rel_err_v\n", max_abs_rel_err_v[-1])
    
    # ---- Serialize to CSV ----
    """
    df = pd.DataFrame({
        "shots": shots_list,
        "rmse_x": rmse_x, "rmse_v": rmse_v,
        "mae_x": mae_x, "mae_v": mae_v,
        "max_abs_err_x": max_abs_err_x, "max_abs_err_v": max_abs_err_v,
        "rmse_rel_x": rmse_rel_x, "rmse_rel_v": rmse_rel_v,
        "mae_rel_x": mae_rel_x, "mae_rel_v": mae_rel_v,
        "max_abs_rel_err_x": max_abs_rel_err_x, "max_abs_rel_err_v": max_abs_rel_err_v
    })
    """
    df = pd.DataFrame({
        "shots": shots_list,
        "rmse_x": rmse_x, "rmse_v": rmse_v,
        "mae_x": mae_x, "mae_v": mae_v,
        "max_abs_err_x": max_abs_err_x, "max_abs_err_v": max_abs_err_v,
    })
    df.to_csv("error_metrics.csv", index=False)
    print("✅ Saved error_metrics.csv")
    """
        # --------- DO PLOTS ----------
        if do_plotting:
            print("PROGRAM INFO: Plotting.")
            data_dict:dict = {
                    "t_grid":t_grid,
                    "x_csim":x_csim,
                    "v_csim":v_csim,
                    "x_qsim":x_qsim,
                    "v_qsim":v_qsim,
                    "abs_err_x":abs_err_x,
                    "abs_err_v":abs_err_v,
                    "rmse_x":rmse_x,
                    "rmse_v":rmse_v,
                    "mae_x":mae_x,
                    "mae_v":mae_v
            }
            ani = do_plots(do_dict, data_dict)
        
        # ---- Save quantum simulation series to CSV (signed position & velocity) ----
        if do_save_data:
            print("PROGRAM INFO: Saving Data.")
            df = pd.DataFrame({
                't': t_grid,
                'x_qsim': x_qsim,
                'v_qsim': v_qsim
            })
            df.to_csv('quantum_timeseries.csv', index=False)
            print("✅ Saved: quantum_timeseries.csv")
    """

