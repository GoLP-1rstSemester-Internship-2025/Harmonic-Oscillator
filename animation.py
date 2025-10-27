import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# ----------------------
# Spring function
# ----------------------
def spring_coords(x_pos, n_coils=5, amplitude=0.1):
    """
    Generate zig-zag coordinates for a spring from 0 to x_pos
    """
    xs = np.linspace(0, x_pos, 2*n_coils + 1)
    ys = np.zeros_like(xs)
    ys[1::2] = amplitude
    ys[2::2] = -amplitude
    ys[-1] = 0
    return xs, ys
# ----------------------
# Init function
# ----------------------
def init():
    spring_line.set_data([], [])
    mass_dot.set_data([], [])
    return spring_line, mass_dot
# ----------------------
# Animation function
# ----------------------
def animate(i:int):
    xi = x[i]
    xs, ys = spring_coords(xi)
    spring_line.set_data(xs, ys)
    mass_dot.set_data([xi], [0])  # <-- wrap in lists
    return spring_line, mass_dot

if __name__ == "__main__":
    df = pd.read_csv('quantum_timeseries.csv')
    x = df.position_quantum
    t = df.t
# ----------------------
# Setup figure
# ----------------------
    Amplitude = max(x)
    fig, ax = plt.subplots(figsize=(8,2))
    ax.set_xlim(-1.5*Amplitude, 1.5*Amplitude)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.set_title('Mass-Spring Animation via 1 Qubit Hamiltonian Simulation')

# Spring line and mass
    spring_line, = ax.plot([], [], lw=2, color='blue')
    mass_dot, = ax.plot([], [], 'ro', markersize=12)
# ----------------------
# Run animation
# ----------------------
    anim = FuncAnimation(fig, animate, frames=len(x), init_func=init, blit=True, interval=t[len(t)-1])
    plt.show()

