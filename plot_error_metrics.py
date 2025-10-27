import pandas as pd
import matplotlib.pyplot as plt

# ---- Load CSV ----
df = pd.read_csv("error_metrics.csv")
shots = df["shots"]

# ---- FIGURE 1: Errors for Position (x) ----
fig_x, ax_x = plt.subplots(figsize=(10, 10))
ax_x.plot(shots, df["rmse_x"], 'o-', label="RMSE x", color='blue')
ax_x.plot(shots, df["mae_x"], 's--', label="MAE x", color='green')
ax_x.plot(shots, df["max_abs_err_x"], 'x-.', label="Max Abs Error x", color='red')
ax_x.set_xlabel("Shots")
ax_x.set_ylabel("Error (x)")
ax_x.set_title("Quantum Harmonic Oscillator: Position (x) Errors")
ax_x.grid(True, alpha=0.3)
ax_x.legend()
fig_x.tight_layout()

# ---- FIGURE 2: Errors for Velocity (v) ----
fig_v, ax_v = plt.subplots(figsize=(10, 10))
ax_v.plot(shots, df["rmse_v"], 'o-', label="RMSE v", color='blue')
ax_v.plot(shots, df["mae_v"], 's--', label="MAE v", color='green')
ax_v.plot(shots, df["max_abs_err_v"], 'x-.', label="Max Abs Error v", color='red')
ax_v.set_xlabel("Shots")
ax_v.set_ylabel("Error (v)")
ax_v.set_title("Quantum Harmonic Oscillator: Velocity (v) Errors")
ax_v.grid(True, alpha=0.3)
ax_v.legend()
fig_v.tight_layout()


fig_x.savefig("error_metrics_x.pdf")
fig_v.savefig("error_metrics_v.pdf")
plt.show()
