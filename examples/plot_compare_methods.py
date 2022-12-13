import pbeis
import pybamm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time as timer
from matplotlib.pyplot import cm
from scipy.fft import fft

# Set plot style ----------------------------------------------------------------------
matplotlib.rc_file("_matplotlibrc", use_default_template=True)

# Set up ------------------------------------------------------------------------------
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"}, name="DFN")
parameter_values = pybamm.ParameterValues("Marquis2019")
frequencies = pbeis.logspace(-4, 4, 30)

# Time domain -------------------------------------------------------------------------
I = 50 * 1e-3  # applied current
number_of_periods = 20
samples_per_period = 16


def current_function(t):
    return I * pybamm.sin(2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t)


parameter_values["Current function [A]"] = current_function

start_time = timer.time()

# Create simulation
sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    solver=pybamm.CasadiSolver(mode="safe without grid"),
)

# Loop over frequencies and solve
impedances_time = []
for frequency in frequencies:
    # Solve
    period = 1 / frequency
    dt = period / samples_per_period
    t_eval = np.array(range(0, 1 + samples_per_period * number_of_periods)) * dt
    sol = sim.solve(t_eval, inputs={"Frequency [Hz]": frequency})
    # Extract final two periods of the solution
    time = sol["Time [s]"].entries[-3 * samples_per_period - 1 :]
    current = sol["Current [A]"].entries[-3 * samples_per_period - 1 :]
    voltage = sol["Terminal voltage [V]"].entries[-3 * samples_per_period - 1 :]
    # FFT
    current_fft = fft(current)
    voltage_fft = fft(voltage)
    # Get index of first harmonic
    idx = np.argmax(np.abs(current_fft))
    impedance = -voltage_fft[idx] / current_fft[idx]
    impedances_time.append(impedance)

end_time = timer.time()
time_elapsed = end_time - start_time
print("Time domain method: ", time_elapsed, "s")

# Frequency domain ---------------------------------------------------------------------
methods = ["direct", "prebicgstab"]
names = ["Gaussian elimination", "preBicgSTAB"]
impedances_freqs = []
for method in methods:
    start_time = timer.time()
    eis_sim = pbeis.EISSimulation(model, parameter_values=parameter_values)
    impedances_freq = eis_sim.solve(frequencies, method)
    end_time = timer.time()
    time_elapsed = end_time - start_time
    print(f"Frequency domain ({method}): ", time_elapsed, "s")
    impedances_freqs.append(impedances_freq)

# Comparison plots ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.4, 3))
colors = cm.tab10(np.linspace(0, 1, 10))
markers = ["o", "d", "s"]

# plot brute force approach
ax[0] = pbeis.nyquist_plot(
    impedances_time,
    ax=ax[0],
    label="Brute force",
    linestyle="-",
    marker=markers[0],
    color=colors[0],
)

# plot frequency domain approach
for i, name in enumerate(names):
    # nyquist
    ax[0] = pbeis.nyquist_plot(
        impedances_freqs[i],
        ax=ax[0],
        label=f"Frequency domain \n ({name})",
        linestyle="--",
        marker=markers[i + 1],
        color=colors[i + 1],
    )
    # calculate difference w.r.t. brute force method
    diffs = [
        (
            2
            * np.sqrt((zt.real - zf.real) ** 2 + (zt.imag - zf.imag) ** 2)
            / (
                np.sqrt(zt.real**2 + zt.imag**2)
                + np.sqrt(zf.real**2 + zf.imag**2)
            )
        )
        * 100
        for zt, zf in zip(impedances_time, impedances_freqs[i])
    ]
    # plot difference
    ax[1].plot(
        frequencies[: len(diffs)],
        diffs,
        label=f"Frequency domain \n ({name})",
        linestyle="--",
        marker=markers[i + 1],
        color=colors[i + 1],
    )

ax[1].set_xlabel(r"$\omega$ [Hz]")
ax[1].set_ylabel("Difference [%]")
ax[1].set_xscale("log")
ax[0].legend()
plt.tight_layout()
plt.savefig(f"figures/{model.name}_time_vs_freq.pdf", dpi=300)
plt.show()
