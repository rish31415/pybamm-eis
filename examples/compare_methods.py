import pbeis
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time as timer
from scipy.fft import fft


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 3.6,
    }
)


# Set up
model = pybamm.lithium_ion.SPM(options={"surface form": "differential"}, name="DFN")

parameter_values = pybamm.ParameterValues("Marquis2019")

frequencies = pbeis.logspace(-4, 4, 30)

# Time domain
I = 50 * 1e-3
number_of_periods = 20
samples_per_period = 16
plot = False  # whether to plot results inside the loop


def current_function(t):
    return I * pybamm.sin(2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t)


parameter_values["Current function [A]"] = current_function

start_time = timer.time()

sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    solver=pybamm.CasadiSolver(mode="safe without grid"),
)

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
    # Plot
    if plot:
        x = np.linspace(0, 1 / dt, len(current_fft))
        _, ax = plt.subplots(2, 2)
        ax[0, 0].plot(time, current)
        ax[0, 1].plot(time, voltage)
        ax[1, 0].plot(x, np.abs(current_fft))
        ax[1, 1].plot(x, np.abs(voltage_fft))
        ax[1, 0].set_xlim([0, frequency * 3])
        ax[1, 1].set_xlim([0, frequency * 3])
        plt.show()

end_time = timer.time()
time_elapsed = end_time - start_time
print("Time domain method: ", time_elapsed, "s")

# Frequency domain
methods = ["direct"]  # , "prebicgstab"]
impedances_freqs = []
for method in methods:
    start_time = timer.time()
    eis_sim = pbeis.EISSimulation(model, parameter_values=parameter_values)
    impedances_freq = eis_sim.solve(frequencies, method)
    end_time = timer.time()
    time_elapsed = end_time - start_time
    print(f"Frequency domain ({method}): ", time_elapsed, "s")
    impedances_freqs.append(impedances_freq)

# Compare
_, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
ax[0] = pbeis.nyquist_plot(
    impedances_time, linestyle="-", ax=ax[0], label="Time", alpha=0.7
)
for i, method in enumerate(methods):
    ax[0] = pbeis.nyquist_plot(
        impedances_freqs[i],
        linestyle="-",
        ax=ax[0],
        label=f"Frequency ({method})",
        alpha=0.7,
    )
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

    ax[1].plot(
        frequencies,
        diffs,
        linestyle="-",
        marker="o",
        label=f"Frequency ({method})",
        alpha=0.7,
    )

ax[1].set_xlabel(r"$\omega$ [Hz]")
ax[1].set_ylabel(r"diff [%]")
ax[1].set_xscale("log")
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.savefig(f"figures/{model.name}_time_vs_freq.pdf", dpi=300)
plt.show()
