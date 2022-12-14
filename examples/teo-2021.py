#
# Compare results to SEIS data from the paper:
#
#   Dynamic Electrochemical Impedance Spectroscopy of Lithium-ion
#   Batteries: Revealing Underlying Physics through Efficient Joint Time-
#   Frequency Modeling, Linnette Teo et al 2021 J. Electrochem. Soc. 168 010526
#
# Note: the data were extracted using WebPlotDigitizer, so may not be perfect.
#

import pbeis
import pybamm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pyplot import cm
from scipy.fft import fft
from parameters.Teo2021 import get_parameter_values
import os

brute_force = False

# Change to parent directory (pybamm-eis)
os.chdir(pbeis.__path__[0] + "/..")

# Load model (DFN with capacitance)
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})

# load parameters
parameter_values = pybamm.ParameterValues(get_parameter_values())
if brute_force:
    I = 50 * 1e-3
    number_of_periods = 20
    samples_per_period = 16

    def current_function(t):
        return I * pybamm.sin(2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t)


# helper function to set state of charge
c_n_max = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
c_p_max = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
sto_n_min = parameter_values["Negative electrode lithiation at 0% SOC"]
sto_n_max = parameter_values["Negative electrode lithiation at 100% SOC"]
sto_p_min = parameter_values["Positive electrode lithiation at 100% SOC"]
sto_p_max = parameter_values["Positive electrode lithiation at 0% SOC"]


def set_initial_concentrations(soc):
    if soc < 0 or soc > 1:
        raise ValueError("Target SOC should be between 0 and 1")

    sto_n = (sto_n_max - sto_n_min) * soc + sto_n_min
    sto_p = sto_p_max - (sto_p_max - sto_p_min) * soc

    parameter_values.update(
        {
            "Initial concentration in negative electrode [mol.m-3]": sto_n * c_n_max,
            "Initial concentration in positive electrode [mol.m-3]": sto_p * c_p_max,
        },
        check_already_exists=False,
    )


# Choose frequencies and calculate impedance, looping over states of charge
frequencies = pbeis.logspace(-3, 4, 20)
_, ax = plt.subplots()
socs = [0.95, 0.65, 0.45, 0.05]
colors = iter(cm.tab10(np.linspace(0, 1, 10)))

for soc in socs:
    c = next(colors)

    # Set SOC
    set_initial_concentrations(soc)

    # Compute impedance using frequency domain method...
    eis_sim = pbeis.EISSimulation(model, parameter_values)
    impedances = eis_sim.solve(frequencies)
    z = impedances * 1000  # Ohm -> mOhm.m2  (with unit cross-sectional area)
    # Add to Nyquist plot (in [mOhm.m2])
    pbeis.nyquist_plot(
        z,
        ax=ax,
        label=f"Gaussian elimination {int(soc*100)}% SOC",
        linestyle="--",
        marker="None",
        c=c,
    )

    # ... and using brute force method
    if brute_force:
        parameter_values["Current function [A]"] = current_function
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
        z_time = (
            impedances_time * 1000
        )  # Ohm -> mOhm.m2  (with unit cross-sectional area)
        # Add to Nyquist plot (in [mOhm.m2])
        pbeis.nyquist_plot(
            z_time,
            ax=ax,
            label=f"Brute force {int(soc*100)}% SOC",
            linestyle="-",
            marker="None",
            c=c,
        )

    # Add data to plot
    data = pd.read_csv(f"data/Teo-SOC{int(soc*100):02d}.csv").to_numpy()
    ax.plot(
        data[:, 0],
        data[:, 1],
        linestyle="None",
        marker="x",
        label=f"Data ({int(soc*100)}% SOC)",
        c=c,
    )
    ax.plot(
        data[:, 0] - data[-1, 0],
        data[:, 1],
        linestyle="None",
        marker="o",
        label=f"Shifted data ({int(soc*100)}% SOC)",
        c=c,
    )

# Show plot
ax.set_xlabel(r"$Z_\mathrm{Re}$ [mOhm m${}^2$]")
ax.set_ylabel(r"$-Z_\mathrm{Im}$ [mOhm m${}^2$]")
ax.legend()
plt.show()
