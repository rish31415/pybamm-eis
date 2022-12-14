import pbeis
import pybamm
import matplotlib.pyplot as plt
from parameters.Teo2021 import get_parameter_values
import os

# Change to parent directory (pybamm-eis)
os.chdir(pbeis.__path__[0] + "/..")

# Load model (DFN with capacitance)
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})

# load parameters
parameter_values = pybamm.ParameterValues(get_parameter_values())

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


set_initial_concentrations(1)


# Choose frequencies and calculate impedance
frequencies = pbeis.logspace(-3, 4, 30)

# Loop over parameters
labels = ["Chen2020", "Marquis2019"]
params = [pybamm.ParameterValues(label) for label in labels]
labels.append("Teo2021")
params.append(parameter_values)

_, ax = plt.subplots()

for param, label in zip(params, labels):
    eis_sim = pbeis.EISSimulation(model, param)
    impedances = eis_sim.solve(frequencies)
    Ly = param["Electrode width [m]"]
    Lz = param["Electrode height [m]"]
    z = impedances * 1000 * Ly * Lz  # Ohm -> mOhm.m2
    pbeis.nyquist_plot(
        z,
        ax=ax,
        label=label,
        linestyle="-",
        marker="None",
    )

# Show plot
ax.set_xlabel(r"$Z_\mathrm{Re}$ [mOhm m${}^2$]")
ax.set_ylabel(r"$-Z_\mathrm{Im}$ [mOhm m${}^2$]")
ax.legend()
plt.show()
