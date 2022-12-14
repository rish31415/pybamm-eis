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
teo_params = pybamm.ParameterValues(get_parameter_values())
chen_params = pybamm.ParameterValues("Chen2020")

# helper function to set state of charge
c_n_max = teo_params["Maximum concentration in negative electrode [mol.m-3]"]
c_p_max = teo_params["Maximum concentration in positive electrode [mol.m-3]"]
sto_n_min = teo_params["Negative electrode lithiation at 0% SOC"]
sto_n_max = teo_params["Negative electrode lithiation at 100% SOC"]
sto_p_min = teo_params["Positive electrode lithiation at 100% SOC"]
sto_p_max = teo_params["Positive electrode lithiation at 0% SOC"]


def set_initial_concentrations(soc):
    if soc < 0 or soc > 1:
        raise ValueError("Target SOC should be between 0 and 1")

    sto_n = (sto_n_max - sto_n_min) * soc + sto_n_min
    sto_p = sto_p_max - (sto_p_max - sto_p_min) * soc

    teo_params.update(
        {
            "Initial concentration in negative electrode [mol.m-3]": sto_n * c_n_max,
            "Initial concentration in positive electrode [mol.m-3]": sto_p * c_p_max,
        },
        check_already_exists=False,
    )


set_initial_concentrations(1)

# Switch out parameters to debug
param_list = [
    # "Negative electrode thickness [m]",
    # "Separator thickness [m]",
    # "Positive electrode thickness [m]",
    # "Electrode height [m]",
    # "Electrode width [m]",
    # "Nominal cell capacity [A.h]",
    # "Typical current [A]",
    # "Current function [A]",
    # "Negative electrode conductivity [S.m-1]",
    # "Maximum concentration in negative electrode [mol.m-3]",
    # "Negative electrode diffusivity [m2.s-1]",
    # "Negative electrode OCP [V]",
    # "Negative electrode porosity",
    # "Negative electrode active material volume fraction",
    # "Negative particle radius [m]",
    # "Negative electrode Bruggeman coefficient (electrolyte)",
    # "Negative electrode Bruggeman coefficient (electrode)",
    # "Negative electrode cation signed stoichiometry",
    # "Negative electrode electrons in reaction",
    # "Negative electrode charge transfer coefficient",
    # "Negative electrode double-layer capacity [F.m-2]",
    # "Negative electrode exchange-current density [A.m-2]",
    # "Negative electrode OCP entropic change [V.K-1]",
    "Positive electrode conductivity [S.m-1]",
    # "Maximum concentration in positive electrode [mol.m-3]",
    # "Positive electrode diffusivity [m2.s-1]",
    # "Positive electrode OCP [V]",
    # "Positive electrode porosity",
    # "Positive electrode active material volume fraction",
    # "Positive particle radius [m]",
    # "Positive electrode Bruggeman coefficient (electrolyte)",
    # "Positive electrode Bruggeman coefficient (electrode)",
    # "Positive electrode cation signed stoichiometry",
    # "Positive electrode electrons in reaction",
    # "Positive electrode charge transfer coefficient",
    # "Positive electrode double-layer capacity [F.m-2]",
    # "Positive electrode exchange-current density [A.m-2]",
    # "Positive electrode OCP entropic change [V.K-1]",
    # "Separator porosity",
    # "Separator Bruggeman coefficient (electrolyte)",
    # "Typical electrolyte concentration [mol.m-3]",
    # "Initial concentration in electrolyte [mol.m-3]",
    # "Cation transference number",
    # "1 + dlnf/dlnc",
    # "Electrolyte diffusivity [m2.s-1]",
    # "Electrolyte conductivity [S.m-1]",
    # "Reference temperature [K]",
    # "Ambient temperature [K]",
    # "Initial temperature [K]",
    # "Number of electrodes connected in parallel to make a cell",
    # "Number of cells connected in series to make a battery",
    # "Lower voltage cut-off [V]",
    # "Upper voltage cut-off [V]",
]
for param in param_list:
    teo_params[param] = chen_params[param]


# Choose frequencies and calculate impedance
frequencies = pbeis.logspace(-3, 4, 30)

# Loop over parameters
labels = ["Chen2020", "Teo2021"]
params = [chen_params, teo_params]

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
