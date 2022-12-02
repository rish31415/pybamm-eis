import pybamm
import pbeis
from parameters.Teo2021 import get_parameter_values
import os

# Change to parent directory (pybamm-eis)
os.chdir(pbeis.__path__[0] + "/..")

# Set up
model = pybamm.lithium_ion.SPMe(options={"surface form": "differential"})

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

params = [parameter_values, pybamm.ParameterValues("Chen2020")]

sols = []
for param in params:
    sim = pybamm.Simulation(model, parameter_values=param)
    sol = sim.solve([0, 3600])
    sols.append(sol)

pybamm.dynamic_plot(sols, labels=["Teo2021", "Chen2020"])
