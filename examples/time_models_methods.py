import pbeis
import pybamm
import numpy as np
import time as timer
import pprint
from scipy.fft import fft


def brute_force(model, parameter_values, frequencies):

    # Time domain
    I = 50 * 1e-3
    number_of_periods = 20
    samples_per_period = 16

    def current_function(t):
        return I * pybamm.sin(2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t)

    parameter_values["Current function [A]"] = current_function

    sim = pybamm.Simulation(
        model,
        parameter_values=parameter_values,
        solver=pybamm.CasadiSolver(mode="safe without grid"),
    )
    sim.build()

    start_time = timer.time()
    impedances_time = []
    for frequency in frequencies:
        # Solve
        period = 1 / frequency
        dt = period / samples_per_period
        t_eval = np.array(range(0, 1 + samples_per_period * number_of_periods)) * dt
        sol = sim.solve(t_eval, inputs={"Frequency [Hz]": frequency})
        # Extract final two periods of the solution
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

    return {
        "States": sol.first_state.y.shape[0],
        "Setup time": sol.set_up_time.value,
        "Solve time": time_elapsed,
    }


def frequency_domain(model, parameter_values, frequencies, method):
    eis_sim = pbeis.EISSimulation(model, parameter_values=parameter_values)
    eis_sim.solve(frequencies, method)
    return {
        "States": eis_sim.y0.shape[0],
        "Setup time": eis_sim.set_up_time.value,
        "Solve time": eis_sim.solve_time.value,
    }


# Run comparison
parameter_values = pybamm.ParameterValues("Marquis2019")
parameter_values = pybamm.get_size_distribution_parameters(
    parameter_values, sd_n=0.2, sd_p=0.4
)
frequencies = np.logspace(-4, 2, 30)
methods = ["direct", "prebicgstab"]
models = [
    pybamm.lithium_ion.SPM(options={"surface form": "differential"}, name="SPM"),
    pybamm.lithium_ion.SPMe(options={"surface form": "differential"}, name="SPMe"),
    pybamm.lithium_ion.DFN(options={"surface form": "differential"}, name="DFN"),
    pybamm.lithium_ion.MPM(name="MPM"),
]
results = dict.fromkeys(
    [model.name for model in models], dict.fromkeys(["brute force"] + methods)
)

for model in models:
    results[model.name]["brute force"] = brute_force(
        model, parameter_values, frequencies
    )
    for method in methods:
        results[model.name][method] = frequency_domain(
            model, parameter_values, frequencies, method
        )

pp = pprint.PrettyPrinter(depth=5)
pp.pprint(results)
