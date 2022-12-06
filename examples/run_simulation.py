import pbeis
import pybamm

pybamm.set_logging_level("INFO")

# Load model (DFN with capacitance)
model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})

# Create simulation
eis_sim = pbeis.EISSimulation(model)

# Choose frequencies and calculate impedance
frequencies = pbeis.logspace(-4, 4, 30)
eis_sim.solve(frequencies)

# Generate a Nyquist plot
eis_sim.nyquist_plot()
