import pbeis
import pybamm
import matplotlib
import matplotlib.pyplot as plt
from models.mpm import MPM

# Set plot style ----------------------------------------------------------------------
matplotlib.rc_file("_matplotlibrc", use_default_template=True)


# Load models and parameters ----------------------------------------------------------
models = [
    pybamm.lithium_ion.SPM(options={"surface form": "differential"}, name="SPM"),
    pybamm.lithium_ion.DFN(options={"surface form": "differential"}, name="DFN"),
    MPM(options={"surface form": "differential"}, name="MPM"),
    pybamm.lithium_ion.SPM(
        options={
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 2,
        },
        name="SPM (pouch)",
    ),
    pybamm.lithium_ion.DFN(
        options={
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 2,
        },
        name="DFN (pouch)",
    ),
    MPM(
        options={
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 2,
        },
        name="MPM (pouch)",
    ),
]
parameter_values = pybamm.ParameterValues("Marquis2019")
parameter_values = pybamm.get_size_distribution_parameters(
    parameter_values, sd_n=0.2, sd_p=0.4
)

# Loop over models and calculate impedance ---------------------------------------------
frequencies = pbeis.logspace(-4, 4, 30)
impedances = []
for model in models:
    print(f"Start calculating impedance for {model.name}")
    eis_sim = pbeis.EISSimulation(model, parameter_values=parameter_values)
    impedances_freq = eis_sim.solve(
        frequencies,
    )
    print(f"Finished calculating impedance for {model.name}")
    impedances.append(impedances_freq)

# Plot comparison ----------------------------------------------------------------------
_, ax = plt.subplots(figsize=(6.4, 3))
for i, model in enumerate(models):
    ax = pbeis.nyquist_plot(
        impedances[i],
        ax=ax,
        linestyle="-",
        marker="d",
        label=f"{model.name}",
    )
ax.legend()
plt.tight_layout()
plt.savefig("figures/compare_models.pdf", dpi=300)
plt.show()
