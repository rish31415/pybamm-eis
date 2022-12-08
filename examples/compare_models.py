import pbeis
import pybamm
import matplotlib.pyplot as plt
from models.mpm import MPM

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 3.6,
    }
)


# Load models and parameters
models = [
    pybamm.lithium_ion.SPM(options={"surface form": "differential"}, name="SPM"),
    pybamm.lithium_ion.SPMe(options={"surface form": "differential"}, name="SPMe"),
    MPM(options={"surface form": "differential"}, name="MPM"),
    pybamm.lithium_ion.DFN(options={"surface form": "differential"}, name="DFN"),
    pybamm.lithium_ion.DFN(
        options={"particle size": "distribution", "surface form": "differential"},
        name="MPDFN",
    ),
    pybamm.lithium_ion.SPM(
        {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 2,
        },
        name="SPM Pouch",
    ),
]
parameter_values = pybamm.ParameterValues("Marquis2019")
parameter_values = pybamm.get_size_distribution_parameters(
    parameter_values, sd_n=0.2, sd_p=0.4
)

# Loop over models and calculate impedance
frequencies = pbeis.logspace(-4, 4, 30)
impedances = []
for model in models:
    print(f"Start calculating impedance for {model.name}")
    eis_sim = pbeis.EISSimulation(model, parameter_values=parameter_values)
    impedances_freq = eis_sim.solve(
        frequencies,
    )
    print(f"Finished calculating impedance for {model.name}")
    print(
        "Number of states: ",
        eis_sim.y0.shape[0],
        "Set-up time: ",
        eis_sim.set_up_time,
        "Solve time: ",
        eis_sim.solve_time,
    )
    impedances.append(impedances_freq)

# Compare
_, ax = plt.subplots(figsize=(7, 4))
for i, model in enumerate(models):
    ax = pbeis.nyquist_plot(
        impedances[i], ax=ax, linestyle="-", label=f"{model.name}", alpha=0.7
    )
ax.legend()
plt.tight_layout()
plt.savefig("figures/compare_models.pdf", dpi=300)
plt.show()
