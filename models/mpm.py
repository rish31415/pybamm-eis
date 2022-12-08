#
# Single Particle Model (SPM)
#
import pybamm


class MPM(pybamm.lithium_ion.BaseModel):
    """
    Many-Particle Model (MPM) of a lithium-ion battery with particle-size
    distributions for each electrode, from [1]_.

    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model. For a detailed list of
        options see :class:`~pybamm.BatteryModelOptions`.
    name : str, optional
        The name of the model.
    build :  bool, optional
        Whether to build the model on instantiation. Default is True. Setting this
        option to False allows users to change any number of the submodels before
        building the complete model (submodels cannot be changed after the model is
        built).
    Examples
    --------
    >>> import pybamm
    >>> model = pybamm.lithium_ion.MPM()
    >>> model.name
    'Many-Particle Model'

    References
    ----------
    .. [1] TL Kirk, J Evans, CP Please and SJ Chapman. “Modelling electrode
        heterogeneity in lithium-ion batteries: unimodal and bimodal particle-size
        distributions”.
        In: arXiv preprint arXiv:2006.12208 (2020).

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, options=None, name="MPM", build=True):
        # Necessary/default options
        options = options or {}
        if "particle size" in options and options["particle size"] != "distribution":
            raise pybamm.OptionError(
                "particle size must be 'distribution' for MPM not '{}'".format(
                    options["particle size"]
                )
            )
        elif "surface form" in options and options["surface form"] == "false":
            raise pybamm.OptionError(
                "surface form must be 'algebraic' or 'differential' for MPM not 'false'"
            )
        else:
            surface_form = options.get("surface form", "algebraic")
            options.update(
                {"particle size": "distribution", "surface form": surface_form}
            )

        # For degradation models we use the "x-average", note that for side reactions
        # this is set by "x-average side reactions"
        self.x_average = True

        # Set "x-average side reactions" to "true" if the model is SPM
        x_average_side_reactions = options.get("x-average side reactions")
        if x_average_side_reactions is None and self.__class__ in [
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.MPM,
        ]:
            options["x-average side reactions"] = "true"

        super().__init__(options, name)

        self.set_submodels(build)

    def set_intercalation_kinetics_submodel(self):

        for domain in ["negative", "positive"]:
            electrode_type = self.options.electrode_types[domain]
            if electrode_type == "planar":
                continue

            if self.options["surface form"] == "false":
                inverse_intercalation_kinetics = (
                    self.get_inverse_intercalation_kinetics()
                )
                self.submodels[f"{domain} interface"] = inverse_intercalation_kinetics(
                    self.param, domain, "lithium-ion main", self.options
                )
                self.submodels[
                    f"{domain} interface current"
                ] = pybamm.kinetics.CurrentForInverseButlerVolmer(
                    self.param, domain, "lithium-ion main", self.options
                )
            else:
                intercalation_kinetics = self.get_intercalation_kinetics(domain)
                phases = self.options.phases[domain]
                for phase in phases:
                    submod = intercalation_kinetics(
                        self.param, domain, "lithium-ion main", self.options, phase
                    )
                    self.submodels[f"{domain} {phase} interface"] = submod
                if len(phases) > 1:
                    self.submodels[
                        f"total {domain} interface"
                    ] = pybamm.kinetics.TotalMainKinetics(
                        self.param, domain, "lithium-ion main", self.options
                    )

    def set_particle_submodel(self):
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue

            particle = getattr(self.options, domain)["particle"]
            for phase in self.options.phases[domain]:
                if particle == "Fickian diffusion":
                    submod = pybamm.particle.FickianDiffusion(
                        self.param, domain, self.options, phase=phase, x_average=True
                    )
                elif particle in [
                    "uniform profile",
                    "quadratic profile",
                    "quartic profile",
                ]:
                    submod = pybamm.particle.XAveragedPolynomialProfile(
                        self.param, domain, self.options, phase=phase
                    )
                self.submodels[f"{domain} {phase} particle"] = submod

    def set_solid_submodel(self):
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            self.submodels[
                f"{domain} electrode potential"
            ] = pybamm.electrode.ohm.LeadingOrder(self.param, domain, self.options)

    def set_electrolyte_concentration_submodel(self):
        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte_diffusion.ConstantConcentration(self.param, self.options)

    def set_electrolyte_potential_submodel(self):

        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        if self.options["electrolyte conductivity"] not in ["default", "leading order"]:
            raise pybamm.OptionError(
                "electrolyte conductivity '{}' not suitable for SPM".format(
                    self.options["electrolyte conductivity"]
                )
            )

        if (
            self.options["surface form"] == "false"
            or self.options.electrode_types["negative"] == "planar"
        ):
            self.submodels[
                "leading-order electrolyte conductivity"
            ] = pybamm.electrolyte_conductivity.LeadingOrder(
                self.param, options=self.options
            )
        if self.options["surface form"] == "false":
            surf_model = surf_form.Explicit
        elif self.options["surface form"] == "differential":
            surf_model = surf_form.LeadingOrderDifferential
        elif self.options["surface form"] == "algebraic":
            surf_model = surf_form.LeadingOrderAlgebraic

        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "planar":
                continue
            self.submodels[f"{domain} surface potential difference"] = surf_model(
                self.param, domain, options=self.options
            )

    def set_convection_submodel(self):
        self.submodels[
            "transverse convection"
        ] = pybamm.convection.transverse.NoConvection(self.param, self.options)
        self.submodels[
            "through-cell convection"
        ] = pybamm.convection.through_cell.NoConvection(self.param, self.options)

    @property
    def default_parameter_values(self):
        default_params = super().default_parameter_values
        default_params = pybamm.get_size_distribution_parameters(default_params)
        return default_params
