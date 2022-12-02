import pybamm


def graphite_ocp_Teo2021(sto):
    """
    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    u_eq = (
        0.7222
        + 0.1387 * sto
        + 0.029 * sto**0.5
        - 0.0172 / sto
        + 0.0019 / sto**1.5
        + 0.2808 * pybamm.exp(0.90 - 15 * sto)
        - 0.7984 * pybamm.exp(0.4465 * sto - 0.4108)
    )

    return u_eq


def graphite_electrolyte_exchange_current_density_Teo2021(c_e, c_s_surf, c_s_max, T):
    """
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    k_ref = 5.031e-11  # (mol/s/m2)(m3/mol)**1.5
    F = pybamm.constants.F
    m_ref = k_ref * F  # (A/m2)(m3/mol)**1.5
    return m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def lco_ocp_Teo2021(sto):
    """
    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    u_eq = (
        -4.656
        + 88.669 * sto**2
        - 401.119 * sto**4
        + 342.909 * sto**6
        - 462.471 * sto**8
        + 433.434 * sto**10
    ) / (
        -1
        + 18.933 * sto**2
        - 79.532 * sto**4
        + 37.311 * sto**6
        - 73.083 * sto**8
        + 95.96 * sto**10
    )

    return u_eq


def lco_electrolyte_exchange_current_density_Teo2021(c_e, c_s_surf, c_s_max, T):
    """
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    k_ref = 2.334e-11  # (mol/s/m2)(m3/mol)**1.5
    F = pybamm.constants.F
    m_ref = k_ref * F  # (A/m2)(m3/mol)**1.5
    return m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def electrolyte_conductivity_Teo2021(c_e, T):
    """
    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        4.1253e-2
        + 5.007e-4 * c_e
        - 4.7212e-7 * c_e**2
        + 1.5094e-10 * c_e**3
        - 1.6018e-14 * c_e**4
    )

    return sigma_e


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for a lithium-cobalt-oxide/graphite cell, from the paper

    Dynamic Electrochemical Impedance Spectroscopy of Lithium-ion
    Batteries: Revealing Underlying Physics through Efficient Joint Time-
    Frequency Modeling, Linnette Teo et al 2021 J. Electrochem. Soc. 168 010526

    and references therein.
    """

    return {
        # cell (unit cross-sectional area)
        "Negative electrode thickness [m]": 88e-06,
        "Separator thickness [m]": 25e-06,
        "Positive electrode thickness [m]": 80e-06,
        "Electrode height [m]": 1,
        "Electrode width [m]": 1,
        "Nominal cell capacity [A.h]": 29.0,
        "Typical current [A]": 29.0,
        "Current function [A]": 29.0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 48.24,  # 100 * (1-0.485-0.0326)
        "Maximum concentration in negative electrode [mol.m-3]": 30555,
        "Negative electrode diffusivity [m2.s-1]": 3.9e-14,
        "Negative electrode OCP [V]": graphite_ocp_Teo2021,
        "Negative electrode porosity": 0.485,
        "Negative electrode active material volume fraction": 0.4824,  # 1-0.485-0.0326
        "Negative particle radius [m]": 2e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 4,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode cation signed stoichiometry": -1.0,
        "Negative electrode electrons in reaction": 1.0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.7,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_electrolyte_exchange_current_density_Teo2021,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        "Negative electrode lithiation at 100% SOC": 0.85512,
        "Negative electrode lithiation at 0% SOC": 0.00819,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 59,  # 100 * (1-0.385-0.025)
        "Maximum concentration in positive electrode [mol.m-3]": 51554,
        "Positive electrode diffusivity [m2.s-1]": 1e-14,
        "Positive electrode OCP [V]": lco_ocp_Teo2021,
        "Positive electrode porosity": 0.385,
        "Positive electrode active material volume fraction": 0.59,  # 1-0.385-0.025
        "Positive particle radius [m]": 2e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 4,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode cation signed stoichiometry": -1.0,
        "Positive electrode electrons in reaction": 1.0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.1,
        "Positive electrode exchange-current density [A.m-2]"
        "": lco_electrolyte_exchange_current_density_Teo2021,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        "Positive electrode lithiation at 100% SOC": 0.49551,
        "Positive electrode lithiation at 0% SOC": 0.94719,
        # separator
        "Separator porosity": 0.724,
        "Separator Bruggeman coefficient (electrolyte)": 4,
        # electrolyte
        "Typical electrolyte concentration [mol.m-3]": 1000.0,
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.363,
        "1 + dlnf/dlnc": 1.0,
        "Electrolyte diffusivity [m2.s-1]": 7.5e-10,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Teo2021,
        # experiment
        "Reference temperature [K]": 298.15,
        "Ambient temperature [K]": 298.15,
        "Initial temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.5,
        "Upper voltage cut-off [V]": 4.2,
    }
