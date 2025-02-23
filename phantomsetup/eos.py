"""Equation of state."""

# The type ignore comments are due to the following bug:
# https://github.com/python/mypy/issues/4975

from typing import Any, Dict

import numpy as np

from . import defaults
from . import constants

ieos_label = {
    1: 'isothermal',
    2: 'adiabatic/polytropic',
    3: 'locally isothermal disc',
    6: 'locally isothermal disc centered on sink particle',
    7: 'z-dependent locally isothermal eos',
    8: 'barotropic',
    9: 'piecewise polytrope',
    10: 'MESA',
    11: 'isothermal with zero pressure',
    14: 'locally isothermal binary disc',
}

ieos_has = {
    'mu': (1, 2, 3, 6, 7, 8, 9, 10, 11, 14),
    'polyk': (1, 2, 3, 6, 7, 8, 9, 10, 11, 14),
    'gamma': (2, 8, 9),
    'qfacdisc': (3, 6, 7, 14),
}

ieos_isothermal = {1, 3, 6, 7, 11, 14}


class EquationOfState:
    """Equation of state for gas.

    Parameters
    ----------
    ieos : int
        The equation of state as represented by the following integers:
            1: 'isothermal'
            2: 'adiabatic/polytropic'
            3: 'locally isothermal disc'
            6: 'locally isothermal disc centered on sink particle'
            7: 'z-dependent locally isothermal eos'
            8: 'barotropic'
            9: 'piecewise polytrope'
            10: 'MESA'
            11: 'isothermal with zero pressure'
            14: 'locally isothermal binary disc'
    """

    def __init__(self, ieos: int, **kwargs) -> None:

        if ieos not in ieos_label:
            raise ValueError(f'ieos={ieos} does not exist')
        # if ieos > 3:
        #     raise NotImplementedError('ieos > 3 not available currently')

        self.ieos = ieos

        self.parameters: Dict[str, Any] = {
            key: None for key in ('mu', 'polyk', 'gamma', 'qfacdisc')
        }

        # Set defaults
        for parameter in self.parameters.keys():
            if ieos in ieos_has[parameter]:  # type: ignore
                if parameter == 'polyk':
                    self.parameters[parameter] = 2 / 3 * defaults.HEADER['RK2']
                else:
                    try:
                        self.parameters[parameter] = defaults.HEADER[parameter]
                    except KeyError:
                        self.parameters[parameter] = defaults.RUN_OPTIONS.config[
                            parameter
                        ].value

        # Set values from kwargs
        for parameter in self.parameters:
            if parameter in kwargs:
                if ieos not in ieos_has[parameter]:  # type: ignore
                    raise ValueError(f'Cannot set {parameter} for ieos={ieos}')
                else:
                    self.parameters[parameter] = kwargs[parameter]

    @property
    def mu(self) -> float:
        """'mu' is the mean molecular weight."""
        return self.parameters['mu']

    @mu.setter
    def mu(self, value: float) -> None:
        if self.ieos not in ieos_has['mu']:  # type: ignore
            raise ValueError(f'ieos={self.ieos} not compatible with setting mu')
        self.parameters['mu'] = value

    @property
    def polyk(self) -> float:
        """'polyk' is a constant of proportionality in the eos.

        Isothermal eos: polyk = (sound speed)^2.
        Adiabatic/polytropic eos: polyk = pressure / rho^(gamma).
        """
        return self.parameters['polyk']

    @polyk.setter
    def polyk(self, value: float) -> None:
        if self.ieos not in ieos_has['polyk']:  # type: ignore
            raise ValueError(f'ieos={self.ieos} not compatible with setting polyk')
        self.parameters['polyk'] = value

    @property
    def gamma(self) -> float:
        """'gamma' is the adiabatic index."""
        return self.parameters['gamma']

    @gamma.setter
    def gamma(self, value: float) -> None:
        if self.ieos not in ieos_has['gamma']:  # type: ignore
            raise ValueError(f'ieos={self.ieos} not compatible with setting gamma')
        self.parameters['gamma'] = value

    @property
    def qfacdisc(self) -> float:
        """'qfacdisc' is the 'q' exponent of locally isothermal disc.

        Sound speed is proportional to radius^(-q).
        """

        return self.parameters['qfacdisc']




    @qfacdisc.setter
    def qfacdisc(self, value: float) -> None:
        if self.ieos not in ieos_has['qfacdisc']:  # type: ignore
            raise ValueError(f'ieos={self.ieos} not compatible with setting qfacdisc')
        self.parameters['qfacdisc'] = value


def polyk_for_locally_isothermal_disc_mine(
    T0: float,
    q_index: float,
    reference_radius: float,
    stellar_mass: float,
    gravitational_constant: float,
    aspect_ratio: float,
) -> float:
    """Get polyk for a locally isothermal disc.

    Parameters
    ----------
    q_index
        The index in the sound speed power law such that
            H ~ (R / R_reference) ^ (3/2 - q).
    aspect_ratio
        The aspect ratio (H/R) at the reference radius.
    reference_radius
        The radius at which the aspect ratio is given.
    stellar_mass
        The mass of the central object the disc is orbiting.
    gravitational_constant
        The gravitational constant.
    """
    cs_ref = np.sqrt((constants.k_b*T0)/(defaults._RUN_OPTIONS['mu']*constants.m_p))
    omega_ref = np.sqrt(constants.gravitational_constant * stellar_mass*constants.solarm / (reference_radius*constants.au)**3)

    return (aspect_ratio * np.sqrt(gravitational_constant * stellar_mass / reference_radius) * reference_radius ** q_index) **2 # Polytropic constant, polyk

def polyk_for_locally_isothermal_disc(
    q_index: float,
    reference_radius: float,
    aspect_ratio: float,
    stellar_mass: float,
    gravitational_constant: float,
) -> float:
    """Get polyk for a locally isothermal disc.
    Parameters
    ----------
    q_index
        The index in the sound speed power law such that
        .. math:: H \sim (R / R_{\mathrm{reference}})^{3/2 - q}
    aspect_ratio
        The aspect ratio (H/R) at the reference radius.
    reference_radius
        The radius at which the aspect ratio is given.
    stellar_mass
        The mass of the central object the disc is orbiting.
    gravitational_constant
        The gravitational constant.
    """
    return (
        aspect_ratio
        * np.sqrt(gravitational_constant * stellar_mass / reference_radius)
        * reference_radius ** q_index
    ) ** 2


def get_aspect_ratio_new(
    T0: float,
    R0_temp: float,
    Tinf: float,
    my_temp_exp: float,
    reference_radius: float,
    stellar_mass: float,
    gravitational_constant: float,
) -> float:

    T_ref = np.sqrt(T0**2*(((reference_radius)**2+(R0_temp)**2)**-my_temp_exp)+Tinf**2) # The temperature at the reference radius
    cs_ref = np.sqrt((constants.k_b*T_ref)/(defaults._RUN_OPTIONS['mu']*constants.m_p)) # The sound speed at the reference radius
    omega_ref = np.sqrt(constants.gravitational_constant * stellar_mass*constants.solarm / (reference_radius*constants.au)**3)
    return(cs_ref/omega_ref)/(reference_radius*constants.au) # Aspect ratio at the reference radius 
