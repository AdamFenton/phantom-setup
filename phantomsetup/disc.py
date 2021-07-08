"""Accretion disc."""

from __future__ import annotations

import functools
from typing import Callable, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy import integrate, spatial, stats, special

from . import constants
from .particles import Particles
from . import defaults
import random
import matplotlib.pyplot as plt

class Disc(Particles):
    """Accretion disc.

    Parameters
    ----------
    particle_type
        The integer particle type.
    number_of_particles
        The number of particles.
    disc_mass
        The total disc mass.
    density_distribution
        The surface density as a function of radius.
    radius_range
        The range of radii as (R_min, R_max).
    q_index
        The index in the sound speed power law such that
            H ~ (R / R_reference) ^ (3/2 - q).
    aspect_ratio
        The aspect ratio at the reference radius.
    reference_radius
        The radius at which the aspect ratio is given.
    stellar_mass
        The mass of the central object the disc is orbiting.
    gravitational_constant
        The gravitational constant.

    Optional Parameters
    -------------------
    centre_of_mass
        The centre of mass of the disc, i.e. around which position
        is it rotating.
    rotation_axis
        An axis around which to rotate the disc.
    rotation_angle
        The angle to rotate around the rotation_axis.
    extra_args
        Extra arguments to pass to density_distribution.
    pressureless
        Set to True if the particles are pressureless, i.e. dust.

    Examples
    --------
    TODO: add examples
    """

    def __init__(
        self,
        *,
        particle_type: int,
        T0: float,
        Tinf: float,
        R0: float,
        R0_temp: float,
        radius_max: float,
        number_of_particles: float,
        disc_mass: float,
        density_distribution: Callable[[float], float],
        radius_range: Tuple[float, float],
        q_index: float,
        qfacdisc: float,
        my_temp_exp:float,
        p_index: float,
        aspect_ratio: float,
        reference_radius: float,
        stellar_mass: float,
        gravitational_constant: float,
        centre_of_mass: Tuple[float, float, float] = None,
        rotation_axis: Union[Tuple[float, float, float], ndarray] = None,
        rotation_angle: float = None,
        extra_args: tuple = None,
        pressureless: bool = False,
    ):
        super().__init__()

        particle_mass = disc_mass / number_of_particles

        position, smoothing_length,temperature = self._set_positions_mine(
            number_of_particles=number_of_particles,
            T0 = T0,
            Tinf = Tinf,
            R0 = R0,
            R0_temp = R0_temp,
            stellar_mass = stellar_mass,
            disc_mass=disc_mass,
            density_distribution=density_distribution,
            radius_range=radius_range,
            q_index=q_index,
            qfacdisc = qfacdisc,
            my_temp_exp = my_temp_exp,
            p_index=p_index,
            aspect_ratio=aspect_ratio,
            reference_radius=reference_radius,
            centre_of_mass=centre_of_mass,
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle,
            extra_args=extra_args,
        )

        velocity = self._set_velocities(

            position=position,
            number_of_particles = number_of_particles,
            disc_mass = disc_mass,
            density_distribution=density_distribution, # I have added this 04/02/2021
            radius_range=radius_range, # I have added this 04/02/2021
            radius_max=radius_max, # I have added this 14/05/2021
            stellar_mass=stellar_mass,
            gravitational_constant=gravitational_constant,
            q_index=q_index,
            aspect_ratio=aspect_ratio,
            reference_radius=reference_radius,
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle,
            pressureless=pressureless,
            extra_args=extra_args,
        )

        self.add_particles(
            particle_type=particle_type,
            particle_mass=particle_mass,
            position=position,
            velocity=velocity,
            temperature=temperature, # I have added this 24/02/21
            smoothing_length=smoothing_length,
        )

    def _set_positions_mine(
        self,
        *,
        number_of_particles: float,
        T0: float,
        Tinf: float,
        R0: float,
        R0_temp: float,
        stellar_mass: float,
        disc_mass: float,
        density_distribution: Callable[[float], float],
        radius_range: Tuple[float, float],
        q_index: float,
        qfacdisc:float,
        my_temp_exp:float,
        p_index: float,
        aspect_ratio: float,
        reference_radius: float,
        hfact: float = 1.5,
        centre_of_mass: Tuple[float, float, float] = None,
        rotation_axis: Union[Tuple[float, float, float], ndarray] = None,
        rotation_angle: float = None,
        extra_args: tuple = None,
    ) -> Tuple[ndarray, ndarray]:
        """Set the disc particle positions.

        Parameters
        ----------
        number_of_particles
            The number of particles.
        disc_mass
            The total disc mass.
        density_distribution
            The surface density as a function of radius.
        radius_range
            The range of radii as (R_min, R_max).
        q_index
            The index in the sound speed power law such that
                H ~ (R / R_reference) ^ (3/2 - q).
        aspect_ratio
            The aspect ratio at the reference radius.
        reference_radius
            The radius at which the aspect ratio is given.

        Optional Parameters
        -------------------
        hfact
            The smoothing length factor. Default is 1.2, as for cubic.
        centre_of_mass
            The centre of mass of the disc, i.e. around which position
            is it rotating.
        rotation_axis
            An axis around which to rotate the disc.
        rotation_angle
            The angle to rotate around the rotation_axis.
        extra_args
            Extra arguments to pass to density_distribution.

        Returns
        -------
        position : ndarray
            The particle positions.
        smoothing_length : ndarray
            The particle smoothing lengths.
        """
        # TODO: I will need to change the eay I calculate the self gravtitating scale height so it avoids using temperature directly. Instead using the aspect ratio.

        if (rotation_axis is not None) ^ (rotation_angle is not None):
            raise ValueError(
                'Must specify rotation_angle and rotation_axis to perform rotation'
            )

        particle_mass = disc_mass / number_of_particles

        if rotation_axis is not None:
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation = spatial.transform.Rotation.from_rotvec(
                rotation_angle * rotation_axis
            )

        if centre_of_mass is None:
            centre_of_mass = (0.0, 0.0, 0.0)

        r_min = radius_range[0]
        r_max = radius_range[1]
        size = number_of_particles

        xi = np.sort(np.random.uniform(r_min, r_max, size))
        if extra_args is not None:
            p = density_distribution(xi, *extra_args)
        else:
            p = density_distribution(xi)
        p /= np.sum(p)


        p=p_index

        random_number_for_R = np.random.uniform(low=0, high=1, size=size)

        omega_in  = (r_min**2)/(R0**2)
        omega_out = (r_max**2)/(R0**2)
        omega = (((1.+omega_in)**(1.-(p/2.)))+(random.random())*((1.+omega_out)**(1.-(p/2.))-(1.+omega_in)**(1.-(p/2.))))**(2./(2.-p)) - 1.
        position_allocation= np.zeros(shape=(size,2))
        position_allocation[:,0] = (((1.+omega_in)**(1.-(p/2.)))+random_number_for_R*((1.+omega_out)**(1.-(p/2.))-(1.+omega_in)**(1.-(p/2.))))**(2./(2.-p)) - 1.
        position_allocation[:,1] = R0*np.sqrt(position_allocation[:,0])
        r = position_allocation[:,1]


        phi = np.random.rand(size) * 2 * np.pi
        AU = constants.au

        stellar_mass = 1 # HARDCODED

        temperature= np.sqrt(T0**2*((((r*AU)**2+(R0_temp*AU)**2)/(AU**2))**-my_temp_exp)+Tinf**2) # KELVIN
        cs = np.sqrt((constants.k_b*temperature)/(defaults._RUN_OPTIONS['mu']*constants.m_p)) # CM/S
        omega_mine = np.sqrt(constants.gravitational_constant * stellar_mass*constants.solarm / (r*constants.au)**3)
        H = (cs/omega_mine)/AU

        random_num = np.random.uniform(0, 1, size)

        sigma = density_distribution(r, *extra_args) # THIS IS IN SOLAR MASS PER AU SQUARED

        Q_toomre =(cs * omega_mine)/(np.pi * constants.gravitational_constant * sigma * ((constants.solarm)/(constants.au**2)) ) # CGS units
        SGG_H = (np.sqrt(np.pi/8) * (cs/omega_mine) * (np.sqrt((1/(Q_toomre**2))+(8/np.pi)) - (1/Q_toomre))/AU)

        z = (np.sqrt(2) * SGG_H * special.erfinv((2*random_num)-1)) # THIS IS IN AU
        position = np.array([r * np.cos(phi), r * np.sin(phi), z]).T

        rho_0 = (sigma/np.sqrt(2*np.pi)) * (1/(SGG_H)) # THIS IS IN SOLAR MASS PER AU CUBED
        density = rho_0  * np.exp(-(((z)**2)/(2*((SGG_H)**2))))# THIS IS IN SOLAR MASS PER AU CUBED

        smoothing_length = hfact * (particle_mass / density) ** (1 / 3)

        if rotation_axis is not None:
            position = rotation.apply(position)

        position += centre_of_mass

        return position, smoothing_length,temperature # I have edited this (added temperature) 24/02/21





    def _set_positions_original(
        self,
        *,
        T0: float,
        Tinf: float,
        R0: float,
        stellar_mass: float,
        number_of_particles: float,
        disc_mass: float,
        density_distribution: Callable[[float], float],
        radius_range: Tuple[float, float],
        q_index: float,
        aspect_ratio: float,
        reference_radius: float,
        hfact: float = 1.2,
        centre_of_mass: Tuple[float, float, float] = None,
        rotation_axis: Union[Tuple[float, float, float], ndarray] = None,
        rotation_angle: float = None,
        extra_args: tuple = None,
    ) -> Tuple[ndarray, ndarray]:
        """Set the disc particle positions.

        Parameters
        ----------
        number_of_particles
            The number of particles.
        disc_mass
            The total disc mass.
        density_distribution
            The surface density as a function of radius.
        radius_range
            The range of radii as (R_min, R_max).
        q_index
            The index in the sound speed power law such that
                H ~ (R / R_reference) ^ (3/2 - q).
        aspect_ratio
            The aspect ratio at the reference radius.
        reference_radius
            The radius at which the aspect ratio is given.

        Optional Parameters
        -------------------
        hfact
            The smoothing length factor. Default is 1.2, as for cubic.
        centre_of_mass
            The centre of mass of the disc, i.e. around which position
            is it rotating.
        rotation_axis
            An axis around which to rotate the disc.
        rotation_angle
            The angle to rotate around the rotation_axis.
        extra_args
            Extra arguments to pass to density_distribution.

        Returns
        -------
        position : ndarray
            The particle positions.
        smoothing_length : ndarray
            The particle smoothing lengths.
        """
        # TODO:
        # - set particle mass from disc mass, or toomre q, or something else
        # - add warps
        # - support for external forces
        # - add correction for self-gravity

        if (rotation_axis is not None) ^ (rotation_angle is not None):
            raise ValueError(
                'Must specify rotation_angle and rotation_axis to perform rotation'
            )

        particle_mass = disc_mass / number_of_particles

        if rotation_axis is not None:
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation = spatial.transform.Rotation.from_rotvec(
                rotation_angle * rotation_axis
            )

        if centre_of_mass is None:
            centre_of_mass = (0.0, 0.0, 0.0)

        r_min = radius_range[0]
        r_max = radius_range[1]
        size = number_of_particles

        xi = np.sort(np.random.uniform(r_min, r_max, size))
        if extra_args is not None:
            p = density_distribution(xi, *extra_args)
        else:
            p = density_distribution(xi)
        p /= np.sum(p)

        r = np.random.choice(xi, size=size, p=p)
        AU = constants.au

        phi = np.random.rand(size) * 2 * np.pi
        H = (
            reference_radius ** (q_index - 1 / 2)
            * aspect_ratio
            * r ** (3 / 2 - q_index)
        )
        random_num = np.random.uniform(0, 1, size)
        z = np.random.normal(scale=H)


        position = np.array([r * np.cos(phi), r * np.sin(phi), z]).T

        integrated_mass = integrate.quad(
            lambda x: 2 * np.pi * x * density_distribution(x, *extra_args),
            radius_range[0],
            radius_range[1],
        )[0]

        normalization = disc_mass / integrated_mass
        sigma = normalization * density_distribution(r, *extra_args)

        density = (sigma) * np.exp(-0.5 * (z / H) ** 2) / (H * np.sqrt(2 * np.pi))



        smoothing_length = hfact * (particle_mass / density) ** (1 / 3)

        if rotation_axis is not None:
            position = rotation.apply(position)

        position += centre_of_mass

        return position, smoothing_length, temperature # I have edited this (added temperature) 24/02/21

    def _set_velocities(
        self,
        *,
        number_of_particles: float,
        disc_mass: float,
        radius_max: float,
        position: ndarray,
        stellar_mass: float,
        gravitational_constant: float,
        density_distribution: Callable[[float], float],
        q_index: float,
        aspect_ratio: float,
        reference_radius: float,
        radius_range: Tuple[float, float],
        rotation_axis: Union[Tuple[float, float, float], ndarray] = None,
        rotation_angle: float = None,
        pressureless: bool = False,
        extra_args: tuple = None,
    ) -> ndarray:
        """Set the disc particle velocities.

        Parameters
        ----------
        position
            The particle positions.
        stellar_mass
            The mass of the central object the disc is orbiting.
        gravitational_constant
            The gravitational constant.
        q_index
            The index in the sound speed power law such that
                H ~ (R / R_reference) ^ (3/2 - q).
        aspect_ratio
            The aspect ratio at the reference radius.
        reference_radius
            The radius at which the aspect ratio is given.

        Optional Parameters
        -------------------
        rotation_axis
            An axis around which to rotate the disc.
        rotation_angle
            The angle to rotate around the rotation_axis.
        pressureless
            Set to True if the particles are pressureless, i.e. dust.

        Returns
        -------
        velocity : ndarray
            The particle velocities.
        """
        if (rotation_axis is not None) ^ (rotation_angle is not None):
            raise ValueError(
                'Must specify rotation_angle and rotation_axis to perform rotation'
            )

        if rotation_axis is not None:
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation = spatial.transform.Rotation.from_rotvec(
                rotation_angle * rotation_axis
            )
        radius = np.sqrt(position[:, 0] ** 2 + position[:, 1] ** 2)
        phi = np.arctan2(position[:, 1], position[:, 0])
        # phi = [random.uniform(-np.pi, np.pi) for _ in range(100)] # I have added this 08/03/2021

        n_bins = 100

        particle_mass = disc_mass / number_of_particles

        hist = np.histogram(radius, bins = n_bins,range=(1,radius_max)) # this 100 needs changing to r_out (not hard coded)



        mass_in_bin = hist[0] * particle_mass
        cumulative_mass = np.cumsum(mass_in_bin)
        cumulative_mass = np.insert(cumulative_mass, 0, 0., axis=0) # This makes sure that there is no mass contained between R = 0 and R = R_in (also makes sure the arrays are the same length)
        velocity = np.sqrt(gravitational_constant * (stellar_mass+cumulative_mass) / hist[1])

        where_are_particles = np.digitize(radius,hist[1])
        ## Οkay up to here
        in_1 = where_are_particles - 1
        in_2 = where_are_particles
        r_1 = hist[1][in_1]
        r_2 = hist[1][in_2]
        v_1 = velocity[in_1]
        v_2 = velocity[in_2]


        omega = v_1 + ((v_1-v_2)/(r_1-r_2))*(radius-r_1)


        if not pressureless:
            h_over_r = aspect_ratio * (radius / reference_radius) ** (1 / 2 - q_index) # I have added this 08/03/2021
            # h_over_r = aspect_ratio * (radius / reference_radius) ** (1 / 2 - q_index)
            v_phi = omega * np.sqrt(1 - h_over_r ** 2)
        else:
            v_phi = omega


        v_z = np.zeros_like(radius) # I have added this 08/03/2021


        velocity = np.array([-v_phi * np.sin(phi), v_phi * np.cos(phi), v_z]).T


        if rotation_axis is not None:
            velocity = rotation.apply(velocity)

        return velocity


def smoothing_length_on_scale_height(
    radius: ndarray,
    smoothing_length: ndarray,
    reference_radius: float,
    aspect_ratio: float,
    q_index: float,
    sample_number: int = None,
):
    """Calculate the average smoothing length on scale height.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    if sample_number is None:
        bins = 10
    else:
        bins = sample_number

    binned_rh = stats.binned_statistic(radius, smoothing_length, bins=bins)

    r = binned_rh.bin_edges
    r = r[:-1] + (r[1:] - r[:-1]) / 2

    h = binned_rh.statistic

    H = reference_radius ** (q_index - 1 / 2) * aspect_ratio * r ** (3 / 2 - q_index)

    return h / H


def keplerian_angular_velocity(
    radius: Union[float, ndarray],
    mass: Union[float, ndarray],
    gravitational_constant: float = None,
) -> Union[float, ndarray]:
    """Keplerian angular velocity Omega.

    Parameters
    ----------
    radius
        The distance from the central object.
    mass
        The central object mass.
    gravitational_constant
        The gravitational constant in appropriate units.
    """
    if gravitational_constant is None:
        gravitational_constant = constants.gravitational_constant
    return np.sqrt(gravitational_constant * mass / radius ** 3)


def add_gap(
    orbital_radius: float, gap_width: float
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Decorate by adding a gap in a density distribution.

    The gap is a step function. I.e. the density is zero within the gap.

    Parameters
    ----------
    radius_planet
        The planet radius.
    gap_width
        The gap width centered on the planet.

    Returns
    -------
    callable
        The density distribution with the added gap.
    """

    def wrapper_outer(distribution):
        @functools.wraps(distribution)
        def wrapper_inner(radius, *extra_args):

            result = distribution(radius, *extra_args)
            gap_radii = np.logical_and(
                orbital_radius - 0.5 * gap_width < radius,
                radius < orbital_radius + 0.5 * gap_width,
            )

            if isinstance(result, ndarray):
                result[gap_radii] = 0.0
            elif gap_radii:
                result = 0.0

            return result

        return wrapper_inner

    return wrapper_outer


def power_law(
    radius: Union[float, ndarray], reference_radius: float, p_index: float
) -> Union[float, ndarray]:
    """Power law distribution.

    (R / R_ref)^(-p)

    Parameters
    ----------
    radius
        The radius at which to evaulate the function.
    reference_radius
        The reference radius
    p_index
        The exponent in the power law.

    Returns
    -------
    float
        The surface density at the specified radius.
    """

    ref, p = reference_radius, p_index
    return radius * (radius / ref) ** (-p)


def power_law_with_zero_inner_boundary(
    radius: Union[float, ndarray],
    inner_radius: float,
    reference_radius: float,
    p_index: float,
) -> Union[float, ndarray]:
    """Power law distribution with zero inner boundary condition.

    (R / R_ref)^(-p) * [1 - sqrt(R_inner / R)]

    Parameters
    ----------
    radius
        The radius at which to evaulate the function.
    inner_radius
        The inner radius.
    reference_radius
        The reference radius.
    p_index
        The exponent in the power law.

    Returns
    -------
    float
        The surface density at the specified radius.
    """
    ref, inner, p = reference_radius, inner_radius, p_index
    return (radius / ref) ** (-p) * (1 - np.sqrt(inner / radius))


def self_similar_accretion_disc(
    radius: Union[float, ndarray], radius_critical: float, gamma: float
) -> Union[float, ndarray]:
    """Lynden-Bell and Pringle (1974) self-similar solution.

    (R / R_crit)^(-y) * exp[-(R / R_crit) ^ (2 - y)]

    Parameters
    ----------
    radius
        The radius at which to evaulate the function.
    radius_critical
        The critical radius for the exponential taper.
    gamma
        The exponent in the power law.

    Returns
    -------
    float
        The surface density at the specified radius.
    """
    rc, y = radius_critical, gamma

    return (radius / rc) ** (-y) * np.exp(-((radius / rc) ** (2 - y)))
    # return  radius **(-y) * (1-np.sqrt(0.25/radius))




def my_surface_density(
    radius: Union[float, ndarray], p_index: float, disc_mass: float,R0: float,radius_min: float,radius_max: float,
) -> Union[float, ndarray]:

    p,M_disc  = p_index, disc_mass
    sigma_0  = (((2-p)/(2*np.pi*R0**2))*(M_disc))/(((R0**2+radius_max**2)/(R0**2))**(1-(p/2))-((R0**2+radius_min**2)/(R0**2))**(1-(p/2))) # CGS units
    # a = (sigma_0*((R0**2/(R0**2+(radius)**2)))**(p/2))
    return (sigma_0*((R0**2/(R0**2+(radius)**2)))**(p/2))
    # print (sigma_0*((R0**2/(R0**2+(radius)**2)))**(p/2))




def get_sigma_0(
    radius: Union[float, ndarray], p_index: float, disc_mass: float,R0: float,radius_min: float,radius_max: float,
) -> Union[float, ndarray]:

    p,M_disc  = p_index, disc_mass

    return ((((2-p)/(2*np.pi*R0**2))*M_disc)/(((R0**2+radius_max**2)/(R0**2))**(1-(p/2))-((R0**2+radius_min**2)/(R0**2))**(1-(p/2))))


def self_similar_accretion_disc_with_zero_inner_boundary(
    radius: Union[float, ndarray],
    radius_inner: float,
    radius_critical: float,
    gamma: float,
) -> Union[float, ndarray]:
    """Self-similar solution with a zero inner boundary condition.

    (R / R_crit)^(-y) * exp[-(R / R_crit) ^ (2 - y)]

    Parameters
    ----------
    radius
        The radius at which to evaulate the function.
    inner_radius
        The inner radius.
    radius_critical
        The critical radius for the exponential taper.
    gamma
        The exponent in the power law.

    Returns
    -------
    float
        The surface density at the specified radius.
    """
    inner, rc, y = radius_inner, radius_critical, gamma
    return (
        (radius / rc) ** (-y)
        * np.exp(-((radius / rc) ** (2 - y)))
        * (1 - np.sqrt(inner / radius))
    )
