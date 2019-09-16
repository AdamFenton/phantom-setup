"""
Setup an accretion disc around a sink particle.
"""

import matplotlib.pyplot as plt
import numpy as np
import phantomsetup
from phantomsetup import defaults
from phantomsetup.disc import accretion_disc_self_similar, add_gap
from phantomsetup.eos import polyk_for_locally_isothermal_disc
from phantomsetup.orbits import hill_sphere_radius
from phantomsetup.units import unit_string_to_cgs

# ------------------------------------------------------------------------------------ #
# Script options

plot_setup = True

# ------------------------------------------------------------------------------------ #
# Constants

igas = defaults.particle_type['igas']

# ------------------------------------------------------------------------------------ #
# Parameters

prefix = 'disc'

number_of_particles = 1_000_000
particle_type = igas

alpha_artificial = 0.1

length_unit = unit_string_to_cgs('au')
mass_unit = unit_string_to_cgs('solarm')

radius_min = 10.0
radius_max = 200.0

disc_mass = 0.01

gravitational_constant = 1.0
stellar_mass = 1.0
stellar_accretion_radius = 5.0
stellar_position = (0.0, 0.0, 0.0)
stellar_velocity = (0.0, 0.0, 0.0)

ieos = 3
q_index = 0.75
aspect_ratio = 0.05
reference_radius = 10.0

radius_critical = 100.0
gamma = 1.0

planet_mass = 0.001
planet_position = (100.0, 0.0, 0.0)
planet_accretion_radius_fraction_hill_radius = 0.25

orbital_radius = np.sqrt(planet_position[0] ** 2 + planet_position[1] ** 2)
planet_hill_radius = hill_sphere_radius(orbital_radius, planet_mass, stellar_mass)
planet_accretion_radius = (
    planet_accretion_radius_fraction_hill_radius * planet_hill_radius
)
gap_width = planet_hill_radius
planet_velocity = np.sqrt(gravitational_constant * stellar_mass / orbital_radius)


# ------------------------------------------------------------------------------------ #
# Surface density distribution

# We use the Lynden-Bell and Pringle (1974) self-similar solution, i.e.
# a power law with an exponential taper. Plus we add a gap, as a step
# function at the planet location.


@add_gap(orbital_radius=orbital_radius, gap_width=gap_width)
def density_distribution(radius, radius_critical, gamma):
    """Surface density distribution.

    Self-similar disc solution with a gap added.
    """
    return accretion_disc_self_similar(radius, radius_critical, gamma)


args = (radius_critical, gamma)

# ------------------------------------------------------------------------------------ #
# Instantiate Setup object

setup = phantomsetup.Setup()
setup.prefix = prefix

# ------------------------------------------------------------------------------------ #
# Set units

setup.set_units(
    length=length_unit, mass=mass_unit, gravitational_constant_is_unity=True
)

# ------------------------------------------------------------------------------------ #
# Set equation of state

polyk = polyk_for_locally_isothermal_disc(
    q_index, reference_radius, aspect_ratio, stellar_mass, gravitational_constant
)

setup.set_equation_of_state(ieos=ieos, polyk=polyk)

# ------------------------------------------------------------------------------------ #
# Set viscosity to disc viscosity

setup.set_dissipation(disc_viscosity=True, alpha=alpha_artificial)

# ------------------------------------------------------------------------------------ #
# Add star

setup.add_sink(
    mass=stellar_mass,
    accretion_radius=stellar_accretion_radius,
    position=stellar_position,
    velocity=stellar_velocity,
)

# ------------------------------------------------------------------------------------ #
# Add disc

setup.add_disc(
    particle_type=particle_type,
    number_of_particles=number_of_particles,
    disc_mass=disc_mass,
    density_distribution=density_distribution,
    radius_range=(radius_min, radius_max),
    q_index=q_index,
    aspect_ratio=aspect_ratio,
    reference_radius=reference_radius,
    stellar_mass=stellar_mass,
    gravitational_constant=gravitational_constant,
    args=(radius_critical, gamma),
)

# ------------------------------------------------------------------------------------ #
# Add planet

setup.add_sink(
    mass=planet_mass,
    accretion_radius=planet_accretion_radius,
    position=planet_position,
    velocity=planet_velocity,
)

# ------------------------------------------------------------------------------------ #
# Write dump file and in file

setup.write_dump_file()
setup.write_in_file()

# ------------------------------------------------------------------------------------ #
# Plot some quantities

if plot_setup:

    fig, ax = plt.subplots()
    ax.plot(setup.x[::10], setup.y[::10], 'k.', ms=0.5)
    for sink in setup.sinks:
        ax.plot(sink.position[0], sink.position[1], 'ro')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')

    fig, ax = plt.subplots()
    ax.plot(setup.R[::10], setup.z[::10], 'k.', ms=0.5)
    ax.set_xlabel('$R$')
    ax.set_ylabel('$z$')
    ax.set_aspect('equal')
    ax.set_ylim(bottom=2 * setup.z.min(), top=2 * setup.z.max())

    fig, ax = plt.subplots()
    ax.plot(setup.R[::10], setup.vphi[::10], 'k.', ms=0.5)
    ax.set_xlabel('$R$')
    ax.set_ylabel('$v_{phi}$')

    plt.show()
