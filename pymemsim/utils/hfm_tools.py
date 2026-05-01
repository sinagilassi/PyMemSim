# import libs
import logging
from typing import List, Tuple
import numpy as np
from pythermodb_settings.models import CustomProp
import pycuc
# locals
from .unit_tools import to_m

# NOTE: logger setup
logger = logging.getLogger(__name__)

# SECTION: calculate membrane surface area per unit length


def calculate_surface_area_per_unit_length(
        number_of_fibers: CustomProp,
        fiber_outer_diameter: CustomProp,
        output_unit: str = 'm'
) -> CustomProp:
    """
    Calculate the surface area per unit length of a hollow fiber membrane module.

    Parameters
    ----------
    number_of_fibers : CustomProp
        The number of fibers in the membrane module.
    fiber_outer_diameter : CustomProp
        The outer diameter of the fibers in the membrane module.
    output_unit : str, optional
        The desired output unit for the surface area per unit length (default is 'm').

    Returns
    -------
    CustomProp
        The surface area per unit length of the membrane module, with value in default unit m2/m.
    """
    try:
        # convert fiber outer diameter to meters
        fiber_outer_diameter_m = to_m(
            fiber_outer_diameter.value,
            fiber_outer_diameter.unit
        )

        # calculate surface area per unit length
        # ! m2/m = number of fibers * pi * fiber outer diameter (m)
        res = number_of_fibers.value * np.pi * fiber_outer_diameter_m

        # convert surface area per unit length to m2/m
        if output_unit == 'm':
            surface_area_per_unit_length = CustomProp(
                value=res,
                unit='m'
            )
        else:
            surface_area_per_unit_length = CustomProp(
                value=pycuc.convert_from_to(res, 'm', output_unit),
                unit=output_unit
            )

        return surface_area_per_unit_length
    except Exception as e:
        logger.error(f"Error calculating surface area per unit length: {e}")
        raise

# SECTION: calculate total membrane surface area


def calculate_total_surface_area(
        number_of_fibers: CustomProp,
        fiber_outer_diameter: CustomProp,
        membrane_length: CustomProp,
        output_unit: str = 'm2'
) -> CustomProp:
    """
    Calculate the total surface area of a hollow fiber membrane module.

    Parameters
    ----------
    number_of_fibers : CustomProp
        The number of fibers in the membrane module.
    fiber_outer_diameter : CustomProp
        The outer diameter of the fibers in the membrane module.
    membrane_length : CustomProp
        The length of the membrane fibers in the module.
    output_unit : str, optional
        The desired output unit for the total surface area (default is 'm2').

    Returns
    -------
    CustomProp
        The total surface area of the membrane module, with value in default unit m2.
    """
    try:
        # calculate surface area per unit length
        # ! m2/m
        surface_area_per_unit_length = calculate_surface_area_per_unit_length(
            number_of_fibers,
            fiber_outer_diameter,
            'm'
        )

        # convert membrane length to meters
        # ! m
        membrane_length_m = to_m(
            membrane_length.value,
            membrane_length.unit
        )

        # calculate total surface area
        # ! m2 = surface area per unit length (m2/m) * membrane length (m)
        total_surface_area = surface_area_per_unit_length.value * membrane_length_m

        # convert total surface area to m2
        if output_unit == 'm2':
            total_surface_area_prop = CustomProp(
                value=total_surface_area,
                unit='m2'
            )
        else:
            total_surface_area_prop = CustomProp(
                value=pycuc.convert_from_to(
                    total_surface_area,
                    'm2',
                    output_unit
                ),
                unit=output_unit
            )

        return total_surface_area_prop
    except Exception as e:
        logger.error(f"Error calculating total surface area: {e}")
        raise


# SECTION: calculate packing fraction
def calculate_packing_fraction(
        number_of_fibers: CustomProp,
        fiber_outer_diameter: CustomProp,
        module_inner_diameter: CustomProp,
) -> CustomProp:
    """
    Calculate packing fraction of hollow fiber module.

    phi = N * do^2 / dm^2

    Returns dimensionless CustomProp.

    Parameters
    ----------
    number_of_fibers : CustomProp
        Number of fibers in the module.
    fiber_outer_diameter : CustomProp
        Outer diameter of the fibers.
    module_inner_diameter : CustomProp
        Inner diameter of the module.

    Returns
    -------
    CustomProp
        Packing fraction of the module, dimensionless.
    """
    try:
        do_m = to_m(fiber_outer_diameter.value, fiber_outer_diameter.unit)
        dm_m = to_m(module_inner_diameter.value, module_inner_diameter.unit)

        if dm_m <= 0:
            raise ValueError("module_inner_diameter must be positive.")

        phi = number_of_fibers.value * do_m**2 / dm_m**2

        return CustomProp(value=phi, unit='-')

    except Exception as e:
        logger.error(f"Error calculating packing fraction: {e}")
        raise

# SECTION: calculate porosity


def calculate_porosity(
        number_of_fibers: CustomProp,
        fiber_outer_diameter: CustomProp,
        module_inner_diameter: CustomProp,
) -> CustomProp:
    """
    Calculate module porosity.

    epsilon = 1 - phi

    where phi is the packing fraction calculated by calculate_packing_fraction().

    Parameters
    ----------
    number_of_fibers : CustomProp
        Number of fibers in the module.
    fiber_outer_diameter : CustomProp
        Outer diameter of the fibers.
    module_inner_diameter : CustomProp
        Inner diameter of the module.

    Returns
    -------
    CustomProp
        Porosity of the module, dimensionless.
    """
    try:
        phi = calculate_packing_fraction(
            number_of_fibers,
            fiber_outer_diameter,
            module_inner_diameter
        )

        epsilon = 1.0 - phi.value

        return CustomProp(value=epsilon, unit='-')

    except Exception as e:
        logger.error(f"Error calculating porosity: {e}")
        raise

# SECTION: calculate module cross-sectional area


def calculate_module_cross_section_area(
        module_inner_diameter: CustomProp,
        output_unit: str = 'm2'
) -> CustomProp:
    """
    Calculate module cross-sectional area.

    A = pi * dm^2 / 4

    Parameters
    ----------
    module_inner_diameter : CustomProp
        Inner diameter of the module.
    output_unit : str, optional
        Desired output unit for the cross-sectional area (default is 'm2').

    Returns
    -------
    CustomProp
        Cross-sectional area of the module, in the specified output unit.
    """
    try:
        dm_m = to_m(module_inner_diameter.value, module_inner_diameter.unit)

        area_m2 = np.pi * dm_m**2 / 4.0

        if output_unit in ['m2', 'm^2']:
            return CustomProp(value=area_m2, unit='m2')

        return CustomProp(
            value=pycuc.convert_from_to(area_m2, 'm2', output_unit),
            unit=output_unit
        )

    except Exception as e:
        logger.error(f"Error calculating module cross-sectional area: {e}")
        raise

# SECTION: calculate total fiber cross-sectional area


def calculate_total_fiber_cross_section_area(
        number_of_fibers: CustomProp,
        fiber_outer_diameter: CustomProp,
        output_unit: str = 'm2'
) -> CustomProp:
    """
    Calculate total fiber cross-sectional area.

    Af = N * pi * do^2 / 4

    Parameters
    ----------
    number_of_fibers : CustomProp
        Number of fibers in the module.
    fiber_outer_diameter : CustomProp
        Outer diameter of the fibers.
    output_unit : str, optional
        Desired output unit for the total fiber cross-sectional area (default is 'm2').

    Returns
    -------
    CustomProp
        Total fiber cross-sectional area, in the specified output unit.
    """
    try:
        do_m = to_m(fiber_outer_diameter.value, fiber_outer_diameter.unit)

        area_m2 = number_of_fibers.value * np.pi * do_m**2 / 4.0

        if output_unit in ['m2', 'm^2']:
            return CustomProp(value=area_m2, unit='m2')

        return CustomProp(
            value=pycuc.convert_from_to(area_m2, 'm2', output_unit),
            unit=output_unit
        )

    except Exception as e:
        logger.error(
            f"Error calculating total fiber cross-sectional area: {e}")
        raise

# SECTION: calculate lumen cross-sectional area


def calculate_lumen_cross_section_area(
        number_of_fibers: CustomProp,
        fiber_inner_diameter: CustomProp,
        output_unit: str = 'm2'
) -> CustomProp:
    """
    Total lumen-side flow area.

    A_lumen = N * pi * di^2 / 4

    Parameters
    ----------
    number_of_fibers : CustomProp
        Number of fibers in the module.
    fiber_inner_diameter : CustomProp
        Inner diameter of the fibers.
    output_unit : str, optional
        Desired output unit for the lumen cross-sectional area (default is 'm2').

    Returns
    -------
    CustomProp
        Total lumen cross-sectional area, in the specified output unit.
    """
    try:
        di_m = to_m(fiber_inner_diameter.value, fiber_inner_diameter.unit)

        area_m2 = number_of_fibers.value * np.pi * di_m**2 / 4.0

        if output_unit in ['m2', 'm^2']:
            return CustomProp(value=area_m2, unit='m2')

        return CustomProp(
            value=pycuc.convert_from_to(area_m2, 'm2', output_unit),
            unit=output_unit
        )

    except Exception as e:
        logger.error(f"Error calculating lumen cross-sectional area: {e}")
        raise

# SECTION: calculate module volume


def calculate_module_volume(
        module_inner_diameter: CustomProp,
        membrane_length: CustomProp,
        output_unit: str = 'm3'
) -> CustomProp:
    """
    Calculate cylindrical module volume.

    V = pi * dm^2 / 4 * L

    Parameters
    ----------
    module_inner_diameter : CustomProp
        Inner diameter of the module.
    membrane_length : CustomProp
        Length of the membrane fibers in the module.
    output_unit : str, optional
        Desired output unit for the module volume (default is 'm3').

    Returns
    -------
    CustomProp
        Volume of the module, in the specified output unit.
    """
    try:
        area = calculate_module_cross_section_area(
            module_inner_diameter,
            output_unit='m2'
        )

        length_m = to_m(membrane_length.value, membrane_length.unit)

        volume_m3 = area.value * length_m

        if output_unit in ['m3', 'm^3']:
            return CustomProp(value=volume_m3, unit='m3')

        return CustomProp(
            value=pycuc.convert_from_to(volume_m3, 'm3', output_unit),
            unit=output_unit
        )

    except Exception as e:
        logger.error(f"Error calculating module volume: {e}")
        raise

# SECTION: calculate shell free cross-sectional area


def calculate_shell_free_cross_section_area(
        number_of_fibers: CustomProp,
        fiber_outer_diameter: CustomProp,
        module_inner_diameter: CustomProp,
        output_unit: str = 'm2'
) -> CustomProp:
    """
    Calculate free shell-side cross-sectional area.

    As = A module - A fibers

    Parameters
    ----------
    number_of_fibers : CustomProp
        Number of fibers in the module.
    fiber_outer_diameter : CustomProp
        Outer diameter of the fibers.
    module_inner_diameter : CustomProp
        Inner diameter of the module.
    output_unit : str, optional
        Desired output unit for the shell free cross-sectional area (default is 'm2').

    Returns
    -------
    CustomProp
        Free shell-side cross-sectional area, in the specified output unit.
    """
    try:
        module_area = calculate_module_cross_section_area(
            module_inner_diameter,
            output_unit='m2'
        )

        fiber_area = calculate_total_fiber_cross_section_area(
            number_of_fibers,
            fiber_outer_diameter,
            output_unit='m2'
        )

        shell_area_m2 = module_area.value - fiber_area.value

        if output_unit in ['m2', 'm^2']:
            return CustomProp(value=shell_area_m2, unit='m2')

        return CustomProp(
            value=pycuc.convert_from_to(shell_area_m2, 'm2', output_unit),
            unit=output_unit
        )

    except Exception as e:
        logger.error(f"Error calculating shell free cross-sectional area: {e}")
        raise

# SECTION: Calculate flow rate from velocity and cross-sectional area


def calculate_flow_rate_from_velocity(
        cross_section_area: CustomProp,
        velocity: CustomProp,
        output_unit: str = 'm3/s'
) -> CustomProp:
    """
    Calculate volumetric flow rate from velocity and cross-sectional area.

    Q = u * A

    Parameters
    ----------
    cross_section_area : CustomProp
        Cross-sectional area of the flow channel.
    velocity : CustomProp
        Fluid velocity.
    output_unit : str, optional
        Desired output unit for the flow rate (default is 'm3/s').

    Returns
    -------
    CustomProp
        Volumetric flow rate, in the specified output unit.
    """
    try:
        area_m2 = pycuc.convert_from_to(
            cross_section_area.value,
            cross_section_area.unit,
            'm2'
        )

        velocity_m_s = pycuc.convert_from_to(
            velocity.value,
            velocity.unit,
            'm/s'
        )

        q_m3_s = velocity_m_s * area_m2

        if output_unit == 'm3/s':
            return CustomProp(value=q_m3_s, unit='m3/s')

        return CustomProp(
            value=pycuc.convert_from_to(q_m3_s, 'm3/s', output_unit),
            unit=output_unit
        )

    except Exception as e:
        logger.error(f"Error calculating flow rate from velocity: {e}")
        raise


def calculate_molar_flow_rate_ideal_gas(
        volumetric_flow_rate: CustomProp,
        pressure: CustomProp,
        temperature: CustomProp,
        universal_gas_constant: float = 8.314,  # J/mol.K
        output_unit: str = 'mol/s'
) -> CustomProp:
    """
    Calculate molar flow rate from volumetric flow rate using the ideal gas law.

    F = P Q / R T

    Parameters
    ----------
    volumetric_flow_rate : CustomProp
        Volumetric flow rate of the gas.
    pressure : CustomProp
        Absolute pressure of the gas.
    temperature : CustomProp
        Absolute temperature of the gas.
    universal_gas_constant : float, optional
        Universal gas constant in J/mol.K (default is 8.314).
    output_unit : str, optional
        Desired output unit for the molar flow rate (default is 'mol/s').

    Returns
    -------
    CustomProp
        Molar flow rate, in the specified output unit.
    """
    try:
        q_m3_s = pycuc.convert_from_to(
            volumetric_flow_rate.value,
            volumetric_flow_rate.unit,
            'm3/s'
        )

        p_pa = pycuc.convert_from_to(
            pressure.value,
            pressure.unit,
            'Pa'
        )

        t_k = pycuc.convert_from_to(
            temperature.value,
            temperature.unit,
            'K'
        )

        R = universal_gas_constant  # J/mol.K

        f_mol_s = p_pa * q_m3_s / (R * t_k)

        if output_unit == 'mol/s':
            return CustomProp(value=f_mol_s, unit='mol/s')

        return CustomProp(
            value=pycuc.convert_from_to(f_mol_s, 'mol/s', output_unit),
            unit=output_unit
        )

    except Exception as e:
        logger.error(f"Error calculating molar flow rate: {e}")
        raise


def calculate_volumetric_flow_rate_ideal_gas(
        molar_flow_rate: CustomProp,
        pressure: CustomProp,
        temperature: CustomProp,
        universal_gas_constant: float = 8.314,  # J/mol.K
        output_unit: str = 'm3/s'
) -> CustomProp:
    """
    Calculate volumetric flow rate from molar flow rate using the ideal gas law.

    Q = F R T / P

    Parameters
    ----------
    molar_flow_rate : CustomProp
        Molar flow rate of the gas.
    pressure : CustomProp
        Absolute pressure of the gas.
    temperature : CustomProp
        Absolute temperature of the gas.
    universal_gas_constant : float, optional
        Universal gas constant in J/mol.K (default is 8.314).
    output_unit : str, optional
        Desired output unit for the volumetric flow rate (default is 'm3/s').

    Returns
    -------
    CustomProp
        Volumetric flow rate, in the specified output unit.
    """
    try:
        f_mol_s = pycuc.convert_from_to(
            molar_flow_rate.value,
            molar_flow_rate.unit,
            'mol/s'
        )

        p_pa = pycuc.convert_from_to(
            pressure.value,
            pressure.unit,
            'Pa'
        )

        t_k = pycuc.convert_from_to(
            temperature.value,
            temperature.unit,
            'K'
        )

        R = universal_gas_constant  # J/mol.K

        q_m3_s = f_mol_s * R * t_k / p_pa

        if output_unit == 'm3/s':
            return CustomProp(value=q_m3_s, unit='m3/s')

        return CustomProp(
            value=pycuc.convert_from_to(q_m3_s, 'm3/s', output_unit),
            unit=output_unit
        )

    except Exception as e:
        logger.error(f"Error calculating volumetric flow rate: {e}")
        raise


def calculate_laminar_lumen_qmax_from_pressure_drop(
        number_of_fibers: CustomProp,
        fiber_inner_diameter: CustomProp,
        membrane_length: CustomProp,
        viscosity: CustomProp,
        max_pressure_drop: CustomProp,
        output_unit: str = 'm3/s'
) -> CustomProp:
    """
    Calculate maximum lumen-side flow rate from pressure drop using the Hagen-Poiseuille equation.

    Q_total = ΔP * pi * di^4 * N / (128 * mu * L)

    Parameters
    ----------
    number_of_fibers : CustomProp
        Number of fibers in the module.
    fiber_inner_diameter : CustomProp
        Inner diameter of the fibers.
    membrane_length : CustomProp
        Length of the membrane fibers.
    viscosity : CustomProp
        Dynamic viscosity of the fluid.
    max_pressure_drop : CustomProp
        Maximum allowable pressure drop along the fiber length.
    output_unit : str, optional
        Desired output unit for the flow rate (default is 'm3/s').

    Returns
    -------
    CustomProp
        Maximum allowable total volumetric flow rate, in the specified output unit.
    """
    try:
        di_m = to_m(fiber_inner_diameter.value, fiber_inner_diameter.unit)
        length_m = to_m(membrane_length.value, membrane_length.unit)

        mu_pa_s = pycuc.convert_from_to(
            viscosity.value,
            viscosity.unit,
            'Pa.s'
        )

        dp_pa = pycuc.convert_from_to(
            max_pressure_drop.value,
            max_pressure_drop.unit,
            'Pa'
        )

        q_m3_s = (
            dp_pa
            * np.pi
            * di_m**4
            * number_of_fibers.value
            / (128.0 * mu_pa_s * length_m)
        )

        if output_unit == 'm3/s':
            return CustomProp(value=q_m3_s, unit='m3/s')

        return CustomProp(
            value=pycuc.convert_from_to(q_m3_s, 'm3/s', output_unit),
            unit=output_unit
        )

    except Exception as e:
        logger.error(f"Error calculating lumen qmax from pressure drop: {e}")
        raise


def estimate_membrane_permeation_capacity(
        number_of_fibers: CustomProp,
        fiber_outer_diameter: CustomProp,
        membrane_length: CustomProp,
        feed_pressure: CustomProp,
        permeate_pressure: CustomProp,
        permeance: dict[str, CustomProp],
        feed_mole_fraction: dict,
) -> Tuple[CustomProp, dict]:
    """
    Rough screening estimate:

    N_perm_i = Pi * A_m * Δp_i

    where:
        Pi      : mol/m2/s/Pa
        A_m     : m2
        Δp_i    : y_feed_i * P_feed - 0 * P_perm

    This is intentionally simple and conservative for validation.

    Parameters
    ----------
    number_of_fibers : CustomProp
        Number of fibers in the module.
    fiber_outer_diameter : CustomProp
        Outer diameter of the fibers.
    membrane_length : CustomProp
        Length of the membrane fibers.
    feed_pressure : CustomProp
        Pressure on the feed side.
    permeate_pressure : CustomProp
        Pressure on the permeate side.
    permeance : dict[str, CustomProp]
        Dictionary of permeance values for each component, in units of mol/s.m2.Pa or GPU.
    feed_mole_fraction : dict
        Dictionary of mole fractions for each component in the feed.
    """
    try:
        area = calculate_total_surface_area(
            number_of_fibers=number_of_fibers,
            fiber_outer_diameter=fiber_outer_diameter,
            membrane_length=membrane_length,
            output_unit='m'
        )

        # In your current file, total_surface_area uses unit='m'
        # although physically it is m2.
        area_m2 = area.value

        pf_pa = pycuc.convert_from_to(
            feed_pressure.value,
            feed_pressure.unit,
            'Pa'
        )

        pp_pa = pycuc.convert_from_to(
            permeate_pressure.value,
            permeate_pressure.unit,
            'Pa'
        )

        capacity = {}

        for comp, Pi in permeance.items():
            # value
            Pi_value = Pi.value
            # unit
            Pi_unit = Pi.unit.strip()

            # check units
            if Pi_unit.lower() not in ['mol/s.m2.pa', 'gpu']:
                raise ValueError(
                    f"Unsupported permeance unit for component {comp}: {Pi_unit}. Supported units are 'mol/s.m2.Pa' and 'GPU'.")

            # convert GPU to mol/s.m2.Pa if necessary
            if Pi_unit.lower() == 'gpu':
                # 1 GPU = 3.35e-10 mol/s.m2.Pa
                Pi_value = Pi_value * 3.35e-10

            yi = feed_mole_fraction.get(comp, 0.0)

            driving_force = max(yi * pf_pa - 0.0 * pp_pa, 0.0)

            capacity[comp] = CustomProp(
                value=Pi_value * area_m2 * driving_force,
                unit='mol/s'
            )

        total_capacity = sum(v.value for v in capacity.values())

        return CustomProp(value=total_capacity, unit='mol/s'), capacity

    except Exception as e:
        logger.error(f"Error estimating membrane permeation capacity: {e}")
        raise


def calculate_hfm_feed_flow_rate_bounds(
        number_of_fibers: CustomProp,
        fiber_inner_diameter: CustomProp,
        fiber_outer_diameter: CustomProp,
        membrane_length: CustomProp,
        feed_pressure: CustomProp,
        feed_temperature: CustomProp,
        permeate_pressure: CustomProp,
        viscosity: CustomProp,
        permeance: dict,
        feed_mole_fraction: dict,
        velocity_min: CustomProp = CustomProp(value=0.01, unit='m/s'),
        velocity_max: CustomProp = CustomProp(value=10.0, unit='m/s'),
        max_pressure_drop: CustomProp = CustomProp(value=20_000.0, unit='Pa'),
        theta_max: float = 0.8,
) -> dict:
    """
    Recommended feed-flow bounds for HFM.

    Lower bound:
        max(min velocity limit, membrane capacity / theta_max)

    Upper bound:
        min(max velocity limit, pressure-drop limit)

    Parameters
    ----------
    number_of_fibers : CustomProp
        Number of fibers in the module.
    fiber_inner_diameter : CustomProp
        Inner diameter of the fibers.
    fiber_outer_diameter : CustomProp
        Outer diameter of the fibers.
    membrane_length : CustomProp
        Length of the membrane fibers.
    feed_pressure : CustomProp
        Pressure on the feed side.
    feed_temperature : CustomProp
        Temperature of the feed gas.
    permeate_pressure : CustomProp
        Pressure on the permeate side.
    viscosity : CustomProp
        Dynamic viscosity of the feed gas.
    permeance : dict
        Dictionary of permeance values for each component, in units of mol/s.m2.Pa or GPU.
    feed_mole_fraction : dict
        Dictionary of mole fractions for each component in the feed (must sum to 1).
    velocity_min : CustomProp, optional
        Minimum lumen-side fluid velocity (default is 0.01 m/s).
    velocity_max : CustomProp, optional
        Maximum lumen-side fluid velocity (default is 10.0 m/s).
    max_pressure_drop : CustomProp, optional
        Maximum allowable pressure drop along the fiber length (default is 20000 Pa).
    theta_max : float, optional
        Maximum allowable stage cut used to set the lower flow-rate bound (default is 0.8).

    Returns
    -------
    dict
        Dictionary containing:
        - ``lumen_cross_section_area`` : total lumen flow area (m2)
        - ``q_min_velocity`` : minimum flow rate from velocity constraint (m3/s)
        - ``q_max_velocity`` : maximum flow rate from velocity constraint (m3/s)
        - ``q_max_pressure_drop`` : maximum flow rate from pressure-drop constraint (m3/s)
        - ``q_min_capacity`` : minimum flow rate from membrane capacity constraint (m3/s)
        - ``q_min_recommended`` : final recommended minimum volumetric flow rate (m3/s)
        - ``q_max_recommended`` : final recommended maximum volumetric flow rate (m3/s)
        - ``f_min_recommended`` : final recommended minimum molar flow rate (mol/s)
        - ``f_max_recommended`` : final recommended maximum molar flow rate (mol/s)
        - ``estimated_total_permeation_capacity`` : total permeation capacity (mol/s)
        - ``estimated_component_capacity`` : per-component permeation capacity (mol/s)
        - ``theta_max`` : stage-cut limit used
        - ``is_feasible_range`` : True if the recommended min < max
    """
    try:
        if not 0.0 < theta_max < 1.0:
            raise ValueError("theta_max must be between 0 and 1.")

        if abs(sum(feed_mole_fraction.values()) - 1.0) > 1e-6:
            raise ValueError("Feed mole fractions must sum to 1.")

        lumen_area = calculate_lumen_cross_section_area(
            number_of_fibers=number_of_fibers,
            fiber_inner_diameter=fiber_inner_diameter,
            output_unit='m2'
        )

        q_min_velocity = calculate_flow_rate_from_velocity(
            cross_section_area=lumen_area,
            velocity=velocity_min,
            output_unit='m3/s'
        )

        q_max_velocity = calculate_flow_rate_from_velocity(
            cross_section_area=lumen_area,
            velocity=velocity_max,
            output_unit='m3/s'
        )

        q_max_dp = calculate_laminar_lumen_qmax_from_pressure_drop(
            number_of_fibers=number_of_fibers,
            fiber_inner_diameter=fiber_inner_diameter,
            membrane_length=membrane_length,
            viscosity=viscosity,
            max_pressure_drop=max_pressure_drop,
            output_unit='m3/s'
        )

        total_capacity, component_capacity = estimate_membrane_permeation_capacity(
            number_of_fibers=number_of_fibers,
            fiber_outer_diameter=fiber_outer_diameter,
            membrane_length=membrane_length,
            feed_pressure=feed_pressure,
            permeate_pressure=permeate_pressure,
            permeance=permeance,
            feed_mole_fraction=feed_mole_fraction
        )

        f_min_capacity = CustomProp(
            value=total_capacity.value / theta_max,
            unit='mol/s'
        )

        q_min_capacity = calculate_volumetric_flow_rate_ideal_gas(
            molar_flow_rate=f_min_capacity,
            pressure=feed_pressure,
            temperature=feed_temperature,
            output_unit='m3/s'
        )

        q_min_final_value = max(
            q_min_velocity.value,
            q_min_capacity.value
        )

        q_max_final_value = min(
            q_max_velocity.value,
            q_max_dp.value
        )

        q_min_final = CustomProp(
            value=q_min_final_value,
            unit='m3/s'
        )

        q_max_final = CustomProp(
            value=q_max_final_value,
            unit='m3/s'
        )

        f_min_final = calculate_molar_flow_rate_ideal_gas(
            volumetric_flow_rate=q_min_final,
            pressure=feed_pressure,
            temperature=feed_temperature,
            output_unit='mol/s'
        )

        f_max_final = calculate_molar_flow_rate_ideal_gas(
            volumetric_flow_rate=q_max_final,
            pressure=feed_pressure,
            temperature=feed_temperature,
            output_unit='mol/s'
        )

        return {
            "lumen_cross_section_area": lumen_area,

            "q_min_velocity": q_min_velocity,
            "q_max_velocity": q_max_velocity,
            "q_max_pressure_drop": q_max_dp,
            "q_min_capacity": q_min_capacity,

            "q_min_recommended": q_min_final,
            "q_max_recommended": q_max_final,

            "f_min_recommended": f_min_final,
            "f_max_recommended": f_max_final,

            "estimated_total_permeation_capacity": total_capacity,
            "estimated_component_capacity": component_capacity,

            "theta_max": theta_max,
            "is_feasible_range": q_min_final.value < q_max_final.value,
        }

    except Exception as e:
        logger.error(f"Error calculating HFM feed flow rate bounds: {e}")
        raise


def validate_hfm_feed_flow_rate(
        feed_flow_rate: CustomProp,
        bounds: dict,
) -> bool:
    """
    Validate feed molar flow rate before solving the HFM model.

    Raises a ValueError if the feed flow rate falls outside the recommended
    bounds computed by calculate_hfm_feed_flow_rate_bounds().

    Parameters
    ----------
    feed_flow_rate : CustomProp
        Feed molar flow rate to validate.
    bounds : dict
        Dictionary returned by calculate_hfm_feed_flow_rate_bounds(), containing
        'f_min_recommended' and 'f_max_recommended' keys.

    Returns
    -------
    bool
        True if the feed flow rate is within the recommended bounds.

    Raises
    ------
    ValueError
        If the feed flow rate is below the minimum or above the maximum recommended value.
    """
    try:
        f_mol_s = pycuc.convert_from_to(
            feed_flow_rate.value,
            feed_flow_rate.unit,
            'mol/s'
        )

        f_min = bounds["f_min_recommended"].value
        f_max = bounds["f_max_recommended"].value

        if f_mol_s < f_min:
            raise ValueError(
                f"Feed flow rate is too low. "
                f"Given {f_mol_s:.4e} mol/s, "
                f"minimum recommended is {f_min:.4e} mol/s."
            )

        if f_mol_s > f_max:
            raise ValueError(
                f"Feed flow rate is too high. "
                f"Given {f_mol_s:.4e} mol/s, "
                f"maximum recommended is {f_max:.4e} mol/s."
            )

        return True

    except Exception as e:
        logger.error(f"Error validating HFM feed flow rate: {e}")
        raise


def validate_stage_cut(
        stage_cut: float,
        theta_max: float = 0.95,
) -> bool:
    """
    Validate the stage cut after solving the HFM model.

    Checks that the stage cut is physically meaningful (non-negative, at most 1)
    and does not exceed the recommended maximum.

    Parameters
    ----------
    stage_cut : float
        Stage cut (theta = permeate flow / feed flow), dimensionless.
    theta_max : float, optional
        Maximum allowable stage cut (default is 0.95).

    Returns
    -------
    bool
        True if the stage cut is within acceptable bounds.

    Raises
    ------
    ValueError
        If the stage cut is negative, greater than 1, or exceeds theta_max.
    """
    if stage_cut < 0.0:
        raise ValueError("Stage cut is negative. Check flux sign convention.")

    if stage_cut > 1.0:
        raise ValueError(
            "Stage cut is greater than 1. "
            "The retentate flow became unphysical."
        )

    if stage_cut > theta_max:
        raise ValueError(
            f"Stage cut is too high: {stage_cut:.4f}. "
            f"Recommended maximum is {theta_max:.4f}."
        )

    return True
