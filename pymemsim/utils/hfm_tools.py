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
        output_unit: str = 'm'
) -> CustomProp:
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
        if output_unit == 'm':
            total_surface_area_prop = CustomProp(
                value=total_surface_area,
                unit='m'
            )
        else:
            total_surface_area_prop = CustomProp(
                value=pycuc.convert_from_to(
                    total_surface_area,
                    'm',
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

# SECTION: calculate module volume


def calculate_module_volume(
        module_inner_diameter: CustomProp,
        membrane_length: CustomProp,
        output_unit: str = 'm3'
) -> CustomProp:
    """
    Calculate cylindrical module volume.

    V = pi * dm^2 / 4 * L
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
