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

# SECTION: calculate porosity
