# import libs
import logging
from typing import List, Tuple
import numpy as np
from pythermodb_settings.models import CustomProp
from ..utils.unit_tools import to_m


class HFMModule:
    def __init__(
        self,
        number_of_fibers: CustomProp,
        fiber_outer_diameter: CustomProp,
        fiber_inner_diameter: CustomProp,
        fiber_length: CustomProp,
        shell_diameter: CustomProp
    ):
        # set attributes
        self.number_of_fibers = number_of_fibers
        self.fiber_outer_diameter = fiber_outer_diameter
        self.fiber_inner_diameter = fiber_inner_diameter
        self.fiber_length = fiber_length
        self.shell_diameter = shell_diameter

        # calculate all properties of the membrane module
        self.properties = self.calculate()

    # SECTION: calculate all properties of the membrane module

    def calculate(self):
        # NOTE: number of fibers
        Nf = self.number_of_fibers.value

        # NOTE: fiber outer diameter
        # ! m
        do = to_m(
            self.fiber_outer_diameter.value,
            self.fiber_outer_diameter.unit
        )

        # NOTE: fiber inner diameter
        # ! m
        di = to_m(
            self.fiber_inner_diameter.value,
            self.fiber_inner_diameter.unit
        )

        # NOTE: fiber length
        # ! m
        L = to_m(self.fiber_length.value, self.fiber_length.unit)

        # NOTE: shell diameter
        # ! m
        Ds = to_m(self.shell_diameter.value, self.shell_diameter.unit)

        # NOTE: cross-sectional area
        # ! m2
        # ? module cross area = pi * (shell diameter / 2)^2
        module_cross_area = np.pi * (Ds / 2)**2

        # ? single fiber cross-sectional area = pi * (fiber inner diameter / 2)^2
        single_fiber_cross_area = np.pi * (do / 2)**2

        # ? total fiber cross-sectional area = number of fibers * pi * (fiber outer diameter / 2)^2
        total_fiber_cross_area = Nf * single_fiber_cross_area

        # NOTE: packing & porosity
        # ! -
        # ? packing = total fiber cross-sectional area / module cross-sectional area
        packing_fraction = total_fiber_cross_area / module_cross_area

        # ? porosity = 1 - packing_fraction
        porosity = 1 - packing_fraction

        # NOTE: membrane surface area
        # ! m2
        # ? single fiber surface area = pi * fiber outer diameter * fiber length
        single_fiber_area = np.pi * do * L

        # ? total fiber surface area = number of fibers * single fiber surface area
        total_membrane_area = Nf * single_fiber_area

        # NOTE: volume of module
        # ! m3
        # ? module volume = module cross-sectional area * fiber length
        module_volume = module_cross_area * L

        # ! m2
        # ? shell free area = module cross-sectional area - total fiber cross-sectional area
        shell_free_area = module_cross_area - total_fiber_cross_area

        # NOTE: area per unit length
        # ! m2/m
        # ? area per unit length = total membrane surface area / fiber length
        area_per_unit_length = total_membrane_area / L

        return {
            'module_cross_area': module_cross_area,
            'single_fiber_cross_area': single_fiber_cross_area,
            'total_fiber_cross_area': total_fiber_cross_area,
            'packing_fraction': packing_fraction,
            'porosity': porosity,
            'single_fiber_area': single_fiber_area,
            'total_membrane_area': total_membrane_area,
            'module_volume': module_volume,
            'shell_free_area': shell_free_area,
            'area_per_unit_length': area_per_unit_length
        }
