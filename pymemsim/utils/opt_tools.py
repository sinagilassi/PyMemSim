# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, ComponentKey, CustomProp, Pressure, Temperature, CustomProperty
from pythermodb_settings.utils import set_component_id
from pyreactlab_core.models.reaction import Reaction
# locals
from ..models.rate_exp import ReactionRateExpression
from ..models.rate_exp_refs import rArgs, rParams, rRet, rXs
from .unit_tools import to_m3, to_K

# NOTE: logger setup
logger = logging.getLogger(__name__)

# SECTION:


def calc_heat_exchange(
    temperature: float,
    jacket_temperature: float,
    heat_transfer_area: float,
    heat_transfer_coefficient: float,
    reactor_volume: float
) -> float:
    """
    Calculate the heat exchange with the surroundings based on the current temperature of the system.

    Parameters
    ----------
    temperature : float
        Current temperature of the system [K].
    jacket_temperature : float
        Temperature of the jacket or surroundings [K].
    heat_transfer_area : float
        Area available for heat transfer [m2].
    heat_transfer_coefficient : float
        Heat transfer coefficient [W/m2.K].
    reactor_volume : float
        Volume of the reactor [m3].

    Returns
    -------
    float
        Heat exchange with the surroundings [W] or [J/s].
    """
    # NOTE: Convert units if necessary
    T = temperature
    T_s = jacket_temperature
    A = heat_transfer_area
    U = heat_transfer_coefficient
    Vr = reactor_volume

    # ! calculate heat exchange using the formula: Q = U * A * (T_s - T)
    # unit check: U [W/m^2.K], A [m^2], T_s [K], temp [K] => Q [W/m^3] or [J/s.m^3]
    return U * A * (T_s - T) / Vr


# SECTION: Set feed mole specification
def set_component_X(
    components: List[Component],
    prop_name: str,
    component_key: ComponentKey
) -> Tuple[Dict[str, CustomProp], np.ndarray]:
    """
    Set the components property based on the specified property in the X dictionary of each component.

    Parameters
    ----------
    components : List[Component]
        A list of Component objects representing the components in the reaction.
    prop_name : str
        The name of the property to be set for each component (e.g., "mole", "concentration").
    component_key : ComponentKey
        A ComponentKey object representing the key to be used for the components in the model source.

    Returns
    -------
    Tuple[Dict[str, CustomProp], np.ndarray]
        A dictionary with component names as keys and CustomProp objects and a numpy array of the mole values for each component.
    """
    # NOTE: set feed mole specification
    X_spec = {}
    X_spec_list = []

    # iterate over components and set feed mole specification
    for component in components:
        # NOTE: get component name using component key
        comp_name = set_component_id(component, component_key)

        # get mole specification for the component
        X_prop_checker = component.X['name'] == prop_name
        if (
            X_prop_checker is None or
            not X_prop_checker
        ):
            raise ValueError(
                f"Component '{comp_name}' does not have the specified property '{prop_name}' in its X dictionary.")

        # NOTE: get the property value and unit
        X_prop = component.X

        # add
        n = CustomProperty(
            value=X_prop['value'],
            unit=X_prop['unit'],
            symbol=X_prop['symbol'],
        )

        # NOTE: add component mole specification to feed mole specification dictionary
        X_spec[comp_name] = n
        X_spec_list.append(n.value)

    # to array
    X_spec_array = np.array(X_spec_list)

    return X_spec, X_spec_array
