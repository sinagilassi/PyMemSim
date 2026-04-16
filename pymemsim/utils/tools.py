# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, cast
from pythermodb_settings.models import Component, ComponentKey, CustomProp, Pressure, Temperature, CustomProperty
from pythermodb_settings.utils import set_component_id, build_components_mapper


def find_components_property(
        components: List[Component],
        prop_values: Dict[str, float],
        component_key: ComponentKey,
):
    """
    Find the specified property values for the components based on the component key.

    Parameters
    ----------
    components : List[Component]
        A list of Component objects representing the components in the reaction.
    prop_values : Dict[str, float]
        A dictionary with component names as keys and the property values to be extracted as values.
    component_key : ComponentKey
        A ComponentKey object representing the key to be used for the components in the model source.

    Returns
    -------
    Tuple[Dict[str, CustomProp], np.ndarray]
        A dictionary with component names as keys and CustomProp objects containing the extracted property values and a numpy array of the extracted property values for each component.

    Raises
    ------
    ValueError
        If a component specified in the prop_values dictionary is not found in the components list based on the component key.
    """
    # NOTE: extract the specified property values for the components based on the component key
    extracted_values = []
    extracted_values_comp = {}

    # iterate through components
    for comp in components:
        comp_id = set_component_id(comp, component_key)
        if comp_id in prop_values:
            # store
            extracted_values.append(prop_values[comp_id])
            extracted_values_comp[comp_id] = prop_values[comp_id]
        else:
            raise ValueError(
                f"Component '{comp_id}' not found in the provided property values."
            )

    # to array
    extracted_values = np.array(extracted_values, dtype=float)

    return extracted_values, extracted_values_comp


# SECTION: collect keys
def collect_keys(
        data: Dict[str, Any]) -> List[str]:
    """Extract keys"""
    keys = set(data.keys())
    # lower case keys for easier access
    keys = [
        key.lower().strip() for key in keys
    ]
    return keys

# SECTION: config property values for components


def config_components_property(
        component_ids: List[str],
        prop_source: Dict[str, Any],
        unit_conversion_func: Callable[[float, str], float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Configure the specified property values for the components based on the component IDs and the property source.

    Parameters
    ----------
    component_ids : List[str]
        A list of component IDs representing the components for which the property values need to be configured.
    prop_source : Dict[str, Any]
        A dictionary with component names as keys and CustomProp objects containing the property values and units as values.
    unit_conversion_func : Callable[[float, str], float]
        A function that takes a value and its unit as input and returns the value converted to the desired unit.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        A numpy array of the configured property values for each component and a dictionary with component names as keys and the configured property values as values.

    Raises
    ------
    ValueError
        If a component specified in the component_ids list is not found in the prop_source dictionary.
    """
    # iterate through components and extract property values
    prop_values = []
    prop_values_comp = {}

    for comp_id in component_ids:
        if comp_id in prop_source:
            # component source
            dt_src = prop_source[comp_id]

            # >> extract value and unit
            dt_value: float = dt_src.get("value")
            dt_unit: str = dt_src.get("unit")

            # conversion
            prop_value_converted = unit_conversion_func(dt_value, dt_unit)

            # store
            prop_values.append(prop_value_converted)
            prop_values_comp[comp_id] = prop_value_converted
        else:
            raise ValueError(
                f"Component '{comp_id}' not found in the provided property source."
            )

    # to array
    prop_values_array = np.array(prop_values, dtype=float)

    return prop_values_array, prop_values_comp


# SECTION: Component references
def generate_component_references(
        components: List[Component],
        component_key: ComponentKey
) -> Dict[str, Any]:
    """
    Generate component references based on the components and the component key. This method creates a mapping of component IDs, formula-state representations, and other relevant references for the components in the model source.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the generated component references, including:
        - component_num: The number of components.
        - component_ids: A list of component IDs generated based on the component key.
        - component_formula_state: A list of formula-state representations for the components.
        - component_mapper: A dictionary mapping component IDs to their corresponding component keys for different properties.
        - component_id_to_index: A dictionary mapping component IDs to their corresponding indices in the components list.
    """
    # NOTE: numbers
    component_num = len(components)

    # NOTE: Create component ID list
    component_ids: list[str] = [
        set_component_id(
            component=comp,
            component_key=cast(ComponentKey, component_key)
        )
        for comp in components
    ]

    # >>> formula-state
    component_formula_state: list[str] = [
        set_component_id(
            component=component,
            component_key='Formula-State'
        )
        for component in components
    ]

    # NOTE: build component mapper
    component_mapper: Dict[str, Dict[ComponentKey, str]] = build_components_mapper(
        components=components,
        component_key=cast(ComponentKey, component_key)
    )

    # >> index mapping
    component_id_to_index: dict[str, int] = {
        comp_id: idx for idx, comp_id in enumerate(component_ids)
    }

    return {
        "component_num": component_num,
        "component_ids": component_ids,
        "component_formula_state": component_formula_state,
        "component_mapper": component_mapper,
        "component_id_to_index": component_id_to_index
    }


# SECTION: smooth floor function
def smooth_floor(x: float | np.ndarray, xmin: float, s: float) -> float | np.ndarray:
    """
    Smooth approximation of ``max(x, xmin)`` using a numerically stable softplus.

    Parameters
    ----------
    x : float | np.ndarray
        Value(s) to floor.
    xmin : float
        Minimum smooth floor value.
    s : float
        Smoothing width. Smaller values approach a hard floor.
    """
    if s <= 0.0:
        raise ValueError("smooth_floor requires s > 0.")

    z = (np.asarray(x, dtype=float) - xmin) / s
    y = xmin + s * np.logaddexp(0.0, z)

    if np.isscalar(x):
        return float(y)
    return y

# SECTION: solver configuration for reactor simulation


def configure_solver_options(
        solver_options: Optional[Dict[str, Any]] = None,
        default_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Configure solver options for reactor simulation by merging user-provided options with default options.

    Parameters
    ----------
    solver_options : Optional[Dict[str, Any]]
        A dictionary of user-provided solver options to override the defaults.
    default_options : Optional[Dict[str, Any]]
        A dictionary of default solver options to be used if not overridden by the user.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the merged solver options to be used for reactor simulation.
    """
    # Set default options if not provided
    if default_options is None:
        default_options = {
            "method": "BDF",
            "rtol": 1e-6,
            "atol": 1e-9,
        }

    # Merge user-provided options with defaults
    if solver_options is not None:
        merged_options = {**default_options, **solver_options}
    else:
        merged_options = default_options

    return merged_options
