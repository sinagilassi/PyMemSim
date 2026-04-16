# import libs
import logging
from typing import Any, Dict, List, Optional, Tuple
import pycuc
from pythermodb_settings.models import Component, ComponentKey, CustomProperty
from pythermodb_settings.utils import set_component_id, build_component_mapper
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.models.component_models import ComponentEquationSource
# locals


# NOTE: logger setup
logger = logging.getLogger(__name__)

# SECTION: Extract data from source


def ext_component_dt(
        component_id: str,
        prop_name: str,
        source: Source
) -> Optional[Dict[str, Any]]:
    """
    Extract data from the source for a given component ID and property name.

    Parameters
    ----------
    component_id : str
        The ID of the component for which to extract data.
    prop_name : str
        The name of the property to extract.
    source : Source
        The source from which to extract the data.

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary containing the extracted data, or None if an error occurs.
    """
    try:
        # NOTE: extract data
        data = source.data_extractor(
            component_id=component_id,
            prop_name=prop_name
        )
        return data
    except Exception as e:
        logger.error(
            f"Error extracting data for component_id: {component_id}, prop_name: {prop_name} - {e}")
        return None


def ext_components_dt(
        component_ids: List[str],
        prop_name: str,
        source: Source
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Extract data for multiple component IDs from the source for a given property name.

    Parameters
    ----------
    component_ids : List[str]
        A list of component IDs for which to extract data.
    prop_name : str
        The name of the property to extract.
    source : Source
        The source from which to extract the data.

    Returns
    -------
    Optional[Dict[str, Dict[str, Any]]]
        A dictionary mapping component IDs to their extracted data, or None if an error occurs.
    """
    try:
        # NOTE: extract data for each component ID
        data_dict = {}
        for component_id in component_ids:
            data = ext_component_dt(
                component_id=component_id,
                prop_name=prop_name,
                source=source
            )
            if data is not None:
                data_dict[component_id] = data
        return data_dict
    except Exception as e:
        logger.error(
            f"Error extracting data for component_ids: {component_ids}, prop_name: {prop_name} - {e}")
        return None

# SECTION: Extract equation from source


def ext_component_eq(
        component: Component,
        prop_name: str,
        component_key: ComponentKey,
        component_keys: List[ComponentKey],
        source: Source
) -> Optional[ComponentEquationSource]:
    """
    Extract an equation from the source for a given component and property name.

    Parameters
    ----------
    component : Component
        The component for which to extract the equation.
    prop_name : str
        The name of the property for which to extract the equation.
    component_key : ComponentKey
        The component key to use for setting component IDs.
    component_keys : List[ComponentKey]
        A list of component keys to use for setting component IDs.
    source : Source
        The source from which to extract the equation.

    Returns
    -------
    Optional[ComponentEquationSource]
        A ComponentEquationSource object containing the extracted equation, or None if an error occurs.
    """
    try:
        # NOTE: extract equation
        eq_src = source.eq_builder(
            components=[component],
            prop_name=prop_name,
            component_keys=component_keys,
        )

        # >> check
        if eq_src is None:
            logger.warning(
                f"No equation found for component: {component}, prop_name: {prop_name}")
            return None

        # >> component id
        component_id = set_component_id(
            component=component,
            component_key=component_key
        )

        # NOTE: create ComponentEquationSource
        component_eq_src: ComponentEquationSource | None = eq_src.get(
            component_id
        )

        # >> check
        if component_eq_src is None:
            logger.warning(
                f"No equation source found for component_id: {component_id}, prop_name: {prop_name}")
            return None

        return component_eq_src
    except Exception as e:
        logger.error(
            f"Error extracting equation for component: {component}, prop_name: {prop_name} - {e}")
        return None


def ext_components_eq(
        components: List[Component],
        prop_name: str,
        component_key: ComponentKey,
        component_mapper: Dict[str, Dict[ComponentKey, str]],
        source: Source
) -> Optional[Dict[str, ComponentEquationSource]]:
    """
    Extract equations from the source for multiple components and a given property name.

    Parameters
    ----------
    components : List[Component]
        A list of components for which to extract equations.
    prop_name : str
        The name of the property for which to extract equations.
    component_key : ComponentKey
        The component key to use for setting component IDs.
    source : Source
        The source from which to extract the equations.

    Returns
    -------
    Optional[Dict[str, ComponentEquationSource]]
        A dictionary mapping component IDs to their extracted ComponentEquationSource objects, or None if an error occurs.
    """
    try:
        # NOTE: extract equation for each component
        eq_dict = {}

        for component in components:
            component_id = set_component_id(
                component=component,
                component_key=component_key
            )

            # configure component keys
            mapper_ = component_mapper.get(component_id, {})
            # >> check
            if not mapper_:
                logger.warning(
                    f"No component keys found for component_id: {component_id} in component_mapper")
                continue

            # >>> component keys
            component_keys = list(mapper_.keys())

            # >> extract equation source for component
            component_eq_src = ext_component_eq(
                component=component,
                prop_name=prop_name,
                component_key=component_key,
                component_keys=component_keys,
                source=source
            )
            if component_eq_src is not None:

                eq_dict[component_id] = component_eq_src

        return eq_dict
    except Exception as e:
        logger.error(
            f"Error extracting equations for components: {components}, prop_name: {prop_name} - {e}")
        return None

# SECTION: Execute equation from source


def exec_component_eq(
        component_eq_src: ComponentEquationSource,
        inputs: Dict[str, Any],
        output_unit: Optional[str] = None
) -> Optional[CustomProperty]:
    """
    Execute the equation from the ComponentEquationSource with the given inputs.

    Parameters
    ----------
    component_eq_src : ComponentEquationSource
        The ComponentEquationSource object containing the equation to execute.
    inputs : Dict[str, Any]
        A dictionary of input values to be passed to the equation
    output_unit : Optional[str]
        An optional string specifying the desired output unit for the result. If provided, the result will be converted to this unit if possible.

    Returns
    -------
    Optional[CustomProperty]
        The result of executing the equation, or None if an error occurs.
    """
    try:
        # NOTE: source
        # ! either TableEquation or MoziEquation
        eq_src = component_eq_src.source

        # NOTE: execute equation
        # >> check has cal method
        if not hasattr(eq_src, "cal"):
            logger.error(
                f"Equation source does not have a 'cal' method: {eq_src}")
            return None

        # NOTE: check inputs units
        # ! all inputs created automatically
        eq_inputs = component_eq_src.inputs
        # ! units
        eq_input_units: Dict[str, str] = {
            k: v.get('unit', '') for k, v in eq_inputs.items()
        }
        eq_input_values: Dict[str, float] = {
            k: v.get('value', 0.0) for k, v in eq_inputs.items()
        }

        # iterate through the expected inputs and convert if necessary
        for input_symbol, input_unit in eq_input_units.items():

            # >> check input value exists
            if input_symbol not in inputs.keys():
                continue  # skip if input value is not provided

            # set
            input_src = inputs[input_symbol]
            value_ = input_src['value']
            unit_ = input_src['unit']

            # >>> add to input values dict
            eq_input_values[input_symbol] = value_

            # NOTE: convert input to expected unit if specified
            if (
                input_unit is not None and
                input_unit != unit_
            ):
                try:
                    # convert to same unit for consistency
                    converted_value = pycuc.convert_from_to(
                        value=value_,
                        from_unit=unit_,
                        to_unit=input_unit
                    )

                    # >>> update input values dict with converted value
                    eq_input_values[input_symbol] = converted_value
                except Exception as e:
                    logger.error(
                        f"Error converting input '{input_symbol}' to required unit '{input_unit}': {e}")
                    return None

        # result from equation source
        # ! Use ** to unpack the inputs dictionary as keyword arguments for the cal method
        res_src = eq_src.cal(
            message="",
            decimal_accuracy=4,
            **eq_input_values
        )

        # extract value, unit, symbol
        res = {}
        if res_src is not None:
            if res_src['value'] is not None and isinstance(res_src['value'], (str, float, int)):
                res['value'] = float(res_src['value'])
            if res_src['unit'] is not None and isinstance(res_src['unit'], str):
                res['unit'] = res_src['unit']
            if res_src['symbol'] is not None and isinstance(res_src['symbol'], str):
                res['symbol'] = res_src['symbol']

        # NOTE: convert to output unit if specified
        if (
            output_unit is not None and
            'value' in res and
            'unit' in res
        ):
            try:
                converted_value = pycuc.convert_from_to(
                    value=res['value'],
                    from_unit=res['unit'],
                    to_unit=output_unit
                )
                res['value'] = converted_value
                res['unit'] = output_unit
            except Exception as e:
                logger.error(
                    f"Error converting result to output unit: {output_unit} - {e}")
                return None

        # >> convert
        res = CustomProperty(**res)
        return res
    except Exception as e:
        logger.error(f"Error executing equation - {e}")
        return None
