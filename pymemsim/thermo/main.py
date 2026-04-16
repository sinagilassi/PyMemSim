# import libs
import logging
from typing import List, Dict, Any, cast, Optional
from pythermodb_settings.models import Component, ComponentKey, CustomProperty
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.thermo import Source
from pyreactsim_core.models.rate_exp import ReactionRateExpression
# locals
from ..models.heat import HeatTransferOptions
from ..sources.thermo_model_source import ThermoModelSource
from ..sources.thermo_source import ThermoSource
from ..utils.tools import generate_component_references
# ! membrane options
from ..models.hfm import HollowFiberMembraneOptions


# NOTE: logger setup
logger = logging.getLogger(__name__)


# SECTION: Build thermo source
def build_thermo_source(
    components: List[Component],
    model_source: ModelSource,
    thermo_inputs: Dict[str, Any],
    unit_options: HollowFiberMembraneOptions,
    heat_transfer_options: HeatTransferOptions,
    reaction_rates: List[ReactionRateExpression],
    component_key: ComponentKey,
) -> ThermoSource:
    """
    Builds a ThermoSource instance using the provided components, model inputs, reactor options, heat transfer options, reaction rates, and component key.

    Parameters
    ----------
    components : List[Component]
        A list of Component objects representing the chemical components involved in the model source.
    model_source : ModelSource
        A ModelSource object containing the source of the model to be used in the simulation.
    thermo_inputs : Dict[str, Any]
        A dictionary of model inputs, where the keys are the names of the inputs and the values
    unit_options : HollowFiberMembraneOptions
        A HollowFiberMembraneOptions object containing the options for the hollow fiber membrane model, which includes the modeling type, pressure modes, phase, gas model, heat capacity modes, density mode, formation enthalpy mode, molecular weight mode, and reaction enthalpy mode. These options will be used to configure the thermo source for the membrane simulation.
        the inputs for reactor simulation.
    heat_transfer_options : HeatTransferOptions
        A HeatTransferOptions object containing the inputs for heat transfer in the batch reactor simulation.
    reaction_rates : List[ReactionRateExpression]
        A list of reaction rate expressions, where each expression is represented as a ReactionRateExpression object containing information about the reaction and its rate expression.
    component_key : ComponentKey
        A ComponentKey object representing the key to be used for the components in the model source.

    Returns
    -------
    ThermoSource
        A ThermoSource instance built using the provided inputs.
    """
    # SECTION: Validation
    # ! thermo inputs
    thermo_inputs = thermo_inputs if thermo_inputs is not None else {}

    # SECTION: Generate component references
    component_refs = generate_component_references(
        components=components,
        component_key=component_key
    )

    # SECTION: Build thermo source
    # NOTE: create source
    source = Source(
        model_source=model_source,
        component_key=component_key,
    )

    thermo_source = ThermoSource(
        components=components,
        source=source,
        model_source=model_source,
        thermo_inputs=thermo_inputs,
        unit_options=unit_options,
        heat_transfer_options=heat_transfer_options,
        reaction_rates=reaction_rates,
        component_refs=component_refs,
        component_key=component_key,
    )

    return thermo_source

# SECTION: build thermo source from
