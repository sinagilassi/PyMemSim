# import libs
import logging
import numpy as np
from typing import List, Dict, Any, cast, Optional
from pythermodb_settings.models import Component, Temperature, Pressure, CustomProperty, ComponentKey
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.models.component_models import ComponentEquationSource
from pyThermoCalcDB.docs.thermo import build_hsg_properties
from pyThermoCalcDB.core import HSGProperties
from pyreactsim_core.models.rate_exp import ReactionRateExpression
# locals
from ..sources.interface import (
    ext_components_dt,
    ext_components_eq,
    exec_component_eq
)
from ..models.heat import HeatTransferOptions
from ..utils.tools import config_components_property
from ..utils.unit_tools import to_J_per_mol, to_g_per_mol
# ! membrane options
from ..models.hfm import HollowFiberMembraneOptions

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoModelSource:
    # NOTE: Attributes
    # ! sources
    Cp_IG_src: Dict[str, ComponentEquationSource] = {}
    Cp_LIQ_src: Dict[str, ComponentEquationSource] = {}
    rho_LIQ_src: Dict[str, ComponentEquationSource] = {}
    EnFo_IG_298_src: Dict[str, Dict[str, Any]] = {}
    MW_src: Dict[str, Dict[str, Any]] = {}
    # ! properties
    EnFo_IG_298: np.ndarray = np.array([])
    EnFo_IG_298_comp: Dict[str, float] = {}
    MW: np.ndarray = np.array([])
    MW_comp: Dict[str, float] = {}

    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_source: ModelSource,
        thermo_inputs: Dict[str, Any],
        unit_options: HollowFiberMembraneOptions,
        heat_transfer_options: HeatTransferOptions,
        reaction_rates: List[ReactionRateExpression],
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        """
        Initializes the ThermoModelSource instance with the provided components, source, component key, and component mapper.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the model source.
        source : Source
            A Source object containing information about the source of the data or equations used in the model source.
        model_source : ModelSource
            A ModelSource object containing information about the model source, including its name, description, and other relevant details.
        thermo_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values. This can include feed specifications, initial conditions, or any other relevant parameters needed for the simulations.
        unit_options : HollowFiberMembraneOptions
            A unit options object containing phase and thermodynamic configuration.
        heat_transfer_options : HeatTransferOptions
            A HeatTransferOptions object containing the inputs for heat transfer in the batch reactor simulation.
        reaction_rates : List[ReactionRateExpression]
            A list of reaction rate expressions, where each expression is represented as a ReactionRateExpression object containing information about the reaction and its rate expression.
        component_refs : Dict[str, Any]
            A dictionary containing references for the components, which can include mappings of component IDs, formulas, and other relevant information for the components in the model source.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the model source.
        """
        # NOTE: Set attributes
        self.components = components
        self.source = source
        self.model_source = model_source
        self.thermo_inputs = thermo_inputs
        self.unit_options = unit_options
        self.heat_transfer_options = heat_transfer_options
        self.reaction_rates = reaction_rates
        self.component_refs = component_refs
        self.component_key = component_key

        # ! component refs
        self.component_ids = component_refs['component_ids']
        self.component_formula_state = component_refs['component_formula_state']
        self.component_mapper = component_refs['component_mapper']

        # ! phase
        self.phase = unit_options.phase

        # SECTION: Extract property equation sources
        if self.heat_transfer_options.heat_transfer_mode == "non-isothermal":
            # check heat capacity mode
            if self.unit_options.gas_heat_capacity_mode == "temperature-dependent":
                # NOTE: extract heat capacity equation source for the components from the model source
                self.Cp_IG_src: Dict[str, ComponentEquationSource] = self.prop_eq_src(
                    prop_name="Cp_IG"
                )

            # NOTE: Enthalpy of formation at 298 K for ideal gas
            if self.unit_options.ideal_gas_formation_enthalpy_mode == "model_source":
                # extract data
                self.EnFo_IG_298_src: Dict[str, Dict[str, Any]] = self.prop_dt_src(
                    component_ids=self.component_ids,
                    prop_name="EnFo_IG"
                )

                # ! values in J/mol
                (
                    self.EnFo_IG_298,
                    self.EnFo_IG_298_comp
                ) = config_components_property(
                    component_ids=self.component_ids,
                    prop_source=self.EnFo_IG_298_src,
                    unit_conversion_func=to_J_per_mol
                )

        if self.phase == "liquid":
            # MW source
            self.MW_src = self.prop_dt_src(
                component_ids=self.component_ids,
                prop_name="MW"
            )

            # ! values in g/mol
            (
                self.MW,
                self.MW_comp
            ) = config_components_property(
                component_ids=self.component_ids,
                prop_source=self.MW_src,
                unit_conversion_func=to_g_per_mol
            )

            # NOTE: density
            if self.unit_options.liquid_density_mode == "temperature-dependent":
                # NOTE: extract density equation source for the components from the model source
                self.rho_LIQ_src: Dict[str, ComponentEquationSource] = self.prop_eq_src(
                    prop_name="rho_LIQ"
                )

            # NOTE: heat capacity
            if self.heat_transfer_options.heat_transfer_mode == "non-isothermal":
                if self.unit_options.liquid_heat_capacity_mode == "temperature-dependent":
                    # extract heat capacity equation source for the components from the model source
                    self.Cp_LIQ_src: Dict[str, ComponentEquationSource] = self.prop_eq_src(
                        prop_name="Cp_LIQ"
                    )

    # ! Extract property equation source for components

    def prop_eq_src(self, prop_name: str) -> Dict[str, ComponentEquationSource]:
        """
        Extracts the property equation for the components from the source and returns it as a dictionary.

        Returns
        -------
        Dict[str, ComponentEquationSource]
            A dictionary where the keys are component IDs and the values are ComponentEquationSource objects
        """
        # NOTE: Extract property equation source for all components
        eq_src = ext_components_eq(
            components=self.components,
            prop_name=prop_name,
            source=self.source,
            component_key=cast(ComponentKey, self.component_key),
            component_mapper=self.component_mapper
        )
        # >> check
        if eq_src is None:
            logger.error("Failed to extract property equation for components.")
            return {}

        return eq_src

    # ! Extract property data source for components
    def prop_dt_src(
            self,
            component_ids: List[str],
            prop_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts the property derivative equation for the components from the source and returns it as a dictionary.

        Returns
        -------
        Dict[str, ComponentEquationSource]
            A dictionary where the keys are component IDs and the values are ComponentEquationSource objects
        """
        # NOTE: Extract property derivative equation source for all components
        dt_src = ext_components_dt(
            component_ids=component_ids,
            prop_name=prop_name,
            source=self.source,
        )
        # >> check
        if dt_src is None:
            logger.error(
                "Failed to extract property derivative equation for components.")
            return {}

        return dt_src

    # SECTION: Model source configurations
    def _get_args_units(
            self,
            eq_src: ComponentEquationSource
    ):
        # res
        res = {}

        # iterate over inputs
        for name, details in eq_src.arg_mappings.items():
            # get unit
            unit = details.get('unit', '')
            # symbol
            symbol = details.get('symbol', '')

            res[symbol] = unit

        return res

    def _get_return_unit(
            self,
            eq_src: ComponentEquationSource,
            symbol: str
    ) -> str:
        # res
        res = {}

        # iterate over returns
        for name, details in eq_src.returns.items():
            # get unit
            unit = details.get('unit', '')
            # symbol
            symbol = details.get('symbol', '')

            res[symbol] = unit

        # check
        if symbol not in res:
            logger.warning(
                f"Symbol {symbol} not found in equation source returns. Returning empty string for unit.")
            return ""

        return res[symbol]

    def _get_inputs(
            self,
            eq_src: ComponentEquationSource,
            except_args: Optional[List[str]] = None
    ):
        # res
        res = {}

        # iterate over inputs
        for name, details in eq_src.inputs.items():

            # get symbol
            symbol = details.get('symbol', '')

            if except_args is not None and symbol in except_args:
                continue

            # get value from component source
            value = details.get('value', None)

            res[symbol] = value

        return res

    # SECTION: HSG properties
    def _config_components_hsg_properties(
            self,
            temperature: float
    ) -> Dict[str, HSGProperties]:
        # build properties
        component_hsg_properties = {}

        for i, comp_id in enumerate(self.component_formula_state):
            # build
            hsg_props = build_hsg_properties(
                component=self.components[i],
                model_source=self.model_source
            )

            # >> check
            if hsg_props is None:
                logger.error(
                    f"Failed to build HSG properties for component {comp_id}.")
                raise ValueError(
                    f"Failed to build HSG properties for component {comp_id}.")

            component_hsg_properties[comp_id] = hsg_props

        return component_hsg_properties
