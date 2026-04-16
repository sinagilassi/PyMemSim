# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, ComponentKey, CustomProp, CustomProperty, Temperature, Pressure
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.models.component_models import ComponentEquationSource
from pyreactlab_core.models.reaction import Reaction
from pyThermoCalcDB.reactions.reactions import dH_rxn_STD
from pyThermoCalcDB.docs.thermo import calc_En_IG_ref, calc_En, calc_En_IG_ref_hsg, calc_En_hsg
from pyThermoCalcDB.reactions.source import dH_rxn_STD as dH_rxn_reactions
from pyThermoCalcDB.models import ComponentEnthalpy
from pyreactsim_core.models.rate_exp import ReactionRateExpression
# locals
from .thermo_model_inputs import ThermoModelInputs
from .thermo_model_source import ThermoModelSource
from .thermo_reaction import ThermoReaction
from .interface import (
    exec_component_eq
)

from ..utils.unit_tools import to_K, to_J_per_mol, to_g_per_mol
from ..utils.tools import find_components_property, collect_keys
from ..models.heat import HeatTransferOptions
from ..models import GasModel
from .thermo_calc import ThermoCalc
# ! membrane options
from ..models.hfm import HollowFiberMembraneOptions

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoSourceCore(ThermoCalc):
    """
    THermo class for handling thermodynamic calculations and properties related to chemical reactions and processes. This class provides methods for calculating various thermodynamic properties, such as heat capacities, enthalpies, and entropies, as well as methods for performing energy balance calculations in chemical systems.
    """
    # NOTE: Attributes
    # reference temperature
    T_ref = Temperature(value=298.15, unit="K")
    T_ref_K = 298.15
    # reference pressure
    P_ref = Pressure(value=101325, unit="Pa")

    def __init__(
        self,
        components: List[Component],
        source: Source,
        thermo_inputs: Dict[str, Any],
        unit_options: HollowFiberMembraneOptions,
        heat_transfer_options: HeatTransferOptions,
        reaction_rates: List[ReactionRateExpression],
        thermo_model_source: ThermoModelSource,
        thermo_model_inputs: ThermoModelInputs,
        thermo_reaction: ThermoReaction,
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        """
        Initializes the THermo instance with default properties and settings for thermodynamic calculations.
        """
        # LINK: ThermoCalc initialization
        ThermoCalc.__init__(self)

        # SECTION: Set attributes
        self.components = components
        self.source = source
        self.thermo_inputs = thermo_inputs
        self.unit_options = unit_options
        self.heat_transfer_options = heat_transfer_options
        self.reaction_rates = reaction_rates
        self.component_refs = component_refs
        self.component_key = component_key

        # SECTION: set model source, model inputs, and reaction
        # ! model source
        self.thermo_model_source = thermo_model_source
        # ! model inputs
        self.thermo_model_inputs = thermo_model_inputs
        # ! reaction
        self.thermo_reaction = thermo_reaction

        # NOTE: component refs
        self.component_ids = component_refs['component_ids']
        self.component_formula_state = component_refs['component_formula_state']
        self.component_mapper = component_refs['component_mapper']
        self.component_id_to_index = component_refs['component_id_to_index']

        # SECTION: Process model configuration

        # NOTE: model source
        model_source: ModelSource | None = self.source.model_source
        if model_source is None:
            raise ValueError(
                "Model source is required for thermodynamic calculations."
            )
        self.model_source = model_source

        # # SECTION: reactions
        self.reactions: List[Reaction] = self.thermo_reaction.build_reactions()

        # ! hsg reactions
        self.hsg_reactions = self.thermo_reaction.hsg_reactions

        # SECTION: Process model configuration
        # ! model inputs keys
        self.thermo_inputs_keys = collect_keys(self.thermo_inputs)

        # SECTION: Reactor configuration
        # ! heat capacity modes
        self.gas_heat_capacity_mode = unit_options.gas_heat_capacity_mode
        # ! liquid heat capacity mode
        self.liquid_heat_capacity_mode = unit_options.liquid_heat_capacity_mode
        # ! phase
        self.phase = unit_options.phase
        # ! density mode
        self.liquid_density_mode = unit_options.liquid_density_mode
        # ! ideal gas enthalpy of formation mode
        self.ideal_gas_formation_enthalpy_mode = unit_options.ideal_gas_formation_enthalpy_mode
        # ! molecular weight mode
        self.molecular_weight_mode = unit_options.molecular_weight_mode
        # ! reaction enthalpy mode
        self.reaction_enthalpy_mode = unit_options.reaction_enthalpy_mode

        # SECTION: heat transfer options
        # ! heat transfer mode
        self.heat_transfer_mode = self.heat_transfer_options.heat_transfer_mode

        # SECTION: Thermodynamic properties

        # ! Ideal Gas Heat Capacity at reference temperature (e.g., 298 K)
        self.Cp_IG_src: Dict[
            str,
            ComponentEquationSource
        ] = self.thermo_model_source.Cp_IG_src
        # >> constant heat capacity
        # ! to J/mol.K
        self.gas_heat_capacity_constant_values = self.thermo_model_inputs.gas_heat_capacity_constant_values
        self.gas_heat_capacity_constant_comp = self.thermo_model_inputs.gas_heat_capacity_constant_comp

        # NOTE: calculate heat capacity change for the reactions using the constant heat capacity values
        # ! to J/K
        self.dCp_rxns = self.calc_dCp_IG()

        # SECTION: Ideal Gas Enthalpy of formation at 298 K
        self.EnFo_IG_298_src: Dict[
            str,
            Dict[str, Any]
        ] = self.thermo_model_source.EnFo_IG_298_src
        # ! values in J/mol
        if len(self.EnFo_IG_298_src) > 0:
            # >> from model source
            self.EnFo_IG_298 = self.thermo_model_source.EnFo_IG_298
            self.EnFo_IG_298_comp = self.thermo_model_source.EnFo_IG_298_comp
        else:
            # >> from model inputs
            self.EnFo_IG_298 = self.thermo_model_inputs.EnFo_IG_298
            self.EnFo_IG_298_comp = self.thermo_model_inputs.EnFo_IG_298_comp

        # dH_rxn at 298 K
        # ! values in J/mol
        self.dH_rxns_298 = self.calc_dH_rxns_298()

        # SECTION: molecular weight (MW)
        self.MW_src: Dict[
            str,
            Dict[str, Any]
        ] = self.thermo_model_source.MW_src
        # ! values in g/mol
        self.MW = self.thermo_model_source.MW
        self.MW_comp = self.thermo_model_source.MW_comp

        # SECTION: liquid density
        self.rho_LIQ_src: Dict[
            str,
            ComponentEquationSource
        ] = self.thermo_model_source.rho_LIQ_src
        # ! values in g/m3
        self.liquid_density_constant_values = self.thermo_model_inputs.liquid_density_constant_values
        self.liquid_density_constant_comp = self.thermo_model_inputs.liquid_density_constant_comp

        # SECTION: heat capacity at liquid phase (Cp_LIQ)
        self.Cp_LIQ_src: Dict[
            str,
            ComponentEquationSource
        ] = self.thermo_model_source.Cp_LIQ_src
        # ! values in J/mol.K
        self.liquid_heat_capacity_constant_values = self.thermo_model_inputs.liquid_heat_capacity_constant_values
        self.liquid_heat_capacity_constant_comp = self.thermo_model_inputs.liquid_heat_capacity_constant_comp

    # SECTION: Thermodynamic property calculations
    # ! Calculate heat capacity at ideal gas for the components (Cp_IG)

    def calc_Cp_IG(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the ideal gas heat capacity (Cp_IG) in J/mol.K for the components in the batch reactor at the specified temperature.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the ideal gas heat capacity (Cp_IG) for the components in the batch reactor.

        Returns
        -------
        np.ndarray
            An array of ideal gas heat capacity (Cp_IG) values for the components in the batch reactor, calculated at the specified temperature.
        """
        # NOTE: temperature in K

        # NOTE: calculate heat capacity at ideal gas for the components based on the heat capacity mode
        if self.gas_heat_capacity_mode == "temperature-dependent":
            # NOTE: calculate temperature-dependent heat capacity
            # ! to J/mol.K
            Cp_IG_values = self.calc_Cp_IG_real(
                inputs={
                    "T": {
                        "value": temperature.value,
                        "unit": temperature.unit
                    }
                },
            )
        elif self.gas_heat_capacity_mode == "constant":
            # NOTE: use constant heat capacity from model inputs
            # ! J/mol.K
            Cp_IG_values = self.gas_heat_capacity_constant_values
        else:
            raise ValueError(
                f"Invalid heat_capacity_mode '{self.gas_heat_capacity_mode}'. Must be 'temperature-dependent' or 'constant'."
            )

        # >> check heat capacity values
        if Cp_IG_values is None:
            raise ValueError(
                "Heat capacity values could not be calculated or retrieved."
            )

        return Cp_IG_values

    # ! Heat capacity at ideal gas at temperature T (Cp_IG)
    def calc_Cp_IG_real(
            self,
            inputs: Dict[str, Any],
    ) -> np.ndarray:
        """
        Calculate the ideal gas heat capacity (Cp_IG) in J/mol.K for the components in the batch reactor at the specified temperature using temperature-dependent heat capacity values.
        """
        # NOTE: extract Cp_IG at reference temperature (e.g., 298 K)
        Cp_IG_ref: Dict[str, CustomProperty] = {}
        for comp in self.component_ids:
            eq_src = self.Cp_IG_src.get(comp)
            if eq_src is None:
                raise ValueError(
                    f"No Cp_IG source found for component: {comp}"
                )

            # >> execute equation to get Cp_IG at reference conditions
            cp_value = exec_component_eq(
                component_eq_src=eq_src,
                inputs=inputs,
                output_unit="J/mol.K"
            )
            if cp_value is None:
                raise ValueError(
                    f"Failed to extract Cp_IG value for component: {comp}"
                )

            # store
            Cp_IG_ref[comp] = cp_value

        # NOTE: build a list of Cp_IG values for all components
        Cp_IG_values = []

        # iterate over components
        for comp in self.component_ids:
            cp_value = Cp_IG_ref.get(comp)
            if cp_value is None:
                raise ValueError(
                    f"No Cp_IG value found for component: {comp}")
            Cp_IG_values.append(cp_value.value)

        # >> to numpy
        Cp_IG_values = np.array(Cp_IG_values, dtype=float)

        return Cp_IG_values

    # ! Calculate heat capacity at liquid phase for the components (Cp_LIQ)
    def calc_Cp_LIQ(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the liquid phase heat capacity (Cp_LIQ) in J/mol.K for the components in the batch reactor at the specified temperature.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the liquid phase heat capacity (Cp_LIQ) for the components in the batch reactor.

        Returns
        -------
        np.ndarray
            An array of liquid phase heat capacity (Cp_LIQ) values for the components in the batch reactor, calculated at the specified temperature.
        """
        # NOTE: calculate heat capacity at liquid phase for the components based on the heat capacity mode
        if self.liquid_heat_capacity_mode == "temperature-dependent":
            # NOTE: calculate temperature-dependent heat capacity
            Cp_LIQ_values = self.calc_Cp_LIQ_real(
                inputs={
                    "T": {
                        "value": temperature.value,
                        "unit": temperature.unit
                    }
                },
            )
        elif self.liquid_heat_capacity_mode == "constant":
            # NOTE: use constant heat capacity from model inputs
            # ! J/mol.K
            Cp_LIQ_values = self.liquid_heat_capacity_constant_values
        else:
            raise ValueError(
                f"Invalid heat_capacity_mode '{self.liquid_heat_capacity_mode}'. Must be 'temperature-dependent' or 'constant'."
            )

        # >> check heat capacity values
        if Cp_LIQ_values is None:
            raise ValueError(
                "Heat capacity values could not be calculated or retrieved."
            )

        return Cp_LIQ_values

    # ! Heat capacity at liquid phase at temperature T (Cp_LIQ)
    def calc_Cp_LIQ_real(
            self,
            inputs: Dict[str, Any],
    ) -> np.ndarray:
        """
        Calculate the liquid phase heat capacity (Cp_LIQ) in J/mol.K for the components in the batch reactor at the specified temperature using temperature-dependent heat capacity values.
        """
        # NOTE: extract Cp_LIQ at reference temperature (e.g., 298 K)
        Cp_LIQ_ref: Dict[str, CustomProperty] = {}
        for comp in self.component_ids:
            eq_src = self.Cp_LIQ_src.get(comp)
            if eq_src is None:
                raise ValueError(
                    f"No Cp_LIQ source found for component: {comp}"
                )

            # >> execute equation to get Cp_LIQ at reference conditions
            # ! to J/mol.K
            cp_value = exec_component_eq(
                component_eq_src=eq_src,
                inputs=inputs,
                output_unit="J/mol.K"
            )
            if cp_value is None:
                raise ValueError(
                    f"Failed to extract Cp_LIQ value for component: {comp}"
                )

            # store
            Cp_LIQ_ref[comp] = cp_value

        # NOTE: build a list of Cp_LIQ values for all components
        Cp_LIQ_values = []

        # iterate over components
        for comp in self.component_ids:
            cp_value = Cp_LIQ_ref.get(comp)
            if cp_value is None:
                raise ValueError(
                    f"No Cp_LIQ value found for component: {comp}")
            Cp_LIQ_values.append(cp_value.value)

        # >> to numpy
        Cp_LIQ_values = np.array(Cp_LIQ_values, dtype=float)

        return Cp_LIQ_values

    # ! Calculate change in heat capacity at ideal gas for the reactions (ΔCp_IG)

    def calc_dCp_IG(
            self,
    ):
        """
        Calculate the change in heat capacity at ideal gas (ΔCp_IG) in J/K for the reactions as:
            ΔCp_rxn = sum(nu_i * Cp_IG_i) for all components i

        Returns
        -------
        np.ndarray
            An array of changes in heat capacity at ideal gas (ΔCp_IG) for each reaction.
        """
        # res
        dCp = []

        # NOTE: check heat capacity constant
        if (
            self.gas_heat_capacity_constant_comp is None or
            self.gas_heat_capacity_constant_values is None or
            len(self.gas_heat_capacity_constant_comp) == 0 or
            len(self.gas_heat_capacity_constant_values) == 0
        ):
            return np.array(dCp, dtype=float)

        # NOTE: calculate for non-isothermal case
        # iterate over reactions
        for rxn in self.reactions:
            # >> calculate reaction enthalpy for the reaction at 298 K
            # stoichiometry matrix
            nu = rxn.reaction_stoichiometry_matrix
            nu = np.array(nu, dtype=float)

            # components
            components = rxn.available_components
            # >> check
            if components is None:
                raise ValueError(
                    f"No components found for reaction: {rxn.name}")

            # heat capacity at ideal gas at reference temperature (e.g., 298 K) for the components
            # ! J/mol.K
            Cp_IG_values, _ = find_components_property(
                components=components,
                prop_values=self.gas_heat_capacity_constant_comp,
                component_key="Formula-State"
            )

            # calculate mix heat capacity change for the reaction using the formula:
            # ΔCp_rxn = sum(nu_i * Cp_IG_i) for all components i
            dCp_rxn = np.sum(nu * Cp_IG_values)

            # >> check
            if dCp_rxn is None:
                raise ValueError(
                    f"Failed to calculate heat capacity change for reaction: {rxn.name}"
                )

            dCp.append(dCp_rxn)

        # NOTE: convert to numpy array
        res = np.array(dCp, dtype=float)

        return res

    # ! Calculate reaction enthalpies (ΔH) for reactions at 298 K
    def calc_dH_rxns_298(
            self,
    ):
        """
        Calculate the reaction enthalpies (ΔH) in J for the reactions in the reactor at 298 K using the provided reaction rates and components.

        Returns
        -------
        np.ndarray
            An array of reaction enthalpies (ΔH) for the reactions in the reactor, calculated at 298 K.
        """
        # res
        dH_rxns = []

        # NOTE: isothermal case
        if self.heat_transfer_mode == "isothermal":
            return np.array(dH_rxns, dtype=float)

        # NOTE: non-isothermal case
        # iterate over reactions
        for rxn in self.reactions:
            # >> calculate reaction enthalpy for the reaction at 298 K
            # stoichiometry matrix
            nu = rxn.reaction_stoichiometry_matrix
            nu = np.array(nu, dtype=float)

            # components
            components = rxn.available_components
            # >> check
            if components is None:
                raise ValueError(
                    f"No components found for reaction: {rxn.name}"
                )

            # Enthalpy of formation at 298 K for the components
            # ! in J/mol
            EnFo_IG_298_values, _ = find_components_property(
                components=components,
                prop_values=self.EnFo_IG_298_comp,
                component_key=cast(ComponentKey, self.component_key)
            )

            # calc
            # ! in J
            dH_rxn = EnFo_IG_298_values @ nu

            # >> check
            if dH_rxn is None:
                raise ValueError(
                    f"Failed to calculate reaction enthalpy for reaction: {rxn.name}"
                )

            dH_rxns.append(dH_rxn)

        # NOTE: convert to numpy array
        res = np.array(dH_rxns, dtype=float)

        return res

    # ! Calculate reaction enthalpies (ΔH) for reactions at temperature T
    def calc_dH_rxns(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the reaction enthalpies (ΔH) for the reactions in the reactor.

        """
        # ! calculate heat generated by reactions: Q_rxn = V Σ_k [(-ΔH_k) r_k]
        # V[m3], ΔH[J/mol], r[mol/m3.s] => Q_rxn [J/s] or [W]
        # ??? ΔH_k
        if self.gas_heat_capacity_mode == "temperature-dependent":
            # NOTE: calculate temperature-dependent enthalpy of formation
            delta_h = self.calc_dH_rxns_real(
                temperature=temperature
            )
        elif self.gas_heat_capacity_mode == "constant":
            # NOTE: use constant enthalpy of formation from model inputs
            delta_h = self.calc_dH_rxns_linear(
                temperature=temperature,
            )
        else:
            raise ValueError(
                f"Invalid heat_capacity_mode '{self.gas_heat_capacity_mode}'. Must be 'temperature-dependent' or 'constant'."
            )

        # >> check enthalpy of formation values
        if delta_h is None:
            raise ValueError(
                "Enthalpy of formation values could not be calculated or retrieved."
            )

        return delta_h

    # ! Calculate reaction enthalpies (ΔH) for reactions at temperature T using temperature-dependent heat capacity values
    def calc_dH_rxns_real(
            self,
            temperature: Temperature,
    ) -> np.ndarray:
        """
        Calculate the reaction enthalpies (ΔH) in J/mol for the reactions in the gas-phase batch reactor using the provided reaction rates and components.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the reaction enthalpies (ΔH) for the reactions
            in the gas-phase batch reactor.

        Returns
        -------
        np.ndarray
            An array of reaction enthalpies (ΔH) for the reactions in the gas-phase batch reactor, calculated at the specified temperature.
        """
        # create model source
        model_source = self.source.model_source
        # >> check
        if model_source is None:
            raise ValueError(
                "Model source is required to calculate reaction enthalpies."
            )

        # res
        dH_rxns = []

        # iterate over reactions
        for i, rxn in enumerate(self.reactions):
            # >> calculate reaction enthalpy for the reaction at the specified temperature
            dH_rxn = dH_rxn_STD(
                reaction=rxn,
                temperature=temperature,
                model_source=model_source,
                hsg_reaction=self.hsg_reactions[i]
            )

            # >> check
            if dH_rxn is None:
                raise ValueError(
                    f"Failed to calculate reaction enthalpy for reaction: {rxn.name}")

            # ! set unit to J/mol
            if dH_rxn.unit != "J/mol":
                # convert
                dH_rxn_converted = to_J_per_mol(
                    value=dH_rxn.value,
                    unit=dH_rxn.unit
                )
                # set
                dH_rxn = CustomProp(
                    value=dH_rxn_converted,
                    unit="J/mol"
                )

            dH_rxns.append(dH_rxn)

        # NOTE: convert to numpy array
        res = [dH.value for dH in dH_rxns]
        res = np.array(res, dtype=float)

        return res

    # ! Calculate reaction enthalpies (ΔH) for reactions at temperature T using constant heat capacity values
    def calc_dH_rxns_linear(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the average reaction enthalpies (ΔH) for the reactions as:
            ΔH_rxn_avg = ΔH_rxn_298 + sum(nu_i * Cp_IG_i * (T - 298)) for all components i

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the average reaction enthalpies (ΔH) for the reactions
            in the reactor.

        Returns
        -------
        np.ndarray
            An array of average reaction enthalpies (ΔH) for the reactions in the gas-phase batch reactor, calculated at the specified temperature.
        """
        # res
        dH_rxns = []

        # iterate over reactions
        for rxn, dH_rxn_298, dCp_mix in zip(self.reactions, self.dH_rxns_298, self.dCp_rxns):
            # nu
            nu = rxn.reaction_stoichiometry_matrix
            nu = np.array(nu, dtype=float)

            # calculate average reaction enthalpy using the formula:
            # ΔH_rxn_avg = ΔH_rxn_298 + dCp_mix * (T - 298)
            T = to_K(temperature.value, temperature.unit)
            dH_rxn_avg = dH_rxn_298 + dCp_mix * (T - self.T_ref_K)

            # >> check
            if dH_rxn_avg is None:
                raise ValueError(
                    f"Failed to calculate average reaction enthalpy for reaction: {rxn.name}")

            dH_rxns.append(dH_rxn_avg)

        # NOTE: convert to numpy array
        res = np.array(dH_rxns, dtype=float)

        return res

    # SECTION: Property configuration methods
    # ! set enthalpy of formation unit to J/mol

    def _config_EnFo_IG_unit(
            self,
    ) -> Dict[str, float]:
        """
        Configure the unit for the enthalpy of formation at ideal gas at 298 K (EnFo_IG_298) in J/mol for the components in the batch reactor based on the model source and reactor configuration.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are component IDs and the values are the enthalpy of formation values for the components in J/mol.
        """
        # NOTE: extract EnFo_IG_298 at reference temperature (e.g., 298 K) and convert to output unit
        res = {}

        # normalized to unit
        for comp in self.component_ids:
            dt_src = self.EnFo_IG_298_src.get(comp)
            if dt_src is None:
                raise ValueError(
                    f"No EnFo_IG_298 source found for component: {comp}")

            # >> extract EnFo_IG_298 value
            dt_value = dt_src.get("value")
            dt_unit = dt_src.get("unit")

            # >> check
            if (
                dt_value is None or
                dt_unit is None
            ):
                raise ValueError(
                    f"Failed to extract EnFo_IG_298 value or unit for component: {comp}"
                )

            # ! to J/mol
            if dt_unit != "J/mol":
                # >> convert to output unit
                dt_value = to_J_per_mol(
                    value=dt_value,
                    unit=dt_unit
                )

            res[comp] = dt_value

        return res

    # ! set molecular weight unit to g/mol
    def _config_MW_unit(
            self,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Configure the unit for the molecular weight (MW) for the components in the batch reactor based on the model source and reactor configuration.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, float]]
            A tuple containing:
            - An array of molecular weight values for the components in g/mol.
            - A dictionary where the keys are component IDs and the values are the molecular weight values for the components in g/mol.
        """
        # NOTE: extract MW at reference temperature (e.g., 298 K) and convert to output unit
        res = []
        res_comp = {}

        # normalized to unit
        for comp in self.component_ids:
            dt_src = self.MW_src.get(comp)
            if dt_src is None:
                raise ValueError(
                    f"No MW source found for component: {comp}")

            # >> extract MW value
            dt_value = dt_src.get("value")
            dt_unit = dt_src.get("unit")

            # >> check
            if (
                dt_value is None or
                dt_unit is None
            ):
                raise ValueError(
                    f"Failed to extract MW value or unit for component: {comp}")

            # >> convert to output unit
            dt_value_converted = to_g_per_mol(
                value=dt_value,
                unit=dt_unit
            )

            res_comp[comp] = dt_value_converted
            res.append(dt_value_converted)

        # convert to numpy array
        res = np.array(res, dtype=float)

        return res, res_comp

    # ! Calculate liquid density (rho_LIQ) for the components at temperature T
    def calc_rho_LIQ(
            self,
            temperature: Temperature,
    ) -> np.ndarray:
        """
        Calculate the density of the liquid either using a constant value from model inputs or using a temperature-dependent equation of state.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the liquid density.

        Returns
        -------
        np.ndarray
            An array of density values for the liquid phase, calculated at the specified temperature.
        """
        # NOTE: calculate density based on the density mode
        if self.liquid_density_mode == "temperature-dependent":
            # NOTE: calculate temperature-dependent density
            # ! to g/m3
            rho_LIQ_values = self.calc_rho_LIQ_real(
                inputs={
                    "T": {
                        "value": temperature.value,
                        "unit": temperature.unit
                    }
                },
            )
        elif self.liquid_density_mode == "constant":
            # NOTE: use constant density from model inputs
            # ! in g/m3
            rho_LIQ_values = self.liquid_density_constant_values
        else:
            raise ValueError(
                f"Invalid density_mode '{self.liquid_density_mode}'. Must be 'temperature-dependent' or 'constant'."
            )

        # >> check density values
        if rho_LIQ_values is None:
            raise ValueError(
                "Density values could not be calculated or retrieved."
            )

        return rho_LIQ_values

    # ! Calculate liquid density (rho_LIQ)
    def calc_rho_LIQ_real(
            self,
            inputs: Dict[str, Any],
    ) -> np.ndarray:
        """
        Calculate the density of the liquid phase in g/m3 for the components in the batch reactor at the specified temperature using temperature-dependent equations of state.
        """
        # NOTE: extract density at reference temperature (e.g., 298 K)
        rho_LIQ_ref: Dict[str, CustomProperty] = {}
        for comp in self.component_ids:
            eq_src = self.rho_LIQ_src.get(comp)
            if eq_src is None:
                raise ValueError(
                    f"No density source found for component: {comp}"
                )

            # >> execute equation to get density at reference conditions
            rho_value = exec_component_eq(
                component_eq_src=eq_src,
                inputs=inputs,
                output_unit="g/m3"
            )
            if rho_value is None:
                raise ValueError(
                    f"Failed to extract density value for component: {comp}"
                )

            # store
            rho_LIQ_ref[comp] = rho_value

        # NOTE: build a list of density values for all components
        rho_LIQ_values = []

        # iterate over components
        for comp in self.component_ids:
            rho_value = rho_LIQ_ref.get(comp)
            if rho_value is None:
                raise ValueError(
                    f"No density value found for component: {comp}")
            rho_LIQ_values.append(rho_value.value)

        # >> to numpy
        rho_LIQ_values = np.array(rho_LIQ_values, dtype=float)

        return rho_LIQ_values

    # SECTION: Liquid Enthalpy
    # ! Calculate liquid phase enthalpy (En_LIQ)
    def calc_En_LIQ_ref(
            self,
            temperature: Temperature,
    ) -> Tuple[Dict[str, CustomProp], np.ndarray]:
        """
        Calculate the liquid phase enthalpy (En_LIQ) in J/mol for the components in the reactor at the specified temperature using the ideal gas reference state.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the liquid phase enthalpy (En_LIQ) for the components in the batch reactor.

        Returns
        -------
        Tuple[Dict[str, CustomProp], np.ndarray]
            A tuple containing:
            - A dictionary where the keys are component IDs and the values are the liquid phase enthalpy values for the components in J/mol, calculated at the specified temperature.
            - An array of liquid phase enthalpy (En_LIQ) values for the components in the batch reactor, calculated at the specified temperature.
        """
        # NOTE: check heat transfer mode
        if self.heat_transfer_mode == "isothermal":
            return {}, np.array([])

        # NOTE: calculate liquid phase enthalpy for the components based on the heat capacity mode
        res = {}
        res_array = []

        # iterate over components
        for i, comp in enumerate(self.component_ids):
            # >> component
            component_ = self.components[i]
            # >> component id [Name-Formula]
            component_id_ = self.component_formula_state[i]

            # >> calculate liquid phase enthalpy for the component at the specified temperature
            En_LIQ_res: ComponentEnthalpy | None = calc_En_IG_ref(
                component=component_,
                temperature=temperature,
                model_source=self.model_source,
                component_key="Name-Formula"
            )

            # >> check
            if En_LIQ_res is None:
                raise ValueError(
                    f"Failed to calculate liquid phase enthalpy for component: {comp}"
                )

            # ! set unit to J/mol
            if En_LIQ_res.unit != "J/mol":
                # convert
                En_LIQ_value_converted = to_J_per_mol(
                    value=En_LIQ_res.value,
                    unit=En_LIQ_res.unit
                )
                # set
                En_LIQ = CustomProp(
                    value=En_LIQ_value_converted,
                    unit="J/mol"
                )
            else:
                En_LIQ = CustomProp(
                    value=En_LIQ_res.value,
                    unit=En_LIQ_res.unit
                )

            res[component_id_] = En_LIQ
            # add to array
            res_array.append(En_LIQ.value)

        # >> check enthalpy values
        if res is None:
            raise ValueError(
                "Enthalpy values could not be calculated or retrieved."
            )

        # to numpy array
        res_array = np.array(res_array, dtype=float)

        return res, res_array

    # SECTION: Reaction Enthalpy calculations (liquid phase)
    def calc_dH_rxns_LIQ(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the reaction enthalpies (ΔH) for the reactions in the liquid phase reactor.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the reaction enthalpies (ΔH) for the reactions in the liquid phase  reactor.

        Returns
        -------
        np.ndarray
            An array of reaction enthalpies (ΔH) for the reactions in the liquid phase reactor, calculated at the specified temperature.
        """
        # NOTE: calculate based on reaction enthalpy mode
        if self.reaction_enthalpy_mode == "ideal_gas":
            # NOTE: calculate reaction enthalpies using ideal gas reference state
            dH_rxns = self.calc_dH_rxns_IG_ref(
                temperature=temperature
            )
        elif self.reaction_enthalpy_mode == "liquid":
            # NOTE: calculate reaction enthalpies using liquid phase enthalpy
            # dH_rxns = self.calc_dH_rxns_LIQ_real(
            #     temperature=temperature
            # )
            raise NotImplementedError(
                "Calculation of reaction enthalpies using liquid phase enthalpy is not implemented yet."
            )
        else:
            raise ValueError(
                f"Invalid reaction_enthalpy_mode '{self.reaction_enthalpy_mode}'. Must be 'ideal_gas' or 'liquid'."
            )

        return dH_rxns

    # ! Calculate reaction enthalpies (ΔH) for reactions at temperature T using ideal gas reference state

    def calc_dH_rxns_IG_ref(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the reaction enthalpy (ΔH) in J/mol for all reaction using the ideal gas reference state.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the reaction enthalpy (ΔH) for the reactions in the batch reactor.

        Returns
        -------
        float
            The reaction enthalpy (ΔH) for the specified reaction at the specified temperature, calculated using the ideal gas reference state.
        """
        # NOTE: calculate reaction enthalpy using ideal gas reference state
        res = []

        # NOTE: calculate liquid phase enthalpy for the components at reference temperature (e.g., 298 K)
        # ! in J/mol
        En_LIQ_comp, _ = self.calc_En_LIQ_ref(temperature=temperature)

        # NOTE: calculate reaction enthalpy for each reaction using the ideal gas reference state
        # iterate over reactions
        for rxn in self.reactions:
            # >> calculate reaction enthalpy for the reaction using ideal gas reference state
            dH_rxn = dH_rxn_reactions(
                reaction=rxn,
                H_i_IG=En_LIQ_comp,
            )

            # >> check
            if dH_rxn is None:
                raise ValueError(
                    f"Failed to calculate reaction enthalpy for reaction: {rxn.name}"
                )

            # ! set unit to J/mol
            if dH_rxn.unit != "J/mol":
                # convert
                dH_rxn_converted = to_J_per_mol(
                    value=dH_rxn.value,
                    unit=dH_rxn.unit
                )
                # set
                dH_rxn = CustomProp(
                    value=dH_rxn_converted,
                    unit="J/mol"
                )

            # store
            res.append(dH_rxn.value)

        # >> check enthalpy values
        if res is None:
            raise ValueError(
                "Reaction enthalpy values could not be calculated or retrieved."
            )

        # NOTE: convert to numpy array
        res = np.array(res, dtype=float)

        return res

    # SECTION: Calculate Enthalpy Stream
    def calc_En_IG(
            self,
            temperature: Temperature,
    ) -> Tuple[Dict[str, CustomProp], np.ndarray]:
        """
        Calculate the ideal gas enthalpy (En_IG) in J/mol for the components in the reactor at the specified temperature.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the ideal gas enthalpy (En_IG) for the components in the reactor.

        Returns
        -------
        Tuple[Dict[str, CustomProp], np.ndarray]
            A tuple containing:
            - A dictionary where the keys are component IDs and the values are the ideal gas enthalpy values for the components in J/mol, calculated at the specified temperature.
            - An array of ideal gas enthalpy (En_IG) values for the components in the batch reactor, calculated at the specified temperature.
        """
        # NOTE: calculate ideal gas enthalpy for the components based on the heat capacity mode
        res = []
        res_comp = {}

        # iterate over components
        for i, comp in enumerate(self.component_ids):
            # >> component
            component_ = self.components[i]
            # >> component id [Name-Formula]
            component_id_ = self.component_formula_state[i]

            # >> calculate ideal gas enthalpy for the component at the specified temperature
            En_IG_res: ComponentEnthalpy | None = calc_En(
                component=component_,
                temperature=temperature,
                model_source=self.model_source,
                component_key="Name-Formula"
            )

            # >> check
            if En_IG_res is None:
                raise ValueError(
                    f"Failed to calculate ideal gas enthalpy for component: {comp}"
                )

            # ! set unit to J/mol
            if En_IG_res.unit != "J/mol":
                # convert
                En_IG_value_converted = to_J_per_mol(
                    value=En_IG_res.value,
                    unit=En_IG_res.unit
                )
                # set
                En_IG = CustomProp(
                    value=En_IG_value_converted,
                    unit="J/mol"
                )
            else:
                En_IG = CustomProp(
                    value=En_IG_res.value,
                    unit=En_IG_res.unit
                )

            # add
            res_comp[component_id_] = En_IG
            res.append(En_IG.value)

        # >> check enthalpy values
        if res_comp is None:
            raise ValueError(
                "Enthalpy values could not be calculated or retrieved."
            )

        # convert to numpy array
        res = np.array(res, dtype=float)

        return res_comp, res
