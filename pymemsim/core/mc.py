# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Literal
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey, Volume
# ! locals
from ..utils.unit_tools import to_m3, to_Pa, to_K, to_W_per_m2_K, to_m2, to_W, to_mol_per_s
from ..utils.tools import collect_keys
from ..models.hfm import HollowFiberMembraneOptions
from ..models.heat import HeatTransferOptions


class MembraneCore:
    def __init__(
        self,
        components: List[Component],
        model_inputs: Dict[str, Any],
        heat_transfer_options: HeatTransferOptions,
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        """
        Initializes the BatchReactor instance with the provided components, source, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the batch reactor.
        model_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values. This can include feed specifications, initial conditions, or any other relevant parameters needed for the simulations.
        heat_transfer_options : HeatTransferOptions
            A HeatTransferOptions object containing the options for configuring the heat transfer in the batch reactor, such as heat transfer mode, jacket temperature, heat transfer coefficient, and heat transfer area.
        component_refs : Dict[str, Any]
            A dictionary containing references for the components, which can include component IDs, formula states, mappers, and any other relevant information needed for the simulations.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the batch reactor.
        """
        # NOTE: Set attributes
        self.components = components
        self.model_inputs = model_inputs
        self.heat_transfer_options = heat_transfer_options
        self.component_refs = component_refs
        self.component_key = component_key

        # SECTION: Heat exchange configuration
        self.heat_transfer_mode = heat_transfer_options.heat_transfer_mode
        self.jacket_temperature = heat_transfer_options.jacket_temperature
        self.heat_transfer_coefficient = heat_transfer_options.heat_transfer_coefficient
        self.heat_transfer_area = heat_transfer_options.heat_transfer_area
        self.heat_rate = heat_transfer_options.heat_rate

        # SECTION: component IDs and related properties
        # ! component refs
        self.component_num = component_refs['component_num']
        self.component_ids = component_refs['component_ids']
        self.component_formula_state = component_refs['component_formula_state']
        self.component_mapper = component_refs['component_mapper']
        self.component_id_to_index = component_refs['component_id_to_index']

        # SECTION: Process model configuration
        # lower case keys for easier access
        self.model_inputs_keys = collect_keys(self.model_inputs)

    def _extract_value_unit(self, value: Any) -> Tuple[float, str]:
        """
        Extract a numeric value and unit from scalar/model/mapping input.

        Returns
        -------
        Tuple[float, str]
            (value, unit). Unit defaults to empty string when unavailable.
        """
        if isinstance(value, (int, float)):
            return float(value), ""

        if isinstance(value, dict):
            if "value" in value:
                return float(value["value"]), str(value.get("unit", ""))
            raise ValueError("Mapping input must contain a 'value' key.")

        if hasattr(value, "value"):
            v = float(getattr(value, "value"))
            u = str(getattr(value, "unit", ""))
            return v, u

        raise ValueError(f"Unsupported value type: {type(value)}")

    def _to_mol_per_s_value(self, value: Any) -> float:
        """
        Convert supported scalar/model/mapping input to mol/s.
        """
        raw_value, raw_unit = self._extract_value_unit(value)
        unit = raw_unit.strip()
        if unit == "":
            # default assumption for naked numbers
            return float(raw_value)
        return to_mol_per_s(value=raw_value, unit=unit)

    # SECTION: Model Inputs configuration
    # NOTE: is isothermal
    @property
    def is_isothermal(self) -> bool:
        """Check if the heat transfer mode is isothermal."""
        return self.heat_transfer_mode == "isothermal"

    # NOTE: temperature configuration
    # ! [K]

    def config_temperature(
            self,
    ):
        if "temperature" in self.model_inputs_keys:
            temperature_: Temperature = self.model_inputs["temperature"]
            temperature_value = to_K(
                temperature_.value,
                temperature_.unit
            )
            # >> update
            temperature = Temperature(
                value=temperature_value,
                unit="K"
            )

            return temperature
        else:
            raise ValueError("Temperature must be provided in model_inputs.")

    # NOTE: pressure configuration
    # ! [Pa]
    def config_pressure(
            self,
    ):
        """Configure the pressure for the batch reactor based on the model inputs."""
        if "pressure" in self.model_inputs_keys:
            pressure_: Pressure = self.model_inputs["pressure"]

            # convert to Pa
            pressure_value = to_Pa(
                pressure_.value,
                pressure_.unit
            )
            # >> update
            pressure = Pressure(
                value=pressure_value,
                unit="Pa"
            )

            return pressure
        else:
            raise ValueError("Pressure must be provided in model_inputs.")

    # NOTE: reactor volume configuration
    # ! [m3]
    def config_reactor_volume(
            self,
    ):
        """Configure the reactor volume for the batch reactor based on the model inputs."""
        if 'reactor_volume' not in self.model_inputs_keys:
            raise ValueError(
                "Reactor volume must be provided in model_inputs."
            )

        # set reactor volume from model inputs
        reactor_volume = self.model_inputs['reactor_volume']

        if reactor_volume is not None:
            reactor_volume_value = to_m3(
                reactor_volume.value,
                reactor_volume.unit
            )
            # >> update
            reactor_volume = Volume(
                value=reactor_volume_value,
                unit="m3"
            )

            return reactor_volume
        else:
            raise ValueError(
                "Reactor volume must be provided for constant volume mode.")

    # NOTE: heat transfer mode configuration
    def config_heat_transfer_mode(
            self,
            temperature_value: float,
    ) -> Tuple[Optional[float], float]:
        """
        Configure the temperature fixed and initial temperature based on the heat transfer mode.
        """
        if self.heat_transfer_mode == "isothermal":
            return temperature_value, temperature_value
        elif self.heat_transfer_mode == "non-isothermal":
            return None, temperature_value
        else:
            raise ValueError(
                "Invalid heat_transfer_mode. Must be 'isothermal' or 'non-isothermal'."
            )

    # NOTE: heat exchange configuration
    def config_heat_exchange(
            self,
    ) -> Tuple[bool, float, float, float]:
        """Configure the heat exchange parameters for the batch reactor based on the model inputs."""
        if (
            self.jacket_temperature is not None and
            self.heat_transfer_coefficient is not None and
            self.heat_transfer_area is not None and
            self.heat_transfer_mode == 'non-isothermal'
        ):
            # >> conversions for heat exchange parameters
            self.jacket_temperature = Temperature(
                value=to_K(
                    self.jacket_temperature.value,
                    self.jacket_temperature.unit
                ),
                unit="K"
            )
            # ! [K]
            jacket_temperature_value = self.jacket_temperature.value

            # >> heat transfer coefficient
            # ! [W/m2.K]
            heat_transfer_coefficient_value = to_W_per_m2_K(
                self.heat_transfer_coefficient.value,
                self.heat_transfer_coefficient.unit
            )

            # >> heat transfer area
            # ! [m2]
            heat_transfer_area_value = to_m2(
                self.heat_transfer_area.value,
                self.heat_transfer_area.unit
            )

            return True, jacket_temperature_value, heat_transfer_coefficient_value, heat_transfer_area_value
        else:
            return False, 0.0, 0.0, 0.0

    # NOTE: mole feed configuration
    # ! [mol]
    def config_mole(
            self
    ):
        """Configure the mole feed for the batch reactor based on the model inputs."""
        # check
        if "mole" not in self.model_inputs_keys:
            raise ValueError("Mole feed must be provided in model_inputs.")

        # NOTE: res
        res = []
        res_comp = {}

        # mole feed value dict
        mole_feed = self.model_inputs["mole"]
        mole_feed_keys = list(mole_feed.keys())

        # iterate through component IDs and extract mole feed values
        for comp_id in self.component_formula_state:
            # check if component ID is in mole feed
            if comp_id in mole_feed_keys:
                value_ = mole_feed[comp_id].value
                unit_ = mole_feed[comp_id].unit

                # add to res
                res.append(value_)
                res_comp[comp_id] = value_

        # convert to array
        res = np.array(res, dtype=float)

        return res_comp, res

    # NOTE: heat rate configuration
    # ! [W/s]
    def config_heat_rate(
            self,
    ) -> Optional[float]:
        """Configure the heat rate for the batch reactor based on the model inputs."""
        if (
            self.heat_rate is not None and
            self.heat_transfer_mode == 'non-isothermal'
        ):
            # >> conversion for heat rate
            # ! [W/s]
            heat_rate_value = to_W(
                self.heat_rate.value,
                self.heat_rate.unit
            )

            return heat_rate_value
        else:
            return None

    # NOTE: mole flow configuration
    # ! [mol/s]
    def config_inlet_mole_flows(
            self,
    ) -> Tuple[Dict[str, float], np.ndarray, float]:
        """
        Configure the inlet mole flows for each component in the reactor
        """
        return self.config_inlet_mole_flows_by_key(
            key="inlet_flows",
            required=True,
            default_component_value=0.0
        )

    def config_inlet_mole_flows_by_key(
            self,
            key: str,
            required: bool = True,
            default_component_value: float = 0.0
    ) -> Tuple[Dict[str, float], np.ndarray, float]:
        """
        Configure component inlet mole flows from an arbitrary model_inputs key.

        Parameters
        ----------
        key : str
            model_inputs key containing {component_id: flow_value}.
        required : bool, optional
            If True, missing key raises ValueError. If False, uses defaults.
        default_component_value : float, optional
            Default mol/s for missing components when not required.
        """
        if key not in self.model_inputs_keys:
            if required:
                raise ValueError(f"'{key}' must be provided in model_inputs.")
            default_array = np.full(
                self.component_num, float(default_component_value), dtype=float
            )
            default_comp = {
                comp_id: float(default_component_value)
                for comp_id in self.component_formula_state
            }
            return default_comp, default_array, float(np.sum(default_array))

        mole_flow = self.model_inputs[key]
        if not isinstance(mole_flow, dict):
            raise ValueError(
                f"'{key}' must be a mapping of component ids to flow values."
            )

        res_comp: Dict[str, float] = {}
        res: List[float] = []
        mole_flow_keys = set(mole_flow.keys())

        for comp_id in self.component_formula_state:
            if comp_id not in mole_flow_keys:
                if required:
                    raise ValueError(
                        f"Missing mole flow entry for component '{comp_id}' in '{key}'."
                    )
                value_ = float(default_component_value)
            else:
                value_ = self._to_mol_per_s_value(mole_flow[comp_id])

            res_comp[comp_id] = value_
            res.append(value_)

        inlet_flow_total = float(sum(res))
        return res_comp, np.array(res, dtype=float), inlet_flow_total

    # NOTE: inlet mole flow total configuration
    # ! [mol/s]
    def config_inlet_mole_flow(
            self,
    ) -> float:
        """
        Configure the total inlet mole flow for the reactor based on the model inputs.
        """
        if "inlet_flow_total" not in self.model_inputs_keys:
            raise ValueError(
                "inlet_flow_total must be provided in model_inputs."
            )

        inlet_flow_total = self.model_inputs["inlet_flow_total"]

        if inlet_flow_total is not None:
            inlet_flow_total_value = to_mol_per_s(
                value=float(inlet_flow_total.value),
                unit=inlet_flow_total.unit
            )

            return inlet_flow_total_value
        else:
            raise ValueError(
                "Total inlet mole flow must be provided for flow configuration."
            )
