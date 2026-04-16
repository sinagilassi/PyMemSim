# import libs
import logging
from collections.abc import Mapping
from typing import Any, Dict, List
from pythermodb_settings.models import Component, ComponentKey, Temperature
# locals
from ..models.heat import HeatTransferOptions
from ..utils.unit_tools import to_K, to_Pa
from .mc import MembraneCore
from ..models.hfm import HollowFiberMembraneOptions


# NOTE: logger setup
logger = logging.getLogger(__name__)
# locals


class HFMCore(MembraneCore):
    """
    Hollow fiber membrane core.
    """

    def __init__(
        self,
        components: List[Component],
        model_inputs: Dict[str, Any],
        unit_options: HollowFiberMembraneOptions,
        heat_transfer_options: HeatTransferOptions,
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        """
        Initialize PFR reactor core with validated/normalized configuration.

        Parameters
        ----------
        components : List[Component]
            Components participating in the reactor model.
        model_inputs : Dict[str, Any]
            User model inputs (inlet flows, inlet temperature, reactor volume, pressure).
        unit_options : PFRReactorOptions
            PFR phase/operation/pressure/thermo options.
        heat_transfer_options : HeatTransferOptions
            Heat-transfer mode and optional jacket/constant-heat settings.
        component_refs : Dict[str, Any]
            Component lookup/index references.
        component_key : ComponentKey
            Key used for component mapping and stoichiometry indexing.
        """
        super().__init__(
            components=components,
            model_inputs=model_inputs,
            heat_transfer_options=heat_transfer_options,
            component_refs=component_refs,
            component_key=component_key,
        )

        # SECTION: reactor configuration
        self.phase = unit_options.phase
        self.operation_mode = unit_options.operation_mode
        self.pressure_mode = unit_options.pressure_mode
        self.gas_model = unit_options.gas_model
        self.gas_heat_capacity_mode = unit_options.gas_heat_capacity_mode
        self.liquid_heat_capacity_mode = unit_options.liquid_heat_capacity_mode
        self.liquid_density_mode = unit_options.liquid_density_mode

        # SECTION: process model configuration
        # ! T_in: inlet temperature [K]
        self.temperature_inlet: Temperature = self.config_inlet_temperature()
        self.temperature_inlet_value = self.temperature_inlet.value
        self._T_in = self.temperature_inlet_value

        # ! V_R: reactor volume [m3]
        self.volume = self.config_reactor_volume()
        self.reactor_volume_value = self.volume.value

        # ! F_in: inlet component molar flow rates [mol/s]
        _, self._F_in, self._F_in_total = self.config_inlet_mole_flows()

        # ! P0: inlet/reference pressure [Pa]
        self._P0 = self._config_pressure_initial()

        # SECTION: heat transfer configuration
        (
            self.heat_exchange,
            self.jacket_temperature_value,
            self.heat_transfer_coefficient_value,
            self.heat_transfer_area_value,
        ) = self.config_heat_exchange()

        self.heat_rate_value = self.config_heat_rate()

        # SECTION: convenience flags
        self.is_constant_pressure = self.pressure_mode == "constant"

        # NOTE: mode flags for current simulation case
        self.is_non_isothermal = self.heat_transfer_mode == "non-isothermal"
        self.is_pressure_state_variable = self.pressure_mode == "state_variable"

    # SECTION: model configuration
    def config_model(
            self
    ):
        if (
            self.phase == "gas" and
            self.pressure_mode == "constant" and
            self.operation_mode == "constant_volume"
        ):
            raise ValueError(
                "Invalid gas PFR setup: operation_mode='constant_volume' is incompatible with pressure_mode='constant'."
            )

    # NOTE: temperature configuration
    def config_inlet_temperature(self) -> Temperature:
        """
        Configure and normalize inlet temperature to Kelvin.

        Accepted input shapes
        ---------------------
        - Temperature model
        - Mapping with {'value', 'unit'}
        - numeric scalar (assumed Kelvin)
        """
        if "inlet_temperature" not in self.model_inputs_keys:
            raise ValueError(
                "inlet_temperature must be provided in model_inputs.")

        inlet_temperature = self.model_inputs["inlet_temperature"]

        if isinstance(inlet_temperature, Temperature):
            value = float(inlet_temperature.value)
            unit = inlet_temperature.unit
        elif isinstance(inlet_temperature, Mapping):
            if "value" not in inlet_temperature or "unit" not in inlet_temperature:
                raise ValueError(
                    "inlet_temperature mapping must contain 'value' and 'unit'."
                )
            value = float(inlet_temperature["value"])
            unit = str(inlet_temperature["unit"])
        else:
            value = float(inlet_temperature)
            unit = "K"

        unit = str(unit).strip()
        if not unit:
            raise ValueError(
                "inlet_temperature unit must be a non-empty string.")

        if unit.upper() != "K":
            value = to_K(value=value, unit=unit)

        return Temperature(value=value, unit="K")

    # NOTE: pressure configuration
    def _config_pressure_initial(self) -> float:
        """
        Configure initial pressure [Pa] for PFR initialization.

        Rules
        -----
        - gas phase:
            pressure is required for:
            * pressure_mode='constant'
            * pressure_mode='shortcut'
            * pressure_mode='state_variable'
        - liquid phase:
            pressure is optional and defaults to 0.0
        """
        if self.phase != "gas":
            # NOTE: liquid PFR does not require pressure state/closure in this version.
            if "inlet_pressure" in self.model_inputs_keys:
                pressure = self.model_inputs["inlet_pressure"]
                return to_Pa(value=float(pressure.value), unit=pressure.unit)
            return 0.0

        # NOTE: gas PFR requires pressure input for all current pressure modes.
        if "inlet_pressure" not in self.model_inputs_keys:
            raise ValueError(
                "pressure must be provided in model_inputs for gas-phase PFR."
            )

        pressure = self.model_inputs["inlet_pressure"]
        return to_Pa(value=float(pressure.value), unit=pressure.unit)
