# import libs
import logging
import numpy as np
from collections.abc import Mapping
from typing import Any, Dict, List
from pythermodb_settings.models import Component, ComponentKey, Temperature, CustomProp
import pycuc
# locals
from ..models.heat import HeatTransferOptions
from ..utils.unit_tools import to_W_per_m2_K, from_gpu_to_mol_per_s_m2_Pa
from .mc import MembraneCore
from ..models.hfm import HollowFiberMembraneOptions, HollowFiberMembraneModuleGeometry
from .hfm_module import HFMModule


# NOTE: logger setup
logger = logging.getLogger(__name__)


class HFMCore(MembraneCore):
    """
    Hollow fiber membrane core with normalized dual-side inputs.
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
        # SECTION: base initialization
        super().__init__(
            components=components,
            model_inputs=model_inputs,
            heat_transfer_options=heat_transfer_options,
            component_refs=component_refs,
            component_key=component_key,
        )

        # SECTION: core options
        self.phase = unit_options.phase
        self.flow_pattern = self._normalize_flow_pattern(
            unit_options.flow_pattern)
        self.operation_mode = getattr(
            unit_options, "operation_mode", "constant_pressure"
        )
        self.feed_pressure_mode = getattr(
            unit_options, "feed_pressure_mode", "constant"
        )
        self.permeate_pressure_mode = getattr(
            unit_options, "permeate_pressure_mode", "constant"
        )
        self.gas_model = unit_options.gas_model
        self.gas_heat_capacity_mode = unit_options.gas_heat_capacity_mode
        self.liquid_heat_capacity_mode = unit_options.liquid_heat_capacity_mode
        self.liquid_density_mode = unit_options.liquid_density_mode

        # NOTE: this pass supports only constant pressure on both sides.
        if (
            self.feed_pressure_mode != "constant" or
            self.permeate_pressure_mode != "constant"
        ):
            raise NotImplementedError(
                "HFM currently supports only constant feed/permeate pressure modes."
            )

        # SECTION: side-specific inlet states with backward compatibility keys
        # ! feed inlet total flow [mol/s] when feed is specified via total flow + composition
        self.feed_inlet_flow = None

        # ! feed mole fractions by component id when feed is specified via total flow + composition
        self.feed_mole_fractions = None

        # ! feed-side inlet flows [mol/s]
        (
            self.feed_inlet_flows_comp,
            self.feed_inlet_flows,
            self.feed_inlet_flow_total
        ) = self._config_feed_inlet_flows()

        # ! permeate-side inlet flows [mol/s]
        (
            self.permeate_inlet_flows_comp,
            self.permeate_inlet_flows,
            self.permeate_inlet_flow_total
        ) = self._config_side_inlet_flows(
            primary_key="permeate_inlet_flows",
            fallback_key=None,
            required=False,
            default_component_value=0.0,
        )

        # ! feed-side inlet temperature [K]
        self.feed_inlet_temperature = self._config_side_temperature(
            primary_key="feed_inlet_temperature",
            fallback_key="inlet_temperature",
            required=True,
            fallback_value_K=None,
        )
        # ! permeate-side inlet temperature [K]
        self.permeate_inlet_temperature = self._config_side_temperature(
            primary_key="permeate_inlet_temperature",
            fallback_key=None,
            required=False,
            fallback_value_K=float(self.feed_inlet_temperature.value),
        )

        # ! feed-side pressure [Pa]
        self.feed_pressure = self._config_side_pressure(
            primary_key="feed_pressure",
            fallback_key="inlet_pressure",
            required=True,
            fallback_value_Pa=None,
        )
        # ! permeate-side pressure [Pa]
        self.permeate_pressure = self._config_side_pressure(
            primary_key="permeate_pressure",
            fallback_key=None,
            required=False,
            fallback_value_Pa=float(self.feed_pressure),
        )

        # SECTION: membrane module properties
        # ! membrane area per unit length [m] (= m2/m)
        self.membrane_area_per_length = self._config_membrane_area_per_length()

        # SECTION: membrane transport and thermal parameters
        # ! overall heat-transfer coefficient [W/m2.K]
        self.overall_heat_transfer_coefficient = self._config_optional_per_area_u(
            key="overall_heat_transfer_coefficient",
            default_value=0.0
        )
        # ! feed-side external heat flux [W/m2]
        self.q_ext_feed = self._config_optional_qext(
            key="q_ext_feed",
            default_value=0.0
        )
        # ! permeate-side external heat flux [W/m2]
        self.q_ext_permeate = self._config_optional_qext(
            key="q_ext_permeate",
            default_value=0.0
        )

        # ! gas transport coefficients aligned with component order
        self.gas_transport_coefficients = self._config_transport_coefficients(
            key="gas_transport_coefficients",
            required=(self.phase == "gas")
        )
        # ! liquid transport coefficients aligned with component order
        self.liquid_transport_coefficients = self._config_transport_coefficients(
            key="liquid_transport_coefficients",
            required=(self.phase == "liquid")
        )

        # SECTION: retained global heat-transfer options (legacy support)
        (
            self.heat_exchange,
            self.jacket_temperature_value,
            self.heat_transfer_coefficient_value,
            self.heat_transfer_area_value,
        ) = self.config_heat_exchange()
        self.heat_rate_value = self.config_heat_rate()

        # NOTE: thermal flags
        self.is_non_isothermal = self.heat_transfer_mode == "non-isothermal"

        # SECTION: final validation
        self.config_model()

    # SECTION: Property accessors for public outputs
    @property
    def is_counter_current(self) -> bool:
        return self.flow_pattern == "counter-current"

    @property
    def is_co_current(self) -> bool:
        return self.flow_pattern == "co-current"

    @property
    def permeate_axial_sign(self) -> int:
        return -1 if self.is_counter_current else 1

    @staticmethod
    def _normalize_flow_pattern(flow_pattern: str) -> str:
        normalized = flow_pattern.strip().lower().replace("-", "").replace("_", "")
        if normalized == "cocurrent":
            return "co-current"
        if normalized == "countercurrent":
            return "counter-current"
        raise ValueError(
            "Invalid flow_pattern. Supported values are "
            "'co-current', 'counter-current', 'cocurrent', 'countercurrent'."
        )

    # SECTION: model validation
    def config_model(self):
        # NOTE: geometric and pressure sanity checks
        if self.membrane_area_per_length <= 0.0:
            raise ValueError(
                "membrane_area_per_length must be positive."
            )
        if self.feed_pressure <= 0.0 or self.permeate_pressure <= 0.0:
            raise ValueError(
                "feed_pressure and permeate_pressure must be positive."
            )

    # SECTION: side-input helpers
    def _config_feed_inlet_flows(self):
        has_explicit_flows = (
            "feed_inlet_flows" in self.model_inputs_keys or
            "inlet_flows" in self.model_inputs_keys
        )
        has_total_flow = "feed_inlet_flow" in self.model_inputs_keys
        has_mole_fractions = "feed_mole_fractions" in self.model_inputs_keys

        # NOTE: keep a single input mode to avoid ambiguous configuration.
        if has_explicit_flows and (has_total_flow or has_mole_fractions):
            raise ValueError(
                "Ambiguous feed inlet specification. Use either "
                "'feed_inlet_flows' (or legacy 'inlet_flows') OR "
                "the pair 'feed_inlet_flow' + 'feed_mole_fractions', not both."
            )

        # NOTE: existing mode: component-wise feed inlet flows.
        if has_explicit_flows:
            return self._config_side_inlet_flows(
                primary_key="feed_inlet_flows",
                fallback_key="inlet_flows",
                required=True,
                default_component_value=0.0,
            )

        # NOTE: new mode: total feed flow + feed mole fractions.
        if has_total_flow or has_mole_fractions:
            if not has_total_flow or not has_mole_fractions:
                missing_key = "feed_inlet_flow" if not has_total_flow else "feed_mole_fractions"
                raise ValueError(
                    f"'{missing_key}' must be provided when using "
                    "'feed_inlet_flow'/'feed_mole_fractions' feed specification mode."
                )

            feed_total_flow = self._to_mol_per_s_value(
                self.model_inputs["feed_inlet_flow"]
            )
            if feed_total_flow < 0.0:
                raise ValueError("'feed_inlet_flow' must be non-negative.")

            feed_mole_fractions_comp, feed_mole_fractions = self._config_feed_mole_fractions()
            feed_flows = feed_total_flow * feed_mole_fractions
            feed_flows_comp = {
                comp_id: float(feed_flows[i])
                for i, comp_id in enumerate(self.component_formula_state)
            }

            self.feed_inlet_flow = float(feed_total_flow)
            self.feed_mole_fractions = feed_mole_fractions_comp

            return (
                feed_flows_comp,
                feed_flows.astype(float),
                float(np.sum(feed_flows)),
            )

        # NOTE: fallback behavior: raise the standard missing-required key error.
        return self._config_side_inlet_flows(
            primary_key="feed_inlet_flows",
            fallback_key="inlet_flows",
            required=True,
            default_component_value=0.0,
        )

    def _config_feed_mole_fractions(self) -> tuple[Dict[str, float], np.ndarray]:
        key = "feed_mole_fractions"
        raw = self.model_inputs[key]
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"'{key}' must be a mapping of component ids to mole fractions."
            )

        x_comp: Dict[str, float] = {}
        x_values: List[float] = []

        for comp_id in self.component_formula_state:
            if comp_id not in raw:
                raise ValueError(
                    f"Missing mole fraction entry for component '{comp_id}' in '{key}'."
                )

            x_i = self._to_dimensionless_fraction(
                raw[comp_id], key=key, component_id=comp_id)
            if x_i < 0.0:
                raise ValueError(
                    f"Mole fraction for component '{comp_id}' in '{key}' must be non-negative."
                )

            x_comp[comp_id] = x_i
            x_values.append(x_i)

        x = np.array(x_values, dtype=float)
        x_sum = float(np.sum(x))
        if x_sum <= 0.0:
            raise ValueError(
                f"Sum of mole fractions in '{key}' must be greater than zero."
            )

        if not np.isclose(x_sum, 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"Mole fractions in '{key}' must sum to 1.0 (current sum: {x_sum})."
            )

        return x_comp, x

    def _to_dimensionless_fraction(self, value: Any, key: str, component_id: str) -> float:
        raw_value, raw_unit = self._extract_value_unit(value)
        unit = str(raw_unit).strip().replace(" ", "")
        if unit not in ("", "-", "1", "mol/mol", "molefraction"):
            raise ValueError(
                f"Unsupported unit '{raw_unit}' for '{key}'[{component_id}]. "
                "Use dimensionless values."
            )
        return float(raw_value)

    def _config_side_inlet_flows(
        self,
        primary_key: str,
        fallback_key: str | None,
        required: bool,
        default_component_value: float
    ):
        # NOTE: prefer explicit side key, fallback to legacy key if provided.
        if primary_key in self.model_inputs_keys:
            return self.config_inlet_mole_flows_by_key(
                key=primary_key,
                required=True,
                default_component_value=default_component_value
            )
        if fallback_key is not None and fallback_key in self.model_inputs_keys:
            return self.config_inlet_mole_flows_by_key(
                key=fallback_key,
                required=True,
                default_component_value=default_component_value
            )
        return self.config_inlet_mole_flows_by_key(
            key=primary_key,
            required=required,
            default_component_value=default_component_value
        )

    # NOTE: configure side temperature with backward-compatible fallback.
    def _config_side_temperature(
        self,
        primary_key: str,
        fallback_key: str | None,
        required: bool,
        fallback_value_K: float | None,
    ) -> Temperature:
        # ! direct side key
        if primary_key in self.model_inputs_keys:
            return Temperature(
                value=self._to_temperature_K(
                    self.model_inputs[primary_key], default_unit="K"),
                unit="K"
            )

        # ! legacy fallback key
        if fallback_key is not None and fallback_key in self.model_inputs_keys:
            return Temperature(
                value=self._to_temperature_K(
                    self.model_inputs[fallback_key], default_unit="K"),
                unit="K"
            )

        # ! missing-key behavior
        if required:
            raise ValueError(
                f"'{primary_key}' must be provided in model_inputs."
            )
        if fallback_value_K is None:
            raise ValueError(
                f"'{primary_key}' is missing and no fallback value is available."
            )
        return Temperature(value=float(fallback_value_K), unit="K")

    # NOTE: configure side pressure with backward-compatible fallback.
    def _config_side_pressure(
        self,
        primary_key: str,
        fallback_key: str | None,
        required: bool,
        fallback_value_Pa: float | None,
    ) -> float:
        # ! direct side key
        if primary_key in self.model_inputs_keys:
            return self._to_pressure_Pa(self.model_inputs[primary_key], default_unit="Pa")

        # ! legacy fallback key
        if fallback_key is not None and fallback_key in self.model_inputs_keys:
            return self._to_pressure_Pa(self.model_inputs[fallback_key], default_unit="Pa")

        # ! missing-key behavior
        if required:
            raise ValueError(
                f"'{primary_key}' must be provided in model_inputs."
            )
        if fallback_value_Pa is None:
            raise ValueError(
                f"'{primary_key}' is missing and no fallback value is available."
            )
        return float(fallback_value_Pa)

    # SECTION: membrane parameter configuration
    # NOTE: configure membrane area-per-length either directly or via module geometry.
    def _config_membrane_area_per_length(self) -> float:
        # case 1
        direct_key = "membrane_area_per_length"
        # case 2
        geometry_keys = (
            "number_of_fibers",
            "fiber_outer_diameter",
            "fiber_inner_diameter",
            "fiber_length",
            "module_diameter",
        )
        # case 3
        geometry_key = "module_geometry"

        # check availability of direct area-per-length input and geometry-based inputs
        has_direct = direct_key in self.model_inputs_keys
        has_geometry = geometry_key in self.model_inputs_keys
        has_any_geometry = any(
            k in self.model_inputs_keys for k in geometry_keys
        )

        # NOTE: case 1: direct area-per-length input with unit handling.
        if has_direct:
            return self._parse_direct_membrane_area_per_length(direct_key)

        # NOTE: case 2: retrieve geometry inputs from separate keys and calculate area-per-length.
        if has_any_geometry:
            return self._calculate_membrane_area_per_length_from_geometry(geometry_keys)

        # NOTE: case 3: retrieve geometry from nested model and calculate area-per-length.
        if has_geometry:
            return self._calculate_membrane_area_per_length_from_geometry_model("module_geometry")

        raise ValueError(
            "Membrane area per unit length must be specified either directly via "
            f"'{direct_key}' or indirectly via geometry inputs ({geometry_keys} or '{geometry_key}')."
        )

    # ! parsing direct area-per-length input with unit handling (expects SI-equivalent units)
    def _parse_direct_membrane_area_per_length(self, key: str) -> float:
        raw = self.model_inputs[key]
        raw_value, raw_unit = self._extract_value_unit(raw)
        unit = str(raw_unit).strip()

        # NOTE: area-per-length has SI unit m (equivalent to m2/m).
        if unit in ("", "m"):
            return float(raw_value)
        else:
            return pycuc.convert_from_to(
                value=float(raw_value),
                from_unit=unit,
                to_unit="m",
            )

    # ! calculating area-per-length from geometry inputs using HFMModule properties
    def _calculate_membrane_area_per_length_from_geometry(self, geometry_keys: tuple[str, ...]) -> float:
        missing = [
            key for key in geometry_keys if key not in self.model_inputs_keys]
        if len(missing) > 0:
            raise ValueError(
                "Incomplete membrane geometry specification. Missing keys: "
                f"{missing}"
            )

        module = HFMModule(
            number_of_fibers=self.model_inputs["number_of_fibers"],
            fiber_outer_diameter=self.model_inputs["fiber_outer_diameter"],
            fiber_inner_diameter=self.model_inputs["fiber_inner_diameter"],
            fiber_length=self.model_inputs["fiber_length"],
            module_diameter=self.model_inputs["module_diameter"],
        )
        return float(module.properties["area_per_unit_length"])

    # ! calculating area-per-length from geometry model
    def _calculate_membrane_area_per_length_from_geometry_model(self, geometry_key: str) -> float:
        if geometry_key not in self.model_inputs_keys:
            raise ValueError(
                f"Membrane geometry model '{geometry_key}' is missing from model_inputs."
            )
        geometry_model = self.model_inputs[geometry_key]
        if not isinstance(geometry_model, HollowFiberMembraneModuleGeometry):
            raise ValueError(
                f"'{geometry_key}' must be an instance of HollowFiberMembraneModuleGeometry."
            )

        # create hfm module
        module = HFMModule(
            number_of_fibers=geometry_model.number_of_fibers,
            fiber_outer_diameter=geometry_model.fiber_outer_diameter,
            fiber_inner_diameter=geometry_model.fiber_inner_diameter,
            fiber_length=geometry_model.fiber_length,
            module_diameter=geometry_model.module_diameter,
        )
        return float(module.properties["area_per_unit_length"])

    # NOTE: configure optional overall heat-transfer coefficient [W/m2.K].
    def _config_optional_per_area_u(self, key: str, default_value: float) -> float:
        if key not in self.model_inputs_keys:
            return float(default_value)
        raw = self.model_inputs[key]
        # ! pydantic-like object with .value/.unit
        if hasattr(raw, "value"):
            value = float(raw.value)
            unit = str(getattr(raw, "unit", "W/m2.K")).strip() or "W/m2.K"
        # ! mapping input {'value', 'unit'}
        elif isinstance(raw, Mapping):
            value = float(raw.get("value", default_value))
            unit = str(raw.get("unit", "W/m2.K")).strip() or "W/m2.K"
        # ! bare numeric (assumed W/m2.K)
        else:
            value = float(raw)
            unit = "W/m2.K"
        return float(to_W_per_m2_K(value=value, unit=unit))

    # NOTE: configure optional side external heat flux [W/m2].
    def _config_optional_qext(self, key: str, default_value: float) -> float:
        """
        Configure side external heat flux [W/m2].

        This keeps unit handling intentionally strict to avoid silent dimensional errors.
        """
        if key not in self.model_inputs_keys:
            return float(default_value)
        raw = self.model_inputs[key]
        # ! pydantic-like object with .value/.unit
        if hasattr(raw, "value"):
            value = float(raw.value)
            unit = str(getattr(raw, "unit", "W/m2")).strip() or "W/m2"
        # ! mapping input {'value', 'unit'}
        elif isinstance(raw, Mapping):
            value = float(raw.get("value", default_value))
            unit = str(raw.get("unit", "W/m2")).strip() or "W/m2"
        # ! bare numeric (assumed W/m2)
        else:
            value = float(raw)
            unit = "W/m2"

        normalized = unit.replace(" ", "")
        if normalized in ("W/m2", "W/m^2"):
            return float(value)
        raise ValueError(
            f"Unsupported unit '{unit}' for {key}. Use 'W/m2'."
        )

    # NOTE: configure component-wise transport coefficients in component order.
    def _config_transport_coefficients(self, key: str, required: bool) -> np.ndarray:
        if key not in self.model_inputs_keys:
            if required:
                raise ValueError(
                    f"'{key}' must be provided in model_inputs for phase '{self.phase}'."
                )
            return np.zeros(self.component_num, dtype=float)

        raw = self.model_inputs[key]
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"'{key}' must be a mapping of component ids to coefficient values."
            )

        # NOTE: build ordered coefficient array according to component_formula_state
        coeffs: List[float] = []
        missing: List[str] = []
        for comp_id in self.component_formula_state:
            if comp_id not in raw:
                missing.append(comp_id)
                coeffs.append(0.0)
                continue

            item = raw[comp_id]
            # ! pydantic-like object with .value
            if (
                hasattr(item, "value") and
                hasattr(item, "unit")
            ):
                # value
                value = float(item.value)
                # unit
                unit = str(item.value).strip()
            # ! mapping input {'value', ...}
            elif isinstance(item, Mapping):
                if "value" not in item:
                    raise ValueError(
                        f"Transport coefficient for '{comp_id}' in '{key}' must contain 'value'."
                    )

                if "unit" not in item:
                    raise ValueError(
                        f"Transport coefficient for '{comp_id}' in '{key}' must contain 'unit'."
                    )

                # set
                value = float(item["value"])
                unit = str(item["unit"]).strip()
            # ! bare numeric
            else:
                raise ValueError(
                    f"Transport coefficient for '{comp_id}' in '{key}' must be a numeric value or an object with a 'value' attribute."
                )

            # NOTE: check unit based on phase and key conventions
            if (
                self.phase == "gas" and
                key == "gas_transport_coefficients"
            ):
                # ! unit check
                if unit.lower() not in ("gpu", "gas permeation unit", "mol/s.m2.pa"):
                    raise ValueError(
                        f"Unsupported unit '{unit}' for gas transport coefficient of component '{comp_id}' in '{key}'. "
                        "Use 'GPU' (Gas Permeation Unit) or 'mol/s.m2.Pa'."
                    )

                # ! unit conversion
                if unit.lower() in ("gpu", "gas permeation unit"):
                    value = from_gpu_to_mol_per_s_m2_Pa(
                        CustomProp(value=value, unit=unit)
                    )
            elif (
                self.phase == "liquid" and
                key == "liquid_transport_coefficients"
            ):
                # ! unit check
                if unit.lower() not in ("m/s", "meter per second"):
                    raise ValueError(
                        f"Unsupported unit '{unit}' for liquid transport coefficient of component '{comp_id}' in '{key}'. "
                        "Use 'm/s' (meter per second)."
                    )

                # ! no conversion needed for m/s
            else:
                raise ValueError(
                    f"Invalid transport coefficient key '{key}' for phase '{self.phase}'."
                )

            # set
            coeffs.append(value)

        # NOTE: enforce completeness only when required for active phase
        if required and len(missing) > 0:
            raise ValueError(
                f"Missing transport coefficients in '{key}' for components: {missing}"
            )

        # NOTE: final non-negativity check
        coeffs_np = np.array(coeffs, dtype=float)
        if np.any(coeffs_np < 0.0):
            raise ValueError(
                f"Transport coefficients in '{key}' must be non-negative.")
        return coeffs_np
