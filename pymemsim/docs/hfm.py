import logging
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from typing import Any, Dict, Optional, cast
from pythermodb_settings.models import ComponentKey
from pythermodb_settings.utils import measure_time
# locals
from ..core.hfmc import HFMCore
from ..core.gas_hfm import GasHFM
from ..core.gas_hfmx import GasHFMX
from ..core.liquid_hfm import LiquidHFM
from ..core.liquid_hfmx import LiquidHFMX
from ..models.hfm import HollowFiberMembraneOptions
from ..models.results import MembraneResult
from ..sources.thermo_source import ThermoSource
from ..utils.tools import configure_solver_options

# NOTE: set logger
logger = logging.getLogger(__name__)


class HFM:
    """
    Hollow Fiber Membrane (HFM) module class for simulating hollow fiber membrane processes based on provided model inputs and thermodynamic data.
    """

    def __init__(
        self,
        model_inputs: Dict[str, Any],
        thermo_source: ThermoSource,
        **kwargs,
    ):
        self.model_inputs = model_inputs
        self.thermo_source = thermo_source

        self.components = thermo_source.components
        self.component_refs = thermo_source.component_refs
        self.component_key = thermo_source.component_key

        self.unit_options = cast(
            HollowFiberMembraneOptions,
            thermo_source.unit_options
        )
        self.heat_transfer_options = thermo_source.heat_transfer_options
        self.phase = self.unit_options.phase
        self.modeling_type = self.unit_options.modeling_type
        self.reaction_rates = thermo_source.reaction_rates

        self.hfm_core = HFMCore(
            components=self.components,
            model_inputs=model_inputs,
            unit_options=self.unit_options,
            heat_transfer_options=self.heat_transfer_options,
            component_refs=self.component_refs,
            component_key=cast(ComponentKey, self.component_key),
        )

        self.module: GasHFM | GasHFMX | LiquidHFM | LiquidHFMX = self._create_module()

    # SECTION: Helper methods
    # NOTE: this method converts the solver state history to physical units for public outputs, especially for scaled models where the state variables are in a non-physical scaled form. It checks the reactor type and applies the appropriate unscaling logic to recover the physical values of flow rates, temperature, and pressure (if applicable) from the scaled state variables.
    def _state_to_physical(self, state: np.ndarray) -> np.ndarray:
        """
        Convert solver state history to physical units for public outputs.
        """
        state_arr = np.asarray(state, dtype=float)
        if state_arr.ndim != 2:
            raise ValueError(
                "Expected state history with shape (n_states, n_points).")

        if not isinstance(self.module, (GasHFMX, LiquidHFMX)):
            return state_arr

        n_points = state_arr.shape[1]
        physical_cols = []
        for j in range(n_points):
            y_scaled = state_arr[:, j]

            if isinstance(self.module, (GasHFMX)):
                ff, fp, tf, tp = self.module._unscale_state(y_scaled)
                y_parts = [ff, fp]
                if self.module.heat_transfer_mode == "non-isothermal":
                    y_parts.append(np.array([tf, tp], dtype=float))
            else:
                ff, fp, tf, tp = self.module._unscale_state(y_scaled)
                y_parts = [ff, fp]
                if self.module.heat_transfer_mode == "non-isothermal":
                    y_parts.append(np.array([tf, tp], dtype=float))

            y_physical = y_parts[0] if len(
                y_parts) == 1 else np.concatenate(y_parts)
            physical_cols.append(y_physical)

        return np.column_stack(physical_cols)

    # NOTE: this method creates the appropriate HFM module instance based on the specified phase and modeling type. It checks the combination of phase (gas or liquid) and modeling type (physical or scale) and instantiates the corresponding class (GasHFM, GasHFMX, LiquidHFM, or LiquidHFMX) with the necessary inputs. If the combination is not implemented, it raises a NotImplementedError.
    def _create_module(self) -> GasHFM | GasHFMX | LiquidHFM | LiquidHFMX:
        if (
            self.phase == "gas" and
            self.modeling_type == "physical"
        ):
            return GasHFM(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                hfm_core=self.hfm_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif (
            self.phase == "gas" and
            self.modeling_type == "scale"
        ):
            return GasHFMX(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                hfm_core=self.hfm_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif (
            self.phase == "liquid" and
            self.modeling_type == "physical"
        ):
            return LiquidHFM(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                hfm_core=self.hfm_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif (
            self.phase == "liquid" and
            self.modeling_type == "scale"
        ):
            return LiquidHFMX(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                hfm_core=self.hfm_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        else:
            raise NotImplementedError(
                f"Module for phase '{self.phase}' and modeling_type '{self.modeling_type}' is not implemented yet."
            )

    # SECTION: Simulation method
    @measure_time
    def simulate(
        self,
        length_span: tuple[float, float],
        solver_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[MembraneResult]:
        """
        Run simulation over the specified volume span with given solver options.

        Parameters
        ----------
        length_span : tuple[float, float]
            The start and end length (z) for the simulation.
        solver_options : Optional[Dict[str, Any]], optional
            A dictionary of solver options to pass to `scipy.integrate.solve_ivp`. If None, default options will be used.
            Supported options include:
            - method: ODE solver method (e.g., 'BDF', 'RK45', etc.)
            - rtol: Relative tolerance for the solver
            - atol: Absolute tolerance for the solver
            - first_step: Initial step size for the solver
            - max_step: Maximum step size for the solver
        **kwargs
            Additional keyword arguments.
            - mode : Literal['silent', 'log', 'attach'], optional
                Mode for time measurement logging. Default is 'silent'.

        Returns
        -------
        Optional[MembraneResult]
            The result of the simulation, including span, state, success flag, and message.

        Notes
        -----
        - The method uses `scipy.integrate.solve_ivp` to solve the ODEs defined by the model.
        - The `mode` keyword argument can be used to control how the execution time is logged:
            - 'silent': No logging of execution time.
            - 'log': Logs the execution time to the logger.
            - 'attach': Logs the execution time and attaches it to the result object.
        - The solver options can be customized by passing a dictionary to `solver_options`. If not provided, default options will be used for the solver. The default values are as:
            - method: 'BDF'
            - rtol: 1e-6
            - atol: 1e-9
        """
        # NOTE: set default solver options if not provided
        configured_solver_options = configure_solver_options(
            solver_options=solver_options
        )

        # NOTE: define ODE function for PFR simulation

        def fun(V, y):
            if isinstance(self.module, (GasHFMX, LiquidHFMX)):
                return self.module.rhs_scaled(V, y)
            elif isinstance(self.module, (GasHFM, LiquidHFM)):
                return self.module.rhs(V, y)
            else:
                raise NotImplementedError(
                    f"ODE function for reactor type '{type(self.module)}' is not implemented yet."
                )

        # NOTE: build initial condition vector
        if isinstance(self.module, (GasHFMX, LiquidHFMX)):
            y0 = self.module.build_y0_scaled()
        elif isinstance(self.module, (GasHFM, LiquidHFM)):
            y0 = self.module.build_y0()
        else:
            raise NotImplementedError(
                f"Initial condition builder for reactor type '{type(self.module)}' is not implemented yet."
            )

        # NOTE: run ODE solver
        sol = solve_ivp(
            fun,
            length_span,
            y0,
            **configured_solver_options,
        )

        # NOTE: check solver success and return results
        if not sol.success:
            logger.error(f"PFR ODE solver failed: {sol.message}")
            return None

        return MembraneResult(
            span=sol.t,
            state=self._state_to_physical(sol.y),
            success=sol.success,
            message=sol.message,
        )
