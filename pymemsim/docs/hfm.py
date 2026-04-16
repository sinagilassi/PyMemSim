import logging
import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, Optional, cast
from pythermodb_settings.models import ComponentKey
from pythermodb_settings.utils import measure_time
# locals
from ..core.gas_hfm import GasHFM
from ..core.gas_hfmx import GasHFM
from ..core.liquid_hfm import LiquidHFM
from ..core.liquid_hfmx import LiquidHFMX
from ..core.hfmc import HFMCore
from ..models.hfm import HollowFiberMembraneOptions
from ..models.results import MembraneResult
from ..sources.thermo_source import ThermoSource
from ..utils.tools import configure_solver_options

# NOTE: set logger
logger = logging.getLogger(__name__)


class PFRReactor:
    """
    PFR reactor interface.
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

        self.pfr_reactor_options = cast(
            PFRReactorOptions,
            thermo_source.reactor_options
        )
        self.heat_transfer_options = thermo_source.heat_transfer_options
        self.phase = self.pfr_reactor_options.phase
        self.modeling_type = self.pfr_reactor_options.modeling_type
        self.reaction_rates = thermo_source.reaction_rates

        self.pfr_reactor_core = PFRReactorCore(
            components=self.components,
            model_inputs=model_inputs,
            pfr_reactor_options=self.pfr_reactor_options,
            heat_transfer_options=self.heat_transfer_options,
            component_refs=self.component_refs,
            component_key=cast(ComponentKey, self.component_key),
        )

        self.reactor: GasPFRReactor | GasPFRReactorX | LiquidPFRReactor | LiquidPFRReactorX = self._create_reactor()

    def _state_to_physical(self, state: np.ndarray) -> np.ndarray:
        """
        Convert solver state history to physical units for public outputs.
        """
        state_arr = np.asarray(state, dtype=float)
        if state_arr.ndim != 2:
            raise ValueError(
                "Expected state history with shape (n_states, n_points).")

        if not isinstance(self.reactor, (GasPFRReactorX, LiquidPFRReactorX)):
            return state_arr

        n_points = state_arr.shape[1]
        physical_cols = []
        for j in range(n_points):
            y_scaled = state_arr[:, j]

            if isinstance(self.reactor, GasPFRReactorX):
                f, temp, p_total = self.reactor._unscale_state(y_scaled)
                y_parts = [f]
                if self.reactor.heat_transfer_mode == "non-isothermal":
                    y_parts.append(np.array([temp], dtype=float))
                if self.reactor.pressure_mode == "state_variable":
                    if p_total is None:
                        raise ValueError(
                            "Scaled gas PFR state_variable pressure could not be recovered."
                        )
                    y_parts.append(np.array([float(p_total)], dtype=float))
            else:
                f, temp = self.reactor._unscale_state(y_scaled)
                y_parts = [f]
                if self.reactor.heat_transfer_mode == "non-isothermal":
                    y_parts.append(np.array([temp], dtype=float))

            y_physical = y_parts[0] if len(
                y_parts) == 1 else np.concatenate(y_parts)
            physical_cols.append(y_physical)

        return np.column_stack(physical_cols)

    def _create_reactor(self) -> GasPFRReactor | GasPFRReactorX | LiquidPFRReactor | LiquidPFRReactorX:
        if self.phase == "gas" and self.modeling_type == "physical":
            return GasPFRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pfr_reactor_core=self.pfr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        if self.phase == "gas" and self.modeling_type == "scale":
            return GasPFRReactorX(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pfr_reactor_core=self.pfr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        if self.phase == "liquid" and self.modeling_type == "physical":
            return LiquidPFRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pfr_reactor_core=self.pfr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        if self.phase == "liquid" and self.modeling_type == "scale":
            return LiquidPFRReactorX(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pfr_reactor_core=self.pfr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )

        raise NotImplementedError(
            f"PFR reactor for phase '{self.phase}' and modeling_type '{self.modeling_type}' is not implemented yet."
        )

    # SECTION: Simulation method
    @measure_time
    def simulate(
        self,
        volume_span: tuple[float, float],
        solver_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[PFRReactorResult]:
        """
        Run PFR simulation over the specified volume span with given solver options.

        Parameters
        ----------
        volume_span : tuple[float, float]
            The start and end volume for the PFR simulation.
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
        Optional[PFRReactorResult]
            The result of the PFR simulation, including volume, state, success flag, and message.

        Notes
        -----
        - The method uses `scipy.integrate.solve_ivp` to solve the ODEs defined by the PFR reactor model.
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
            if isinstance(self.reactor, (GasPFRReactorX, LiquidPFRReactorX)):
                return self.reactor.rhs_scaled(V, y)
            elif isinstance(self.reactor, (GasPFRReactor, LiquidPFRReactor)):
                return self.reactor.rhs(V, y)
            else:
                raise NotImplementedError(
                    f"ODE function for reactor type '{type(self.reactor)}' is not implemented yet."
                )

        # NOTE: build initial condition vector
        if isinstance(self.reactor, (GasPFRReactorX, LiquidPFRReactorX)):
            y0 = self.reactor.build_y0_scaled()
        elif isinstance(self.reactor, (GasPFRReactor, LiquidPFRReactor)):
            y0 = self.reactor.build_y0()
        else:
            raise NotImplementedError(
                f"Initial condition builder for reactor type '{type(self.reactor)}' is not implemented yet."
            )

        # NOTE: run ODE solver
        sol = solve_ivp(
            fun,
            volume_span,
            y0,
            **configured_solver_options,
        )

        # NOTE: check solver success and return results
        if not sol.success:
            logger.error(f"PFR ODE solver failed: {sol.message}")
            return None

        return PFRReactorResult(
            volume=sol.t,
            state=self._state_to_physical(sol.y),
            success=sol.success,
            message=sol.message,
        )
