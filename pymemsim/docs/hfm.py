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
from ..solvers.countercurrent_shooting import solve_countercurrent_shooting
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

        # NOTE: create HFM module
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
        # NOTE: route to the appropriate simulation method based on flow pattern
        if self.hfm_core.is_co_current:
            return self._simulate_cocurrent(
                length_span=length_span,
                solver_options=solver_options,
            )
        if self.hfm_core.is_counter_current:
            return self._simulate_countercurrent(
                length_span=length_span,
                solver_options=solver_options,
            )
        raise ValueError(
            f"Unsupported flow_pattern '{self.hfm_core.flow_pattern}'.")

    # ! define the ODE function for the solver
    def _rhs_point(self, z: float, y: np.ndarray) -> np.ndarray:
        if isinstance(self.module, (GasHFMX, LiquidHFMX)):
            return self.module.rhs_scaled(z, y)
        if isinstance(self.module, (GasHFM, LiquidHFM)):
            return self.module.rhs(z, y)
        raise NotImplementedError(
            f"ODE function for reactor type '{type(self.module)}' is not implemented yet."
        )

    # NOTE: simulate co-current flow by solving an initial value problem (IVP)
    def _simulate_cocurrent(
        self,
        length_span: tuple[float, float],
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[MembraneResult]:
        # configure solver options with defaults
        configured_solver_options = configure_solver_options(
            solver_options=solver_options
        )

        # build initial condition vector y0 based on the module type
        if isinstance(self.module, (GasHFMX, LiquidHFMX)):
            y0 = self.module.build_y0_scaled()
        elif isinstance(self.module, (GasHFM, LiquidHFM)):
            y0 = self.module.build_y0()
        else:
            raise NotImplementedError(
                f"Initial condition builder for reactor type '{type(self.module)}' is not implemented yet."
            )

        # solve the IVP using scipy's solve_ivp
        sol = solve_ivp(
            self._rhs_point,
            length_span,
            y0,
            **configured_solver_options,
        )

        # check if the solver was successful and log an error if it failed
        if not sol.success:
            logger.error(f"HFM co-current IVP solver failed: {sol.message}")
            return None

        # convert the solver state history to physical units and return the result
        return MembraneResult(
            span=sol.t,
            state=self._state_to_physical(sol.y),
            success=sol.success,
            message=sol.message,
        )

    # NOTE: simulate counter-current flow by solving a boundary value problem (BVP)
    def _simulate_countercurrent(
        self,
        length_span: tuple[float, float],
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[MembraneResult]:
        # ! check if the module type supports counter-current simulation
        if not isinstance(self.module, (GasHFM, GasHFMX)):
            raise NotImplementedError(
                "Counter-current simulation is implemented only for gas modules "
                "(GasHFM and GasHFMX) in this release."
            )

        solver_options_local = solver_options.copy() if solver_options is not None else {}
        countercurrent_solver = str(
            solver_options_local.pop("countercurrent_solver", "bvp")
        ).strip().lower()

        if countercurrent_solver == "shooting":
            return solve_countercurrent_shooting(
                module=self.module,
                rhs_point=self._rhs_point,
                state_to_physical=self._state_to_physical,
                length_span=length_span,
                solver_options=solver_options_local,
            )
        if countercurrent_solver != "bvp":
            raise ValueError(
                "Invalid countercurrent_solver. Supported values are 'bvp' and 'shooting'."
            )

        # configure BVP solver options with defaults
        bvp_options = solver_options_local
        mesh_points = int(bvp_options.pop("mesh_points", 80))
        tol = float(bvp_options.pop("tol", 1e-3))
        max_nodes = int(bvp_options.pop("max_nodes", 20000))
        verbose = int(bvp_options.pop("verbose", 0))
        bc_tol = bvp_options.pop("bc_tol", None)
        debug_bc = bool(bvp_options.pop("debug_bc", False))

        # build the mesh
        z_mesh = self.module.build_mesh(
            length_span=length_span,
            mesh_points=mesh_points
        )

        # build the initial guess
        y_guess = self.module.build_initial_guess(z_mesh)

        # >> debug
        if debug_bc:
            bc_guess = np.asarray(self.module.bc(
                y_guess[:, 0], y_guess[:, -1]), dtype=float)
            logger.info(
                "HFM counter-current initial BC residual: norm_inf=%.3e norm_l2=%.3e",
                float(np.max(np.abs(bc_guess))),
                float(np.linalg.norm(bc_guess)),
            )

        # ! define the ODE function for the BVP solver, which can handle both pointwise and vectorized inputs
        def fun(z: np.ndarray, y: np.ndarray) -> np.ndarray:
            z_arr = np.asarray(z, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            if y_arr.ndim == 1:
                return self._rhs_point(float(z_arr), y_arr)

            out = np.empty_like(y_arr)
            for j in range(y_arr.shape[1]):
                out[:, j] = self._rhs_point(float(z_arr[j]), y_arr[:, j])
            return out

        # NOTE: continuation-style retries for difficult coupled BVPs.
        attempts = [
            {"mesh_points": mesh_points, "tol": tol, "max_nodes": max_nodes},
            {"mesh_points": max(mesh_points, 100), "tol": min(
                5.0 * tol, 5e-2), "max_nodes": max(max_nodes, 40000)},
            {"mesh_points": max(mesh_points, 140), "tol": min(
                10.0 * tol, 1e-1), "max_nodes": max(max_nodes, 80000)},
        ]

        sol = None
        for k, attempt in enumerate(attempts, start=1):
            attempt_mesh_points = int(attempt["mesh_points"])
            attempt_tol = float(attempt["tol"])
            attempt_max_nodes = int(attempt["max_nodes"])

            if k == 1:
                z_mesh_k = z_mesh
                y_guess_k = y_guess
            else:
                z_mesh_k = self.module.build_mesh(
                    length_span=length_span,
                    mesh_points=attempt_mesh_points
                )
                if sol is None or getattr(sol, "sol", None) is None:
                    y_guess_k = self.module.build_initial_guess(z_mesh_k)
                else:
                    y_guess_k = sol.sol(z_mesh_k)
                if debug_bc:
                    bc_guess_k = np.asarray(self.module.bc(
                        y_guess_k[:, 0], y_guess_k[:, -1]), dtype=float)
                    logger.info(
                        "HFM counter-current retry %d initial BC residual: norm_inf=%.3e norm_l2=%.3e",
                        k,
                        float(np.max(np.abs(bc_guess_k))),
                        float(np.linalg.norm(bc_guess_k)),
                    )

            solve_bvp_kwargs: Dict[str, Any] = {
                "tol": attempt_tol,
                "max_nodes": attempt_max_nodes,
                "verbose": verbose,
            }
            if bc_tol is not None:
                solve_bvp_kwargs["bc_tol"] = float(bc_tol)

            sol = solve_bvp(
                fun=fun,
                bc=self.module.bc,
                x=z_mesh_k,
                y=y_guess_k,
                **solve_bvp_kwargs,
            )

            if sol.success:
                break

            logger.warning(
                "HFM counter-current BVP attempt %d failed: status=%s message=%s nodes=%d tol=%.3e max_nodes=%d",
                k,
                getattr(sol, "status", "n/a"),
                sol.message,
                len(sol.x),
                attempt_tol,
                attempt_max_nodes,
            )

        if sol is None or not sol.success:
            logger.error(
                "HFM counter-current BVP solver failed after retries: status=%s message=%s nodes=%d",
                getattr(sol, "status", "n/a") if sol is not None else "n/a",
                sol.message if sol is not None else "no solution object",
                len(sol.x) if sol is not None else 0,
            )
            return None

        return MembraneResult(
            span=sol.x,
            state=self._state_to_physical(sol.y),
            success=sol.success,
            message=sol.message,
        )
