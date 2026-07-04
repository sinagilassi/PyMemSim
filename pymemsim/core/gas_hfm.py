# import libs
import logging
import numpy as np
from typing import Dict, List, Tuple, cast
from pythermodb_settings.models import Component, ComponentKey, CustomProperty, Pressure, Temperature
from pyreactsim_core.models.rate_exp import ReactionRateExpression
# locals
from ..configs.constants import R_J_per_mol_K
from ..models.ref import GasModel
from ..sources.thermo_source import ThermoSource
from ..utils.reaction_tools import stoichiometry_mat, stoichiometry_mat_key
from ..utils.thermo_tools import calc_rxn_heat_generation, calc_total_heat_capacity
from .hfmc import HFMCore
from ..utils.tools import smooth_floor
from ..transport.hydrodynamics import (
    gas_fiber_pressure_squared_derivative,
    shell_pressure_derivative_laminar,
)


logger = logging.getLogger(__name__)


def _clip_finite(value: float, lower: float, upper: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return float(upper if value > 0.0 else lower)
    return float(np.clip(value, lower, upper))


def _require_geometry_value(name: str, value: float | None) -> float:
    if value is None:
        raise ValueError(
            f"Pressure-drop geometry is incomplete. Missing: {name}"
        )
    return float(value)


class GasHFM:
    """
    Gas-phase hollow-fiber membrane model (cocurrent, dual-side, constant pressure).
    """
    R = R_J_per_mol_K

    def __init__(
        self,
        components: List[Component],
        reaction_rates: List[ReactionRateExpression],
        thermo_source: ThermoSource,
        hfm_core: HFMCore,
        component_key: ComponentKey,
        **kwargs
    ):
        # NOTE: sets
        self.components = components
        self.component_key = component_key
        self.thermo_source = thermo_source
        self.hfm_core = hfm_core

        self.heat_transfer_mode = hfm_core.heat_transfer_mode
        self.gas_model = hfm_core.gas_model
        self.feed_pressure_mode = hfm_core.feed_pressure_mode
        self.permeate_pressure_mode = hfm_core.permeate_pressure_mode
        self.has_feed_pressure_state = self.feed_pressure_mode == "state_variable"
        self.has_permeate_pressure_state = self.permeate_pressure_mode == "state_variable"

        # NOTE: flow pattern setup
        self.s_p = hfm_core.permeate_axial_sign

        # NOTE: aux parameters
        self.ns = len(components)  # number of species/components

        # NOTE: Normalized dual-side membrane inputs
        # ! feed inlet flows [mol/s]
        self.Ff_in = hfm_core.feed_inlet_flows.astype(float)
        # ! permeate inlet flows [mol/s]
        self.Fp_in = hfm_core.permeate_inlet_flows.astype(float)

        # ! feed inlet temperature [K]
        self.Tf_in = float(hfm_core.feed_inlet_temperature.value)
        # ! permeate inlet temperature [K]
        self.Tp_in = float(hfm_core.permeate_inlet_temperature.value)

        # ! feed pressure [Pa]
        self.Pf = float(hfm_core.feed_pressure)
        # ! permeate pressure [Pa]
        self.Pp = float(hfm_core.permeate_pressure)
        self.pressure_floor = 1.0
        self.temperature_floor = 100.0
        self.max_solver_velocity = 1.0e6
        self.max_solver_density = 1.0e6
        self.max_solver_pressure_derivative = 1.0e12
        self.max_solver_pressure_squared_derivative = 1.0e18

        # ! membrane area per length [m]
        self.a_m = float(hfm_core.membrane_area_per_length)

        # ! pressure-drop geometry
        self.n_fibers = hfm_core.number_of_fibers
        self.d_inner = hfm_core.fiber_inner_diameter
        self.d_outer = hfm_core.fiber_outer_diameter
        self.shell_flow_area = hfm_core.shell_free_area
        self.shell_hydraulic_diameter = hfm_core.shell_hydraulic_diameter

        # ! overall heat transfer coefficient [W/m2.K]
        self.U_m = float(hfm_core.overall_heat_transfer_coefficient)

        # ! external heat source/sink per unit length [W/m]
        self.q_ext_f = float(hfm_core.q_ext_feed)
        self.q_ext_p = float(hfm_core.q_ext_permeate)

        # ! gas transport coefficients, permeability [mol/s.Pa]
        self.Pi = hfm_core.gas_transport_coefficients.astype(float)

        # SECTION: Legacy global heat source fallback (if explicitly provided)
        self.heat_exchange = hfm_core.heat_exchange
        self.heat_transfer_coefficient_value = hfm_core.heat_transfer_coefficient_value
        self.heat_transfer_area_value = hfm_core.heat_transfer_area_value
        self.jacket_temperature_value = hfm_core.jacket_temperature_value
        self.heat_rate_value = hfm_core.heat_rate_value

        # SECTION: Reaction setup
        self.reaction_rates = reaction_rates
        self.reactions = self.thermo_source.thermo_reaction.build_reactions()
        self.reaction_stoichiometry = stoichiometry_mat_key(
            reactions=self.reactions,
            component_key=component_key
        )
        self.reaction_stoichiometry_matrix = stoichiometry_mat(
            reactions=self.reactions,
            components=self.components,
            component_key=component_key,
        )

        # SECTION: Component indexing and reference setup
        self.component_num = self.thermo_source.component_refs["component_num"]
        self.component_formula_state = self.thermo_source.component_refs[
            "component_formula_state"
        ]
        self.component_id_to_index = self.thermo_source.component_refs["component_id_to_index"]

        # SECTION: Validation
        if self.Pi.shape[0] != self.component_num:
            raise ValueError(
                "gas_transport_coefficients length must match component_num.")

        if self.has_feed_pressure_state or self.has_permeate_pressure_state:
            required_geometry = {
                "number_of_fibers": self.n_fibers,
                "fiber_inner_diameter": self.d_inner,
                "shell_flow_area": self.shell_flow_area,
                "shell_hydraulic_diameter": self.shell_hydraulic_diameter,
            }
            missing = [k for k, v in required_geometry.items() if v is None]
            if len(missing) > 0:
                raise ValueError(
                    "Pressure-drop geometry is incomplete. Missing: "
                    f"{missing}"
                )

    # SECTION: Handlers
    # ! inlet flow
    @property
    def F_in(self) -> np.ndarray:
        """Backward-compatible alias for feed inlet flow vector."""
        return self.Ff_in

    # ! build initial state vector
    def build_y0(self) -> np.ndarray:
        """
        State layout:
        y = [Ff_i..., Fp_i..., Pf?, Pp2?, Tf?, Tp?]
        """
        y0_parts: List[np.ndarray] = [
            self.Ff_in.astype(float), self.Fp_in.astype(float)]
        if getattr(self, "has_feed_pressure_state", False):
            y0_parts.append(np.array([self.Pf], dtype=float))
        if getattr(self, "has_permeate_pressure_state", False):
            y0_parts.append(np.array([self.Pp**2], dtype=float))
        if self.heat_transfer_mode == "non-isothermal":
            y0_parts.append(np.array([self.Tf_in, self.Tp_in], dtype=float))
        return np.concatenate(y0_parts)

    # ! boundary conditions for BVP solver
    def bc(self, ya, yb):
        # ya = y(z=0)
        # yb = y(z=L)

        # ns
        ns = self.ns

        # Feed at z=0
        bc_feed = ya[:ns] - self.Ff_in

        # Permeate at z=L
        bc_permeate = yb[ns:2*ns] - self.Fp_in

        bc_parts = [bc_feed, bc_permeate]
        idx = 2 * ns

        if getattr(self, "has_feed_pressure_state", False):
            bc_parts.append(np.array([ya[idx] - self.Pf], dtype=float))
            idx += 1

        if getattr(self, "has_permeate_pressure_state", False):
            p_state = yb[idx] if self.s_p < 0 else ya[idx]
            bc_parts.append(np.array([p_state - self.Pp**2], dtype=float))
            idx += 1

        # NOTE: Add temperature BCs if non-isothermal
        if self.heat_transfer_mode == "non-isothermal":
            # Tf at z=0
            bc_Tf = ya[idx] - self.Tf_in

            # Tp at z=L
            bc_Tp = yb[idx + 1] - self.Tp_in

            bc_parts.append(np.array([bc_Tf, bc_Tp], dtype=float))

        return np.concatenate(bc_parts)

    # ! build mesh
    def build_mesh(
        self,
        length_span: tuple[float, float],
        mesh_points: int = 50,
    ) -> np.ndarray:
        """
        Build mesh for BVP solver.
        """
        z0, z1 = float(length_span[0]), float(length_span[1])
        if z1 <= z0:
            raise ValueError("length_span must satisfy z_end > z_start.")
        if mesh_points < 5:
            raise ValueError("mesh_points must be >= 5 for solve_bvp.")
        return np.linspace(z0, z1, int(mesh_points), dtype=float)

    # ! build initial guess for BVP solver
    def build_initial_guess(self, z_mesh: np.ndarray) -> np.ndarray:
        """
        Build initial guess for BVP solver.
        """
        z_mesh = np.asarray(z_mesh, dtype=float)
        if z_mesh.ndim != 1 or z_mesh.size < 2:
            raise ValueError(
                "z_mesh must be a 1D array with at least two points.")

        n_points = z_mesh.size
        ns = self.component_num
        z0 = float(z_mesh[0])
        z1 = float(z_mesh[-1])
        if z1 <= z0:
            raise ValueError("z_mesh must be strictly increasing.")
        eta = (z_mesh - z0) / (z1 - z0)

        # Feed guess: mild decrease from inlet to outlet.
        ff_out_guess = np.maximum(0.95 * self.Ff_in, 1e-12)
        ff_guess = np.vstack([
            np.linspace(float(self.Ff_in[i]), float(ff_out_guess[i]), n_points)
            for i in range(ns)
        ])

        # Permeate guess: very small positive floor at z=0 to avoid singular
        # mole-fraction denominators, while preserving feed-composition shape.
        # This avoids artificial flat profiles when permeate inlet is zero.
        ff_total_in = max(float(np.sum(self.Ff_in)), 1e-30)
        yf_in = self.Ff_in / ff_total_in
        # NOTE: keep this floor extremely small so it avoids singular
        # normalization without creating a visible flat-profile artifact.
        fp_floor_total = max(1e-30, 1e-14 * ff_total_in)
        fp_floor_vec = fp_floor_total * np.maximum(yf_in, 1e-12)
        fp_start_guess = np.where(
            self.Fp_in > 0.0, 0.2 * self.Fp_in, fp_floor_vec)
        fp_start_guess = np.maximum(fp_start_guess, fp_floor_vec)
        blend = eta ** 1.5
        fp_guess = np.vstack([
            (1.0 - blend) *
            float(fp_start_guess[i]) + blend * float(self.Fp_in[i])
            for i in range(ns)
        ])

        y_parts = [ff_guess, fp_guess]
        if getattr(self, "has_feed_pressure_state", False):
            pf_guess = np.full(n_points, self.Pf, dtype=float)
            y_parts.append(np.vstack([pf_guess]))
        if getattr(self, "has_permeate_pressure_state", False):
            pp2_guess = np.full(n_points, self.Pp**2, dtype=float)
            y_parts.append(np.vstack([pp2_guess]))
        if self.heat_transfer_mode == "non-isothermal":
            tf_out_guess = 0.99 * self.Tf_in + 0.01 * self.Tp_in
            tp_start_guess = 0.99 * self.Tp_in + 0.01 * self.Tf_in
            tf_guess = np.linspace(
                self.Tf_in, tf_out_guess, n_points, dtype=float)
            tp_guess = np.linspace(
                tp_start_guess, self.Tp_in, n_points, dtype=float)
            y_parts.append(np.vstack([tf_guess, tp_guess]))

        return np.vstack(y_parts)

    # SECTION: ODE RHS builder
    def rhs(self, z: float, y: np.ndarray) -> np.ndarray:
        ns = self.component_num

        # NOTE: regularize flows with smooth floor to avoid numerical
        # issues in BVP iterations near zero flow conditions.
        Ff = np.asarray(smooth_floor(
            y[:ns],
            xmin=0.0,
            s=1e-12
        ), dtype=float
        )
        Fp = np.asarray(smooth_floor(
            y[ns:2 * ns],
            xmin=0.0,
            s=1e-12
        ),
            dtype=float
        )

        idx = 2 * ns
        Pf_local = self.Pf
        Pp_local = self.Pp
        Pp2_local = self.Pp**2

        if getattr(self, "has_feed_pressure_state", False):
            Pf_local = max(float(y[idx]), self.pressure_floor)
            idx += 1

        if getattr(self, "has_permeate_pressure_state", False):
            Pp2_local = max(float(y[idx]), self.pressure_floor**2)
            Pp_local = float(np.sqrt(Pp2_local))
            idx += 1

        if self.heat_transfer_mode == "non-isothermal":
            temperature_floor = float(getattr(self, "temperature_floor", 100.0))
            Tf = float(smooth_floor(y[idx], xmin=temperature_floor, s=1e-3))
            Tp = float(smooth_floor(y[idx + 1], xmin=temperature_floor, s=1e-3))
        else:
            Tf = self.Tf_in
            Tp = self.Tp_in

        # NOTE: Build feed-side reaction closure once per integration step.
        # ! feed-side
        temperature_f = Temperature(value=Tf, unit="K")
        pressure_f = Pressure(value=Pf_local, unit="Pa")

        # NOTE: Calculate feed-side molar flow rate, composition, and concentration for reaction calculations
        # ! feed-side molar flow rate [mol/s]
        Ff_total = max(float(np.sum(Ff)), 1e-30)

        # ! feed-side mole fraction vector
        yf = Ff / Ff_total

        # ! feed-side volumetric flow rate [m3/s] (calculated from molar flow rate, temperature, pressure, and gas model)
        qf = self.thermo_source.calc_gas_volumetric_flow_rate(
            molar_flow_rate=Ff_total,
            temperature=Tf,
            pressure=Pf_local,
            R=self.R,
            gas_model=cast(GasModel, self.gas_model)
        )
        qf = max(float(qf), 1e-30)

        # ! feed-side concentration vector [mol/m3]
        Cf = Ff / qf

        # NOTE: Build feed-side partial pressure and concentration dicts for reaction rate calculations
        partial_pressures_std = {
            sp: CustomProperty(value=yf[i] * Pf_local, unit="Pa", symbol="P")
            for i, sp in enumerate(self.component_formula_state)
        }
        concentration_std = {
            sp: CustomProperty(value=Cf[i], unit="mol/m3", symbol="C")
            for i, sp in enumerate(self.component_formula_state)
        }

        # ! Calculate reaction rates based on current feed conditions (if reactions are defined)
        rates_f = np.array([], dtype=float)
        if len(self.reaction_rates) > 0:
            rates_f = self._calc_rates(
                partial_pressures=partial_pressures_std,
                concentration=concentration_std,
                temperature=temperature_f,
                pressure=pressure_f
            )

        # NOTE: Feed-side optional reaction source (legacy basis retention) [mol/s]
        dF_rxn_f = self._build_reaction_source_feed(rates=rates_f)

        # NOTE: Fluxes J_i = Pi_i * (y_f_i P_f - y_p_i P_p)
        # ! [mol/m2.s]
        J = self._calc_fluxes(Ff=Ff, Fp=Fp, Pf=Pf_local, Pp=Pp_local)

        # NOTE: Material balances
        # ! feed side: dFf_i/dz = -a_m * J_i + r_i (reaction source)
        dFf_dz = -self.a_m * J + dF_rxn_f

        # ! permeate side: dFp_i/dz = +a_m * J_i
        # ? for co-current flow, the permeate axial sign is positive where as
        # ? for counter-current flow, the permeate axial sign is negative.
        dFp_dz = self.s_p * self.a_m * J

        # >> Combine derivatives
        out_parts = [dFf_dz, dFp_dz]

        if getattr(self, "has_feed_pressure_state", False):
            out_parts.append(np.array([
                self._calc_feed_shell_pressure_derivative(
                    Ff=Ff,
                    yf=yf,
                    Ff_total=Ff_total,
                    Tf=Tf,
                    Pf=Pf_local,
                )
            ], dtype=float))

        if getattr(self, "has_permeate_pressure_state", False):
            Fp_total_for_pressure = max(float(np.sum(Fp)), 1e-30)
            yp_for_pressure = Fp / Fp_total_for_pressure
            out_parts.append(np.array([
                self._calc_permeate_fiber_pressure_squared_derivative(
                    Fp_total=Fp_total_for_pressure,
                    yp=yp_for_pressure,
                    Tp=Tp,
                )
            ], dtype=float))

        out = np.concatenate(out_parts)

        if self.heat_transfer_mode == "isothermal":
            return out

        # NOTE: Calculate feed-side reaction heat generation based on current rates and temperature
        q_rxn_f = self._reaction_heat_source_feed(
            rates=rates_f,
            temperature=temperature_f
        )

        # NOTE: Energy balances
        dTf_dz, dTp_dz = self._build_temperature_derivatives(
            Ff=Ff,
            Fp=Fp,
            Tf=Tf,
            Tp=Tp,
            q_rxn_f=q_rxn_f
        )
        return np.concatenate([out, np.array([dTf_dz, dTp_dz], dtype=float)])

    # NOTE: calculate fluxes based on current feed/permeate flows and pressures
    def _calc_fluxes(
        self,
        Ff: np.ndarray,
        Fp: np.ndarray,
        Pf: float | None = None,
        Pp: float | None = None,
    ) -> np.ndarray:
        """
        Calculate fluxes based on current feed/permeate flows and pressures using the expression:
        J_i = Pi_i * (y_f_i P_f - y_p_i P_p)

        where y_f_i and y_p_i are the feed and permeate mole fractions calculated from the current flows.

        Parameters
        ----------
        Ff : np.ndarray
            Current feed-side molar flow vector [mol/s].
        Fp : np.ndarray
            Current permeate-side molar flow vector [mol/s].

        Returns
        -------
        np.ndarray
            Flux vector for each component i [mol/m2.s].

        Notes
        -----
        Gas transport coefficient Pi_i is given in units of mol/m2.s.Pa, so the resulting flux is in mol/m2.s.
        """
        # NOTE: regularize total flows with a scale-aware epsilon to avoid
        # singular normalization without imposing a large fixed minimum.
        flow_ref = max(float(np.sum(self.Ff_in)), 1e-30)
        eps_total = max(1e-30, 1e-12 * flow_ref)
        Ff_total = max(float(np.sum(Ff)), eps_total)
        Fp_total = max(float(np.sum(Fp)), eps_total)
        yf = Ff / Ff_total
        yp = Fp / Fp_total
        Pf_local = self.Pf if Pf is None else float(Pf)
        Pp_local = self.Pp if Pp is None else float(Pp)
        return self.Pi * (yf * Pf_local - yp * Pp_local)

    def _calc_feed_shell_pressure_derivative(
        self,
        Ff: np.ndarray,
        yf: np.ndarray,
        Ff_total: float,
        Tf: float,
        Pf: float,
    ) -> float:
        mu_mix = self.thermo_source.calc_Vis_GAS(mole_fractions=yf)
        mu = float(mu_mix.value)
        shell_flow_area = _require_geometry_value(
            "shell_flow_area",
            self.shell_flow_area,
        )
        shell_hydraulic_diameter = _require_geometry_value(
            "shell_hydraulic_diameter",
            self.shell_hydraulic_diameter,
        )
        q_shell = self.thermo_source.calc_gas_volumetric_flow_rate(
            molar_flow_rate=Ff_total,
            temperature=Tf,
            pressure=Pf,
            R=self.R,
            gas_model=cast(GasModel, self.gas_model),
        )
        q_shell = max(float(q_shell), 1e-30)
        u_shell = _clip_finite(
            q_shell / shell_flow_area,
            lower=0.0,
            upper=float(getattr(self, "max_solver_velocity", 1.0e6)),
        )
        mw_mix_kg_per_mol = float(np.sum(yf * self.thermo_source.MW)) / 1000.0
        rho_shell = _clip_finite(
            Pf * mw_mix_kg_per_mol / (self.R * Tf),
            lower=1.0e-30,
            upper=float(getattr(self, "max_solver_density", 1.0e6)),
        )
        dP_dz = shell_pressure_derivative_laminar(
            density=rho_shell,
            velocity=u_shell,
            hydraulic_diameter=shell_hydraulic_diameter,
            mu=mu,
            axial_sign=1.0,
        )
        limit = float(getattr(self, "max_solver_pressure_derivative", 1.0e12))
        return _clip_finite(dP_dz, lower=-limit, upper=limit)

    def _calc_permeate_fiber_pressure_squared_derivative(
        self,
        Fp_total: float,
        yp: np.ndarray,
        Tp: float,
    ) -> float:
        mu_mix = self.thermo_source.calc_Vis_GAS(mole_fractions=yp)
        n_fibers = _require_geometry_value("number_of_fibers", self.n_fibers)
        d_inner = _require_geometry_value("fiber_inner_diameter", self.d_inner)
        dP2_dz = gas_fiber_pressure_squared_derivative(
            mu=float(mu_mix.value),
            molar_flow_rate=Fp_total,
            n_fibers=n_fibers,
            diameter_inner=d_inner,
            temperature=Tp,
            gas_constant=self.R,
            axial_sign=float(self.s_p),
        )
        limit = float(getattr(self, "max_solver_pressure_squared_derivative", 1.0e18))
        return _clip_finite(dP2_dz, lower=-limit, upper=limit)

    def _calc_fluxes_v0(self, Ff: np.ndarray, Fp: np.ndarray) -> np.ndarray:
        Ff_safe = np.asarray(Ff, dtype=float)
        Fp_safe = np.asarray(Fp, dtype=float)

        Ff_total = max(float(np.sum(Ff_safe)), 1e-12)
        Fp_total = max(float(np.sum(Fp_safe)), 1e-12)

        yf = Ff_safe / Ff_total

        # regularized permeate composition
        if np.sum(self.Fp_in) > 1e-12:
            yp_ref = self.Fp_in / np.sum(self.Fp_in)
        else:
            yp_ref = np.ones_like(Fp_safe) / len(Fp_safe)

        alpha = Fp_total / (Fp_total + 1e-8)
        yp_raw = Fp_safe / Fp_total
        yp = alpha * yp_raw + (1.0 - alpha) * yp_ref

        return self.Pi * (yf * self.Pf - yp * self.Pp)

    # NOTE: calculate feed-side reaction source term from precomputed reaction rates
    def _build_reaction_source_feed(self, rates: np.ndarray) -> np.ndarray:
        if rates.size == 0:
            return np.zeros(self.component_num, dtype=float)
        return self._build_stoich_source(rates=rates)

    # NOTE: calculate feed-side reaction heat generation based on current feed flows and temperature
    def _calc_rates(
        self,
        partial_pressures: Dict[str, CustomProperty],
        concentration: Dict[str, CustomProperty],
        temperature: Temperature,
        pressure: Pressure
    ) -> np.ndarray:
        rates = []
        for rate_exp in self.reaction_rates:
            basis = rate_exp.basis
            if basis == "pressure":
                r_k = rate_exp.calc(
                    xi=partial_pressures,
                    temperature=temperature,
                    pressure=pressure
                )
            elif basis == "concentration":
                r_k = rate_exp.calc(
                    xi=concentration,
                    temperature=temperature,
                    pressure=pressure
                )
            else:
                raise ValueError(
                    f"Invalid basis '{basis}' for gas HFM reaction rate expression '{rate_exp.name}'."
                )
            rates.append(float(r_k.value))
        return np.array(rates, dtype=float)

    # ! build reaction source term based on current rates and stoichiometry
    def _build_stoich_source(self, rates: np.ndarray) -> np.ndarray:
        src = np.zeros(self.component_num, dtype=float)
        for k, _ in enumerate(self.reactions):
            r_k = rates[k]
            for sp_name, nu_ik in self.reaction_stoichiometry[k].items():
                i = self.component_id_to_index[sp_name]
                src[i] += nu_ik * r_k
        return src

    # NOTE: calculate feed-side reaction heat generation from precomputed rates
    def _reaction_heat_source_feed(
        self,
        rates: np.ndarray,
        temperature: Temperature
    ) -> float:
        # >> check for empty rates (no reactions defined) and return zero heat generation if so
        if rates.size == 0:
            return 0.0

        # calculate reaction enthalpies at current feed temperature
        delta_h = self.thermo_source.calc_dH_rxns(temperature=temperature)

        # calculate total reaction heat generation per unit length (W/m) based on current rates and reaction enthalpies
        return float(calc_rxn_heat_generation(delta_h=delta_h, rates=rates, reactor_volume=1.0))

    # SECTION: Energy balance builder
    def _build_temperature_derivatives(
        self,
        Ff: np.ndarray,
        Fp: np.ndarray,
        Tf: float,
        Tp: float,
        q_rxn_f: float
    ) -> Tuple[float, float]:
        # NOTE: sets
        # ! feed temperature [K]
        tf_obj = Temperature(value=Tf, unit="K")
        # ! permeate temperature [K]
        tp_obj = Temperature(value=Tp, unit="K")

        # NOTE: calculate heat capacities and heat-capacity flows
        # ! cp_i for feed [J/mol.K]
        cp_f = self.thermo_source.calc_Cp_IG(temperature=tf_obj)
        # ! cp_p for permeate [J/mol.K]
        cp_p = self.thermo_source.calc_Cp_IG(temperature=tp_obj)

        # >>> cp_flow_f for feed
        # ! [W/K]
        cp_flow_f = float(calc_total_heat_capacity(x=Ff, cp=cp_f))
        cp_flow_p = float(calc_total_heat_capacity(x=Fp, cp=cp_p))

        if cp_flow_f <= 1e-16:
            raise ValueError(
                "Feed-side heat-capacity flow is too small or zero."
            )

        # For near-zero permeate flow the permeate temperature equation is ill-conditioned.
        # Keep a stable fallback until permeate flow builds up.
        if cp_flow_p <= 1e-16:
            cp_flow_p = np.inf

        # NOTE: conductive heat transfer across membrane
        # ! [W/m] (positive from feed to permeate)
        q_cond = self.U_m * (Tf - Tp)

        # NOTE: energy balances
        # ! feed side [K/m]
        dTf_dz = self.a_m * (-q_cond + self.q_ext_f) / \
            cp_flow_f + q_rxn_f / cp_flow_f

        # ! permeate side [K/m]
        # NOTE: only conductive transfer changes orientation with flow pattern.
        dTp_dz = self.a_m * (self.s_p * q_cond + self.q_ext_p) / cp_flow_p

        return float(dTf_dz), float(dTp_dz)
