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


logger = logging.getLogger(__name__)


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

        # ! membrane area per length [m]
        self.a_m = float(hfm_core.membrane_area_per_length)

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
        y = [Ff_i..., Fp_i..., Tf?, Tp?]
        """
        y0_parts: List[np.ndarray] = [
            self.Ff_in.astype(float), self.Fp_in.astype(float)]
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

        # NOTE: Add temperature BCs if non-isothermal
        if self.heat_transfer_mode == "non-isothermal":
            # Tf at z=0
            bc_Tf = ya[2*ns] - self.Tf_in

            # Tp at z=L
            bc_Tp = yb[2*ns + 1] - self.Tp_in

            return np.concatenate([bc_feed, bc_permeate, np.array([bc_Tf, bc_Tp], dtype=float)])

        return np.concatenate([bc_feed, bc_permeate])

    # ! build mesh
    def build_mesh(self):
        """
        Build mesh for BVP solver.
        """
        # TODO: Consider mesh refinement
        pass

    # ! build initial guess for BVP solver
    def build_initial_guess(self):
        """
        Build initial guess for BVP solver.
        """
        # TODO: Implement more informed initial guess
        pass

    # SECTION: ODE RHS builder
    def rhs(self, z: float, y: np.ndarray) -> np.ndarray:
        ns = self.component_num
        Ff = np.clip(y[:ns], 0.0, None)
        Fp = np.clip(y[ns:2 * ns], 0.0, None)

        if self.heat_transfer_mode == "non-isothermal":
            Tf = float(y[2 * ns])
            Tp = float(y[2 * ns + 1])
        else:
            Tf = self.Tf_in
            Tp = self.Tp_in

        # NOTE: Build feed-side reaction closure once per integration step.
        # ! feed-side
        temperature_f = Temperature(value=Tf, unit="K")
        pressure_f = Pressure(value=self.Pf, unit="Pa")

        # NOTE: Calculate feed-side molar flow rate, composition, and concentration for reaction calculations
        # ! feed-side molar flow rate [mol/s]
        Ff_total = max(float(np.sum(Ff)), 1e-30)

        # ! feed-side mole fraction vector
        yf = Ff / Ff_total

        # ! feed-side volumetric flow rate [m3/s] (calculated from molar flow rate, temperature, pressure, and gas model)
        qf = self.thermo_source.calc_gas_volumetric_flow_rate(
            molar_flow_rate=Ff_total,
            temperature=Tf,
            pressure=self.Pf,
            R=self.R,
            gas_model=cast(GasModel, self.gas_model)
        )
        qf = max(float(qf), 1e-30)

        # ! feed-side concentration vector [mol/m3]
        Cf = Ff / qf

        # NOTE: Build feed-side partial pressure and concentration dicts for reaction rate calculations
        partial_pressures_std = {
            sp: CustomProperty(value=yf[i] * self.Pf, unit="Pa", symbol="P")
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
        J = self._calc_fluxes(Ff=Ff, Fp=Fp)

        # NOTE: Material balances
        # ! feed side: dFf_i/dz = -a_m * J_i + r_i (reaction source)
        dFf_dz = -self.a_m * J + dF_rxn_f

        # ! permeate side: dFp_i/dz = +a_m * J_i
        # ? for co-current flow, the permeate axial sign is positive where as
        # ? for counter-current flow, the permeate axial sign is negative.
        dFp_dz = self.s_p * self.a_m * J

        # >> Combine derivatives
        out = np.concatenate([dFf_dz, dFp_dz])

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
    def _calc_fluxes(self, Ff: np.ndarray, Fp: np.ndarray) -> np.ndarray:
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
        Ff_total = max(float(np.sum(Ff)), 1e-30)
        Fp_total = max(float(np.sum(Fp)), 1e-30)
        yf = Ff / Ff_total
        yp = Fp / Fp_total
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
        # ? for co-current flow, the conductive heat transfer term is negative where as
        # ? for counter-current flow, the conductive heat transfer term is positive.
        dTp_dz = self.a_m * (self.s_p * q_cond + self.q_ext_p) / cp_flow_p

        return float(dTf_dz), float(dTp_dz)
