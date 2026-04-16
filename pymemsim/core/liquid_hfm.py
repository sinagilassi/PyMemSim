# import libs
import logging
import numpy as np
from typing import Dict, List, Tuple
from pythermodb_settings.models import Component, ComponentKey, CustomProperty, Temperature
from pyreactsim_core.models.rate_exp import ReactionRateExpression
# locals
from ..sources.thermo_source import ThermoSource
from ..utils.reaction_tools import stoichiometry_mat, stoichiometry_mat_key
from ..utils.thermo_tools import calc_rxn_heat_generation, calc_total_heat_capacity
from .hfmc import HFMCore


logger = logging.getLogger(__name__)


class LiquidHFM:
    """
    Liquid-phase hollow-fiber membrane model (cocurrent, dual-side, constant pressure).
    """

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
        self.operation_mode = hfm_core.operation_mode
        self.liquid_density_mode = hfm_core.liquid_density_mode

        # NOTE: Normalized dual-side membrane inputs
        # ! feed inlet flows [mol/s]
        self.Ff_in = hfm_core.feed_inlet_flows.astype(float)
        # ! permeate inlet flows [mol/s]
        self.Fp_in = hfm_core.permeate_inlet_flows.astype(float)

        # ! feed inlet temperature [K]
        self.Tf_in = float(hfm_core.feed_inlet_temperature.value)
        # ! permeate inlet temperature [K]
        self.Tp_in = float(hfm_core.permeate_inlet_temperature.value)

        # ! membrane area per length [m]
        self.a_m = float(hfm_core.membrane_area_per_length)

        # ! overall heat transfer coefficient [W/m2.K]
        self.U_m = float(hfm_core.overall_heat_transfer_coefficient)

        # ! external heat source/sink per unit length [W/m]
        self.q_ext_f = float(hfm_core.q_ext_feed)
        self.q_ext_p = float(hfm_core.q_ext_permeate)

        # ! liquid transport coefficients [m/s]
        self.k_i = hfm_core.liquid_transport_coefficients.astype(float)

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
        self.component_formula_state = self.thermo_source.component_refs["component_formula_state"]
        self.component_id_to_index = self.thermo_source.component_refs["component_id_to_index"]

        # SECTION: Validation
        if self.k_i.shape[0] != self.component_num:
            raise ValueError("liquid_transport_coefficients length must match component_num.")

        # NOTE: Inlet volumetric flows for constant-volume closures per side
        tf_obj = Temperature(value=self.Tf_in, unit="K")
        tp_obj = Temperature(value=self.Tp_in, unit="K")
        rho_f_in = self.thermo_source.calc_rho_LIQ(temperature=tf_obj)
        rho_p_in = self.thermo_source.calc_rho_LIQ(temperature=tp_obj)
        # ! feed inlet volumetric flow [m3/s]
        self.qf_in = self.thermo_source.calc_liquid_volumetric_flow_rate(
            molar_flow_rates=self.Ff_in,
            molecular_weights=self.thermo_source.MW,
            density=rho_f_in
        )
        # ! permeate inlet volumetric flow [m3/s]
        self.qp_in = self.thermo_source.calc_liquid_volumetric_flow_rate(
            molar_flow_rates=self.Fp_in,
            molecular_weights=self.thermo_source.MW,
            density=rho_p_in
        )

    # SECTION: Handlers
    # ! inlet flow
    @property
    def F_in(self) -> np.ndarray:
        """Backward-compatible alias for feed inlet flow vector."""
        return self.Ff_in

    # ! build initial state vector
    def build_y0(self) -> np.ndarray:
        y0_parts: List[np.ndarray] = [self.Ff_in.astype(float), self.Fp_in.astype(float)]
        if self.heat_transfer_mode == "non-isothermal":
            y0_parts.append(np.array([self.Tf_in, self.Tp_in], dtype=float))
        return np.concatenate(y0_parts)

    # SECTION: ODE RHS builder
    def rhs(self, z: float, y: np.ndarray) -> np.ndarray:
        # NOTE: unpack state
        ns = self.component_num
        Ff = np.clip(y[:ns], 0.0, None)
        Fp = np.clip(y[ns:2 * ns], 0.0, None)

        if self.heat_transfer_mode == "non-isothermal":
            Tf = float(y[2 * ns])
            Tp = float(y[2 * ns + 1])
        else:
            Tf = self.Tf_in
            Tp = self.Tp_in

        # NOTE: build temperature objects and densities
        tf_obj = Temperature(value=Tf, unit="K")
        tp_obj = Temperature(value=Tp, unit="K")
        rho_f = self.thermo_source.calc_rho_LIQ(temperature=tf_obj)
        rho_p = self.thermo_source.calc_rho_LIQ(temperature=tp_obj)

        # NOTE: volumetric flow closures
        # ! feed side [m3/s]
        qf = max(self._calc_q_vol(F=Ff, rho_LIQ=rho_f, q_in=float(self.qf_in)), 1e-30)
        # ! permeate side [m3/s]
        qp = max(self._calc_q_vol(F=Fp, rho_LIQ=rho_p, q_in=float(self.qp_in)), 1e-30)

        # NOTE: concentrations and membrane flux
        # ! feed concentration [mol/m3]
        Cf = Ff / qf
        # ! permeate concentration [mol/m3]
        Cp = Fp / qp
        # ! membrane flux [mol/m2.s]
        J = self.k_i * (Cf - Cp)

        # NOTE: feed-side optional reaction source
        dF_rxn_f = self._build_reaction_source_feed(Cf=Cf, Tf=Tf)

        # NOTE: material balances
        # ! feed side [mol/s.m]
        dFf_dz = -self.a_m * J + dF_rxn_f
        # ! permeate side [mol/s.m]
        dFp_dz = +self.a_m * J
        out = np.concatenate([dFf_dz, dFp_dz])

        if self.heat_transfer_mode == "isothermal":
            return out

        # NOTE: energy balances
        dTf_dz, dTp_dz = self._build_temperature_derivatives(
            Ff=Ff,
            Fp=Fp,
            Cf=Cf,
            Tf=Tf,
            Tp=Tp
        )
        return np.concatenate([out, np.array([dTf_dz, dTp_dz], dtype=float)])

    # NOTE: calculate liquid volumetric flow rate from composition and density
    def _calc_q_liquid(self, flow: np.ndarray, rho_LIQ: np.ndarray) -> float:
        return float(self.thermo_source.calc_liquid_volumetric_flow_rate(
            molar_flow_rates=flow,
            molecular_weights=self.thermo_source.MW,
            density=rho_LIQ
        ))

    # NOTE: volumetric flow closure by operation mode
    def _calc_q_vol(self, F: np.ndarray, rho_LIQ: np.ndarray, q_in: float) -> float:
        if self.operation_mode == "constant_volume":
            return float(q_in)
        if self.operation_mode == "constant_pressure":
            return float(self._calc_q_liquid(flow=F, rho_LIQ=rho_LIQ))
        raise ValueError(
            f"Invalid operation_mode '{self.operation_mode}' for liquid HFM."
        )

    # NOTE: calculate reaction rates (liquid concentration basis only)
    def _calc_rates(self, concentration: Dict[str, CustomProperty], temperature: Temperature) -> np.ndarray:
        rates = []
        for rate_exp in self.reaction_rates:
            basis = rate_exp.basis
            if basis != "concentration":
                raise ValueError(
                    f"Invalid basis '{basis}' for liquid HFM reaction rate expression '{rate_exp.name}'. "
                    "Liquid HFM supports basis='concentration' only."
                )
            r_k = rate_exp.calc(
                xi=concentration,
                temperature=temperature,
                pressure=None
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

    # NOTE: calculate feed-side reaction source term based on feed concentrations and temperature
    def _build_reaction_source_feed(self, Cf: np.ndarray, Tf: float) -> np.ndarray:
        if len(self.reaction_rates) == 0:
            return np.zeros(self.component_num, dtype=float)
        concentration_std = {
            sp: CustomProperty(value=Cf[i], unit="mol/m3", symbol="C")
            for i, sp in enumerate(self.component_formula_state)
        }
        rates = self._calc_rates(
            concentration=concentration_std,
            temperature=Temperature(value=Tf, unit="K")
        )
        return self._build_stoich_source(rates=rates)

    # NOTE: calculate feed-side reaction heat generation based on feed concentrations and temperature
    def _reaction_heat_source_feed(self, Cf: np.ndarray, Tf: float) -> float:
        if len(self.reaction_rates) == 0:
            return 0.0
        concentration_std = {
            sp: CustomProperty(value=Cf[i], unit="mol/m3", symbol="C")
            for i, sp in enumerate(self.component_formula_state)
        }
        temperature = Temperature(value=Tf, unit="K")
        rates = self._calc_rates(concentration=concentration_std, temperature=temperature)
        delta_h = self.thermo_source.calc_dH_rxns_LIQ(temperature=temperature)
        return float(calc_rxn_heat_generation(delta_h=delta_h, rates=rates, reactor_volume=1.0))

    # SECTION: Energy balance builder
    def _build_temperature_derivatives(
        self,
        Ff: np.ndarray,
        Fp: np.ndarray,
        Cf: np.ndarray,
        Tf: float,
        Tp: float
    ) -> Tuple[float, float]:
        # NOTE: sets
        # ! feed temperature [K]
        tf_obj = Temperature(value=Tf, unit="K")
        # ! permeate temperature [K]
        tp_obj = Temperature(value=Tp, unit="K")

        # NOTE: calculate heat capacities and heat-capacity flows
        # ! cp_i for feed [J/mol.K]
        cp_f = self.thermo_source.calc_Cp_LIQ(temperature=tf_obj)
        # ! cp_p for permeate [J/mol.K]
        cp_p = self.thermo_source.calc_Cp_LIQ(temperature=tp_obj)

        # >>> cp_flow_f for feed
        # ! [W/K]
        cp_flow_f = float(calc_total_heat_capacity(x=Ff, cp=cp_f))
        cp_flow_p = float(calc_total_heat_capacity(x=Fp, cp=cp_p))

        if cp_flow_f <= 1e-16:
            raise ValueError("Feed-side heat-capacity flow is too small or zero.")

        # For near-zero permeate flow the permeate temperature equation is ill-conditioned.
        # Keep a stable fallback until permeate flow builds up.
        if cp_flow_p <= 1e-16:
            cp_flow_p = np.inf

        # NOTE: reaction heat
        q_rxn_f = self._reaction_heat_source_feed(Cf=Cf, Tf=Tf)

        # NOTE: conductive heat transfer across membrane
        # ! [W/m] (positive from feed to permeate)
        q_cond = self.U_m * (Tf - Tp)

        # NOTE: energy balances
        # ! feed side [K/m]
        dTf_dz = self.a_m * (-q_cond + self.q_ext_f) / cp_flow_f + q_rxn_f / cp_flow_f
        # ! permeate side [K/m]
        dTp_dz = self.a_m * (+q_cond + self.q_ext_p) / cp_flow_p
        return float(dTf_dz), float(dTp_dz)
