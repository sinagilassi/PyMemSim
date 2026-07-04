from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pythermodb_settings.models import CustomProp

from pymemsim.core.gas_hfm import GasHFM
from pymemsim.core.gas_hfmx import GasHFMX
from pymemsim.core.hfmc import HFMCore
from pymemsim.core.liquid_hfm import LiquidHFM
from pymemsim.docs.hfm import HFM
from pymemsim.models.hfm import HollowFiberMembraneOptions
from pymemsim.models.results import MembraneResult


def _build_hfm_with_module(
    module: GasHFM | GasHFMX | LiquidHFM,
    flow_pattern: str,
) -> HFM:
    hfm = HFM.__new__(HFM)
    hfm.module = module
    hfm.hfm_core = SimpleNamespace(
        flow_pattern=flow_pattern,
        is_co_current=(flow_pattern == "co-current"),
        is_counter_current=(flow_pattern == "counter-current"),
    )
    return hfm


class _DummyGasPhysical(GasHFM):
    def __init__(self):
        self.component_num = 1
        self.ns = 1
        self.heat_transfer_mode = "isothermal"
        self.Ff_in = np.array([1.0], dtype=float)
        self.Fp_in = np.array([0.2], dtype=float)
        self.Tf_in = 300.0
        self.Tp_in = 300.0

    def rhs(self, z: float, y: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0], dtype=float)


class _DummyGasScaled(GasHFMX):
    def __init__(self):
        self.component_num = 1
        self.ns = 1
        self.heat_transfer_mode = "isothermal"
        self.Ff_in = np.array([1.0], dtype=float)
        self.Fp_in = np.array([0.2], dtype=float)
        self.Ff_scale = np.array([1.0], dtype=float)
        self.Fp_scale = np.array([0.2], dtype=float)
        self.Tf_in = 300.0
        self.Tp_in = 300.0
        self.Tf_scale_ref = 300.0
        self.Tp_scale_ref = 300.0
        self.T_scale = 100.0

    def rhs_scaled(self, z: float, y_scaled: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0], dtype=float)

    def _unscale_state(self, y_scaled: np.ndarray):
        ff = y_scaled[:1] * self.Ff_scale
        fp = y_scaled[1:2] * self.Fp_scale
        return ff, fp, self.Tf_in, self.Tp_in


class _DummyGasPhysicalNonIsothermal(GasHFM):
    def __init__(self):
        self.component_num = 1
        self.ns = 1
        self.heat_transfer_mode = "non-isothermal"
        self.Ff_in = np.array([1.0], dtype=float)
        self.Fp_in = np.array([0.2], dtype=float)
        self.Tf_in = 330.0
        self.Tp_in = 290.0

    def rhs(self, z: float, y: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=float)


class _DummyLiquid(LiquidHFM):
    def __init__(self):
        self.component_num = 1
        self.heat_transfer_mode = "isothermal"
        self.Ff_in = np.array([1.0], dtype=float)
        self.Fp_in = np.array([0.0], dtype=float)


class _ThermoStub:
    MW = np.array([16.0, 44.0], dtype=float)

    def calc_Cp_IG(self, temperature):
        return np.array([1.0, 1.0], dtype=float)

    def calc_gas_volumetric_flow_rate(self, molar_flow_rate, temperature, pressure, R, gas_model):
        _ = gas_model
        return float(molar_flow_rate * R * temperature / pressure)

    def calc_Vis_GAS(self, mole_fractions):
        _ = mole_fractions
        return CustomProp(value=1.8e-5, unit="Pa.s")


def test_flow_pattern_aliases_normalize_to_canonical():
    assert HFMCore._normalize_flow_pattern("co-current") == "co-current"
    assert HFMCore._normalize_flow_pattern("counter-current") == "counter-current"
    assert HFMCore._normalize_flow_pattern("cocurrent") == "co-current"
    assert HFMCore._normalize_flow_pattern("countercurrent") == "counter-current"

    opt = HollowFiberMembraneOptions(phase="gas", flow_pattern="countercurrent")
    assert opt.flow_pattern == "countercurrent"


def test_cocurrent_ivp_regression_shape_and_success():
    module = _DummyGasPhysical()
    hfm = _build_hfm_with_module(module=module, flow_pattern="co-current")

    res = hfm.simulate(length_span=(0.0, 1.0), solver_options={"max_step": 0.1})
    assert res is not None
    assert res.success is True
    assert res.state.shape[0] == 2
    assert res.state.shape[1] > 1


def test_countercurrent_bvp_converges_physical():
    module = _DummyGasPhysical()
    hfm = _build_hfm_with_module(module=module, flow_pattern="counter-current")

    res = hfm.simulate(
        length_span=(0.0, 1.0),
        solver_options={"mesh_points": 30, "tol": 1e-5, "max_nodes": 5000},
    )
    assert res is not None
    assert res.success is True
    bc_res = module.bc(res.state[:, 0], res.state[:, -1])
    assert np.max(np.abs(bc_res)) < 1e-6


def test_countercurrent_bvp_converges_scaled_and_is_finite():
    module = _DummyGasScaled()
    hfm = _build_hfm_with_module(module=module, flow_pattern="counter-current")

    res = hfm.simulate(
        length_span=(0.0, 1.0),
        solver_options={"mesh_points": 30, "tol": 1e-5, "max_nodes": 5000},
    )
    assert res is not None
    assert res.success is True
    assert np.all(np.isfinite(res.state))
    assert np.all(res.state[:2, :] >= 0.0)


def test_countercurrent_bvp_explicit_option_regression():
    module = _DummyGasPhysical()
    hfm = _build_hfm_with_module(module=module, flow_pattern="counter-current")

    res = hfm.simulate(
        length_span=(0.0, 1.0),
        solver_options={
            "countercurrent_solver": "bvp",
            "mesh_points": 30,
            "tol": 1e-5,
            "max_nodes": 5000,
        },
    )
    assert res is not None
    assert res.success is True


def test_countercurrent_shooting_converges_physical():
    module = _DummyGasPhysical()
    hfm = _build_hfm_with_module(module=module, flow_pattern="counter-current")

    res = hfm.simulate(
        length_span=(0.0, 1.0),
        solver_options={"countercurrent_solver": "shooting"},
    )
    assert res is not None
    assert res.success is True
    bc_res = module.bc(res.state[:, 0], res.state[:, -1])
    assert np.max(np.abs(bc_res[module.ns:2 * module.ns])) < 1e-6


def test_countercurrent_shooting_converges_scaled_and_is_finite():
    module = _DummyGasScaled()
    hfm = _build_hfm_with_module(module=module, flow_pattern="counter-current")

    res = hfm.simulate(
        length_span=(0.0, 1.0),
        solver_options={"countercurrent_solver": "shooting"},
    )
    assert res is not None
    assert res.success is True
    assert np.all(np.isfinite(res.state))
    assert np.all(res.state[:2, :] >= 0.0)


def test_countercurrent_shooting_non_isothermal_terminal_temperature_enforced():
    module = _DummyGasPhysicalNonIsothermal()
    hfm = _build_hfm_with_module(module=module, flow_pattern="counter-current")

    res = hfm.simulate(
        length_span=(0.0, 1.0),
        solver_options={"countercurrent_solver": "shooting"},
    )
    assert res is not None
    assert res.success is True
    bc_res = module.bc(res.state[:, 0], res.state[:, -1])
    assert abs(float(bc_res[-1])) < 1e-6


def test_countercurrent_shooting_failure_returns_none():
    module = _DummyGasPhysical()
    hfm = _build_hfm_with_module(module=module, flow_pattern="counter-current")

    res = hfm.simulate(
        length_span=(0.0, 1.0),
        solver_options={
            "countercurrent_solver": "shooting",
            "shooting_ivp_method": "NOT_A_VALID_METHOD",
            "shooting_max_nfev": 10,
        },
    )
    assert res is None


def test_countercurrent_shooting_route_calls_solver_module():
    module = _DummyGasPhysical()
    hfm = _build_hfm_with_module(module=module, flow_pattern="counter-current")
    fake = MembraneResult(
        span=np.array([0.0, 1.0], dtype=float),
        state=np.array([[1.0, 1.0], [0.2, 0.2]], dtype=float),
        success=True,
        message="ok",
    )

    with patch("pymemsim.docs.hfm.solve_countercurrent_shooting", return_value=fake) as mocked:
        res = hfm.simulate(
            length_span=(0.0, 1.0),
            solver_options={"countercurrent_solver": "shooting"},
        )

    assert mocked.called
    assert res is fake


def test_countercurrent_solver_invalid_option_raises():
    module = _DummyGasPhysical()
    hfm = _build_hfm_with_module(module=module, flow_pattern="counter-current")

    with _expect_raises(ValueError, "Invalid countercurrent_solver"):
        hfm.simulate(
            length_span=(0.0, 1.0),
            solver_options={"countercurrent_solver": "invalid"},
        )


def test_countercurrent_liquid_guardrail():
    module = _DummyLiquid()
    hfm = _build_hfm_with_module(module=module, flow_pattern="counter-current")

    with _expect_raises(NotImplementedError, "only for gas modules"):
        hfm.simulate(length_span=(0.0, 1.0))


def test_countercurrent_zero_permeate_guess_stays_positive_and_finite():
    module = _DummyGasPhysical()
    module.component_num = 2
    module.ns = 2
    module.Ff_in = np.array([0.8, 0.2], dtype=float)
    module.Fp_in = np.array([0.0, 0.0], dtype=float)

    z = np.linspace(0.0, 1.0, 40)
    y_guess = module.build_initial_guess(z)
    fp_guess = y_guess[module.ns:2 * module.ns, :]

    assert np.all(np.isfinite(fp_guess))
    assert np.all(fp_guess >= 0.0)
    # positive away from z=L to keep mole-fraction denominators well conditioned
    assert np.all(fp_guess[:, :-1] > 0.0)


def test_countercurrent_permeate_energy_sign_changes_conduction_only():
    module = GasHFM.__new__(GasHFM)
    module.thermo_source = _ThermoStub()
    module.a_m = 1.0
    module.U_m = 2.0
    module.q_ext_f = 0.0
    module.q_ext_p = 1.0
    module.s_p = -1

    Ff = np.array([2.0, 1.0], dtype=float)
    Fp = np.array([1.5, 0.5], dtype=float)
    Tf = 320.0
    Tp = 300.0
    dTf_dz, dTp_dz = module._build_temperature_derivatives(
        Ff=Ff,
        Fp=Fp,
        Tf=Tf,
        Tp=Tp,
        q_rxn_f=0.0,
    )

    cp_flow_p = float(np.sum(Fp))
    q_cond = module.U_m * (Tf - Tp)
    expected_dTp = module.a_m * (module.s_p * q_cond + module.q_ext_p) / cp_flow_p
    assert abs(dTp_dz - expected_dTp) < 1e-12
    assert np.isfinite(dTf_dz)


def _build_pressure_state_module() -> GasHFM:
    module = GasHFM.__new__(GasHFM)
    module.component_num = 2
    module.ns = 2
    module.heat_transfer_mode = "isothermal"
    module.feed_pressure_mode = "state_variable"
    module.permeate_pressure_mode = "state_variable"
    module.has_feed_pressure_state = True
    module.has_permeate_pressure_state = True
    module.Ff_in = np.array([0.7, 0.3], dtype=float)
    module.Fp_in = np.array([1.0e-12, 1.0e-12], dtype=float)
    module.Tf_in = 300.0
    module.Tp_in = 300.0
    module.Pf = 400000.0
    module.Pp = 100000.0
    module.pressure_floor = 1.0
    module.a_m = 0.1
    module.Pi = np.array([1.0e-9, 2.0e-9], dtype=float)
    module.s_p = 1
    module.R = 8.314462618
    module.gas_model = "ideal"
    module.thermo_source = _ThermoStub()
    module.reaction_rates = []
    module.component_formula_state = ["A-g", "B-g"]
    module.n_fibers = 100.0
    module.d_inner = 1.0e-3
    module.shell_flow_area = 1.0e-3
    module.shell_hydraulic_diameter = 1.0e-2
    return module


def test_gas_hfm_pressure_states_extend_y0_and_rhs():
    module = _build_pressure_state_module()

    y0 = module.build_y0()
    assert y0.shape == (6,)
    assert y0[4] == module.Pf
    assert y0[5] == module.Pp**2

    dy = module.rhs(0.0, y0)
    assert dy.shape == y0.shape
    assert dy[4] < 0.0
    assert dy[5] < 0.0


def test_gas_hfm_flux_uses_local_pressures():
    module = _build_pressure_state_module()
    Ff = np.array([0.5, 0.5], dtype=float)
    Fp = np.array([0.2, 0.8], dtype=float)

    J = module._calc_fluxes(Ff=Ff, Fp=Fp, Pf=300000.0, Pp=50000.0)
    expected = module.Pi * (
        np.array([0.5, 0.5]) * 300000.0
        - np.array([0.2, 0.8]) * 50000.0
    )

    assert np.allclose(J, expected)


def test_physical_result_converts_permeate_pressure_squared_to_pressure():
    module = _build_pressure_state_module()
    hfm = HFM.__new__(HFM)
    hfm.module = module

    state = np.array(
        [
            [0.7, 0.6],
            [0.3, 0.25],
            [1.0e-12, 0.1],
            [1.0e-12, 0.05],
            [400000.0, 390000.0],
            [100000.0**2, 90000.0**2],
        ],
        dtype=float,
    )

    public = hfm._state_to_physical(state)

    assert public.shape == state.shape
    assert np.allclose(public[4], state[4])
    assert np.allclose(public[5], np.array([100000.0, 90000.0]))


def test_scaled_result_converts_pressure_states_to_physical_pressure():
    module = GasHFMX.__new__(GasHFMX)
    module.component_num = 1
    module.ns = 1
    module.has_feed_pressure_state = True
    module.has_permeate_pressure_state = True
    module.heat_transfer_mode = "isothermal"
    module.Ff_scale = np.array([2.0], dtype=float)
    module.Fp_scale = np.array([0.5], dtype=float)
    module.Pf = 400000.0
    module.Pp = 100000.0
    module.Pf_scale = 400000.0
    module.Pp2_scale = 100000.0**2

    hfm = HFM.__new__(HFM)
    hfm.module = module
    state_scaled = np.array(
        [
            [1.0],
            [0.2],
            [0.95],
            [0.81],
        ],
        dtype=float,
    )

    public = hfm._state_to_physical(state_scaled)

    assert public.shape == state_scaled.shape
    assert np.isclose(public[0, 0], 2.0)
    assert np.isclose(public[1, 0], 0.1)
    assert np.isclose(public[2, 0], 380000.0)
    assert np.isclose(public[3, 0], 90000.0)


def test_scaled_gas_temperature_unscale_uses_thermo_safe_floor():
    module = GasHFMX.__new__(GasHFMX)
    module.component_num = 1
    module.ns = 1
    module.has_feed_pressure_state = False
    module.has_permeate_pressure_state = False
    module.heat_transfer_mode = "non-isothermal"
    module.Ff_scale = np.array([1.0], dtype=float)
    module.Fp_scale = np.array([1.0], dtype=float)
    module.Tf_scale_ref = 303.0
    module.Tp_scale_ref = 303.0
    module.T_scale = 100.0
    module.temperature_floor = 100.0

    y_scaled = np.array([1.0, 1.0, -10.0, -10.0], dtype=float)

    _, _, _, _, tf, tp = module._unscale_state_full(y_scaled)

    assert tf >= 100.0
    assert tp >= 100.0


class _expect_raises:
    def __init__(self, exc_type: type[BaseException], contains: str):
        self.exc_type = exc_type
        self.contains = contains

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} to be raised.")
        if not isinstance(exc, self.exc_type):
            raise AssertionError(f"Expected {self.exc_type.__name__}, got {type(exc).__name__}.")
        if self.contains not in str(exc):
            raise AssertionError(
                f"Expected error message to contain '{self.contains}', got '{exc}'."
            )
        return True
