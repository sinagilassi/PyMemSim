from types import SimpleNamespace

import numpy as np

from pymemsim.core.gas_hfm import GasHFM
from pymemsim.core.gas_hfmx import GasHFMX
from pymemsim.core.hfmc import HFMCore
from pymemsim.core.liquid_hfm import LiquidHFM
from pymemsim.docs.hfm import HFM
from pymemsim.models.hfm import HollowFiberMembraneOptions


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


class _DummyLiquid(LiquidHFM):
    def __init__(self):
        self.component_num = 1
        self.heat_transfer_mode = "isothermal"
        self.Ff_in = np.array([1.0], dtype=float)
        self.Fp_in = np.array([0.0], dtype=float)


class _ThermoStub:
    def calc_Cp_IG(self, temperature):
        return np.array([1.0, 1.0], dtype=float)


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
