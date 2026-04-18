from types import SimpleNamespace

import numpy as np
import pytest

from pymemsim.models.results import MembraneResult
from pymemsim.utils.result_tools import analyze_hfm_result


class DummyThermoSource:
    def __init__(self):
        self.MW = np.array([44.0, 28.0], dtype=float)

    def calc_gas_volumetric_flow_rate(
        self,
        molar_flow_rate: float,
        temperature: float,
        pressure: float,
        R: float,
        gas_model: str,
    ) -> float:
        _ = gas_model
        return float(molar_flow_rate * R * temperature / pressure)

    def calc_rho_LIQ(self, temperature):
        _ = temperature
        return np.array([1000.0, 1000.0], dtype=float)

    def calc_liquid_volumetric_flow_rate(self, molar_flow_rates, molecular_weights, density):
        mass_flow_kg_s = np.sum(np.asarray(molar_flow_rates, dtype=float) * np.asarray(molecular_weights, dtype=float)) / 1000.0
        rho_mean = float(np.mean(np.asarray(density, dtype=float)))
        return float(mass_flow_kg_s / rho_mean)


def _build_components():
    return [
        SimpleNamespace(name="A", formula="A", state="g"),
        SimpleNamespace(name="B", formula="B", state="g"),
    ]


def test_analyze_hfm_result_gas_non_isothermal():
    thermo = DummyThermoSource()
    components = _build_components()
    module = SimpleNamespace(
        component_formula_state=["A-g", "B-g"],
        heat_transfer_mode="non-isothermal",
        phase="gas",
        gas_model="ideal",
        R=8.314462618,
        Ff_in=np.array([10.0, 5.0], dtype=float),
        Fp_in=np.array([0.0, 0.0], dtype=float),
        Tf_in=300.0,
        Tp_in=290.0,
        Pf=500000.0,
        Pp=100000.0,
        Pi=np.array([1.0e-8, 2.0e-8], dtype=float),
        U_m=10.0,
        a_m=1.0,
        q_ext_f=0.0,
        q_ext_p=0.0,
    )
    hfm = SimpleNamespace(
        module=module,
        components=components,
        thermo_source=thermo,
        unit_options=SimpleNamespace(
            phase="gas",
            modeling_type="scale",
            feed_pressure_mode="constant",
            permeate_pressure_mode="constant",
        ),
    )

    span = np.array([0.0, 1.0], dtype=float)
    state = np.array(
        [
            [10.0, 8.0],   # Ff_A
            [5.0, 4.0],    # Ff_B
            [0.0, 2.0],    # Fp_A
            [0.0, 1.0],    # Fp_B
            [300.0, 295.0],  # Tf
            [290.0, 292.0],  # Tp
        ],
        dtype=float,
    )
    result = MembraneResult(span=span, state=state, success=True, message="ok")

    analysis = analyze_hfm_result(result=result, hfm_module=hfm, target_component="A-g")

    assert set(analysis.keys()) == {
        "case_definition",
        "streams",
        "performance",
        "profiles",
        "thermal",
        "hydraulic",
        "balances",
        "warnings",
    }
    assert analysis["performance"]["target_component"] == "A-g"
    assert np.isclose(analysis["performance"]["stage_cut_molar"], 3.0 / 15.0)
    assert np.isclose(analysis["performance"]["recoveries_permeate"]["A-g"], 2.0 / 10.0)
    assert np.isclose(analysis["performance"]["purity_permeate_target"], 2.0 / 3.0)
    assert np.isclose(analysis["thermal"]["feed_delta_T_K"], -5.0)
    assert np.isclose(analysis["thermal"]["permeate_delta_T_K"], 2.0)
    assert np.isclose(analysis["balances"]["overall_molar_closure_residual"], 0.0, atol=1e-12)
    assert analysis["profiles"]["flux_profiles_mol_per_m2_s"]["A-g"][0] is not None


def test_analyze_hfm_result_liquid_isothermal():
    thermo = DummyThermoSource()
    components = _build_components()
    module = SimpleNamespace(
        component_formula_state=["A-l", "B-l"],
        heat_transfer_mode="isothermal",
        phase="liquid",
        operation_mode="constant_pressure",
        Ff_in=np.array([2.0, 3.0], dtype=float),
        Fp_in=np.array([0.0, 0.0], dtype=float),
        Tf_in=298.0,
        Tp_in=298.0,
        Pf=200000.0,
        Pp=100000.0,
        k_i=np.array([2.0e-6, 1.0e-6], dtype=float),
        U_m=0.0,
        a_m=1.0,
        q_ext_f=0.0,
        q_ext_p=0.0,
        qf_in=0.001,
        qp_in=0.001,
    )
    hfm = SimpleNamespace(
        module=module,
        components=[
            SimpleNamespace(name="A", formula="A", state="l"),
            SimpleNamespace(name="B", formula="B", state="l"),
        ],
        thermo_source=thermo,
        unit_options=SimpleNamespace(
            phase="liquid",
            modeling_type="physical",
            feed_pressure_mode="constant",
            permeate_pressure_mode="constant",
        ),
    )

    span = np.array([0.0, 1.0], dtype=float)
    state = np.array(
        [
            [2.0, 1.2],  # Ff_A
            [3.0, 2.3],  # Ff_B
            [0.0, 0.8],  # Fp_A
            [0.0, 0.7],  # Fp_B
        ],
        dtype=float,
    )
    result = MembraneResult(span=span, state=state, success=True, message="ok")

    analysis = analyze_hfm_result(result=result, hfm_module=hfm)

    assert analysis["case_definition"]["phase"] == "liquid"
    assert analysis["profiles"]["driving_force_name"] == "delta_concentration_mol_per_m3"
    assert "A-l" in analysis["profiles"]["driving_force_profiles"]
    assert analysis["thermal"]["is_non_isothermal"] is False
    assert analysis["thermal"]["feed_delta_T_K"] is None


def test_analyze_hfm_result_zero_denominator_warnings():
    thermo = DummyThermoSource()
    module = SimpleNamespace(
        component_formula_state=["A-g", "B-g"],
        heat_transfer_mode="isothermal",
        phase="gas",
        gas_model="ideal",
        R=8.314462618,
        Ff_in=np.array([1.0, 0.0], dtype=float),
        Fp_in=np.array([0.0, 0.0], dtype=float),
        Tf_in=300.0,
        Tp_in=300.0,
        Pf=300000.0,
        Pp=100000.0,
        Pi=np.array([1.0e-8, 1.0e-8], dtype=float),
        U_m=0.0,
        a_m=1.0,
        q_ext_f=0.0,
        q_ext_p=0.0,
    )
    hfm = SimpleNamespace(
        module=module,
        components=[
            SimpleNamespace(name="A", formula="A", state="g"),
            SimpleNamespace(name="B", formula="B", state="g"),
        ],
        thermo_source=thermo,
        unit_options=SimpleNamespace(phase="gas"),
    )

    span = np.array([0.0, 1.0], dtype=float)
    state = np.array(
        [
            [1.0, 0.7],  # Ff_A
            [0.0, 0.0],  # Ff_B
            [0.0, 0.3],  # Fp_A
            [0.0, 0.0],  # Fp_B
        ],
        dtype=float,
    )
    result = MembraneResult(span=span, state=state, success=True, message="ok")
    analysis = analyze_hfm_result(result=result, hfm_module=hfm, target_component="B-g")

    assert analysis["performance"]["recoveries_permeate"]["B-g"] is None
    assert analysis["performance"]["enrichment_factor_target"] is None
    assert len(analysis["warnings"]) > 0


def test_analyze_hfm_result_shape_validation():
    module = SimpleNamespace(
        component_formula_state=["A-g", "B-g"],
        heat_transfer_mode="isothermal",
        phase="gas",
        Ff_in=np.array([1.0, 1.0], dtype=float),
        Fp_in=np.array([0.0, 0.0], dtype=float),
        Tf_in=300.0,
        Tp_in=300.0,
        Pf=200000.0,
        Pp=100000.0,
        Pi=np.array([1.0e-8, 1.0e-8], dtype=float),
    )
    hfm = SimpleNamespace(
        module=module,
        components=[
            SimpleNamespace(name="A", formula="A", state="g"),
            SimpleNamespace(name="B", formula="B", state="g"),
        ],
        thermo_source=DummyThermoSource(),
        unit_options=SimpleNamespace(phase="gas"),
    )

    bad_result = MembraneResult(
        span=np.array([0.0, 1.0, 2.0], dtype=float),
        state=np.array(
            [
                [1.0, 0.9],
                [1.0, 0.8],
                [0.0, 0.1],
                [0.0, 0.2],
            ],
            dtype=float,
        ),
        success=True,
        message="ok",
    )
    with pytest.raises(ValueError, match="span/state mismatch"):
        analyze_hfm_result(result=bad_result, hfm_module=hfm)
