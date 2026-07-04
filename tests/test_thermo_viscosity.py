import numpy as np
import pytest
from pythermodb_settings.models import CustomProp

from pymemsim.sources.thermo_source_core import ThermoSourceCore


def test_calc_vis_gas_uses_wilke_mixture():
    thermo = ThermoSourceCore.__new__(ThermoSourceCore)
    thermo.Vis_GAS = [
        CustomProp(value=1.0e-5, unit="Pa.s"),
        CustomProp(value=2.0e-5, unit="Pa.s"),
    ]
    thermo.MW = np.array([16.0, 44.0], dtype=float)

    res = thermo.calc_Vis_GAS(mole_fractions=np.array([0.25, 0.75]))

    assert res.unit == "Pa.s"
    assert 1.0e-5 < float(res.value) < 2.0e-5


def test_calc_vis_liq_uses_log_mixture():
    thermo = ThermoSourceCore.__new__(ThermoSourceCore)
    thermo.Vis_LIQ = [
        CustomProp(value=1.0e-3, unit="Pa.s"),
        CustomProp(value=4.0e-3, unit="Pa.s"),
    ]

    res = thermo.calc_Vis_LIQ(mole_fractions=np.array([0.5, 0.5]))

    assert res.unit == "Pa.s"
    assert float(res.value) == pytest.approx(2.0e-3)


def test_calc_vis_gas_missing_data_raises():
    thermo = ThermoSourceCore.__new__(ThermoSourceCore)
    thermo.Vis_GAS = []
    thermo.MW = np.array([16.0], dtype=float)

    with pytest.raises(ValueError, match="Gas viscosity data is required"):
        thermo.calc_Vis_GAS(mole_fractions=np.array([1.0]))
