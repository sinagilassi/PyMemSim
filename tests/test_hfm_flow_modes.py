import numpy as np

from pymemsim.core.gas_hfm import GasHFM
from pymemsim.core.gas_hfm_co_current import GasHFMCoCurrent
from pymemsim.core.gas_hfm_counter_current import GasHFMCounterCurrent
from pymemsim.core.gas_hfmx import GasHFMXCounterCurrent
from pymemsim.docs.hfm import HFM


def test_gas_hfm_backward_compatibility_alias():
    assert issubclass(GasHFM, GasHFMCoCurrent)


def test_counter_current_bc_and_guess_physical_non_isothermal():
    module = GasHFMCounterCurrent.__new__(GasHFMCounterCurrent)
    module.component_num = 2
    module.heat_transfer_mode = "non-isothermal"
    module.Ff_in = np.array([10.0, 5.0], dtype=float)
    module.Fp_in = np.array([1.0, 2.0], dtype=float)
    module.Tf_in = 320.0
    module.Tp_in = 300.0

    z_mesh = np.linspace(0.0, 1.0, 7)
    y_guess = module.build_bvp_guess(z_mesh)

    assert y_guess.shape == (6, 7)
    assert np.all(y_guess[:4, :] > 0.0)

    ya = np.array([10.0, 5.0, 3.0, 4.0, 320.0, 310.0], dtype=float)
    yb = np.array([8.0, 4.0, 1.0, 2.0, 305.0, 300.0], dtype=float)
    bc_res = module.bc(ya, yb)

    assert bc_res.shape == (6,)
    assert np.allclose(bc_res, np.zeros(6), atol=1e-12)


def test_counter_current_bc_scaled_is_zero_at_boundary_conditions():
    module = GasHFMXCounterCurrent.__new__(GasHFMXCounterCurrent)
    module.component_num = 2
    module.heat_transfer_mode = "non-isothermal"
    module.Ff_in = np.array([6.0, 3.0], dtype=float)
    module.Fp_in = np.array([0.5, 0.2], dtype=float)
    module.Ff_scale = np.array([6.0, 3.0], dtype=float)
    module.Fp_scale = np.array([0.5, 0.2], dtype=float)
    module.Tf_in = 330.0
    module.Tp_in = 290.0
    module.Tf_scale_ref = 330.0
    module.Tp_scale_ref = 290.0
    module.T_scale = 100.0

    # ya corresponds to z=0, yb corresponds to z=L
    ya_scaled = np.array([
        1.0, 1.0,   # Ff/Ff_scale
        3.0, 4.0,   # Fp/Fp_scale (free at z=0)
        0.0, 0.5,   # theta_f, theta_p
    ], dtype=float)
    yb_scaled = np.array([
        0.8, 0.9,   # Ff/Ff_scale (free at z=L)
        1.0, 1.0,   # Fp/Fp_scale = inlet at z=L
        0.2, 0.0,   # theta_f, theta_p
    ], dtype=float)

    bc_res = module.bc(ya_scaled, yb_scaled)
    assert bc_res.shape == (6,)
    assert np.allclose(bc_res, np.zeros(6), atol=1e-12)


def test_hfm_detects_counter_current_modules():
    hfm = HFM.__new__(HFM)
    hfm.module = GasHFMCounterCurrent.__new__(GasHFMCounterCurrent)
    assert hfm._is_counter_current_module() is True
