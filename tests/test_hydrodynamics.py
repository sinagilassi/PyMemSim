import math

import pytest

from pymemsim.transport.hydrodynamics import (
    friction_factor_laminar,
    gas_fiber_pressure_squared_derivative,
    reynolds_number,
    shell_pressure_derivative_laminar,
)


def test_gas_fiber_pressure_squared_derivative():
    res = gas_fiber_pressure_squared_derivative(
        mu=1.8e-5,
        molar_flow_rate=0.2,
        n_fibers=100.0,
        diameter_inner=1.0e-3,
        temperature=300.0,
        gas_constant=8.314462618,
    )
    expected = (
        -256.0
        * 8.314462618
        * 300.0
        * 1.8e-5
        * (0.2 / 100.0)
        / (math.pi * (1.0e-3) ** 4)
    )
    assert res == pytest.approx(expected)


def test_shell_pressure_derivative_laminar():
    re = reynolds_number(
        density=1.2,
        velocity=0.5,
        hydraulic_diameter=0.01,
        mu=1.8e-5,
    )
    assert re == pytest.approx(333.3333333333333)
    assert friction_factor_laminar(re) == pytest.approx(64.0 / re)

    dP_dz = shell_pressure_derivative_laminar(
        density=1.2,
        velocity=0.5,
        hydraulic_diameter=0.01,
        mu=1.8e-5,
    )
    expected = -(64.0 / re) * 1.2 * 0.5**2 / (2.0 * 0.01)
    assert dP_dz == pytest.approx(expected)


def test_hydrodynamics_validation():
    with pytest.raises(ValueError, match="mu must be positive"):
        gas_fiber_pressure_squared_derivative(
            mu=0.0,
            molar_flow_rate=1.0,
            n_fibers=1.0,
            diameter_inner=1.0e-3,
            temperature=300.0,
        )
