# import libs
import logging
import math
from typing import Optional

# NOTE: logger setup
logger = logging.getLogger(__name__)


def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _require_non_negative(name: str, value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} cannot be negative.")
    return value


def hagen_poiseuille_liquid(
    mu: float,
    length: float,
    volumetric_flow_rate: float,
    diameter: float,
    p_in: Optional[float] = None,
) -> float:
    """Calculate liquid Hagen-Poiseuille pressure drop or outlet pressure."""
    mu = _require_positive("mu", mu)
    length = _require_positive("length", length)
    volumetric_flow_rate = _require_non_negative(
        "volumetric_flow_rate", volumetric_flow_rate)
    diameter = _require_positive("diameter", diameter)

    delta_p = 128.0 * mu * length * volumetric_flow_rate / (
        math.pi * diameter**4
    )
    if p_in is None:
        return float(delta_p)
    p_in = _require_positive("p_in", p_in)
    return float(p_in - delta_p)


def hagen_poiseuille_gas_compressible(
    mu: float,
    length: float,
    molar_flow_rate: float,
    diameter: float,
    temperature: float,
    p_in: float,
    gas_constant: float = 8.314462618,
) -> float:
    """Calculate gas outlet pressure from the pressure-squared relation."""
    mu = _require_positive("mu", mu)
    length = _require_positive("length", length)
    molar_flow_rate = _require_non_negative("molar_flow_rate", molar_flow_rate)
    diameter = _require_positive("diameter", diameter)
    temperature = _require_positive("temperature", temperature)
    p_in = _require_positive("p_in", p_in)
    gas_constant = _require_positive("gas_constant", gas_constant)

    pressure_square_drop = (
        256.0
        * mu
        * length
        * gas_constant
        * temperature
        * molar_flow_rate
        / (math.pi * diameter**4)
    )
    p_out_squared = p_in**2 - pressure_square_drop
    if p_out_squared < 0.0:
        raise ValueError(
            "Calculated outlet pressure squared is negative. "
            "Check flow rate, viscosity, length, diameter, and inlet pressure."
        )
    return float(math.sqrt(p_out_squared))


def gas_fiber_pressure_squared_derivative(
    mu: float,
    molar_flow_rate: float,
    n_fibers: float,
    diameter_inner: float,
    temperature: float,
    gas_constant: float = 8.314462618,
    axial_sign: float = 1.0,
) -> float:
    """Return d(P^2)/dz for compressible gas flow in hollow-fiber lumens."""
    mu = _require_positive("mu", mu)
    molar_flow_rate = _require_non_negative("molar_flow_rate", molar_flow_rate)
    n_fibers = _require_positive("n_fibers", n_fibers)
    diameter_inner = _require_positive("diameter_inner", diameter_inner)
    temperature = _require_positive("temperature", temperature)
    gas_constant = _require_positive("gas_constant", gas_constant)
    axial_sign = float(axial_sign)

    dP2_dz = (
        -256.0
        * gas_constant
        * temperature
        * mu
        * (molar_flow_rate / n_fibers)
        / (math.pi * diameter_inner**4)
    )
    return float(axial_sign * dP2_dz)


def reynolds_number(
    density: float,
    velocity: float,
    hydraulic_diameter: float,
    mu: float,
) -> float:
    """Return hydraulic-diameter Reynolds number."""
    density = _require_positive("density", density)
    velocity = _require_non_negative("velocity", velocity)
    hydraulic_diameter = _require_positive(
        "hydraulic_diameter", hydraulic_diameter)
    mu = _require_positive("mu", mu)
    return float(density * velocity * hydraulic_diameter / mu)


def friction_factor_laminar(reynolds: float) -> float:
    """Return Darcy friction factor for laminar flow."""
    reynolds = _require_positive("reynolds", reynolds)
    return float(64.0 / reynolds)


def shell_pressure_derivative_laminar(
    density: float,
    velocity: float,
    hydraulic_diameter: float,
    mu: float,
    axial_sign: float = 1.0,
) -> float:
    """Return dP/dz for laminar shell-side flow."""
    density = _require_positive("density", density)
    velocity = _require_non_negative("velocity", velocity)
    hydraulic_diameter = _require_positive(
        "hydraulic_diameter", hydraulic_diameter)
    mu = _require_positive("mu", mu)
    axial_sign = float(axial_sign)

    re = reynolds_number(
        density=density,
        velocity=velocity,
        hydraulic_diameter=hydraulic_diameter,
        mu=mu,
    )
    f = friction_factor_laminar(re)
    dP_dz = -f * density * velocity**2 / (2.0 * hydraulic_diameter)
    return float(axial_sign * dP_dz)


def pressure_drop_fiber_side(
    phase: str,
    mu: float,
    length: float,
    diameter: float,
    p_in: float,
    volumetric_flow_rate: Optional[float] = None,
    molar_flow_rate: Optional[float] = None,
    temperature: Optional[float] = None,
    gas_constant: float = 8.314462618,
) -> float:
    """Calculate fiber-side outlet pressure for gas or liquid flow."""
    phase = phase.strip().lower()
    if phase == "liquid":
        if volumetric_flow_rate is None:
            raise ValueError(
                "volumetric_flow_rate is required for liquid pressure drop."
            )
        return hagen_poiseuille_liquid(
            mu=mu,
            length=length,
            volumetric_flow_rate=volumetric_flow_rate,
            diameter=diameter,
            p_in=p_in,
        )
    if phase == "gas":
        if molar_flow_rate is None:
            raise ValueError(
                "molar_flow_rate is required for gas pressure drop."
            )
        if temperature is None:
            raise ValueError("temperature is required for gas pressure drop.")
        return hagen_poiseuille_gas_compressible(
            mu=mu,
            length=length,
            molar_flow_rate=molar_flow_rate,
            diameter=diameter,
            temperature=temperature,
            p_in=p_in,
            gas_constant=gas_constant,
        )
    raise ValueError("phase must be either 'liquid' or 'gas'.")


def pressure_drop_shell_side(
    density: float,
    velocity: float,
    hydraulic_diameter: float,
    mu: float,
    length: float,
    p_in: Optional[float] = None,
) -> float:
    """Calculate shell-side laminar pressure drop or outlet pressure."""
    length = _require_positive("length", length)
    dP_dz = shell_pressure_derivative_laminar(
        density=density,
        velocity=velocity,
        hydraulic_diameter=hydraulic_diameter,
        mu=mu,
    )
    delta_p = -dP_dz * length
    if p_in is None:
        return float(delta_p)
    p_in = _require_positive("p_in", p_in)
    return float(p_in - delta_p)
