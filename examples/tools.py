from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class FeedConfig:
    """Container for membrane feed configuration."""
    species_molar_flows: Dict[str, float]   # mol/s
    total_molar_flow: float                 # mol/s
    mole_fractions: Dict[str, float]        # -
    total_volumetric_flow: float | None = None  # m^3/s


def _validate_mole_fractions(mole_fractions: Dict[str, float], tol: float = 1e-8) -> None:
    if not mole_fractions:
        raise ValueError("mole_fractions cannot be empty.")

    total_x = sum(mole_fractions.values())

    if abs(total_x - 1.0) > tol:
        raise ValueError(
            f"Mole fractions must sum to 1.0, but got {total_x:.12f}."
        )

    for comp, x in mole_fractions.items():
        if x < 0.0 or x > 1.0:
            raise ValueError(f"Invalid mole fraction for {comp}: {x}")


def configure_feed_from_molar_flow(
    mole_fractions: Dict[str, float],
    total_molar_flow: float,
) -> FeedConfig:
    """
    Configure feed from total molar flow rate.

    Parameters
    ----------
    mole_fractions : dict
        Example: {"CO2": 0.4, "N2": 0.6}
    total_molar_flow : float
        Total feed molar flow rate [mol/s]

    Returns
    -------
    FeedConfig
    """
    _validate_mole_fractions(mole_fractions)

    if total_molar_flow <= 0.0:
        raise ValueError("total_molar_flow must be positive.")

    species_molar_flows = {
        comp: x * total_molar_flow for comp, x in mole_fractions.items()
    }

    return FeedConfig(
        species_molar_flows=species_molar_flows,
        total_molar_flow=total_molar_flow,
        mole_fractions=mole_fractions.copy(),
        total_volumetric_flow=None,
    )


def configure_feed_from_volumetric_flow(
    mole_fractions: Dict[str, float],
    total_volumetric_flow: float,
    pressure_pa: float,
    temperature_k: float,
    gas_constant: float = 8.314462618,
) -> FeedConfig:
    """
    Configure feed from total volumetric flow rate using ideal gas law.

    Parameters
    ----------
    mole_fractions : dict
        Example: {"CO2": 0.4, "N2": 0.6}
    total_volumetric_flow : float
        Total volumetric flow rate [m^3/s]
    pressure_pa : float
        Feed pressure [Pa]
    temperature_k : float
        Feed temperature [K]
    gas_constant : float
        Gas constant [J/mol/K]

    Returns
    -------
    FeedConfig
    """
    _validate_mole_fractions(mole_fractions)

    if total_volumetric_flow <= 0.0:
        raise ValueError("total_volumetric_flow must be positive.")
    if pressure_pa <= 0.0:
        raise ValueError("pressure_pa must be positive.")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive.")

    total_molar_flow = pressure_pa * \
        total_volumetric_flow / (gas_constant * temperature_k)

    species_molar_flows = {
        comp: x * total_molar_flow for comp, x in mole_fractions.items()
    }

    return FeedConfig(
        species_molar_flows=species_molar_flows,
        total_molar_flow=total_molar_flow,
        mole_fractions=mole_fractions.copy(),
        total_volumetric_flow=total_volumetric_flow,
    )


def cm3s_to_m3s(q_cm3_s: float) -> float:
    """Convert cm^3/s to m^3/s."""
    return q_cm3_s * 1e-6


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    x_feed = {"CO2": 0.4, "N2": 0.6}

    # Case 1: total molar flow is known
    feed1 = configure_feed_from_molar_flow(
        mole_fractions=x_feed,
        total_molar_flow=0.01,  # mol/s
    )
    print("Case 1: from total molar flow")
    print(feed1)
    print()

    # Case 2: total volumetric flow is known
    # Example: Qf = 400 cm^3/s, P = 101325 Pa, T = 298 K
    # qf_cm3_s = 400.0
    # qf_m3_s = cm3s_to_m3s(qf_cm3_s)

    # feed2 = configure_feed_from_volumetric_flow(
    #     mole_fractions=x_feed,
    #     total_volumetric_flow=qf_m3_s,
    #     pressure_pa=101325.0,
    #     temperature_k=298.0,
    # )
    # print("Case 2: from total volumetric flow")
    # print(feed2)
    # print()

    # # Convenient access
    # print("CO2 molar flow [mol/s]:", feed2.species_molar_flows["CO2"])
    # print("N2 molar flow  [mol/s]:", feed2.species_molar_flows["N2"])
