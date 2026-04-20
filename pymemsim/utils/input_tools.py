# import libs
from pythermodb_settings.models import CustomProp
import pycuc
# locals
from ..configs.constants import R_J_per_mol_K

R = R_J_per_mol_K  # Pa·m^3/(mol·K)


def to_m3_per_s(value, unit):
    """
    Convert flow rate to m³/s based on the specified unit.
    """
    return pycuc.convert_from_to(
        value=value,
        from_unit=unit,
        to_unit="m3/s",
    )


def Q_std_to_mol_s(
        Q_std: CustomProp,
        T_std=273.15,
        P_std=101325.0
) -> CustomProp:
    """
    Convert standard volumetric flow [m3/s] → molar flow [mol/s]

    Parameters
    ----------
    Q_std : CustomProp
        Standard volumetric flow with value and unit (e.g., m3/s, L/s).
    T_std : float, optional
        Standard temperature in Kelvin (default is 273.15 K).
    P_std : float, optional
        Standard pressure in Pascals (default is 101325 Pa).

    Returns
    -------
    CustomProp
        Molar flow with value in mol/s and unit "mol/s".
    """
    # calculate molar flow using ideal gas law: n = Q / Vm, where Vm = RT/P
    # calculate standard volumetric flow
    # ! m3/s
    Q_std_m3_s = to_m3_per_s(Q_std.value, Q_std.unit)

    # calculate molar volume at standard conditions: Vm = RT/P
    # ! m3/mol
    Vm_std = R_J_per_mol_K * T_std / P_std

    # calculate molar flow: n = Q / Vm
    # ! mol/s
    res = Q_std_m3_s / Vm_std

    return CustomProp(value=res, unit="mol/s")
