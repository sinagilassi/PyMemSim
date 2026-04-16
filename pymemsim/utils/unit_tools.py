# import libs
import logging
import pycuc


def to_m3(value: float, unit: str) -> float:
    """
    Convert a given value to cubic meters (m3) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'm3', 'L', 'cm3', 'ft3', etc.

    Returns
    -------
    float
        The converted value in cubic meters (m3).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='m3')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to m3: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_Pa(value: float, unit: str) -> float:
    """
    Convert a given value to Pascals (Pa) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'Pa', 'kPa', 'MPa', 'bar', 'atm', etc.

    Returns
    -------
    float
        The converted value in Pascals (Pa).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='Pa')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to Pa: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_K(value: float, unit: str) -> float:
    """
    Convert a given value to Kelvin (K) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'K', 'C', 'F', etc.

    Returns
    -------
    float
        The converted value in Kelvin (K).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='K')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to K: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_J_per_mol_K(value: float, unit: str) -> float:
    """
    Convert a given value to Joules per mole-Kelvin (J/mol.K) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'J/mol.K', 'kJ/mol.K', 'cal/mol.K', etc.

    Returns
    -------
    float
        The converted value in Joules per mole-Kelvin (J/mol.K).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='J/mol.K')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to J/mol.K: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_W_per_m2_K(value: float, unit: str) -> float:
    """
    Convert a given value to Watts per square meter-Kelvin (W/m2.K) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'W/m2.K', 'kW/m2.K', 'cal/cm2.K', etc.

    Returns
    -------
    float
        The converted value in Watts per square meter-Kelvin (W/m2.K).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='W/m2.K')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to W/m2.K: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_m2(value: float, unit: str) -> float:
    """
    Convert a given value to square meters (m2) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'm2', 'cm2', 'ft2', etc.

    Returns
    -------
    float
        The converted value in square meters (m2).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='m2')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to m2: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_J_per_mol(value: float, unit: str) -> float:
    """
    Convert a given value to Joules per mole (J/mol) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'J/mol', 'kJ/mol', 'cal/mol', etc.

    Returns
    -------
    float
        The converted value in Joules per mole (J/mol).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='J/mol')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to J/mol: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_g_per_m3(value: float, unit: str) -> float:
    """
    Convert a given value to grams per cubic meter (g/m3) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'g/m3', 'kg/m3', 'lb/ft3', etc.

    Returns
    -------
    float
        The converted value in grams per cubic meter (g/m3).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='g/m3')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to g/m3: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_kg_per_m3(value: float, unit: str) -> float:
    """
    Convert a given value to kilograms per cubic meter (kg/m3).
    """
    try:
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='kg/m3')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to kg/m3: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_g_per_mol(value: float, unit: str) -> float:
    """
    Convert a given value to grams per mole (g/mol) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'g/mol', 'kg/mol', 'lb/mol', etc.

    Returns
    -------
    float
        The converted value in grams per mole (g/mol).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='g/mol')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to g/mol: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_W(
    value: float,
    unit: str
) -> float:
    """
    Convert a given value to Watts (W) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'W', 'kW', 'cal/s', etc.

    Returns
    -------
    float
        The converted value in Watts (W).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='W')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to W: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_mol_per_s(value: float, unit: str) -> float:
    try:
        return pycuc.convert_from_to(value=value, from_unit=unit, to_unit="mol/s")
    except Exception:
        # fallback to raw value for unit systems that are already mol/s but not
        # explicitly recognized by converter
        return float(value)
