# import libs
from typing import Dict, List, Literal, Optional, Tuple, Union
# local
from .docs import HFM


def hfm_module(unit_name: str,
               flow_mode: Literal[
                   'co-current', 'counter-current', 'cross-flow'
               ],
               energy_analysis: bool = False) -> HFM:
    """
    Create a hollow fiber membrane (HFM) module.

    Parameters
    ----------
    unit_name : str
        Name of the unit.
    flow_mode : str
        Flow mode of the unit. Options are 'co-current', 'counter-current', or 'cross-flow'.
    energy_analysis : bool, optional
        Whether to include energy analysis in the model. Default is False.

    Returns
    -------
    HFM
        HFM module object.
    """
    try:
        # NOTE: check if flow_mode is valid
        if flow_mode not in ['co-current', 'counter-current', 'cross-flow']:
            raise ValueError(
                "Invalid flow mode. Options are 'co-current', 'counter-current', or 'cross-flow'.")

        # NOTE: check if energy_balance is valid
        if not isinstance(energy_analysis, bool):
            raise ValueError(
                "Invalid energy balance. Must be a boolean value.")

        # NOTE: check if unit_name is valid
        if not isinstance(unit_name, str):
            raise ValueError("Invalid unit name. Must be a string.")

        # NOTE: check if unit_name is empty
        if unit_name == "":
            raise ValueError("Invalid unit name. Must be a non-empty string.")

        # build the HFM module
        return HFM(unit_name, flow_mode, energy_analysis)
    except Exception as e:
        raise Exception(f"Error in creation of hfm_module: {e}") from e
