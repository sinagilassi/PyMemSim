# import libs
import logging
from typing import Dict, Any
from pythermodb_settings.utils import measure_time
# locals
# ! source
from .sources.thermo_source import ThermoSource
# ! membrane
from .docs.hfm import HFM


# NOTE: logger setup
logger = logging.getLogger(__name__)

# SECTION: Create hollow fiber membrane module


@measure_time
def create_hfm_module(
    model_inputs: Dict[str, Any],
    thermo_source: ThermoSource,
    **kwargs,
) -> HFM:
    """
    Factory function to create a hollow fiber membrane module (HFM) instance based on the provided model inputs and thermo source.

    Parameters
    ----------
    model_inputs : Dict[str, Any]
        A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values.
        - feed_inlet_flows: Dict[str, CustomProp]
          or feed_inlet_flow + feed_mole_fractions
        - feed temperature: Temperature
        - feed pressure: Pressure
    thermo_source : ThermoSource
        A ThermoSource object containing the thermodynamic source information for the batch reactor simulation.
    **kwargs
        Additional keyword arguments for future extensions.
        - mode : Literal['silent', 'log', 'attach'], optional
                Mode for time measurement logging. Default is 'silent'.

    Returns
    -------
    HFM
        An instance of the HFM class representing the hollow fiber membrane module.
    """
    # NOTE: create HFM instance
    hfm_module = HFM(
        model_inputs=model_inputs,
        thermo_source=thermo_source,
        **kwargs,
    )

    return hfm_module
