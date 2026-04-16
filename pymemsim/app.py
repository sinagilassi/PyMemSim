# import libs
import logging
from typing import Dict, Any
# locals
# ! source
from .sources.thermo_source import ThermoSource
# ! membrane


# NOTE: logger setup
logger = logging.getLogger(__name__)

# SECTION: Create hollow fiber membrane module


def create_hollow_fiber_membrane_module(
    model_inputs: Dict[str, Any],
    thermo_source: ThermoSource,
    **kwargs,
):
    pass
