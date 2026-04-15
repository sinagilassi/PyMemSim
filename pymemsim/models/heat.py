# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias
from pythermodb_settings.models import Component, CustomProp, Volume, Temperature
# locals
from .ref import HeatTransferMode

# SECTION: Heat exchange models


class HeatTransferOptions(BaseModel):
    """
    Options for configuring the heat transfer model.

    Attributes
    ----------
    heat_transfer_mode : HeatTransferMode
        Heat transfer mode (isothermal or non-isothermal).
    heat_transfer_coefficient : CustomProp
        Heat transfer coefficient (required if mode is non-isothermal).
    heat_transfer_area : CustomProp
        Heat transfer area (required if mode is non-isothermal).
    jacket_temperature : Temperature
        Temperature of the jacket (required if mode is non-isothermal).
    """
    heat_transfer_mode: HeatTransferMode = Field(
        ...,
        description="Heat transfer mode (isothermal or non-isothermal)."
    )
    heat_transfer_coefficient: Optional[CustomProp] = Field(
        default=None,
        description="Heat transfer coefficient (required if mode is non-isothermal)."
    )
    heat_transfer_area: Optional[CustomProp] = Field(
        default=None,
        description="Heat transfer area (required if mode is non-isothermal)."
    )
    jacket_temperature: Optional[Temperature] = Field(
        default=None,
        description="Temperature of the jacket (required if mode is non-isothermal)."
    )
    heat_rate: Optional[CustomProp] = Field(
        default=None,
        description="Constant heat rate in W or J/s (optional, can be used instead of heat transfer coefficient and area)."
    )
