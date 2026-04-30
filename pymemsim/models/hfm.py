# import libs
from pydantic import Field, BaseModel, model_validator, ConfigDict
from typing import Literal, Optional
from pythermodb_settings.models import CustomProp
# locals
from .ref import GasModel, UnitPhase, MembraneOptions

# SECTION: Hollow Fiber Membrane Model


class HollowFiberMembraneOptions(MembraneOptions):
    """
    Options for hollow fiber membrane model.

    Attributes
    ----------
    modeling_type : Literal['physical', 'scale']
        Modeling type as physical or scale. The physical model solves ODE states in physical units, while the scale model solves scaled state variables.
    flow_pattern : Literal['co-current', 'counter-current', 'cocurrent', 'countercurrent']
        Flow pattern as co-current/counter-current (canonical) or cocurrent/countercurrent (aliases).
    feed_pressure_mode : Optional[Literal['constant', 'state_variable']]
        Pressure mode as constant and state_variable. The state_variable considers pressure as a variable computes the pressure drop along the unit.
    permeate_pressure_mode : Optional[Literal['constant', 'state_variable']]
        Pressure mode as constant and state_variable. The state_variable considers pressure as a variable computes the pressure drop along the unit.
    phase : UnitPhase
        Phase of the membrane unit (gas or liquid).
    gas_model : GasModel
        Gas model to use (required if phase is gas).
    gas_heat_capacity_mode : Optional[Literal['constant', 'temperature-dependent', 'differential']]
        Gas heat capacity mode as constant, temperature-dependent, and differential.
    liquid_heat_capacity_mode : Optional[Literal['constant', 'temperature-dependent', 'differential']]
        Liquid heat capacity mode as constant, temperature-dependent, and differential.
    liquid_density_mode : Optional[Literal['constant', 'temperature-dependent']]
        Liquid density mode as constant or temperature-dependent.
    ideal_gas_formation_enthalpy_mode : Optional[Literal['model_inputs', 'model_source']]
        Source of gas formation enthalpy as model_inputs or model_source.
    molecular_weight_mode : Optional[Literal['model_inputs', 'model_source']]
        Source of molecular weight as model_inputs or model_source.
    reaction_enthalpy_mode : Optional[Literal['ideal_gas', 'liquid']]
        Mode for reaction enthalpy calculation as ideal_gas or liquid.
    """
    modeling_type: Literal['physical', 'scale'] = Field(
        default="physical",
        description="Modeling type as physical or scale. The physical model solves ODE states in physical units, while the scale model solves scaled state variables."
    )
    flow_pattern: Literal['co-current', 'counter-current', 'cocurrent', 'countercurrent'] = Field(
        default="co-current",
        description=(
            "Flow mode as co-current or counter-current. Alias forms "
            "'cocurrent' and 'countercurrent' are also accepted."
        )
    )
    feed_pressure_mode: Optional[Literal["constant", "state_variable"]] = Field(
        default="constant",
        description="Pressure mode as constant and state_variable. The state_variable considers pressure as a variable computes the pressure drop along the reactor."
    )
    permeate_pressure_mode: Optional[Literal["constant", "state_variable"]] = Field(
        default="constant",
        description="Pressure mode as constant and state_variable. The state_variable considers pressure as a variable computes the pressure drop along the reactor."
    )

# SECTION: hollow fiber membrane module geometry


class HollowFiberMembraneModuleGeometry(BaseModel):
    """
    Geometry for hollow fiber membrane module.

    Attributes
    ----------
    number_of_fibers : CustomProp
        Number of fibers in the module.
    fiber_length : CustomProp
        Length of each fiber with relevant unit, e.g., m, cm.
    fiber_inner_diameter : CustomProp
        Inner diameter of the fibers with relevant unit, e.g., m, cm.
    fiber_outer_diameter : CustomProp
        Outer diameter of the fibers with relevant unit, e.g., m, cm.
    module_diameter : CustomProp
        Diameter of the module with relevant unit, e.g., m, cm.
    """
    number_of_fibers: CustomProp = Field(
        ...,
        description="Number of fibers in the module."
    )
    fiber_length: CustomProp = Field(
        ...,
        description="Length of each fiber with relevant unit, e.g., m, cm"
    )
    fiber_inner_diameter: CustomProp = Field(
        ...,
        description="Inner diameter of the fibers with relevant unit, e.g., m, cm"
    )
    fiber_outer_diameter: CustomProp = Field(
        ...,
        description="Outer diameter of the fibers with relevant unit, e.g., m, cm"
    )
    module_diameter: CustomProp = Field(
        ...,
        description="Diameter of the module with relevant unit, e.g., m, cm"
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_geometry(self):
        if self.fiber_outer_diameter.unit == self.fiber_inner_diameter.unit:
            if self.fiber_outer_diameter.value <= self.fiber_inner_diameter.value:
                raise ValueError(
                    "fiber_outer_diameter must be larger than fiber_inner_diameter."
                )
        return self
