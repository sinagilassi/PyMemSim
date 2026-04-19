# import libs
from pydantic import Field
from typing import Literal, Optional
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
