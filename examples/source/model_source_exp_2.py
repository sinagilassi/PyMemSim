# import packages/modules
import os
from rich import print
from typing import Callable, Dict, Optional, Union, List, Any
import pyThermoDB as ptdb
import pyThermoLinkDB as ptdblink
from pyThermoLinkDB import (
    build_component_model_source,
    build_components_model_source,
    build_model_source
)
from pyThermoLinkDB.models import ComponentModelSource, ModelSource
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource
from pythermodb_settings.models import Component, Pressure, Temperature, CustomProp, Volume, CustomProperty
from pyThermoDB import ComponentThermoDB
from pyThermoDB import build_component_thermodb_from_reference
from pyreactlab_core.models.reaction import Reaction
# locals
from examples.reference import REFERENCE_CONTENT

# check version
print(ptdb.__version__)
print(ptdblink.__version__)

# ====================================================
# SECTION: BUILD COMPONENT THERMODB
# ====================================================
# NOTE: parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))
print(parent_dir)

# NOTE: thermodb directory
thermodb_dir = os.path.join(parent_dir, 'thermodb/gas')
print(thermodb_dir)

# NOTE: create component
# ! propane
# carbon dioxide
CO2 = Component(
    name='carbon dioxide',
    formula='CO2',
    state='g',
)

# Hydrogen
H2 = Component(
    name='hydrogen',
    formula='H2',
    state='g',
)

# methanol
CH3OH = Component(
    name='methanol',
    formula='CH3OH',
    state='g',
)

# water
H2O = Component(
    name='water',
    formula='H2O',
    state='g',
)

# components
components = [CO2, H2, CH3OH, H2O]

# NOTE: ignore state properties
ignore_state_props = ['MW', 'VaPr', 'Cp_IG']

# ====================================================
# SECTION: build components thermodb
# ====================================================
thermodb_components: List[ComponentThermoDB] = []

for comp in components:
    thermodb_component = build_component_thermodb_from_reference(
        component_name=comp.name,
        component_formula=comp.formula,
        component_state=comp.state,
        reference_content=REFERENCE_CONTENT,
        ignore_state_props=ignore_state_props,
    )
    if thermodb_component is None:
        raise ValueError(f"thermodb_component for {comp.name} is None")
    thermodb_components.append(thermodb_component)

# ====================================================
# SECTION: build model source
# ====================================================
# NOTE: with partially matched rules
component_model_source: List[ComponentModelSource] = build_components_model_source(
    components_thermodb=thermodb_components,
    rules=None,
)

# model source
model_source: ModelSource = build_model_source(
    source=component_model_source,
)
# ====================================================
# SECTION: THERMODB LINK CONFIGURATION
# ====================================================

# build datasource & equationsource
datasource = model_source.data_source
equationsource = model_source.equation_source

# ====================================================
# SECTION: model source
# ====================================================
model_source: ModelSource = ModelSource(
    data_source=datasource,
    equation_source=equationsource
)

# print model source
print(model_source)

# SECTION: Create source
source = Source(
    model_source=model_source,
    component_key='Name-Formula',
)
print(source)

# NOTE: Extract data & equations for a specific component
Cp_IG_src: Dict[str, ComponentEquationSource] | None = source.eq_builder(
    components=[CO2],
    prop_name='Cp_IG',
)
print(Cp_IG_src)

if Cp_IG_src is None:
    raise ValueError("Cp_IG_src is None")

CO2_Cp_IG_src: ComponentEquationSource | None = Cp_IG_src.get(
    'carbon dioxide-CO2')
print(CO2_Cp_IG_src)

# execute Cp_IG equation for carbon dioxide at 300 K
if CO2_Cp_IG_src is None:
    raise ValueError("CO2_Cp_IG_src is None")
Cp_IG_eq = CO2_Cp_IG_src.source
# calc
Cp_IG_value = Cp_IG_eq.cal(
    T=300
)
print(f"Cp_IG for carbon dioxide at 300 K: {Cp_IG_value} J/mol.K")
