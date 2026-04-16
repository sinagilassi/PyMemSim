# import packages/modules
import os
from pathlib import Path
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
from pythermodb_settings.models import Component, Pressure, Temperature, CustomProp, Volume, CustomProperty
from pyThermoDB import ComponentThermoDB
from pyThermoDB import build_component_thermodb_from_reference
from pyreactlab_core.models.reaction import Reaction
# locals
from examples.reference_2_liq import REFERENCE_CONTENT

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
thermodb_dir = str(Path(__file__).parent.parent / 'thermodb/liquid')
print(thermodb_dir)

# NOTE: create component
# methanol
CH3OH = Component(
    name='methanol',
    formula='CH3OH',
    state='l',
)

# water
H2O = Component(
    name='water',
    formula='H2O',
    state='l',
)

# acetic acid
CH3COOH = Component(
    name='acetic acid',
    formula='CH3COOH',
    state='l',
)

# methyl acetate
C3H6O2 = Component(
    name='methyl acetate',
    formula='C3H6O2',
    state='l',
)

# hydrogen
H2 = Component(
    name='hydrogen',
    formula='H2',
    state='l',
)

# ethanol
C2H5OH = Component(
    name='ethanol',
    formula='C2H5OH',
    state='l',
)

# components
components = [CH3OH, H2O, CH3COOH, C3H6O2, H2, C2H5OH]

# NOTE: ignore state properties
ignore_state_props = ['MW', 'VaPr', 'Cp_IG', 'Cp_LIQ', 'rho_LIQ', 'EnVap']

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
        thermodb_save=True,
        thermodb_save_path=thermodb_dir,
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
