# import packages/modules
import os
import time
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
thermodb_dir = str(Path(__file__).parent.parent / 'thermodb')
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

# Carbon monoxide
CO = Component(
    name='carbon monoxide',
    formula='CO',
    state='g',
)

# ethylene
C2H4 = Component(
    name='ethylene',
    formula='C2H4',
    state='g',
)

# ethane
C2H6 = Component(
    name='ethane',
    formula='C2H6',
    state='g',
)

# components
components = [CO, H2, CH3OH, H2O, CO2]

# NOTE: ignore state properties
ignore_state_props = ['MW', 'VaPr', 'Cp_IG']

# ====================================================
# SECTION: build components thermodb
# ====================================================
thermodb_components: List[ComponentThermoDB] = []

_thermodb_t0 = time.perf_counter()
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
_thermodb_t1 = time.perf_counter()
print(
    f"[bold cyan]Timing[/bold cyan] build_component_thermodb_from_reference (total): "
    f"{(_thermodb_t1 - _thermodb_t0) * 1000.0:.2f} ms"
)

# ====================================================
# SECTION: build model source
# ====================================================
# NOTE: with partially matched rules
_build_t0 = time.perf_counter()
component_model_source: List[ComponentModelSource] = build_components_model_source(
    components_thermodb=thermodb_components,
    rules=None,
)

# model source
model_source: ModelSource = build_model_source(
    source=component_model_source,
)
_build_t1 = time.perf_counter()
print(
    f"[bold cyan]Timing[/bold cyan] build_components_model_source + build_model_source: "
    f"{(_build_t1 - _build_t0) * 1000.0:.2f} ms"
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
_wrap_t0 = time.perf_counter()
model_source: ModelSource = ModelSource(
    data_source=datasource,
    equation_source=equationsource
)
_wrap_t1 = time.perf_counter()
print(
    f"[bold cyan]Timing[/bold cyan] ModelSource wrapper build: "
    f"{(_wrap_t1 - _wrap_t0) * 1000.0:.2f} ms"
)
