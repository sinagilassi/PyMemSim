# import packages/modules
import logging
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
from pyThermoLinkDB import load_and_build_model_source
from pyThermoLinkDB.models import ComponentModelSource, ModelSource
from pythermodb_settings.models import Component, Pressure, Temperature, CustomProp, Volume, CustomProperty
from pyThermoDB import ComponentThermoDB
from pyThermoDB import build_component_thermodb_from_reference
from pyreactlab_core.models.reaction import Reaction
from pythermodb_settings.models import (
    Component,
    ComponentRule,
    ComponentThermoDBSource,
)

# NOTE: setup logger
logger = logging.getLogger(__name__)

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
# components = [CO2, H2, CH3OH, H2O]

# =======================================
# SECTION: 🌍 LOAD THERMODB
# =======================================
# NOTE: thermodb configurations
# thermodb file
CO2_thermodb_file = os.path.join(
    thermodb_dir,
    'carbon dioxide.pkl'
)
H2_thermodb_file = os.path.join(
    thermodb_dir,
    'hydrogen.pkl'
)
CH3OH_thermodb_file = os.path.join(
    thermodb_dir,
    'methanol.pkl'
)
H2O_thermodb_file = os.path.join(
    thermodb_dir,
    'water.pkl'
)
CO_thermodb_file = os.path.join(
    thermodb_dir,
    'carbon monoxide.pkl'
)

# =======================================
# SECTION: create thermodb source
# ======================================
# NOTE: component thermodb
CO2_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=CO2,
    source=CO2_thermodb_file
)

H2_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=H2,
    source=H2_thermodb_file
)

CH3OH_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=CH3OH,
    source=CH3OH_thermodb_file
)

H2O_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=H2O,
    source=H2O_thermodb_file
)

CO_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=CO,
    source=CO_thermodb_file
)

# NOTE: load and build model source
# NOTE: debug timing for model source build
_build_t0 = time.perf_counter()
# ! with rules
# model_source: ModelSource = load_and_build_model_source(
#     thermodb_sources=[
#         CO2_thermodb,
#         ethanol_thermodb
#     ],
#     rules=thermodb_rules,
#     original_equation_label=False
# )
# print(model_source)

# # ! without rules & original labels is True
# model_source: ModelSource = load_and_build_model_source(
#     thermodb_sources=[
#         CO2_thermodb,
#         ethanol_thermodb
#     ],
#     original_equation_label=True
# )
# print(model_source)

# ! without rules & original labels is False
model_source: ModelSource = load_and_build_model_source(
    thermodb_sources=[
        CO2_thermodb,
        H2_thermodb,
        CH3OH_thermodb,
        H2O_thermodb,
        CO_thermodb
    ],
    original_equation_label=False
)
_build_t1 = time.perf_counter()
print(
    f"[bold cyan]Timing[/bold cyan] load_and_build_model_source: "
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
