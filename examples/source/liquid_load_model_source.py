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

# =======================================
# SECTION: 🌍 LOAD THERMODB
# =======================================
# NOTE: thermodb configurations
# thermodb file
CH3OH_thermodb_file = os.path.join(thermodb_dir, 'methanol.pkl')
H2O_thermodb_file = os.path.join(thermodb_dir, 'water.pkl')
CH3COOH_thermodb_file = os.path.join(thermodb_dir, 'acetic acid.pkl')
C3H6O2_thermodb_file = os.path.join(thermodb_dir, 'methyl acetate.pkl')
H2_thermodb_file = os.path.join(thermodb_dir, 'hydrogen.pkl')
C2H5OH_thermodb_file = os.path.join(thermodb_dir, 'ethanol.pkl')

# =======================================
# SECTION: create thermodb source
# ======================================
# NOTE: component thermodb
CH3OH_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=CH3OH,
    source=CH3OH_thermodb_file
)

H2O_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=H2O,
    source=H2O_thermodb_file
)

CH3COOH_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=CH3COOH,
    source=CH3COOH_thermodb_file
)

C3H6O2_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=C3H6O2,
    source=C3H6O2_thermodb_file
)

H2_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=H2,
    source=H2_thermodb_file
)

C2H5OH_thermodb: ComponentThermoDBSource = ComponentThermoDBSource(
    component=C2H5OH,
    source=C2H5OH_thermodb_file
)

# ====================================================
# SECTION: build model source
# ====================================================
# ! without rules & original labels is False
model_source: ModelSource = load_and_build_model_source(
    thermodb_sources=[
        CH3OH_thermodb,
        H2O_thermodb,
        CH3COOH_thermodb,
        C3H6O2_thermodb,
        H2_thermodb,
        C2H5OH_thermodb,
    ],
    original_equation_label=False
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
