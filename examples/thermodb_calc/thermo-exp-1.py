# import packages/modules
import logging
import warnings
from rich import print
import pyThermoDB as ptdb
import pyThermoLinkDB as ptdblink
from pythermodb_settings.models import Pressure, Temperature, CustomProp, Volume
from pythermodb_settings.utils import set_component_id
# locals
from pyreactsim.models import BatchReactorOptions, HeatTransferOptions
from pyreactsim.thermo import build_thermo_source
from pyreactsim import create_batch_reactor, BatchReactor
from pyThermoCalcDB.docs.thermo import calc_En_IG_ref
from pyThermoCalcDB.reactions.source import dH_rxn_STD
from pyThermoCalcDB.models import ComponentEnthalpy
from pyreactlab_core.models.reaction import Reaction

# NOTE: Batch reactor settings
# ! model sources for liquid phase batch reactor
from examples.source.liquid_model_source_exp_1 import model_source, components

# SECTION: Calculate liquid phase enthalpy for the components at reference temperature (e.g., 298 K)
# ! specify temperature for enthalpy calculation
temperature = Temperature(
    value=398.15,
    unit="K",
)

En_LIQ_comp = {}

# iterate over components
for component in components:
    # component ID
    comp_id = set_component_id(component, component_key='Formula-State')
    # >> calculate liquid phase enthalpy for the component at the specified temperature
    En_LIQ_res: ComponentEnthalpy | None = calc_En_IG_ref(
        component=component,
        model_source=model_source,
        temperature=temperature,
    )
    En_LIQ_comp[comp_id] = En_LIQ_res

# log
print("Liquid phase enthalpy for the components at reference temperature:")
for comp_id, En_LIQ in En_LIQ_comp.items():
    print(f"{comp_id}: {En_LIQ.value} {En_LIQ.unit}")

# NOTE: Reaction
reaction = Reaction(
    name="reaction 1",
    reaction="CH3COOH(l) + CH3OH(l) <=> C3H6O2(l) + H2O(l)",
    components=components
)

# SECTION: Calculate standard enthalpy change of the reaction
dH_rxn = dH_rxn_STD(
    reaction=reaction,
    H_i_IG=En_LIQ_comp,
)
# log
print("Standard enthalpy change of the reaction (dH_rxn_STD):")
print(dH_rxn)
