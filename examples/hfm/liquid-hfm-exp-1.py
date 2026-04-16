# import packages/modules
import logging
import sys
import warnings
from pathlib import Path
from rich import print
import pyThermoDB as ptdb
import pyThermoLinkDB as ptdblink
from pythermodb_settings.models import CustomProp, Temperature, Volume
# locals
from pyreactsim import PFRReactor, create_pfr_reactor
from pyreactsim.models import PFRReactorOptions, HeatTransferOptions
from pyreactsim.thermo import build_thermo_source
# NOTE: example-specific imports
# from examples.source.liquid_model_source_exp_1 import model_source
from examples.rates.esterification_acetic_acid_1 import components, reaction_rates, model_source
from examples.plot.plot_res import plot_pfr_reactor_result

# NOTE: example source and kinetics
# ! add project root and examples root to import path for standalone script execution
PROJECT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# NOTE: CSTR plotting helper
# check version
print(ptdb.__version__)
print(ptdblink.__version__)

# NOTE: silence library warnings/errors for this example run
# warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pyThermoCalcDB", "pyreactlab_core"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

# ====================================================
# SECTION: Inputs
# ====================================================
# ! Case 6: non-isothermal, constant-volume, ideal gas (gas-phase CSTR)

# NOTE: jacket temperature
jacket_temperature = Temperature(
    value=350,
    unit="K",
)

# NOTE: heat transfer coefficient
heat_transfer_coefficient = CustomProp(
    value=100.0,
    unit="W/m2.K",
)

# NOTE: heat transfer area
heat_transfer_area = CustomProp(
    value=2.0,
    unit="m2",
)

# NOTE: reactor options for thermo/source compatibility
cstr_reactor_options = PFRReactorOptions(
    modeling_type="scale",
    phase="liquid",
    operation_mode="constant_volume",
    gas_heat_capacity_mode="temperature-dependent",
    liquid_heat_capacity_mode='temperature-dependent',
    liquid_density_mode='constant',
)

# NOTE: heat transfer options
heat_transfer_options = HeatTransferOptions(
    heat_transfer_mode="non-isothermal",
    heat_transfer_coefficient=heat_transfer_coefficient,
    heat_transfer_area=heat_transfer_area,
    jacket_temperature=jacket_temperature,
)

# ====================================================
# SECTION: thermo inputs
# ====================================================
# NOTE: optional constant gas heat capacities [J/mol.K]
constant_gas_heat_capacity = {
    "CH3OH-l": CustomProp(value=75.3, unit="J/mol.K"),  # methanol
    "H2O-l": CustomProp(value=75.3, unit="J/mol.K"),   # water
    "CH3COOH-l": CustomProp(value=75.3, unit="J/mol.K"),  # acetic acid
    "C3H6O2-l": CustomProp(value=75.3, unit="J/mol.K"),  # methyl acetate
}

# NOTE: optional constant liquid heat capacities [J/mol.K]
constant_liquid_heat_capacity = {
    "CH3OH-l": CustomProp(value=81.1, unit="J/mol.K"),  # methanol
    "H2O-l": CustomProp(value=75.3, unit="J/mol.K"),   # water
    "CH3COOH-l": CustomProp(value=138.0, unit="J/mol.K"),  # acetic acid
    "C3H6O2-l": CustomProp(value=120.0, unit="J/mol.K"),  # methyl acetate
}

# NOTE: constant liquid density (rho_LIQ) for the system in kg/m3
constant_liquid_density = {
    "CH3OH-l": CustomProp(value=786.6, unit="kg/m3"),  # methanol
    "H2O-l": CustomProp(value=997.0, unit="kg/m3"),   # water
    "CH3COOH-l": CustomProp(value=1049.0, unit="kg/m3"),  # acetic acid
    "C3H6O2-l": CustomProp(value=932.0, unit="kg/m3"),  # methyl acetate
}

# ! thermo inputs
thermo_inputs = {
    "gas_heat_capacity": constant_gas_heat_capacity,
    "liquid_heat_capacity": constant_liquid_heat_capacity,
    "liquid_density": constant_liquid_density,
}

# ====================================================
# SECTION: model inputs
# ====================================================
# NOTE: fixed reactor holdup volume [m3]
reactor_volume = Volume(
    value=0.05,
    unit="m3",
)

# NOTE: pressure
pressure = CustomProp(
    value=2,
    unit="bar",
)

# NOTE: initial reactor temperature [K]
initial_temperature = Temperature(
    value=340,
    unit="K",
)

# NOTE: feed stream temperature [K]
inlet_temperature = Temperature(
    value=330,
    unit="K",
)

# NOTE: initial reactor holdup moles [mol]
initial_mole = {
    "CH3COOH-l": CustomProp(value=100.0, unit="mol"),  # acetic acid
    "CH3OH-l": CustomProp(value=100.0, unit="mol"),  # methanol
    "C3H6O2-l": CustomProp(value=0.0, unit="mol"),  # methyl acetate
    "H2O-l": CustomProp(value=0.0, unit="mol"),  # water
}

# NOTE: feed component molar flow rates [mol/s]
feed_mole_flow = {
    "CH3COOH-l": CustomProp(value=0.10, unit="mol/s"),  # acetic acid
    "CH3OH-l": CustomProp(value=0.10, unit="mol/s"),  # methanol
    "C3H6O2-l": CustomProp(value=0.0, unit="mol/s"),  # methyl acetate
    "H2O-l": CustomProp(value=0.0, unit="mol/s"),  # water
}

# NOTE: model inputs for CSTR
# ! constant volume
model_inputs = {
    "inlet_flows": feed_mole_flow,
    "reactor_volume": reactor_volume,
    "inlet_temperature": inlet_temperature,
    "inlet_pressure": pressure,
}

print("[bold green]Model inputs successfully defined![/bold green]")

# ====================================================
# SECTION: build thermo source
# ====================================================
thermo_source = build_thermo_source(
    components=components,
    model_source=model_source,
    thermo_inputs=thermo_inputs,
    reactor_options=cstr_reactor_options,
    heat_transfer_options=heat_transfer_options,
    reaction_rates=reaction_rates,
    component_key="Name-Formula",
)
print("[bold green]Thermo source successfully built![/bold green]")
print(thermo_source)

# ====================================================
# SECTION: create cstr reactor
# ====================================================
_reactor: PFRReactor = create_pfr_reactor(
    model_inputs=model_inputs,
    thermo_source=thermo_source,
)
print("[bold green]reactor successfully created![/bold green]")
print(_reactor)

# NOTE: simulate CSTR
simulation_results = _reactor.simulate(
    volume_span=(0, reactor_volume.value),
    solver_options={
        "method": "Radau",
        "rtol": 1e-6,
        "atol": 1e-9,
        # "max_step": 0.001,
    },
    mode="log"
)
print("[bold green]simulation completed![/bold green]")
# print(simulation_results)

# NOTE: plot results
if simulation_results is not None:
    plot_pfr_reactor_result(
        result=simulation_results,
        components=components,
    )
