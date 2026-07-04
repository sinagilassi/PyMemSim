# import packages/modules
import logging
import sys
import warnings
from pathlib import Path
from pythermodb_settings.models import CustomProp, Temperature
from rich import print
# ! locals
from examples.source.gas_load_model_source import CO2, CH4, model_source
from pymemsim.models import HeatTransferOptions, HollowFiberMembraneModuleGeometry


# NOTE: example source and kinetics
# ! add project root and examples root to import path for standalone script execution
PROJECT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# NOTE: silence library warnings/errors for this example run
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
SUPPRESS_PYMEMSIM_LOGS = False
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pythermocalcdb", "pyreactlab_core"):
    if logger_name == "pymemsim" and not SUPPRESS_PYMEMSIM_LOGS:
        logging.getLogger(logger_name).setLevel(logging.INFO)
        continue
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)


# ====================================================
# SECTION: Define components
# ====================================================
components = [CO2, CH4]

# ====================================================
# SECTION: Inputs
# ====================================================

# NOTE: heat-transfer options
heat_transfer_options = HeatTransferOptions(
    heat_transfer_mode="non-isothermal",
    heat_transfer_coefficient=CustomProp(value=100.0, unit="W/m2.K"),
    heat_transfer_area=CustomProp(value=2.0, unit="m2"),
    jacket_temperature=Temperature(value=330.0, unit="K"),
)

# NOTE: optional thermo inputs
thermo_inputs = {}

# ====================================================
# SECTION: Model Inputs
# ====================================================
# ! method 1
# volumetric flow rate
# feed_volumetric_flow = CustomProp(value=2.5e-4, unit="m3/min")
# convert to molar flow rate at standard conditions using ideal gas law
# feed_molar_flow = Q_std_to_mol_s(feed_volumetric_flow)
# print(f"feed molar flow: {feed_molar_flow.value:.4e} {feed_molar_flow.unit}")
# feed_inlet_flow = CustomProp(value=feed_molar_flow.value, unit="mol/s")

# ! method 2 (alternative): directly specify feed molar flow rate
feed_inlet_flow = CustomProp(value=0.002500, unit="mol/s")

# feed specification mode: feed mole fractions
feed_mole_fractions = {
    "CO2-g": CustomProp(value=0.6, unit=""),
    "CH4-g": CustomProp(value=0.4, unit=""),
}

# feed inlet temperature
feed_inlet_temperature = Temperature(value=298.15, unit="K")

# feed inlet pressure
feed_pressure = CustomProp(value=405, unit="kPa")

# permeate inlet temperature
permeate_inlet_temperature = Temperature(value=298.15, unit="K")

# permeate inlet pressure
permeate_pressure = CustomProp(value=101, unit="kPa")

# NOTE: gas transport coefficients Pi_i (Permeance) for each component i
# ! gpu
gas_transport_coefficients = {
    "CO2-g": CustomProp(value=9.43, unit="GPU"),
    "CH4-g": CustomProp(value=2.63, unit="GPU"),
}

# NOTE: membrane unit geometry
module_geometry = HollowFiberMembraneModuleGeometry(
    number_of_fibers=CustomProp(value=100, unit=""),
    fiber_length=CustomProp(value=15, unit="cm"),
    fiber_inner_diameter=CustomProp(value=0.0389, unit="cm"),
    fiber_outer_diameter=CustomProp(value=0.0735, unit="cm"),
    module_diameter=CustomProp(value=1, unit="cm"),
)

# NOTE: overall heat transfer coefficient (U) for the module, in units of W/m2.K
overall_heat_transfer_coefficient = CustomProp(value=20.0, unit="W/m2.K")

# NOTE: external heat
q_ext_feed = CustomProp(value=0.0, unit="W/m2")
q_ext_permeate = CustomProp(value=0.0, unit="W/m2")

# NOTE: model inputs
model_inputs = {
    # ! dual-side inlet specs
    # ! feed
    "feed_inlet_flow": feed_inlet_flow,
    "feed_mole_fractions": feed_mole_fractions,
    "feed_inlet_temperature": feed_inlet_temperature,
    "feed_pressure": feed_pressure,
    # ! permeate
    "permeate_inlet_temperature": permeate_inlet_temperature,
    "permeate_pressure": permeate_pressure,
    # ! membrane module geometry inputs
    "module_geometry": module_geometry,
    # ! heat transfer parameters
    "overall_heat_transfer_coefficient": overall_heat_transfer_coefficient,
    "q_ext_feed": q_ext_feed,
    "q_ext_permeate": q_ext_permeate,
    "gas_transport_coefficients": gas_transport_coefficients,
}


# NOTE: countercurrent solver method
COUNTERCURRENT_METHOD = "bvp"  # "bvp" | "shooting"


# NOTE: length span for the simulation (in meters)
length_span = (0.0, 0.15)  # [m]
flow_pattern_to_run = "co-current"
target_component = "CO2-g"

# ====================================================
# SECTION: hollow fiber membrane options
# ====================================================
modeling_type = "scale"
phase = "gas"
feed_pressure_mode = "state_variable"
permeate_pressure_mode = "state_variable"
gas_model = "ideal"
