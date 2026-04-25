# import packages/modules
from examples.plot.plot_res import plot_hfm_result
from examples.source.liquid_model_source_exp_1 import components, model_source
from pymemsim.thermo import build_thermo_source
from pymemsim.models import HeatTransferOptions, HollowFiberMembraneOptions, MembraneResult
from pymemsim import HFM, create_hfm_module
import logging
import sys
import warnings
from pathlib import Path

from rich import print
from pythermodb_settings.models import CustomProp, Temperature

# NOTE: example source and kinetics
# ! add project root and examples root to import path for standalone script execution
PROJECT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# locals


# NOTE: silence library warnings/errors for this example run
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pythermocalcdb", "pymemsim", "pyreactlab_core"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)


# ====================================================
# SECTION: Inputs
# ====================================================

# NOTE: membrane unit options
unit_options = HollowFiberMembraneOptions(
    modeling_type="scale",
    phase="liquid",
    feed_pressure_mode="constant",
    permeate_pressure_mode="constant",
    liquid_heat_capacity_mode="temperature-dependent",
    liquid_density_mode="temperature-dependent",
)

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
feed_inlet_flows = {
    "CH3OH-l": CustomProp(value=0.10, unit="mol/s"),
    "H2O-l": CustomProp(value=0.00, unit="mol/s"),
    "CH3COOH-l": CustomProp(value=0.10, unit="mol/s"),
    "C3H6O2-l": CustomProp(value=0.00, unit="mol/s"),
    "H2-l": CustomProp(value=0.00, unit="mol/s"),
    "C2H5OH-l": CustomProp(value=0.00, unit="mol/s"),
}

permeate_inlet_flows = {
    "CH3OH-l": CustomProp(value=0.00, unit="mol/s"),
    "H2O-l": CustomProp(value=0.00, unit="mol/s"),
    "CH3COOH-l": CustomProp(value=0.00, unit="mol/s"),
    "C3H6O2-l": CustomProp(value=0.00, unit="mol/s"),
    "H2-l": CustomProp(value=0.00, unit="mol/s"),
    "C2H5OH-l": CustomProp(value=0.00, unit="mol/s"),
}

# NOTE: liquid transport coefficients k_i [placeholder units expected by model]
liquid_transport_coefficients = {
    "CH3OH-l": CustomProp(value=2.0e-6, unit=""),
    "H2O-l": CustomProp(value=1.0e-6, unit=""),
    "CH3COOH-l": CustomProp(value=1.0e-6, unit=""),
    "C3H6O2-l": CustomProp(value=5.0e-7, unit=""),
    "H2-l": CustomProp(value=2.0e-6, unit=""),
    "C2H5OH-l": CustomProp(value=1.0e-6, unit=""),
}

model_inputs = {
    # NOTE: dual-side inlet specs
    "feed_inlet_flows": feed_inlet_flows,
    "permeate_inlet_flows": permeate_inlet_flows,
    "feed_inlet_temperature": Temperature(value=330.0, unit="K"),
    "permeate_inlet_temperature": Temperature(value=300.0, unit="K"),
    "feed_pressure": CustomProp(value=2.0, unit="bar"),
    "permeate_pressure": CustomProp(value=1.0, unit="bar"),
    # NOTE: membrane parameters
    "membrane_area_per_length": CustomProp(value=10.0, unit="m"),
    "overall_heat_transfer_coefficient": CustomProp(value=20.0, unit="W/m2.K"),
    "q_ext_feed": CustomProp(value=0.0, unit="W/m2"),
    "q_ext_permeate": CustomProp(value=0.0, unit="W/m2"),
    "liquid_transport_coefficients": liquid_transport_coefficients,
}


# ====================================================
# SECTION: Build Thermo Source
# ====================================================
thermo_source = build_thermo_source(
    components=components,
    model_source=model_source,
    thermo_inputs=thermo_inputs,
    unit_options=unit_options,
    heat_transfer_options=heat_transfer_options,
    reaction_rates=[],
    component_key="Name-Formula",
)
print("[bold green]Thermo source successfully built![/bold green]")


# ====================================================
# SECTION: Create HFM Module
# ====================================================
hfm_module: HFM = create_hfm_module(
    model_inputs=model_inputs,
    thermo_source=thermo_source,
)
print("[bold green]HFM module successfully created![/bold green]")


# ====================================================
# SECTION: Simulate
# ====================================================
length_span = (0.0, 1.0)  # [m]

simulation_results: MembraneResult | None = hfm_module.simulate(
    length_span=length_span,
    solver_options={
        "method": "Radau",
        "rtol": 1e-6,
        "atol": 1e-9,
    },
    mode="log",
)
print("[bold green]HFM simulation completed![/bold green]")

if simulation_results is not None:
    print("success:", simulation_results.success)
    print("message:", simulation_results.message)
    print("span points:", len(simulation_results.span))
    print("state shape:", simulation_results.state.shape)
    plot_hfm_result(
        result=simulation_results,
        components=components,
        show=True,
        title_prefix="Liquid HFM",
    )
