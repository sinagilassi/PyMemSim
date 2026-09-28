# Counter-current liquid hollow-fiber membrane example.
import logging
import sys
import warnings
from pathlib import Path

from pythermodb_settings.models import CustomProp, Temperature
from rich import print

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.plot.plot_res import plot_hfm_result
from examples.source.liquid_model_source_exp_1 import components, model_source
from pymemsim import HFM, create_hfm_module
from pymemsim.models import (
    HeatTransferOptions,
    HollowFiberMembraneOptions,
    MembraneResult,
)
from pymemsim.thermo import build_thermo_source
from pymemsim.utils import analyze_hfm_result, print_hfm_result_tables


warnings.filterwarnings("ignore")
for logger_name in (
    "pyThermoDB",
    "pyThermoLinkDB",
    "pythermocalcdb",
    "pyreactlab_core",
):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)


# Select either "bvp" or "shooting".
COUNTERCURRENT_METHOD = "bvp"


unit_options = HollowFiberMembraneOptions(
    modeling_type="scale",
    phase="liquid",
    flow_pattern="counter-current",
    feed_pressure_mode="constant",
    permeate_pressure_mode="constant",
    liquid_heat_capacity_mode="constant",
    liquid_density_mode="constant",
)

heat_transfer_options = HeatTransferOptions(
    heat_transfer_mode="non-isothermal",
    heat_transfer_coefficient=CustomProp(value=100.0, unit="W/m2.K"),
    heat_transfer_area=CustomProp(value=2.0, unit="m2"),
    jacket_temperature=Temperature(value=330.0, unit="K"),
)

feed_inlet_flows = {
    "CH3OH-l": CustomProp(value=0.10, unit="mol/s"),
    "H2O-l": CustomProp(value=0.00, unit="mol/s"),
    "CH3COOH-l": CustomProp(value=0.10, unit="mol/s"),
    "C3H6O2-l": CustomProp(value=0.00, unit="mol/s"),
    "H2-l": CustomProp(value=0.00, unit="mol/s"),
    "C2H5OH-l": CustomProp(value=0.00, unit="mol/s"),
}

# This is the permeate inlet at z=L for counter-current operation.
permeate_inlet_flows = {
    "CH3OH-l": CustomProp(value=1.0e-6, unit="mol/s"),
    "H2O-l": CustomProp(value=1.0e-6, unit="mol/s"),
    "CH3COOH-l": CustomProp(value=1.0e-6, unit="mol/s"),
    "C3H6O2-l": CustomProp(value=1.0e-6, unit="mol/s"),
    "H2-l": CustomProp(value=1.0e-6, unit="mol/s"),
    "C2H5OH-l": CustomProp(value=1.0e-6, unit="mol/s"),
}

liquid_transport_coefficients = {
    "CH3OH-l": CustomProp(value=2.0e-9, unit="m/s"),
    "H2O-l": CustomProp(value=1.0e-9, unit="m/s"),
    "CH3COOH-l": CustomProp(value=1.0e-9, unit="m/s"),
    "C3H6O2-l": CustomProp(value=5.0e-10, unit="m/s"),
    "H2-l": CustomProp(value=2.0e-9, unit="m/s"),
    "C2H5OH-l": CustomProp(value=1.0e-9, unit="m/s"),
}

# The reference source does not provide a valid temperature-dependent liquid
# density correlation for every demonstration component at 330 K. Supply
# constant values so this example remains runnable and self-contained.
thermo_inputs = {
    "liquid_density": {
        "CH3OH-l": CustomProp(value=792.0, unit="kg/m3"),
        "H2O-l": CustomProp(value=997.0, unit="kg/m3"),
        "CH3COOH-l": CustomProp(value=1049.0, unit="kg/m3"),
        "C3H6O2-l": CustomProp(value=932.0, unit="kg/m3"),
        "H2-l": CustomProp(value=70.8, unit="kg/m3"),
        "C2H5OH-l": CustomProp(value=789.0, unit="kg/m3"),
    },
    "liquid_heat_capacity": {
        "CH3OH-l": CustomProp(value=81.1, unit="J/mol.K"),
        "H2O-l": CustomProp(value=75.3, unit="J/mol.K"),
        "CH3COOH-l": CustomProp(value=123.0, unit="J/mol.K"),
        "C3H6O2-l": CustomProp(value=140.0, unit="J/mol.K"),
        "H2-l": CustomProp(value=28.8, unit="J/mol.K"),
        "C2H5OH-l": CustomProp(value=112.4, unit="J/mol.K"),
    },
}

model_inputs = {
    "feed_inlet_flows": feed_inlet_flows,
    "permeate_inlet_flows": permeate_inlet_flows,
    "feed_inlet_temperature": Temperature(value=330.0, unit="K"),
    "permeate_inlet_temperature": Temperature(value=300.0, unit="K"),
    # Pressures are required by the common HFM input contract. Current liquid
    # counter-current operation remains a constant-pressure model.
    "feed_pressure": CustomProp(value=2.0, unit="bar"),
    "permeate_pressure": CustomProp(value=1.0, unit="bar"),
    "membrane_area_per_length": CustomProp(value=0.10, unit="m"),
    "overall_heat_transfer_coefficient": CustomProp(
        value=1.0,
        unit="W/m2.K",
    ),
    "q_ext_feed": CustomProp(value=0.0, unit="W/m2"),
    "q_ext_permeate": CustomProp(value=0.0, unit="W/m2"),
    "liquid_transport_coefficients": liquid_transport_coefficients,
}

thermo_source = build_thermo_source(
    components=components,
    model_source=model_source,
    thermo_inputs=thermo_inputs,
    unit_options=unit_options,
    heat_transfer_options=heat_transfer_options,
    reaction_rates=[],
    component_key="Name-Formula",
)

hfm_module: HFM = create_hfm_module(
    model_inputs=model_inputs,
    thermo_source=thermo_source,
)

if COUNTERCURRENT_METHOD == "bvp":
    solver_options = {
        "countercurrent_solver": "bvp",
        "mesh_points": 80,
        "tol": 1e-3,
        "bc_tol": 1e-3,
        "max_nodes": 20000,
        "debug_bc": True,
    }
elif COUNTERCURRENT_METHOD == "shooting":
    solver_options = {
        "countercurrent_solver": "shooting",
        "shooting_ivp_method": "auto",
        "shooting_ivp_rtol": 1e-6,
        "shooting_ivp_atol": 1e-9,
        "shooting_residual_tol": 1e-3,
        "shooting_multistart": True,
    }
else:
    raise ValueError(
        "COUNTERCURRENT_METHOD must be either 'bvp' or 'shooting'."
    )

length_span = (0.0, 1.0)
print(
    "[bold green]Running counter-current liquid HFM "
    f"with {COUNTERCURRENT_METHOD}...[/bold green]"
)
result: MembraneResult | None = hfm_module.simulate(
    length_span=length_span,
    solver_options=solver_options,
    mode="log",
)

if result is None:
    raise RuntimeError("Liquid counter-current simulation returned None.")

print("success:", result.success)
print("message:", result.message)
print("span points:", len(result.span))
print("state shape:", result.state.shape)

analysis = analyze_hfm_result(
    result=result,
    hfm_module=hfm_module,
    target_component="CH3OH-l",
)
print_hfm_result_tables(analysis)

plot_hfm_result(
    result=result,
    components=components,
    show=True,
    title_prefix="Liquid HFM counter-current",
    basis="flow",
)
