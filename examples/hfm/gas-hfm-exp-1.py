# import packages/modules
import logging
import sys
import warnings
from pathlib import Path
from typing import cast, Literal
from pythermodb_settings.models import CustomProp, Temperature
from rich import print
# ! locals
from examples.source.gas_load_model_source import model_source, CO2, N2
from pymemsim.thermo import build_thermo_source
from pymemsim.models import HeatTransferOptions, HollowFiberMembraneOptions, MembraneResult
from pymemsim import HFM, create_hfm_module
from pymemsim.utils import analyze_hfm_result, print_hfm_result_tables

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
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pyThermoCalcDB", "pymemsim", "pyreactlab_core"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)


# ====================================================
# SECTION: Define components
# ====================================================
components = [CO2, N2]

# ====================================================
# SECTION: Inputs
# ====================================================

# NOTE: heat-transfer options
heat_transfer_options = HeatTransferOptions(
    heat_transfer_mode="isothermal",
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
    "CO2-g": CustomProp(value=0.00067, unit="mol/s"),
    "N2-g": CustomProp(value=0.001, unit="mol/s"),
}

permeate_inlet_flows = {
    "CO2-g": CustomProp(value=0.00, unit="mol/s"),
    "N2-g": CustomProp(value=0.00, unit="mol/s"),
}

# NOTE: gas transport coefficients Pi_i (Permeance) for each component i, in units of mol/s.m2.Pa
gas_transport_coefficients = {
    "CO2-g": CustomProp(value=63.6*3.35e-10, unit="mol/s.m2.Pa"),
    "N2-g": CustomProp(value=3.05*3.35e-10, unit="mol/s.m2.Pa"),
}

model_inputs = {
    # NOTE: dual-side inlet specs
    "feed_inlet_flows": feed_inlet_flows,
    "permeate_inlet_flows": permeate_inlet_flows,
    "feed_inlet_temperature": Temperature(value=298.0, unit="K"),
    "permeate_inlet_temperature": Temperature(value=298.0, unit="K"),
    "feed_pressure": CustomProp(value=404, unit="kPa"),
    "permeate_pressure": CustomProp(value=101, unit="kPa"),
    # NOTE: membrane parameters
    "membrane_area_per_length": CustomProp(value=0.05058, unit="m2/m"),
    "overall_heat_transfer_coefficient": CustomProp(value=20.0, unit="W/m2.K"),
    "q_ext_feed": CustomProp(value=0.0, unit="W/m2"),
    "q_ext_permeate": CustomProp(value=0.0, unit="W/m2"),
    "gas_transport_coefficients": gas_transport_coefficients,
}


def run_case(flow_pattern: str, length_span: tuple[float, float]) -> MembraneResult | None:
    # NOTE: membrane unit options per flow pattern
    unit_options = HollowFiberMembraneOptions(
        modeling_type="scale",
        phase="gas",
        feed_pressure_mode="constant",
        permeate_pressure_mode="constant",
        gas_model="ideal",
        flow_pattern=cast(
            Literal["co-current", "counter-current"], flow_pattern),
    )

    # NOTE: build thermo source
    thermo_source = build_thermo_source(
        components=components,
        model_source=model_source,
        thermo_inputs=thermo_inputs,
        unit_options=unit_options,
        heat_transfer_options=heat_transfer_options,
        reaction_rates=[],
        component_key="Name-Formula",
    )

    # NOTE: create module
    hfm_module: HFM = create_hfm_module(
        model_inputs=model_inputs,
        thermo_source=thermo_source,
    )

    # NOTE: route solver options by flow pattern
    if flow_pattern == "co-current":
        solver_options = {
            "method": "Radau",
            "rtol": 1e-6,
            "atol": 1e-9,
        }
    else:
        solver_options = {
            "mesh_points": 50,
            "tol": 1e-5,
            "max_nodes": 5000,
            "verbose": 0,
        }

    simulation_results: MembraneResult | None = hfm_module.simulate(
        length_span=length_span,
        solver_options=solver_options,
        mode="log",
    )

    print(f"\n[bold cyan]Flow pattern: {flow_pattern}[/bold cyan]")
    if simulation_results is None:
        print("[bold red]Simulation failed (returned None).[/bold red]")
        return None

    print("success:", simulation_results.success)
    print("message:", simulation_results.message)
    print("span points:", len(simulation_results.span))
    print("state shape:", simulation_results.state.shape)

    analysis = analyze_hfm_result(
        result=simulation_results,
        hfm_module=hfm_module,
        target_component="CO2-g",
    )
    print_hfm_result_tables(analysis)

    return simulation_results


# SETUP: run cases
length_span = (0.0, 0.63)  # [m]

print("[bold green]Running gas HFM example for both flow patterns...[/bold green]")
# run_case(flow_pattern="co-current", length_span=length_span)
run_case(flow_pattern="counter-current", length_span=length_span)
