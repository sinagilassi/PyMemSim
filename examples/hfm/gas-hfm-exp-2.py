# import packages/modules
import logging
import sys
import warnings
from pathlib import Path
from typing import cast, Literal
from pythermodb_settings.models import CustomProp, Temperature
from rich import print
# ! locals
from examples.source.gas_load_model_source import model_source, CO2, CH4
from examples.plot.plot_res import plot_hfm_result, plot_hfm_permeate_flow_profile
from pymemsim.thermo import build_thermo_source
from pymemsim.models import HeatTransferOptions, HollowFiberMembraneOptions, MembraneResult
from pymemsim import HFM, create_hfm_module
from pymemsim.utils import analyze_hfm_result, print_hfm_result_tables
from pymemsim.utils import Q_std_to_mol_s, to_m3_per_s

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
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pyThermoCalcDB", "pyreactlab_core"):
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
# volumetric flow rate
feed_volumetric_flow = CustomProp(value=1e-4, unit="m3/min")
# convert to molar flow rate at standard conditions using ideal gas law
feed_molar_flow = Q_std_to_mol_s(feed_volumetric_flow)
print(f"feed molar flow: {feed_molar_flow.value:.4e} {feed_molar_flow.unit}")

# feed specification mode: total molar flow + feed mole fractions
feed_inlet_flow = CustomProp(value=feed_molar_flow.value, unit="mol/s")
feed_mole_fractions = {
    "CO2-g": CustomProp(value=0.6, unit=""),
    "CH4-g": CustomProp(value=0.4, unit=""),
}

# permeate inlet flow (can be set to zero or a small value to avoid numerical issues with zero flow)
# permeate_inlet_flows = {
#     "CO2-g": CustomProp(value=0.000001, unit="mol/s"),
#     "CH4-g": CustomProp(value=0.000001, unit="mol/s"),
# }

# NOTE: gas transport coefficients Pi_i (Permeance) for each component i, in units of mol/s.m2.Pa
gas_transport_coefficients = {
    "CO2-g": CustomProp(value=31.60*3.35e-10, unit="mol/s.m2.Pa"),
    "CH4-g": CustomProp(value=8.81*3.35e-10, unit="mol/s.m2.Pa"),
}

model_inputs = {
    # NOTE: dual-side inlet specs
    # ! feed
    "feed_inlet_flow": feed_inlet_flow,
    "feed_mole_fractions": feed_mole_fractions,
    "feed_inlet_temperature": Temperature(value=338.15, unit="K"),
    "feed_pressure": CustomProp(value=405, unit="kPa"),
    # ! permeate
    "permeate_inlet_temperature": Temperature(value=338.15, unit="K"),
    "permeate_pressure": CustomProp(value=101, unit="kPa"),
    # NOTE: membrane parameters
    "membrane_area_per_length": CustomProp(value=0.231, unit="m2/m"),
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
            "mesh_points": 120,
            "tol": 1e-2,
            "bc_tol": 1e-2,
            "max_nodes": 50000,
            "verbose": 2,
            "debug_bc": True,
        }
    print(
        f"[bold yellow]solver options ({flow_pattern}):[/bold yellow] {solver_options}")

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
length_span = (0.0, 0.15)  # [m]
flow_pattern_to_run = "counter-current"

print("[bold green]Running gas HFM example for both flow patterns...[/bold green]")
res_case = run_case(flow_pattern=flow_pattern_to_run, length_span=length_span)

if res_case is not None:
    plot_hfm_result(
        result=res_case,
        components=components,
        show=True,
        title_prefix=f"Gas HFM {flow_pattern_to_run}",
        basis="flow",
    )
    # plot_hfm_permeate_flow_profile(
    #     result=res_case,
    #     components=components,
    #     show=True,
    #     title=f"Gas HFM {flow_pattern_to_run}: Permeate Flow Profile",
    # )
