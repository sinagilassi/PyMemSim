# import packages/modules
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import cast, Literal
from pythermodb_settings.models import CustomProp, Temperature
from rich import print
# ! locals
from examples.plot.plot_res import plot_hfm_result, plot_hfm_permeate_flow_profile
from pymemsim.thermo import build_thermo_source
from pymemsim.models import HollowFiberMembraneOptions, MembraneResult
from pymemsim import HFM, create_hfm_module
from pymemsim.utils import analyze_hfm_result, print_hfm_result_tables, save_hfm_analysis_txt
# ! inputs
from examples.validation.inputs_case_1 import (
    components,
    heat_transfer_options,
    thermo_inputs,
    model_inputs,
    model_source,
    COUNTERCURRENT_METHOD,
    length_span,
    flow_pattern_to_run
)


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


def run_case(flow_pattern: str, length_span: tuple[float, float]) -> MembraneResult | None:
    # NOTE: membrane unit options per flow pattern
    unit_options = HollowFiberMembraneOptions(
        modeling_type="scale",
        phase="gas",
        feed_pressure_mode="state_variable",
        permeate_pressure_mode="state_variable",
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
    cocurrent_solver_options = {
        "method": "Radau",
        "rtol": 1e-6,
        "atol": 1e-9,
    }
    countercurrent_bvp_solver_options = {
        "countercurrent_solver": "bvp",
        "mesh_points": 120,
        "tol": 1e-3,
        "bc_tol": 1e-3,
        "max_nodes": 50000,
        "verbose": 2,
        "debug_bc": True,
    }
    countercurrent_shooting_solver_options = {
        "countercurrent_solver": "shooting",
        "shooting_ivp_method": "auto",
        "shooting_ivp_rtol": 1e-6,
        "shooting_ivp_atol": 1e-9,
        "shooting_max_nfev": 1200,
        "shooting_ftol": 1e-8,
        "shooting_xtol": 1e-8,
        "shooting_gtol": 1e-8,
        "shooting_residual_tol": 1e-3,
        "shooting_multistart": True,
        "shooting_penalty": 1e3,
        "shooting_debug": True,
    }

    if flow_pattern == "co-current":
        solver_options = cocurrent_solver_options
    else:
        if COUNTERCURRENT_METHOD == "bvp":
            solver_options = countercurrent_bvp_solver_options
        elif COUNTERCURRENT_METHOD == "shooting":
            solver_options = countercurrent_shooting_solver_options
        else:
            raise ValueError(
                "Invalid COUNTERCURRENT_METHOD. Supported values are 'bvp' and 'shooting'."
            )
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
    print("\n[bold magenta]Analysis of results:[/bold magenta]")
    print(analysis)
    print_hfm_result_tables(analysis)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    flow_pattern_id = flow_pattern.replace(" ", "-")
    analysis_file = EXAMPLES_DIR / "results" / \
        f"gas-hfm-exp-4-{flow_pattern_id}-{run_id}.txt"
    save_hfm_analysis_txt(analysis=analysis, file_path=analysis_file)
    print(f"[bold green]Saved analysis to:[/bold green] {analysis_file}")

    return simulation_results


print("[bold green]Running gas HFM example for both flow patterns...[/bold green]")
res_case = run_case(flow_pattern=flow_pattern_to_run, length_span=length_span)

if res_case is not None:
    plot_hfm_result(
        result=res_case,
        components=components,
        show=False,
        title_prefix=f"Gas HFM {flow_pattern_to_run}",
        basis="flow",
    )
    # plot_hfm_permeate_flow_profile(
    #     result=res_case,
    #     components=components,
    #     show=True,
    #     title=f"Gas HFM {flow_pattern_to_run}: Permeate Flow Profile",
    # )
