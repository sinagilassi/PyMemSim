import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Sequence
from matplotlib.figure import Figure

from pymemsim.models import MembraneResult


def _component_labels(components: Sequence[Any]) -> list[str]:
    labels: list[str] = []
    for comp in components:
        formula = getattr(comp, "formula", None)
        state = getattr(comp, "state", None)
        name = getattr(comp, "name", None)
        if formula is not None and state is not None:
            labels.append(f"{formula}-{state}")
        elif formula is not None:
            labels.append(str(formula))
        elif name is not None:
            labels.append(str(name))
        else:
            labels.append("component")
    return labels


def plot_hfm_result(
    result: MembraneResult,
    components: Sequence[Any],
    show: bool = True,
    title_prefix: str = "HFM",
) -> tuple[Figure, np.ndarray]:
    """
    Plot membrane/HFM results from MembraneResult.

    Supported state layouts
    -----------------------
    - Isothermal dual-side: [Ff_i..., Fp_i...]            => (2*ns, n_points)
    - Non-isothermal dual-side: [Ff_i..., Fp_i..., Tf,Tp] => (2*ns+2, n_points)
    """
    span = np.asarray(result.span, dtype=float)
    state = np.asarray(result.state, dtype=float)

    if state.ndim != 2:
        raise ValueError("result.state must be 2D with shape (n_states, n_points).")
    if span.shape[0] != state.shape[1]:
        raise ValueError(
            f"span/state mismatch: len(span)={span.shape[0]}, state points={state.shape[1]}."
        )

    ns = len(components)
    labels = _component_labels(components)
    n_states = state.shape[0]

    dual_side_isothermal = n_states == 2 * ns
    dual_side_non_isothermal = n_states == 2 * ns + 2
    if not dual_side_isothermal and not dual_side_non_isothermal:
        raise ValueError(
            f"Unsupported MembraneResult state shape {state.shape}. "
            f"Expected (2*ns,n) or (2*ns+2,n) with ns={ns}."
        )

    Ff = state[:ns, :]
    Fp = state[ns:2 * ns, :]
    Tf = state[2 * ns, :] if dual_side_non_isothermal else None
    Tp = state[2 * ns + 1, :] if dual_side_non_isothermal else None

    nrows = 2 if dual_side_isothermal else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.5 * nrows), sharex=True)
    axes = np.atleast_1d(axes)

    for i in range(ns):
        axes[0].plot(span, Ff[i, :], label=labels[i])
    axes[0].set_ylabel("Feed Flow [mol/s]")
    axes[0].set_title(f"{title_prefix}: Feed-Side Flows")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    for i in range(ns):
        axes[1].plot(span, Fp[i, :], label=labels[i])
    axes[1].set_ylabel("Permeate Flow [mol/s]")
    axes[1].set_title(f"{title_prefix}: Permeate-Side Flows")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    if dual_side_non_isothermal and Tf is not None and Tp is not None:
        axes[2].plot(span, Tf, label="T_feed")
        axes[2].plot(span, Tp, label="T_permeate")
        axes[2].set_ylabel("Temperature [K]")
        axes[2].set_title(f"{title_prefix}: Temperature Profiles")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Axial Coordinate z")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, axes


def plot_membrane_result(
    result: MembraneResult,
    components: Sequence[Any],
    show: bool = True,
    title_prefix: str = "Membrane",
) -> tuple[Figure, np.ndarray]:
    return plot_hfm_result(
        result=result,
        components=components,
        show=show,
        title_prefix=title_prefix,
    )
