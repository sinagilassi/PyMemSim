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
    basis: str = "flow",
) -> tuple[Figure, np.ndarray]:
    """
    Plot membrane/HFM results from MembraneResult.

    Supported state layouts
    -----------------------
    - Isothermal dual-side: [Ff_i..., Fp_i...]            => (2*ns, n_points)
    - Non-isothermal dual-side: [Ff_i..., Fp_i..., Tf,Tp] => (2*ns+2, n_points)
    """
    if basis not in ("flow", "mole_fraction"):
        raise ValueError("basis must be 'flow' or 'mole_fraction'.")

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

    if basis == "mole_fraction":
        def _safe_comp(flows: np.ndarray) -> np.ndarray:
            totals = np.sum(flows, axis=0)
            # NOTE: mask mole fractions where total flow is too small to be
            # physically interpretable; prevents floor-driven flat trends.
            tol = max(1e-30, 1e-10 * float(np.max(np.abs(totals))))
            safe = np.where(np.abs(totals) > tol, totals, 1.0)
            y = flows / safe
            y[:, np.abs(totals) <= tol] = np.nan
            return y

        feed_plot = _safe_comp(Ff)
        perm_plot = _safe_comp(Fp)
        y_label = "Mole Fraction [-]"
        feed_title = f"{title_prefix}: Feed-Side Mole Fractions"
        perm_title = f"{title_prefix}: Permeate-Side Mole Fractions"
    else:
        feed_plot = Ff
        perm_plot = Fp
        y_label = "Flow [mol/s]"
        feed_title = f"{title_prefix}: Feed-Side Flows"
        perm_title = f"{title_prefix}: Permeate-Side Flows"

    nrows = 2 if dual_side_isothermal else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.5 * nrows), sharex=True)
    axes = np.atleast_1d(axes)

    for i in range(ns):
        axes[0].plot(span, feed_plot[i, :], label=labels[i])
    axes[0].set_ylabel(y_label)
    axes[0].set_title(feed_title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    for i in range(ns):
        axes[1].plot(span, perm_plot[i, :], label=labels[i])
    axes[1].set_ylabel(y_label)
    axes[1].set_title(perm_title)
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
    basis: str = "flow",
) -> tuple[Figure, np.ndarray]:
    return plot_hfm_result(
        result=result,
        components=components,
        show=show,
        title_prefix=title_prefix,
        basis=basis,
    )


def plot_hfm_permeate_flow_profile(
    result: MembraneResult,
    components: Sequence[Any],
    show: bool = True,
    title: str = "HFM: Permeate Flow Profile",
) -> tuple[Figure, plt.Axes]:
    """
    Plot permeate-side component flows and total permeate flow vs axial coordinate.
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
    if state.shape[0] < 2 * ns:
        raise ValueError("Unsupported state shape for permeate-flow plotting.")

    labels = _component_labels(components)
    Fp = np.asarray(state[ns:2 * ns, :], dtype=float)
    Fp_total = np.sum(Fp, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    for i in range(ns):
        ax.plot(span, Fp[i, :], label=f"{labels[i]} permeate", linewidth=1.8)
    ax.plot(span, Fp_total, label="Permeate total", linewidth=2.2, color="black", linestyle="--")

    ax.set_xlabel("Axial Coordinate z")
    ax.set_ylabel("Permeate Flow [mol/s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def plot_hfm_flow_pattern_comparison(
    cocurrent_result: MembraneResult,
    countercurrent_result: MembraneResult,
    components: Sequence[Any],
    show: bool = True,
    title_prefix: str = "HFM Comparison",
    basis: str = "mole_fraction",
) -> tuple[Figure, np.ndarray]:
    """
    Plot co-current and counter-current HFM results on shared axes.

    Notes
    -----
    Both results are plotted against their native span coordinate.
    """
    if basis not in ("mole_fraction", "flow"):
        raise ValueError("basis must be 'mole_fraction' or 'flow'.")

    co_span = np.asarray(cocurrent_result.span, dtype=float)
    cc_span = np.asarray(countercurrent_result.span, dtype=float)
    co_state = np.asarray(cocurrent_result.state, dtype=float)
    cc_state = np.asarray(countercurrent_result.state, dtype=float)

    if co_state.ndim != 2 or cc_state.ndim != 2:
        raise ValueError("Both result.state values must be 2D with shape (n_states, n_points).")
    if co_span.shape[0] != co_state.shape[1]:
        raise ValueError("Co-current span/state mismatch.")
    if cc_span.shape[0] != cc_state.shape[1]:
        raise ValueError("Counter-current span/state mismatch.")

    ns = len(components)
    labels = _component_labels(components)

    co_non_iso = co_state.shape[0] == 2 * ns + 2
    cc_non_iso = cc_state.shape[0] == 2 * ns + 2
    if co_state.shape[0] not in (2 * ns, 2 * ns + 2):
        raise ValueError("Unsupported co-current state layout.")
    if cc_state.shape[0] not in (2 * ns, 2 * ns + 2):
        raise ValueError("Unsupported counter-current state layout.")

    nrows = 3 if (co_non_iso or cc_non_iso) else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes)

    co_ff = co_state[:ns, :]
    co_fp = co_state[ns:2 * ns, :]
    cc_ff = cc_state[:ns, :]
    cc_fp = cc_state[ns:2 * ns, :]

    if basis == "mole_fraction":
        def _safe_comp(flows: np.ndarray) -> np.ndarray:
            totals = np.sum(flows, axis=0)
            safe = np.where(np.abs(totals) > 1e-30, totals, 1.0)
            y = flows / safe
            y[:, np.abs(totals) <= 1e-30] = 0.0
            return y

        co_ff_plot = _safe_comp(co_ff)
        co_fp_plot = _safe_comp(co_fp)
        cc_ff_plot = _safe_comp(cc_ff)
        cc_fp_plot = _safe_comp(cc_fp)
        y_label = "Mole Fraction [-]"
        feed_title = f"{title_prefix}: Feed-Side Mole Fractions"
        perm_title = f"{title_prefix}: Permeate-Side Mole Fractions"
    else:
        co_ff_plot = co_ff
        co_fp_plot = co_fp
        cc_ff_plot = cc_ff
        cc_fp_plot = cc_fp
        y_label = "Flow [mol/s]"
        feed_title = f"{title_prefix}: Feed-Side Flows"
        perm_title = f"{title_prefix}: Permeate-Side Flows"

    for i in range(ns):
        axes[0].plot(co_span, co_ff_plot[i, :], label=f"{labels[i]} (co)", linestyle="-")
        axes[0].plot(cc_span, cc_ff_plot[i, :], label=f"{labels[i]} (counter)", linestyle="--")
    axes[0].set_ylabel(y_label)
    axes[0].set_title(feed_title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8, ncol=2)

    for i in range(ns):
        axes[1].plot(co_span, co_fp_plot[i, :], label=f"{labels[i]} (co)", linestyle="-")
        axes[1].plot(cc_span, cc_fp_plot[i, :], label=f"{labels[i]} (counter)", linestyle="--")
    axes[1].set_ylabel(y_label)
    axes[1].set_title(perm_title)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8, ncol=2)

    if nrows == 3:
        if co_non_iso:
            axes[2].plot(co_span, co_state[2 * ns, :], label="T_feed (co)", linestyle="-")
            axes[2].plot(co_span, co_state[2 * ns + 1, :], label="T_perm (co)", linestyle="-")
        if cc_non_iso:
            axes[2].plot(cc_span, cc_state[2 * ns, :], label="T_feed (counter)", linestyle="--")
            axes[2].plot(cc_span, cc_state[2 * ns + 1, :], label="T_perm (counter)", linestyle="--")
        axes[2].set_ylabel("Temperature [K]")
        axes[2].set_title(f"{title_prefix}: Temperature Profiles")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Axial Coordinate z")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
