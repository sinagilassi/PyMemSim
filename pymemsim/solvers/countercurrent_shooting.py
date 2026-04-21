import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from ..core.gas_hfm import GasHFM
from ..core.gas_hfmx import GasHFMX
from ..models.results import MembraneResult


logger = logging.getLogger(__name__)


def solve_countercurrent_shooting(
    *,
    module: GasHFM | GasHFMX,
    rhs_point: Callable[[float, np.ndarray], np.ndarray],
    state_to_physical: Callable[[np.ndarray], np.ndarray],
    length_span: tuple[float, float],
    solver_options: Optional[Dict[str, Any]] = None,
) -> Optional[MembraneResult]:
    """
    Solve counter-current boundary conditions with a shooting method.
    """
    options = dict(solver_options or {})
    z0, z1 = float(length_span[0]), float(length_span[1])
    if z1 <= z0:
        raise ValueError("length_span must satisfy z_end > z_start.")

    ns = int(module.component_num)
    non_isothermal = module.heat_transfer_mode == "non-isothermal"

    shooting_debug = bool(options.pop("shooting_debug", False))
    shooting_penalty = float(options.pop("shooting_penalty", 1e3))
    shooting_residual_tol = float(options.pop("shooting_residual_tol", 1e-3))
    seed_mesh_points = int(options.pop("shooting_seed_mesh_points", 30))
    shooting_multistart = bool(options.pop("shooting_multistart", True))

    ivp_method_raw = str(options.pop("shooting_ivp_method", "BDF")).strip()
    ivp_rtol = float(options.pop("shooting_ivp_rtol", 1e-6))
    ivp_atol = float(options.pop("shooting_ivp_atol", 1e-9))
    ivp_max_step = options.pop("shooting_ivp_max_step", None)

    shooting_max_nfev = options.pop("shooting_max_nfev", None)
    if shooting_max_nfev is not None:
        shooting_max_nfev = int(shooting_max_nfev)
    shooting_ftol = float(options.pop("shooting_ftol", 1e-8))
    shooting_xtol = float(options.pop("shooting_xtol", 1e-8))
    shooting_gtol = float(options.pop("shooting_gtol", 1e-8))

    if isinstance(module, GasHFMX):
        y_template = np.asarray(module.build_y0_scaled(), dtype=float)
    else:
        y_template = np.asarray(module.build_y0(), dtype=float)

    z_seed = module.build_mesh(
        length_span=length_span,
        mesh_points=max(5, seed_mesh_points),
    )
    y_seed = np.asarray(module.build_initial_guess(z_seed), dtype=float)

    flow_floor = 1e-14
    s0_flow = np.maximum(y_seed[ns:2 * ns, 0], flow_floor)
    s0_parts = [s0_flow]
    if non_isothermal:
        s0_parts.append(np.array([float(y_seed[2 * ns + 1, 0])], dtype=float))
    s0 = np.concatenate(s0_parts)

    flow_ref = max(float(np.sum(np.abs(y_template[:ns]))), 1e-12)
    s_ref_flow = np.maximum(np.abs(s0[:ns]), 1e-6 * flow_ref)
    s_ref_parts = [s_ref_flow]
    if non_isothermal:
        s_ref_parts.append(np.array([max(abs(float(s0[-1])), 1.0)], dtype=float))
    s_ref = np.concatenate(s_ref_parts)

    lb_raw = np.full_like(s0, -np.inf, dtype=float)
    ub_raw = np.full_like(s0, np.inf, dtype=float)
    lb_raw[:ns] = flow_floor

    u0 = s0 / s_ref
    lb = lb_raw / s_ref
    ub = ub_raw / s_ref

    flow_res_scale = max(float(np.sum(np.abs(y_template[:ns]))), 1e-12)
    res_scales = [np.full(ns, flow_res_scale, dtype=float)]
    if non_isothermal:
        temp_scale = max(abs(float(y_template[2 * ns] - y_template[2 * ns + 1])), 1.0)
        res_scales.append(np.array([temp_scale], dtype=float))
    residual_scales = np.concatenate(res_scales)

    stats = {
        "residual_evals": 0,
        "ivp_failures": 0,
    }

    if ivp_method_raw.lower() == "auto":
        ivp_method_candidates = ["BDF", "Radau"]
    else:
        ivp_method_candidates = [ivp_method_raw]

    def build_y0_from_shooting(s_raw: np.ndarray) -> np.ndarray:
        y0 = np.array(y_template, dtype=float, copy=True)
        y0[ns:2 * ns] = np.maximum(s_raw[:ns], flow_floor)
        if non_isothermal:
            y0[2 * ns + 1] = float(s_raw[-1])
        return y0

    def extract_terminal_residual(y0: np.ndarray, yb: np.ndarray) -> np.ndarray:
        bc_full = np.asarray(module.bc(y0, yb), dtype=float)
        terminal = [bc_full[ns:2 * ns]]
        if non_isothermal:
            terminal.append(np.array([bc_full[-1]], dtype=float))
        return np.concatenate(terminal)

    def integrate_once(
        s_raw: np.ndarray,
        *,
        ivp_method: str,
        full_trajectory: bool,
    ) -> tuple[Optional[Any], np.ndarray, Optional[str]]:
        y0 = build_y0_from_shooting(s_raw)
        ivp_kwargs: Dict[str, Any] = {
            "method": ivp_method,
            "rtol": ivp_rtol,
            "atol": ivp_atol,
        }
        if ivp_max_step is not None:
            ivp_kwargs["max_step"] = float(ivp_max_step)
        if not full_trajectory:
            ivp_kwargs["t_eval"] = np.array([z1], dtype=float)

        try:
            sol = solve_ivp(
                rhs_point,
                (z0, z1),
                y0,
                **ivp_kwargs,
            )
        except Exception as exc:
            return None, y0, f"{type(exc).__name__}: {exc}"

        if not sol.success:
            return None, y0, f"IVP failed: {sol.message}"
        if sol.y.size == 0 or not np.all(np.isfinite(sol.y)):
            return None, y0, "IVP produced non-finite state."
        if abs(float(sol.t[-1]) - z1) > max(1e-12, 1e-9 * abs(z1 - z0)):
            return None, y0, "IVP did not reach terminal boundary."
        return sol, y0, None

    def residual_func(u: np.ndarray, *, ivp_method: str) -> np.ndarray:
        stats["residual_evals"] += 1
        s_raw = np.asarray(u, dtype=float) * s_ref
        sol, y0, err = integrate_once(
            s_raw,
            ivp_method=ivp_method,
            full_trajectory=False,
        )
        if err is not None or sol is None:
            stats["ivp_failures"] += 1
            if shooting_debug:
                logger.debug("Counter-current shooting IVP failure: %s", err)
            return np.full_like(u, shooting_penalty, dtype=float)

        residual_raw = extract_terminal_residual(y0, sol.y[:, -1])
        if not np.all(np.isfinite(residual_raw)):
            stats["ivp_failures"] += 1
            return np.full_like(u, shooting_penalty, dtype=float)

        residual_scaled = residual_raw / residual_scales
        if shooting_debug:
            logger.debug(
                "Counter-current shooting residual eval=%d norm_inf=%.3e",
                stats["residual_evals"],
                float(np.max(np.abs(residual_scaled))),
            )
        return residual_scaled

    feed_guess = np.maximum(y_template[:ns], flow_floor)
    inlet_guess = np.maximum(y_template[ns:2 * ns], flow_floor)
    flow_candidates = [s0[:ns], inlet_guess, 0.25 * feed_guess, 0.75 * feed_guess]
    temp_candidates = [float(s0[-1])] if non_isothermal else []
    if non_isothermal:
        temp_candidates.extend([float(y_template[2 * ns + 1]), float(y_template[2 * ns])])

    u_candidates: list[np.ndarray] = []
    for flows in flow_candidates:
        candidate_parts = [np.maximum(np.asarray(flows, dtype=float), flow_floor)]
        if non_isothermal:
            for temp_val in temp_candidates:
                candidate_parts_with_t = candidate_parts + [np.array([temp_val], dtype=float)]
                s_candidate = np.concatenate(candidate_parts_with_t)
                u_candidates.append(np.clip(s_candidate / s_ref, lb, ub))
        else:
            s_candidate = np.concatenate(candidate_parts)
            u_candidates.append(np.clip(s_candidate / s_ref, lb, ub))

    # NOTE: add scaled variants around the seed to avoid early local minima.
    multipliers = [1.0, 3.0, 10.0, 0.3]
    for mult in multipliers:
        u_candidates.append(np.clip(mult * u0, lb, ub))

    # NOTE: keep order while removing exact duplicates.
    unique_candidates: list[np.ndarray] = []
    for cand in u_candidates:
        if not any(np.allclose(cand, seen, rtol=0.0, atol=1e-14) for seen in unique_candidates):
            unique_candidates.append(cand)
    if not shooting_multistart:
        unique_candidates = [u0]

    best_lsq = None
    best_sol_final = None
    best_final_norm_inf = np.inf

    for ivp_method in ivp_method_candidates:
        for idx, u_start in enumerate(unique_candidates, start=1):
            for lsq_method in ("trf", "dogbox"):
                try:
                    lsq = least_squares(
                        lambda u: residual_func(u, ivp_method=ivp_method),
                        u_start,
                        bounds=(lb, ub),
                        method=lsq_method,
                        x_scale="jac",
                        max_nfev=shooting_max_nfev,
                        ftol=shooting_ftol,
                        xtol=shooting_xtol,
                        gtol=shooting_gtol,
                    )
                except Exception as exc:
                    if shooting_debug:
                        logger.debug(
                            "Counter-current shooting ivp=%s start=%d lsq=%s failed: %s: %s",
                            ivp_method,
                            idx,
                            lsq_method,
                            type(exc).__name__,
                            exc,
                        )
                    continue

                s_opt = np.asarray(lsq.x, dtype=float) * s_ref
                sol_final, y0_final, final_err = integrate_once(
                    s_opt,
                    ivp_method=ivp_method,
                    full_trajectory=True,
                )
                if final_err is not None or sol_final is None:
                    if shooting_debug:
                        logger.debug(
                            "Counter-current shooting ivp=%s start=%d lsq=%s final IVP failed: %s",
                            ivp_method,
                            idx,
                            lsq_method,
                            final_err,
                        )
                    continue

                final_residual_raw = extract_terminal_residual(y0_final, sol_final.y[:, -1])
                final_residual_scaled = final_residual_raw / residual_scales
                final_norm_inf = float(np.max(np.abs(final_residual_scaled)))
                if shooting_debug:
                    logger.debug(
                        "Counter-current shooting ivp=%s start=%d lsq=%s status=%d nfev=%d final_norm_inf=%.3e",
                        ivp_method,
                        idx,
                        lsq_method,
                        int(lsq.status),
                        int(lsq.nfev),
                        final_norm_inf,
                    )

                if final_norm_inf < best_final_norm_inf:
                    best_final_norm_inf = final_norm_inf
                    best_lsq = lsq
                    best_sol_final = sol_final

                if bool(lsq.success) and np.isfinite(final_norm_inf) and final_norm_inf <= shooting_residual_tol:
                    break
            if best_lsq is not None and bool(best_lsq.success) and np.isfinite(best_final_norm_inf) and best_final_norm_inf <= shooting_residual_tol:
                break
        if best_lsq is not None and bool(best_lsq.success) and np.isfinite(best_final_norm_inf) and best_final_norm_inf <= shooting_residual_tol:
            break

    if best_lsq is None or best_sol_final is None:
        logger.error("HFM counter-current shooting failed: no valid shooting trajectory found.")
        return None

    lsq = best_lsq
    sol_final = best_sol_final
    final_norm_inf = float(best_final_norm_inf)
    converged = bool(lsq.success) and np.isfinite(final_norm_inf) and final_norm_inf <= shooting_residual_tol
    message = (
        "counter-current shooting: "
        f"success={converged} status={int(lsq.status)} "
        f"nfev={int(lsq.nfev)} residual_norm_inf={final_norm_inf:.3e} "
        f"residual_evals={int(stats['residual_evals'])} ivp_failures={int(stats['ivp_failures'])}"
    )

    if not converged:
        logger.error("HFM counter-current shooting failed: %s", message)
        return None

    logger.info("HFM counter-current shooting converged: %s", message)
    return MembraneResult(
        span=sol_final.t,
        state=state_to_physical(sol_final.y),
        success=True,
        message=message,
    )
