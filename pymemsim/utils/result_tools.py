from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pythermodb_settings.models import Temperature

from ..models.results import MembraneResult


def _component_labels(components: Sequence[Any], fallback_count: int) -> List[str]:
    labels: List[str] = []
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

    if len(labels) < fallback_count:
        labels.extend([f"component_{i + 1}" for i in range(len(labels), fallback_count)])
    return labels[:fallback_count]


def _warn_once(warnings: List[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _safe_ratio(
    numerator: float,
    denominator: float,
    eps: float,
    warnings: List[str],
    warning_message: str,
) -> Optional[float]:
    if abs(denominator) <= eps:
        _warn_once(warnings, warning_message)
        return None
    return float(numerator / denominator)


def _safe_composition(flows: np.ndarray, totals: np.ndarray, eps: float) -> np.ndarray:
    totals_safe = np.where(np.abs(totals) > eps, totals, 1.0)
    y = flows / totals_safe
    mask = np.abs(totals) <= eps
    if np.any(mask):
        y[:, mask] = 0.0
    return y


def _resolve_target_index(
    target_component: Optional[str],
    component_ids: Sequence[str],
    components: Sequence[Any],
    Ff_in: np.ndarray,
    warnings: List[str],
) -> int:
    if target_component is None:
        return int(np.argmax(Ff_in)) if Ff_in.size > 0 else 0

    normalized = str(target_component).strip().lower()

    for i, cid in enumerate(component_ids):
        if normalized == str(cid).strip().lower():
            return i

    for i, comp in enumerate(components):
        formula = str(getattr(comp, "formula", "")).strip().lower()
        name = str(getattr(comp, "name", "")).strip().lower()
        if normalized == formula or normalized == name:
            return i

    _warn_once(
        warnings,
        f"target_component '{target_component}' was not found; defaulted to largest feed component.",
    )
    return int(np.argmax(Ff_in)) if Ff_in.size > 0 else 0


def _compute_gas_volumetric_flow_profile(
    thermo_source: Any,
    gas_model: str,
    R: float,
    total_flow: np.ndarray,
    temperature: np.ndarray,
    pressure: float,
) -> np.ndarray:
    out = np.zeros_like(total_flow, dtype=float)
    for j in range(total_flow.shape[0]):
        out[j] = float(
            thermo_source.calc_gas_volumetric_flow_rate(
                molar_flow_rate=float(total_flow[j]),
                temperature=float(temperature[j]),
                pressure=float(pressure),
                R=float(R),
                gas_model=gas_model,
            )
        )
    return out


def _compute_liquid_volumetric_flow_profile(
    thermo_source: Any,
    operation_mode: str,
    flow_profile: np.ndarray,
    temperature_profile: np.ndarray,
    q_inlet: float,
) -> np.ndarray:
    out = np.zeros(flow_profile.shape[1], dtype=float)
    if operation_mode == "constant_volume":
        out[:] = float(q_inlet)
        return out

    for j in range(flow_profile.shape[1]):
        rho = thermo_source.calc_rho_LIQ(
            temperature=Temperature(value=float(temperature_profile[j]), unit="K")
        )
        out[j] = float(
            thermo_source.calc_liquid_volumetric_flow_rate(
                molar_flow_rates=flow_profile[:, j],
                molecular_weights=thermo_source.MW,
                density=rho,
            )
        )
    return out


def analyze_hfm_result(
    result: MembraneResult,
    hfm_module: Any,
    target_component: Optional[str] = None,
    eps: float = 1e-30,
) -> Dict[str, Any]:
    span = np.asarray(result.span, dtype=float)
    state = np.asarray(result.state, dtype=float)

    if state.ndim != 2:
        raise ValueError("result.state must be 2D with shape (n_states, n_points).")
    if span.shape[0] != state.shape[1]:
        raise ValueError(
            f"span/state mismatch: len(span)={span.shape[0]}, state points={state.shape[1]}."
        )

    module = getattr(hfm_module, "module", hfm_module)
    thermo_source = getattr(hfm_module, "thermo_source", getattr(module, "thermo_source", None))
    components = list(getattr(hfm_module, "components", getattr(module, "components", [])))

    fallback_count = len(components) if len(components) > 0 else max(state.shape[0] // 2, 1)
    component_ids = list(
        getattr(module, "component_formula_state", _component_labels(components, fallback_count))
    )
    ns = len(component_ids)
    expected_isothermal = 2 * ns
    expected_non_isothermal = 2 * ns + 2
    if state.shape[0] not in (expected_isothermal, expected_non_isothermal):
        raise ValueError(
            f"Unsupported state shape {state.shape}. Expected (2*ns,n) or (2*ns+2,n) with ns={ns}."
        )

    has_temperature_state = state.shape[0] == expected_non_isothermal
    phase = getattr(getattr(hfm_module, "unit_options", None), "phase", None)
    if phase is None:
        phase = getattr(module, "phase", "gas")
    phase = str(phase)

    Ff = np.clip(state[:ns, :], 0.0, None)
    Fp = np.clip(state[ns:2 * ns, :], 0.0, None)
    Ff_total = np.sum(Ff, axis=0)
    Fp_total = np.sum(Fp, axis=0)

    Tf_profile = (
        np.asarray(state[2 * ns, :], dtype=float)
        if has_temperature_state
        else np.full_like(span, float(getattr(module, "Tf_in", np.nan)), dtype=float)
    )
    Tp_profile = (
        np.asarray(state[2 * ns + 1, :], dtype=float)
        if has_temperature_state
        else np.full_like(span, float(getattr(module, "Tp_in", np.nan)), dtype=float)
    )

    yf = _safe_composition(Ff, Ff_total, eps=eps)
    yp = _safe_composition(Fp, Fp_total, eps=eps)

    warnings: List[str] = []

    flow_pattern = getattr(
        getattr(hfm_module, "hfm_core", None),
        "flow_pattern",
        getattr(getattr(hfm_module, "unit_options", None), "flow_pattern", None),
    )
    normalized_flow_pattern = (
        str(flow_pattern).strip().lower().replace("-", "").replace("_", "")
        if flow_pattern is not None
        else ""
    )
    s_p = float(getattr(module, "s_p", 1.0))
    is_counter_current = (
        normalized_flow_pattern == "countercurrent"
        or (normalized_flow_pattern == "" and s_p < 0.0)
    )
    perm_in_idx = -1 if is_counter_current else 0
    perm_out_idx = 0 if is_counter_current else -1

    Ff_in = Ff[:, 0]
    Ff_out = Ff[:, -1]
    Fp_in = Fp[:, perm_in_idx]
    Fp_out = Fp[:, perm_out_idx]
    Ff_in_total = float(Ff_total[0])
    Ff_out_total = float(Ff_total[-1])
    Fp_in_total = float(Fp_total[perm_in_idx])
    Fp_out_total = float(Fp_total[perm_out_idx])

    yf_in = yf[:, 0]
    yf_out = yf[:, -1]
    yp_out = yp[:, perm_out_idx]

    target_index = _resolve_target_index(
        target_component=target_component,
        component_ids=component_ids,
        components=components,
        Ff_in=Ff_in,
        warnings=warnings,
    )
    target_id = component_ids[target_index] if len(component_ids) > 0 else None

    qf_profile: Optional[np.ndarray] = None
    qp_profile: Optional[np.ndarray] = None
    qf_in: Optional[float] = None
    qf_out: Optional[float] = None
    qp_in: Optional[float] = None
    qp_out: Optional[float] = None
    flux_profile: Optional[np.ndarray] = None
    driving_force_profile: Optional[np.ndarray] = None
    driving_force_name: Optional[str] = None

    if phase == "gas":
        Pf = float(getattr(module, "Pf", np.nan))
        Pp = float(getattr(module, "Pp", np.nan))
        Pi = np.asarray(getattr(module, "Pi", np.zeros(ns, dtype=float)), dtype=float)

        driving_force_profile = yf * Pf - yp * Pp
        driving_force_name = "delta_partial_pressure_Pa"
        flux_profile = Pi[:, None] * driving_force_profile

        if thermo_source is not None:
            gas_model = str(getattr(module, "gas_model", "ideal"))
            R_value = float(getattr(module, "R", 8.31446261815324))
            qf_profile = _compute_gas_volumetric_flow_profile(
                thermo_source=thermo_source,
                gas_model=gas_model,
                R=R_value,
                total_flow=Ff_total,
                temperature=Tf_profile,
                pressure=Pf,
            )
            qp_profile = _compute_gas_volumetric_flow_profile(
                thermo_source=thermo_source,
                gas_model=gas_model,
                R=R_value,
                total_flow=Fp_total,
                temperature=Tp_profile,
                pressure=Pp,
            )
            qf_in = float(qf_profile[0])
            qf_out = float(qf_profile[-1])
            qp_in = float(qp_profile[perm_in_idx])
            qp_out = float(qp_profile[perm_out_idx])
    elif phase == "liquid":
        k_i = np.asarray(getattr(module, "k_i", np.zeros(ns, dtype=float)), dtype=float)
        operation_mode = str(getattr(module, "operation_mode", "constant_pressure"))

        if (
            thermo_source is not None
            and hasattr(thermo_source, "calc_liquid_volumetric_flow_rate")
            and hasattr(thermo_source, "calc_rho_LIQ")
            and hasattr(thermo_source, "MW")
        ):
            qf_profile = _compute_liquid_volumetric_flow_profile(
                thermo_source=thermo_source,
                operation_mode=operation_mode,
                flow_profile=Ff,
                temperature_profile=Tf_profile,
                q_inlet=float(getattr(module, "qf_in", 0.0)),
            )
            qp_profile = _compute_liquid_volumetric_flow_profile(
                thermo_source=thermo_source,
                operation_mode=operation_mode,
                flow_profile=Fp,
                temperature_profile=Tp_profile,
                q_inlet=float(getattr(module, "qp_in", 0.0)),
            )
            qf_in = float(qf_profile[0])
            qf_out = float(qf_profile[-1])
            qp_in = float(qp_profile[perm_in_idx])
            qp_out = float(qp_profile[perm_out_idx])

            qf_safe = np.where(np.abs(qf_profile) > eps, qf_profile, 1.0)
            qp_safe = np.where(np.abs(qp_profile) > eps, qp_profile, 1.0)
            Cf = Ff / qf_safe
            Cp = Fp / qp_safe
            Cf[:, np.abs(qf_profile) <= eps] = 0.0
            Cp[:, np.abs(qp_profile) <= eps] = 0.0

            driving_force_profile = Cf - Cp
            driving_force_name = "delta_concentration_mol_per_m3"
            flux_profile = k_i[:, None] * driving_force_profile
        else:
            _warn_once(
                warnings,
                "Liquid volumetric flow profile could not be computed (missing thermo methods).",
            )
    else:
        _warn_once(warnings, f"Unsupported phase '{phase}' for advanced post-analysis.")

    stage_cut_molar = _safe_ratio(
        numerator=max(Fp_out_total - Fp_in_total, 0.0),
        denominator=Ff_in_total,
        eps=eps,
        warnings=warnings,
        warning_message="Stage-cut (molar) is undefined because feed inlet total flow is near zero.",
    )
    stage_cut_volumetric = None
    if qf_in is not None and qp_out is not None:
        stage_cut_volumetric = _safe_ratio(
            numerator=max(float(qp_out) - float(qp_in or 0.0), 0.0),
            denominator=float(qf_in),
            eps=eps,
            warnings=warnings,
            warning_message="Stage-cut (volumetric) is undefined because feed inlet volumetric flow is near zero.",
        )

    if hasattr(module, "Fp_scale"):
        fp_scale = np.asarray(getattr(module, "Fp_scale"), dtype=float)
        # smooth_floor at scaled permeate uses s=1e-12 in GasHFMX.
        floor_est = float(np.log(2.0) * 1e-12 * np.sum(fp_scale))
        if Fp_out_total <= 10.0 * floor_est:
            _warn_once(
                warnings,
                "Permeate outlet flow is close to the numerical floor; stage-cut may be floor-dominated.",
            )

    recoveries_permeate: Dict[str, Optional[float]] = {}
    retentions_retentate: Dict[str, Optional[float]] = {}
    component_residuals: Dict[str, Optional[float]] = {}
    for i, cid in enumerate(component_ids):
        recoveries_permeate[cid] = _safe_ratio(
            numerator=float(Fp_out[i]),
            denominator=float(Ff_in[i]),
            eps=eps,
            warnings=warnings,
            warning_message=f"Permeate recovery for '{cid}' is undefined because feed inlet component flow is near zero.",
        )
        retentions_retentate[cid] = _safe_ratio(
            numerator=float(Ff_out[i]),
            denominator=float(Ff_in[i]),
            eps=eps,
            warnings=warnings,
            warning_message=f"Retentate retention for '{cid}' is undefined because feed inlet component flow is near zero.",
        )
        component_residuals[cid] = _safe_ratio(
            numerator=float((Ff_in[i] + Fp_in[i]) - (Ff_out[i] + Fp_out[i])),
            denominator=float(abs(Ff_in[i] + Fp_in[i])),
            eps=eps,
            warnings=warnings,
            warning_message=f"Component closure for '{cid}' is undefined because total component inlet flow is near zero.",
        )

    purity_permeate_target = _safe_ratio(
        numerator=float(Fp_out[target_index]) if target_id is not None else 0.0,
        denominator=float(Fp_out_total),
        eps=eps,
        warnings=warnings,
        warning_message="Target permeate purity is undefined because permeate outlet total flow is near zero.",
    )
    purity_retentate_target = _safe_ratio(
        numerator=float(Ff_out[target_index]) if target_id is not None else 0.0,
        denominator=float(Ff_out_total),
        eps=eps,
        warnings=warnings,
        warning_message="Target retentate purity is undefined because retentate outlet total flow is near zero.",
    )
    y_feed_target = float(yf_in[target_index]) if target_id is not None else 0.0
    enrichment_factor_target = _safe_ratio(
        numerator=float(purity_permeate_target) if purity_permeate_target is not None else 0.0,
        denominator=float(y_feed_target),
        eps=eps,
        warnings=warnings,
        warning_message="Target enrichment factor is undefined because target feed composition is near zero.",
    )

    separation_factors: Dict[str, Optional[float]] = {}
    for i in range(ns):
        for j in range(ns):
            if i == j:
                continue
            key = f"{component_ids[i]}/{component_ids[j]}"
            num = float(yp_out[i]) * float(yf_in[j])
            den = float(yp_out[j]) * float(yf_in[i])
            separation_factors[key] = _safe_ratio(
                numerator=num,
                denominator=den,
                eps=eps,
                warnings=warnings,
                warning_message=f"Separation factor '{key}' is undefined due to near-zero denominator.",
            )

    avg_flux_by_component: Dict[str, Optional[float]] = {}
    outlet_flux_by_component: Dict[str, Optional[float]] = {}
    inlet_flux_by_component: Dict[str, Optional[float]] = {}
    if flux_profile is not None:
        for i, cid in enumerate(component_ids):
            inlet_flux_by_component[cid] = float(flux_profile[i, 0])
            outlet_flux_by_component[cid] = float(flux_profile[i, -1])
            avg_flux_by_component[cid] = float(np.trapezoid(flux_profile[i, :], span) / (span[-1] - span[0])) if span[-1] != span[0] else float(flux_profile[i, 0])
    else:
        for cid in component_ids:
            inlet_flux_by_component[cid] = None
            outlet_flux_by_component[cid] = None
            avg_flux_by_component[cid] = None
        _warn_once(warnings, "Flux profile is unavailable for this case.")

    q_cond_profile = None
    q_cond_avg = None
    q_cond_integrated = None
    delta_Tf = None
    delta_Tp = None
    max_side_temperature_gap = None
    if has_temperature_state:
        U_m = float(getattr(module, "U_m", 0.0))
        q_cond_profile = U_m * (Tf_profile - Tp_profile)
        q_cond_avg = float(np.mean(q_cond_profile))
        q_cond_integrated = float(np.trapezoid(q_cond_profile, span))
        delta_Tf = float(Tf_profile[-1] - Tf_profile[0])
        delta_Tp = float(Tp_profile[-1] - Tp_profile[0])
        max_side_temperature_gap = float(np.max(np.abs(Tf_profile - Tp_profile)))
    else:
        _warn_once(warnings, "Thermal profile metrics are unavailable for isothermal results.")

    Pf = getattr(module, "Pf", None)
    Pp = getattr(module, "Pp", None)
    feed_pressure_drop = None
    permeate_pressure_drop = None
    avg_transmembrane_pressure_diff = None
    Pf_profile = None
    Pp_profile = None
    if Pf is not None and Pp is not None:
        Pf_value = float(Pf)
        Pp_value = float(Pp)
        Pf_profile = np.full_like(span, Pf_value, dtype=float)
        Pp_profile = np.full_like(span, Pp_value, dtype=float)
        feed_pressure_drop = float(Pf_value - Pf_value)
        permeate_pressure_drop = float(Pp_value - Pp_value)
        avg_transmembrane_pressure_diff = float(np.mean(Pf_profile - Pp_profile))
    else:
        _warn_once(warnings, "Hydraulic pressure metrics are unavailable (missing pressure attributes).")

    total_molar_closure_residual = _safe_ratio(
        numerator=float((Ff_in_total + Fp_in_total) - (Ff_out_total + Fp_out_total)),
        denominator=float(abs(Ff_in_total + Fp_in_total)),
        eps=eps,
        warnings=warnings,
        warning_message="Overall molar closure is undefined because total inlet flow is near zero.",
    )
    feed_basis_molar_closure_residual = _safe_ratio(
        numerator=float(Ff_in_total - (Ff_out_total + Fp_out_total - Fp_in_total)),
        denominator=float(abs(Ff_in_total)),
        eps=eps,
        warnings=warnings,
        warning_message="Feed-basis molar closure is undefined because feed inlet flow is near zero.",
    )

    defined_component_residuals = [abs(v) for v in component_residuals.values() if v is not None]
    max_component_molar_closure_residual = (
        float(max(defined_component_residuals)) if len(defined_component_residuals) > 0 else None
    )

    def _stream_dict(
        total_molar_flow: float,
        component_flows: np.ndarray,
        composition: np.ndarray,
        temperature_value: float,
        pressure_value: Optional[float],
        volumetric_flow: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "total_molar_flow_mol_per_s": float(total_molar_flow),
            "component_molar_flows_mol_per_s": {
                cid: float(component_flows[i]) for i, cid in enumerate(component_ids)
            },
            "mole_fractions": {
                cid: float(composition[i]) for i, cid in enumerate(component_ids)
            },
            "temperature_K": float(temperature_value),
            "pressure_Pa": float(pressure_value) if pressure_value is not None else None,
            "volumetric_flow_m3_per_s": float(volumetric_flow) if volumetric_flow is not None else None,
        }

    profiles: Dict[str, Any] = {
        "span_m": span.tolist(),
        "feed_component_molar_flow_profiles_mol_per_s": {
            cid: Ff[i, :].tolist() for i, cid in enumerate(component_ids)
        },
        "permeate_component_molar_flow_profiles_mol_per_s": {
            cid: Fp[i, :].tolist() for i, cid in enumerate(component_ids)
        },
        "feed_total_molar_flow_profile_mol_per_s": Ff_total.tolist(),
        "permeate_total_molar_flow_profile_mol_per_s": Fp_total.tolist(),
        "feed_mole_fraction_profiles": {
            cid: yf[i, :].tolist() for i, cid in enumerate(component_ids)
        },
        "permeate_mole_fraction_profiles": {
            cid: yp[i, :].tolist() for i, cid in enumerate(component_ids)
        },
        "feed_temperature_profile_K": Tf_profile.tolist() if has_temperature_state else None,
        "permeate_temperature_profile_K": Tp_profile.tolist() if has_temperature_state else None,
        "feed_pressure_profile_Pa": Pf_profile.tolist() if Pf_profile is not None else None,
        "permeate_pressure_profile_Pa": Pp_profile.tolist() if Pp_profile is not None else None,
        "feed_volumetric_flow_profile_m3_per_s": qf_profile.tolist() if qf_profile is not None else None,
        "permeate_volumetric_flow_profile_m3_per_s": qp_profile.tolist() if qp_profile is not None else None,
        "driving_force_name": driving_force_name,
        "driving_force_profiles": (
            {cid: driving_force_profile[i, :].tolist() for i, cid in enumerate(component_ids)}
            if driving_force_profile is not None
            else None
        ),
        "flux_profiles_mol_per_m2_s": (
            {cid: flux_profile[i, :].tolist() for i, cid in enumerate(component_ids)}
            if flux_profile is not None
            else None
        ),
    }

    return {
        "case_definition": {
            "phase": phase,
            "flow_pattern": flow_pattern,
            "modeling_type": getattr(getattr(hfm_module, "unit_options", None), "modeling_type", None),
            "heat_transfer_mode": getattr(module, "heat_transfer_mode", None),
            "operation_mode": getattr(module, "operation_mode", None),
            "feed_pressure_mode": getattr(getattr(hfm_module, "unit_options", None), "feed_pressure_mode", None),
            "permeate_pressure_mode": getattr(getattr(hfm_module, "unit_options", None), "permeate_pressure_mode", None),
            "module_length_m": float(span[-1] - span[0]) if span.size > 1 else 0.0,
            "membrane_area_per_length_m": float(getattr(module, "a_m", np.nan)),
            "component_ids": component_ids,
            "solver_success": bool(result.success),
            "solver_message": result.message,
            "solver_points": int(span.shape[0]),
        },
        "streams": {
            "feed_inlet": _stream_dict(
                total_molar_flow=Ff_in_total,
                component_flows=Ff_in,
                composition=yf_in,
                temperature_value=float(Tf_profile[0]),
                pressure_value=float(Pf) if Pf is not None else None,
                volumetric_flow=qf_in,
            ),
            "retentate_outlet": _stream_dict(
                total_molar_flow=Ff_out_total,
                component_flows=Ff_out,
                composition=yf_out,
                temperature_value=float(Tf_profile[-1]),
                pressure_value=float(Pf) if Pf is not None else None,
                volumetric_flow=qf_out,
            ),
            "permeate_outlet": _stream_dict(
                total_molar_flow=Fp_out_total,
                component_flows=Fp_out,
                composition=yp_out,
                temperature_value=float(Tp_profile[perm_out_idx]),
                pressure_value=float(Pp) if Pp is not None else None,
                volumetric_flow=qp_out,
            ),
        },
        "performance": {
            "target_component": target_id,
            "stage_cut_molar": stage_cut_molar,
            "stage_cut_volumetric": stage_cut_volumetric,
            "recoveries_permeate": recoveries_permeate,
            "retentions_retentate": retentions_retentate,
            "purity_permeate_target": purity_permeate_target,
            "purity_retentate_target": purity_retentate_target,
            "enrichment_factor_target": enrichment_factor_target,
            "separation_factors": separation_factors,
            "flux_inlet_mol_per_m2_s": inlet_flux_by_component,
            "flux_outlet_mol_per_m2_s": outlet_flux_by_component,
            "flux_average_mol_per_m2_s": avg_flux_by_component,
        },
        "profiles": profiles,
        "thermal": {
            "is_non_isothermal": bool(has_temperature_state),
            "feed_delta_T_K": delta_Tf,
            "permeate_delta_T_K": delta_Tp,
            "max_side_temperature_gap_K": max_side_temperature_gap,
            "conductive_heat_transfer_profile_W_per_m2": q_cond_profile.tolist() if q_cond_profile is not None else None,
            "conductive_heat_transfer_average_W_per_m2": q_cond_avg,
            "conductive_heat_transfer_integrated_W_per_m": q_cond_integrated,
            "external_heat_flux_feed_W_per_m2": float(getattr(module, "q_ext_f", 0.0)),
            "external_heat_flux_permeate_W_per_m2": float(getattr(module, "q_ext_p", 0.0)),
        },
        "hydraulic": {
            "feed_pressure_drop_Pa": feed_pressure_drop,
            "permeate_pressure_drop_Pa": permeate_pressure_drop,
            "average_transmembrane_pressure_difference_Pa": avg_transmembrane_pressure_diff,
        },
        "balances": {
            "overall_molar_closure_residual": total_molar_closure_residual,
            "feed_basis_molar_closure_residual": feed_basis_molar_closure_residual,
            "component_molar_closure_residuals": component_residuals,
            "max_component_molar_closure_residual": max_component_molar_closure_residual,
            "energy_closure_residual": None,
        },
        "warnings": warnings,
    }


def analyze_membrane_result(
    result: MembraneResult,
    hfm_module: Any,
    target_component: Optional[str] = None,
    eps: float = 1e-30,
) -> Dict[str, Any]:
    return analyze_hfm_result(
        result=result,
        hfm_module=hfm_module,
        target_component=target_component,
        eps=eps,
    )


def build_hfm_result_table_template(analysis: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    streams = analysis.get("streams", {})
    performance = analysis.get("performance", {})
    balances = analysis.get("balances", {})
    thermal = analysis.get("thermal", {})

    feed = streams.get("feed_inlet", {})
    retentate = streams.get("retentate_outlet", {})
    permeate = streams.get("permeate_outlet", {})

    rows_stream = [
        {
            "metric": "Total molar flow [mol/s]",
            "feed": str(feed.get("total_molar_flow_mol_per_s")),
            "retentate": str(retentate.get("total_molar_flow_mol_per_s")),
            "permeate": str(permeate.get("total_molar_flow_mol_per_s")),
        },
        {
            "metric": "Temperature [K]",
            "feed": str(feed.get("temperature_K")),
            "retentate": str(retentate.get("temperature_K")),
            "permeate": str(permeate.get("temperature_K")),
        },
        {
            "metric": "Pressure [Pa]",
            "feed": str(feed.get("pressure_Pa")),
            "retentate": str(retentate.get("pressure_Pa")),
            "permeate": str(permeate.get("pressure_Pa")),
        },
        {
            "metric": "Volumetric flow [m3/s]",
            "feed": str(feed.get("volumetric_flow_m3_per_s")),
            "retentate": str(retentate.get("volumetric_flow_m3_per_s")),
            "permeate": str(permeate.get("volumetric_flow_m3_per_s")),
        },
        {
            "metric": "Component flows [mol/s]",
            "feed": str(feed.get("component_molar_flows_mol_per_s")),
            "retentate": str(retentate.get("component_molar_flows_mol_per_s")),
            "permeate": str(permeate.get("component_molar_flows_mol_per_s")),
        },
        {
            "metric": "Mole fractions",
            "feed": str(feed.get("mole_fractions")),
            "retentate": str(retentate.get("mole_fractions")),
            "permeate": str(permeate.get("mole_fractions")),
        },
    ]

    rows_global_perf = [
        {"metric": "Stage cut (molar)", "value": str(performance.get("stage_cut_molar"))},
        {"metric": "Stage cut (volumetric)", "value": str(performance.get("stage_cut_volumetric"))},
    ]

    rows_target_perf = [
        {"metric": "Target component", "value": str(performance.get("target_component"))},
        {"metric": "Target permeate purity", "value": str(performance.get("purity_permeate_target"))},
        {"metric": "Target retentate purity", "value": str(performance.get("purity_retentate_target"))},
        {"metric": "Target enrichment factor", "value": str(performance.get("enrichment_factor_target"))},
        {"metric": "Recoveries (permeate)", "value": str(performance.get("recoveries_permeate"))},
        {"metric": "Retentions (retentate)", "value": str(performance.get("retentions_retentate"))},
    ]

    rows_balance = [
        {"metric": "Overall molar closure", "value": str(balances.get("overall_molar_closure_residual"))},
        {"metric": "Feed-basis molar closure", "value": str(balances.get("feed_basis_molar_closure_residual"))},
        {"metric": "Max component closure", "value": str(balances.get("max_component_molar_closure_residual"))},
        {"metric": "Component closures", "value": str(balances.get("component_molar_closure_residuals"))},
    ]

    rows_thermal = [
        {"metric": "Non-isothermal", "value": str(thermal.get("is_non_isothermal"))},
        {"metric": "Feed delta T [K]", "value": str(thermal.get("feed_delta_T_K"))},
        {"metric": "Permeate delta T [K]", "value": str(thermal.get("permeate_delta_T_K"))},
        {"metric": "Max side gap [K]", "value": str(thermal.get("max_side_temperature_gap_K"))},
        {"metric": "q_cond avg [W/m2]", "value": str(thermal.get("conductive_heat_transfer_average_W_per_m2"))},
    ]

    return {
        "stream_summary": rows_stream,
        "global_performance_summary": rows_global_perf,
        "target_performance_summary": rows_target_perf,
        "balance_summary": rows_balance,
        "thermal_summary": rows_thermal,
    }


def print_hfm_result_tables(analysis: Dict[str, Any]) -> None:
    from rich import print as rich_print
    from rich.table import Table

    template = build_hfm_result_table_template(analysis)

    stream_table = Table(title="HFM Stream Summary")
    stream_table.add_column("Metric", style="cyan")
    stream_table.add_column("Feed")
    stream_table.add_column("Retentate")
    stream_table.add_column("Permeate")
    for row in template["stream_summary"]:
        stream_table.add_row(row["metric"], row["feed"], row["retentate"], row["permeate"])
    rich_print(stream_table)

    for section_name, section_title in [
        ("global_performance_summary", "HFM Global Performance"),
        ("target_performance_summary", "HFM Target Component Performance"),
        ("balance_summary", "HFM Balance Summary"),
        ("thermal_summary", "HFM Thermal Summary"),
    ]:
        table = Table(title=section_title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        for row in template[section_name]:
            table.add_row(row["metric"], row["value"])
        rich_print(table)
