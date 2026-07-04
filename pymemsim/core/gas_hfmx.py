import numpy as np
from typing import Optional, Tuple

from .gas_hfm import GasHFM
from ..utils.tools import smooth_floor


class GasHFMX(GasHFM):
    """
    Scaled gas-phase HFM model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Ff_scale = np.maximum(self.Ff_in.astype(float), 1e-8)
        # NOTE: for counter-current with zero permeate inlet, avoid tiny permeate scales
        # that can make reported flows floor-dominated.
        fp_scale_fallback = np.maximum(0.1 * self.Ff_in.astype(float), 1e-8)
        self.Fp_scale = np.maximum(
            np.where(self.Fp_in > 1e-12, self.Fp_in, fp_scale_fallback),
            1e-8
        )

        self.Tf_scale_ref = float(self.Tf_in)
        self.Tp_scale_ref = float(self.Tp_in)
        self.T_scale = 100.0
        self.Pf_scale = max(float(self.Pf), 1.0)
        self.Pp2_scale = max(float(self.Pp**2), 1.0)

    # NOTE: override base class methods for scaling
    def build_y0_scaled(self) -> np.ndarray:
        y0_parts = [
            self.Ff_in.astype(float) / self.Ff_scale,
            self.Fp_in.astype(float) / self.Fp_scale,
        ]
        if getattr(self, "has_feed_pressure_state", False):
            y0_parts.append(np.array([self.Pf / self.Pf_scale], dtype=float))
        if getattr(self, "has_permeate_pressure_state", False):
            y0_parts.append(np.array([self.Pp**2 / self.Pp2_scale], dtype=float))
        if self.heat_transfer_mode == "non-isothermal":
            theta_f0 = (float(self.Tf_in) - self.Tf_scale_ref) / self.T_scale
            theta_p0 = (float(self.Tp_in) - self.Tp_scale_ref) / self.T_scale
            y0_parts.append(np.array([theta_f0, theta_p0], dtype=float))
        return np.concatenate(y0_parts)

    # NOTE: build initial guess by scaling the physical initial guess
    def build_initial_guess(self, z_mesh: np.ndarray) -> np.ndarray:
        y_guess_physical = super().build_initial_guess(z_mesh)

        ns = self.component_num
        y_guess_scaled = np.array(y_guess_physical, dtype=float, copy=True)
        y_guess_scaled[:ns, :] = y_guess_physical[:ns, :] / \
            self.Ff_scale[:, None]
        y_guess_scaled[ns:2 * ns, :] = y_guess_physical[ns:2 *
                                                        ns, :] / self.Fp_scale[:, None]

        idx = 2 * ns
        if getattr(self, "has_feed_pressure_state", False):
            y_guess_scaled[idx, :] = y_guess_physical[idx, :] / self.Pf_scale
            idx += 1
        if getattr(self, "has_permeate_pressure_state", False):
            y_guess_scaled[idx, :] = y_guess_physical[idx, :] / self.Pp2_scale
            idx += 1

        if self.heat_transfer_mode == "non-isothermal":
            y_guess_scaled[idx, :] = (y_guess_physical[idx, :] -
                                      self.Tf_scale_ref) / self.T_scale
            y_guess_scaled[idx + 1, :] = (
                y_guess_physical[idx + 1, :] - self.Tp_scale_ref) / self.T_scale

        return y_guess_scaled

    # NOTE: build BC residuals by unscaling the state and comparing to scaled inlet conditions
    def bc(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        ns = self.ns
        bc_feed = ya[:ns] - (self.Ff_in / self.Ff_scale)
        bc_permeate = yb[ns:2 * ns] - (self.Fp_in / self.Fp_scale)
        bc_parts = [bc_feed, bc_permeate]
        idx = 2 * ns

        if getattr(self, "has_feed_pressure_state", False):
            bc_parts.append(
                np.array([ya[idx] - (self.Pf / self.Pf_scale)], dtype=float)
            )
            idx += 1

        if getattr(self, "has_permeate_pressure_state", False):
            p_state = yb[idx] if self.s_p < 0 else ya[idx]
            bc_parts.append(
                np.array([p_state - (self.Pp**2 / self.Pp2_scale)], dtype=float)
            )
            idx += 1

        if self.heat_transfer_mode == "non-isothermal":
            theta_f_in = (self.Tf_in - self.Tf_scale_ref) / self.T_scale
            theta_p_in = (self.Tp_in - self.Tp_scale_ref) / self.T_scale
            bc_tf = ya[idx] - theta_f_in
            bc_tp = yb[idx + 1] - theta_p_in
            bc_parts.append(np.array([bc_tf, bc_tp], dtype=float))

        return np.concatenate(bc_parts)

    # NOTE: unscale the state to physical units, compute physical RHS, then rescale the RHS for the solver
    def _unscale_state(self, y_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        Ff, Fp, _Pf, _Pp2, Tf, Tp = self._unscale_state_full(y_scaled)
        return Ff, Fp, Tf, Tp

    def _unscale_state_full(self, y_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        ns = self.component_num
        idx = 2 * ns

        Ff = np.asarray(
            smooth_floor(y_scaled[:ns], xmin=0.0, s=1e-9),
            dtype=float
        ) * self.Ff_scale
        Fp = np.asarray(
            smooth_floor(y_scaled[ns:2 * ns], xmin=0.0, s=1e-14),
            dtype=float
        ) * self.Fp_scale

        Pf = float(getattr(self, "Pf", 1.0))
        Pp = float(getattr(self, "Pp", 1.0))
        Pp2 = float(Pp**2)
        if getattr(self, "has_feed_pressure_state", False):
            Pf = float(smooth_floor(
                y_scaled[idx] * self.Pf_scale,
                xmin=1.0,
                s=1e-3,
            ))
            idx += 1
        if getattr(self, "has_permeate_pressure_state", False):
            Pp2 = float(smooth_floor(
                y_scaled[idx] * self.Pp2_scale,
                xmin=1.0,
                s=1e-3,
            ))
            idx += 1

        if self.heat_transfer_mode == "non-isothermal":
            theta_f = float(y_scaled[idx])
            theta_p = float(y_scaled[idx + 1])
            Tf = float(smooth_floor(self.Tf_scale_ref +
                       self.T_scale * theta_f, xmin=1.0, s=1e-3))
            Tp = float(smooth_floor(self.Tp_scale_ref +
                       self.T_scale * theta_p, xmin=1.0, s=1e-3))
        else:
            Tf = float(getattr(self, "Tf_in", 0.0))
            Tp = float(getattr(self, "Tp_in", 0.0))

        return Ff, Fp, Pf, Pp2, Tf, Tp

    # NOTE: helper method to scale the RHS for the solver
    def _scale_rhs(
        self,
        dFf_dz: np.ndarray,
        dFp_dz: np.ndarray,
        dPf_dz: Optional[float] = None,
        dPp2_dz: Optional[float] = None,
        dTf_dz: Optional[float] = None,
        dTp_dz: Optional[float] = None
    ) -> np.ndarray:
        out = [dFf_dz / self.Ff_scale, dFp_dz / self.Fp_scale]

        if dPf_dz is not None:
            out.append(np.array([dPf_dz / self.Pf_scale], dtype=float))
        if dPp2_dz is not None:
            out.append(np.array([dPp2_dz / self.Pp2_scale], dtype=float))

        if dTf_dz is not None and dTp_dz is not None:
            out.append(
                np.array([dTf_dz / self.T_scale, dTp_dz / self.T_scale], dtype=float))

        return np.concatenate(out)

    # NOTE: override base class RHS to handle scaling
    def rhs_physical(self, z: float, y: np.ndarray) -> np.ndarray:
        return super().rhs(z, y)

    # NOTE: override base class RHS to handle scaling
    def rhs_scaled(self, z: float, y_scaled: np.ndarray) -> np.ndarray:
        ns = self.component_num
        Ff, Fp, Pf, Pp2, Tf, Tp = self._unscale_state_full(y_scaled)

        y_parts = [Ff, Fp]
        if getattr(self, "has_feed_pressure_state", False):
            y_parts.append(np.array([Pf], dtype=float))
        if getattr(self, "has_permeate_pressure_state", False):
            y_parts.append(np.array([Pp2], dtype=float))
        if self.heat_transfer_mode == "non-isothermal":
            y_parts.append(np.array([Tf, Tp], dtype=float))
        y_physical = np.concatenate(y_parts)

        dy_physical_dz = self.rhs_physical(z, y_physical)

        dFf_dz = dy_physical_dz[:ns]
        dFp_dz = dy_physical_dz[ns:2 * ns]
        idx = 2 * ns

        dPf_dz = None
        dPp2_dz = None
        if getattr(self, "has_feed_pressure_state", False):
            dPf_dz = float(dy_physical_dz[idx])
            idx += 1
        if getattr(self, "has_permeate_pressure_state", False):
            dPp2_dz = float(dy_physical_dz[idx])
            idx += 1

        if self.heat_transfer_mode == "isothermal":
            return self._scale_rhs(
                dFf_dz=dFf_dz,
                dFp_dz=dFp_dz,
                dPf_dz=dPf_dz,
                dPp2_dz=dPp2_dz,
            )

        dTf_dz = float(dy_physical_dz[idx])
        dTp_dz = float(dy_physical_dz[idx + 1])
        return self._scale_rhs(
            dFf_dz=dFf_dz,
            dFp_dz=dFp_dz,
            dPf_dz=dPf_dz,
            dPp2_dz=dPp2_dz,
            dTf_dz=dTf_dz,
            dTp_dz=dTp_dz
        )
