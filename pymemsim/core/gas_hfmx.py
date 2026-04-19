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
        self.Fp_scale = np.maximum(np.where(self.Fp_in > 0.0, self.Fp_in, 1e-8), 1e-8)

        self.Tf_scale_ref = float(self.Tf_in)
        self.Tp_scale_ref = float(self.Tp_in)
        self.T_scale = 100.0

    def build_y0_scaled(self) -> np.ndarray:
        y0_parts = [
            self.Ff_in.astype(float) / self.Ff_scale,
            self.Fp_in.astype(float) / self.Fp_scale,
        ]
        if self.heat_transfer_mode == "non-isothermal":
            theta_f0 = (float(self.Tf_in) - self.Tf_scale_ref) / self.T_scale
            theta_p0 = (float(self.Tp_in) - self.Tp_scale_ref) / self.T_scale
            y0_parts.append(np.array([theta_f0, theta_p0], dtype=float))
        return np.concatenate(y0_parts)

    def build_initial_guess(self, z_mesh: np.ndarray) -> np.ndarray:
        y_guess_physical = super().build_initial_guess(z_mesh)

        ns = self.component_num
        y_guess_scaled = np.array(y_guess_physical, dtype=float, copy=True)
        y_guess_scaled[:ns, :] = y_guess_physical[:ns, :] / self.Ff_scale[:, None]
        y_guess_scaled[ns:2 * ns, :] = y_guess_physical[ns:2 * ns, :] / self.Fp_scale[:, None]

        if self.heat_transfer_mode == "non-isothermal":
            y_guess_scaled[2 * ns, :] = (y_guess_physical[2 * ns, :] - self.Tf_scale_ref) / self.T_scale
            y_guess_scaled[2 * ns + 1, :] = (y_guess_physical[2 * ns + 1, :] - self.Tp_scale_ref) / self.T_scale

        return y_guess_scaled

    def bc(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        ns = self.ns
        bc_feed = ya[:ns] - (self.Ff_in / self.Ff_scale)
        bc_permeate = yb[ns:2 * ns] - (self.Fp_in / self.Fp_scale)

        if self.heat_transfer_mode == "non-isothermal":
            theta_f_in = (self.Tf_in - self.Tf_scale_ref) / self.T_scale
            theta_p_in = (self.Tp_in - self.Tp_scale_ref) / self.T_scale
            bc_tf = ya[2 * ns] - theta_f_in
            bc_tp = yb[2 * ns + 1] - theta_p_in
            return np.concatenate([bc_feed, bc_permeate, np.array([bc_tf, bc_tp], dtype=float)])

        return np.concatenate([bc_feed, bc_permeate])

    def _unscale_state(self, y_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        ns = self.component_num
        idx = 2 * ns

        Ff = np.asarray(
            smooth_floor(y_scaled[:ns], xmin=0.0, s=1e-9),
            dtype=float
        ) * self.Ff_scale
        Fp = np.asarray(
            smooth_floor(y_scaled[ns:2 * ns], xmin=0.0, s=1e-9),
            dtype=float
        ) * self.Fp_scale

        if self.heat_transfer_mode == "non-isothermal":
            theta_f = float(y_scaled[idx])
            theta_p = float(y_scaled[idx + 1])
            Tf = float(smooth_floor(self.Tf_scale_ref + self.T_scale * theta_f, xmin=1.0, s=1e-3))
            Tp = float(smooth_floor(self.Tp_scale_ref + self.T_scale * theta_p, xmin=1.0, s=1e-3))
        else:
            Tf = float(self.Tf_in)
            Tp = float(self.Tp_in)

        return Ff, Fp, Tf, Tp

    def _scale_rhs(
        self,
        dFf_dz: np.ndarray,
        dFp_dz: np.ndarray,
        dTf_dz: Optional[float] = None,
        dTp_dz: Optional[float] = None
    ) -> np.ndarray:
        out = [dFf_dz / self.Ff_scale, dFp_dz / self.Fp_scale]

        if dTf_dz is not None and dTp_dz is not None:
            out.append(np.array([dTf_dz / self.T_scale, dTp_dz / self.T_scale], dtype=float))

        return np.concatenate(out)

    def rhs_physical(self, z: float, y: np.ndarray) -> np.ndarray:
        return super().rhs(z, y)

    def rhs_scaled(self, z: float, y_scaled: np.ndarray) -> np.ndarray:
        ns = self.component_num
        Ff, Fp, Tf, Tp = self._unscale_state(y_scaled)

        y_parts = [Ff, Fp]
        if self.heat_transfer_mode == "non-isothermal":
            y_parts.append(np.array([Tf, Tp], dtype=float))
        y_physical = np.concatenate(y_parts)

        dy_physical_dz = self.rhs_physical(z, y_physical)

        dFf_dz = dy_physical_dz[:ns]
        dFp_dz = dy_physical_dz[ns:2 * ns]

        if self.heat_transfer_mode == "isothermal":
            return self._scale_rhs(dFf_dz=dFf_dz, dFp_dz=dFp_dz)

        dTf_dz = float(dy_physical_dz[2 * ns])
        dTp_dz = float(dy_physical_dz[2 * ns + 1])
        return self._scale_rhs(
            dFf_dz=dFf_dz,
            dFp_dz=dFp_dz,
            dTf_dz=dTf_dz,
            dTp_dz=dTp_dz
        )
