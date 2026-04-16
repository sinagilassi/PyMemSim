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
