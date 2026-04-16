import numpy as np
from typing import Optional, Tuple

# locals
from .gas_hfm import GasHFM
from ..utils.tools import smooth_floor


class GasHFMX(GasHFM):
    """
    Scaled gas-phase plug-flow reactor model.

    Physical balances are inherited from GasPFRReactor; only solver-state
    scaling/unscaling is added.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Species scaling from inlet component molar flows.
        self.F_scale = np.maximum(self.F_in.astype(float), 1e-8)

        # T_scale_ref is the numerical scaling center, not thermodynamic reference state.
        self.T_scale_ref = float(self._T_in)
        self.T_scale = 100.0  # K

        # Pressure scaling for state-variable pressure mode.
        self.P_ref = float(self._P0)
        self.P_scale = max(abs(self.P_ref), 1e3)

    def build_y0_scaled(self) -> np.ndarray:
        """
        Build scaled inlet state vector for solve_ivp.
        """
        f0_scaled = self.F_in.astype(float) / self.F_scale
        y0_parts = [f0_scaled]

        if self.heat_transfer_mode == "non-isothermal":
            theta0 = (float(self._T_in) - self.T_scale_ref) / self.T_scale
            y0_parts.append(np.array([theta0], dtype=float))

        if self.pressure_mode == "state_variable":
            pi0 = (float(self._P0) - self.P_ref) / self.P_scale
            y0_parts.append(np.array([pi0], dtype=float))

        if len(y0_parts) == 1:
            return y0_parts[0]
        return np.concatenate(y0_parts)

    def _unscale_state(self, y_scaled: np.ndarray) -> Tuple[np.ndarray, float, Optional[float]]:
        """
        Convert scaled state vector to physical units.
        """
        ns = self.component_num
        idx = ns

        F = np.asarray(
            smooth_floor(y_scaled[:ns], xmin=0.0, s=1e-9),
            dtype=float
        ) * self.F_scale

        if self.heat_transfer_mode == "non-isothermal":
            theta = float(y_scaled[idx])
            idx += 1
            temp = self.T_scale_ref + self.T_scale * theta
            temp = float(smooth_floor(temp, xmin=1.0, s=1e-3))
        else:
            temp = float(self._T_in)

        p_total: Optional[float] = None
        if self.pressure_mode == "state_variable":
            pi = float(y_scaled[idx])
            p_total = self.P_ref + self.P_scale * pi
            p_total = float(smooth_floor(p_total, xmin=1.0, s=1e-3))

        return F, temp, p_total

    def _scale_rhs(
        self,
        dF_dV: np.ndarray,
        dT_dV: Optional[float] = None,
        dP_dV: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert physical derivatives to scaled derivatives.
        """
        out = [dF_dV / self.F_scale]

        if dT_dV is not None:
            out.append(np.array([dT_dV / self.T_scale], dtype=float))

        if dP_dV is not None:
            out.append(np.array([dP_dV / self.P_scale], dtype=float))

        if len(out) == 1:
            return out[0]
        return np.concatenate(out)

    def rhs_physical(self, V: float, y: np.ndarray) -> np.ndarray:
        """
        Physical RHS wrapper.
        """
        return super().rhs(V, y)

    def rhs_scaled(self, V: float, y_scaled: np.ndarray) -> np.ndarray:
        """
        Scaled RHS for solve_ivp.
        """
        ns = self.component_num

        F, temp, p_total = self._unscale_state(y_scaled)

        y_parts = [F]
        if self.heat_transfer_mode == "non-isothermal":
            y_parts.append(np.array([temp], dtype=float))
        if self.pressure_mode == "state_variable":
            y_parts.append(np.array([float(p_total)], dtype=float))
        y_physical = y_parts[0] if len(
            y_parts) == 1 else np.concatenate(y_parts)

        dy_physical_dV = self.rhs_physical(V, y_physical)

        idx = ns
        dF_dV = dy_physical_dV[:ns]
        dT_dV: Optional[float] = None
        dP_dV: Optional[float] = None

        if self.heat_transfer_mode == "non-isothermal":
            dT_dV = float(dy_physical_dV[idx])
            idx += 1

        if self.pressure_mode == "state_variable":
            dP_dV = float(dy_physical_dV[idx])

        return self._scale_rhs(dF_dV=dF_dV, dT_dV=dT_dV, dP_dV=dP_dV)
