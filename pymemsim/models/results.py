# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional


class MembraneResult(BaseModel):
    """Container for membrane simulation outputs."""

    span: Any = Field(
        ...,
        description="Span coordinate points returned by the ODE solver."
    )
    state: Any = Field(
        ...,
        description="State matrix returned by the ODE solver (n_states, n_points)."
    )
    success: bool = Field(
        ...,
        description="Whether the ODE solver finished successfully."
    )
    message: str = Field(
        default="",
        description="Solver status message."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span": self.span,
            "state": self.state,
            "success": self.success,
            "message": self.message,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]
