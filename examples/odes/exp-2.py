from typing import Any, cast
from diffeqpy import de as _de

# diffeqpy.de replaces itself dynamically from Julia packages at runtime.
de: Any = cast(Any, _de)


def robertson(u, p, t):
    y1, y2, y3 = u

    dy1 = -0.04*y1 + 1e4*y2*y3
    dy2 = 0.04*y1 - 1e4*y2*y3 - 3e7*y2**2
    dy3 = 3e7*y2**2

    return [dy1, dy2, dy3]


u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 1e5)

prob = de.ODEProblem(robertson, u0, tspan)

# 🔥 Use stiff solver
sol = de.solve(prob, de.Rodas5())

print(sol.u[-1])
