from typing import Any, cast
from rich import print
from diffeqpy import de as _de

# diffeqpy.de replaces itself dynamically from Julia packages at runtime.
de: Any = cast(Any, _de)


def f(u, p, t):
    return -u


u0 = 1.0
tspan = (0.0, 5.0)

# Create ODE problem - u0 must be an array-like
prob = de.ODEProblem(f, [u0], tspan)

# Solve with specific solver
sol = de.solve(prob, de.Tsit5())
# print(sol)

print("t:", sol.t)
print("u:", sol.u)
print("success:", sol.retcode)
