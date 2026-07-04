# import libs
from rich import print
from pythermodb_settings.models import CustomProp
from pymemsim.utils.hfm_tools import calculate_hfm_feed_flow_rate_bounds, validate_hfm_feed_flow_rate

# NOTE: permeance values in GPU (Gas Permeation Units), where 1 GPU = 3.35e-10 mol/s.m2.Pa
permeance = {
    "O2": CustomProp(value=9.30, unit="GPU"),
    "N2": CustomProp(value=1.80, unit="GPU"),
}

# NOTE: feed mole fractions for each component (must sum to 1)
z_feed = {
    "O2": CustomProp(value=0.205, unit="-"),
    "N2": CustomProp(value=0.795, unit="-"),
}

# NOTE: calculate feed flow rate bounds for a hollow fiber membrane system based on the provided parameters
bounds = calculate_hfm_feed_flow_rate_bounds(
    number_of_fibers=CustomProp(value=368, unit=""),
    fiber_inner_diameter=CustomProp(value=0.0080, unit="cm"),
    fiber_outer_diameter=CustomProp(value=0.0160, unit="cm"),
    fiber_length=CustomProp(value=25, unit="cm"),
    feed_temperature=CustomProp(value=296.15, unit="K"),
    feed_pressure=CustomProp(value=690000.0, unit="Pa"),
    permeate_pressure=CustomProp(value=100000.0, unit="Pa"),
    viscosity=CustomProp(value=1.8e-5, unit="Pa.s"),
    permeance=permeance,
    feed_mole_fraction=z_feed,
    velocity_min=CustomProp(value=0.01, unit="m/s"),
    velocity_max=CustomProp(value=10.0, unit="m/s"),
    max_pressure_drop=CustomProp(value=20000.0, unit="Pa"),
    theta_max=0.8,
)

print(bounds["f_min_recommended"])
print(bounds["f_max_recommended"])
print(bounds["is_feasible_range"])

# validate_hfm_feed_flow_rate(
#     feed_flow_rate=CustomProp(value=1.0e-5, unit="mol/s"),
#     bounds=bounds,
# )
