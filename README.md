# PyMemSim

![PyMemSim](https://drive.google.com/uc?export=view&id=1qSO27JxSgVyms5BriNdphkXI7TLJMgWq)

![Downloads](https://img.shields.io/pypi/dm/PyMemSim)
![PyPI](https://img.shields.io/pypi/v/PyMemSim)
![Python Version](https://img.shields.io/pypi/pyversions/PyMemSim.svg)
![License](https://img.shields.io/pypi/l/PyMemSim)
![Read the Docs](https://img.shields.io/readthedocs/PyMemSim)

**PyMemSim** is a Python package for membrane-process simulation, with a current focus on hollow-fiber membrane (HFM) models.

## Package Functionalities

Current capabilities include:

- Hollow-fiber membrane simulation through the `HFM` interface and `create_hfm_module(...)` factory.
- Gas-phase and liquid-phase HFM modeling.
- `physical` and `scale` modeling modes.
- Isothermal and non-isothermal simulation setup.
- Dual-side inlet specification (feed/permeate flows, temperatures, pressures).
- Constant-pressure operation for feed and permeate sides.
- Component-wise transport coefficients for gas and liquid systems.
- Solver-based simulation (`scipy.integrate.solve_ivp` for IVP and `scipy.integrate.solve_bvp` via `solver_bvp` for BVP) with configurable solver options.

## Installation

Install from PyPI:

```bash
pip install PyMemSim
```

Quick version check:

```python
import pymemsim as pms
print(pms.__version__)
```

## Usage Examples

For a complete end-to-end setup (thermo source, options, model inputs, module creation, and simulation), see:

- `examples/hfm/gas-hfm-exp-1.py`
- `examples/hfm/gas-hfm-exp-2.py`

You can also review:

- `examples/hfm/liquid-hfm-exp-1.py`

### Co-current vs Counter-current (Gas HFM)

The gas HFM examples show how to switch the membrane flow arrangement through:

- `HollowFiberMembraneOptions(..., flow_pattern="co-current")`
- `HollowFiberMembraneOptions(..., flow_pattern="counter-current")`

Reference examples:

- `examples/hfm/gas-hfm-exp-1.py`:
  co-current case (`flow_pattern_to_run = "co-current"`), solved with IVP-style solver options (`Radau`, `rtol`, `atol`).
- `examples/hfm/gas-hfm-exp-2.py`:
  counter-current case (`flow_pattern_to_run = "counter-current"`), solved with `solver_bvp` and BVP-style options (`mesh_points`, `tol`, `bc_tol`, `max_nodes`).

You can run either arrangement in both files by changing `flow_pattern_to_run` to `"co-current"` or `"counter-current"`.

## Development Status

PyMemSim is under active development. APIs, model options, and behaviors may change in future releases.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software in your own applications or projects. However, if you choose to use this app in another app or software, please ensure that my name, Sina Gilassi, remains credited as the original author. This includes retaining any references to the original repository or documentation where applicable. By doing so, you help acknowledge the effort and time invested in creating this project.

## FAQ

For any question, contact me on [LinkedIn](https://www.linkedin.com/in/sina-gilassi/)
