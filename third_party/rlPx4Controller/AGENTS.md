# Repository Guidelines

## Project Structure & Module Organization
Core C++ controller logic lives in `include/` (headers) and `src/` (native test/demo executable). Python bindings are in `bind/` via pybind11, and the Python package is `rlPx4Controller/` (including `traj_tools/`). Simulation and validation scripts are grouped under `test/` by scenario (`flight_test/`, `simple_sim/`, `joystick_test/`, `ploy_traj_test/`). Documentation sources are in `docs/source/`.

## Build, Test, and Development Commands
- `python -m pip install -e .` builds and installs the pybind11 extensions in editable mode for local development.
- `python setup.py build_ext --inplace` compiles extension modules into the workspace without reinstalling.
- `cmake -S . -B build && cmake --build build` builds the C++ executable target from `CMakeLists.txt`.
- `python test/simple_sim/sim.py` runs a basic simulation path.
- `python test/flight_test/flight_test.py` runs flight-control integration checks (script-based, not pytest-driven).
- `make -C docs html` builds Sphinx docs into `docs/build/html`.

## Coding Style & Naming Conventions
For C++, follow modern C++ Core Guidelines: prefer RAII, `const` by default, `enum class`, no raw owning pointers, and explicit ownership semantics. Keep headers self-contained and avoid `using namespace` in headers. Naming is primarily `snake_case` for functions/variables and descriptive class names (for example `Px4RateController`, `SimplePositionController`). In Python, follow PEP 8 style where practical and keep test scripts scenario-focused.

## Testing Guidelines
This repository currently relies on executable/script tests rather than a unified test runner. Add new checks under the closest scenario folder in `test/` and name files by behavior (example: `parallel_flight_test_pos.py`). For C++ logic changes, validate both native build success and at least one Python integration script. If behavior changes affect trajectories or control loops, include before/after plots or logs in the PR.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects, sometimes with bracket tags (for example `[update] readme`, `[delete] yawrate`). Keep commits focused and scoped to one change area. PRs should include: purpose, key files touched, local commands run, and any control/simulation evidence (screenshots, plots, or metric snippets) when runtime behavior changes.
