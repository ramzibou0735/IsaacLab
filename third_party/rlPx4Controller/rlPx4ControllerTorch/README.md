# rlPx4ControllerTorch

`rlPx4ControllerTorch` is a sibling package for `rlPx4Controller` that ports the
parallel PX4-like controllers to a torch-native C++ backend built on ATen.

## Build

Use the IsaacLab Python environment that already provides the matching PyTorch
toolchain:

```bash
/home/ramzi/IsaacLab/env_isaaclab/bin/python setup.py build_ext --inplace
```

If you install it in editable mode, use the active environment's torch build
instead of an isolated build environment:

```bash
/home/ramzi/IsaacLab/env_isaaclab/bin/python -m pip install --no-build-isolation -e .
```

## Import

```python
from rlPx4ControllerTorch.pyParallelControl import ParallelPosControl
```
