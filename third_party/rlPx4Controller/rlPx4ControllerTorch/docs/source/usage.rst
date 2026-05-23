Usage
=====

Build the extension with the IsaacLab Python environment:

.. code-block:: bash

   /home/ramzi/IsaacLab/env_isaaclab/bin/python setup.py build_ext --inplace

If your selected Python environment has ``pip`` available, use the same torch
build and disable build isolation for editable installs.

.. code-block:: bash

   python -m pip install --no-build-isolation -e .

Import the parallel controllers from the new package:

.. code-block:: python

   import torch
   from rlPx4ControllerTorch.pyParallelControl import ParallelPosControl

   controller = ParallelPosControl(1024)
   pos = torch.zeros((1024, 3), dtype=torch.float32)
   quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(1024, 1)
   vel = torch.zeros((1024, 3), dtype=torch.float32)
   ang_vel = torch.zeros((1024, 3), dtype=torch.float32)

   controller.set_status(pos, quat, vel, ang_vel, 0.01)
   commands = controller.update(torch.zeros((1024, 4), dtype=torch.float32))

The wrapper accepts torch tensors only. When ``device`` is omitted, it chooses
``cuda`` when available and otherwise ``cpu``.
