Compatibility
=============

The new package keeps the controller class names, method names, argument order,
input shapes, output shapes, and quaternion convention from the existing
``rlPx4Controller.pyParallelControl`` module.

Import path
-----------

Old:

.. code-block:: python

   from rlPx4Controller.pyParallelControl import ParallelPosControl

New:

.. code-block:: python

   from rlPx4ControllerTorch.pyParallelControl import ParallelPosControl

Behavior notes
--------------

- The implementation preserves the existing controller defaults, clamp ranges,
  and mixer behavior.
- Hover thrust remains hardcoded to ``0.1533`` during status updates, matching
  the current repository behavior.
- Only obvious indexing bugs from the legacy parallel controller wrappers were
  corrected during the port.
- The Python boundary is ``torch.Tensor`` only.
