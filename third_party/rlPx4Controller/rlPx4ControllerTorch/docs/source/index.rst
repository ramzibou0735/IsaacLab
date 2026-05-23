rlPx4ControllerTorch
===================

``rlPx4ControllerTorch`` is a torch-native sibling package for the parallel
controllers in ``rlPx4Controller``. It preserves the public controller class
surface while moving the batched math to ATen tensors so the same code path can
run on CPU or CUDA devices.

.. toctree::
   :maxdepth: 2

   usage
   compatibility
   api
