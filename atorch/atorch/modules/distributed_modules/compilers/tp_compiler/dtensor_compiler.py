# A Tensor Parallel Compiler that compiles a model into DTensor implementation.
# Module parameters are sharded with distribute_module api.
# tensors flowing through the graph are resharded with DTensor.redistributed api.
