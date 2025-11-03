# Unified Extension Guide

This section focuses on the extension implementation patterns: how to
setup a user extension implement.

## Key flow

1. Create a subclass of Extension and override the desired methods.
2. Register the subclass using entrypoints in setup.py:
       entry_points={
           'dlrover.unified.extension': [
               'my_extension = my_module:MyExtension',
           ],
       }.


## Example

```python
# step1: create a user extension
# test/a.py
from dlrover.python.unified.controller.extension import ManagerExtension

class XXXExtension(ManagerExtension):
    def xxx(self):
        return "xxx"

# step2: set into entrypoints
...
entry_points={
           'dlrover.unified.extension': [
               'xxx_extension = test.a:XXXExtension',
           ],
       }
...
```

## Supported Extension Point

The currently supported extension points are shown in the table below:

| Extension        | method(point)              | description                                                                                                                                                                    | default implement                                                                                                                |
|------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| ManagerExtension | [relaunch_nodes_impl](https://github.com/intelligent-machine-learning/dlrover/blob/917b4c329fca2208370df4b8a9c0a26ba348a8d9/dlrover/python/unified/controller/extension.py#L54C15-L54C34) | Logic for replacing Ray Nodes: This is because Ray deployments can reside on Kubernetes or physical machines, and the operational mode can be either job mode or cluster mode. | No replacement of Ray Nodes will be performed. The node replacement requirements in the fault tolerance process will be ignored. |

