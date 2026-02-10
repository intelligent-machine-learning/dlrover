# How to Implement Dynamic Fault Tolerance in DLRover

## Background

The current fault tolerance mechanism in DLRover primarily falls into two categories:
 - Process-level fault toleran
 - Pod-level fault tolerance

This means that regardless of the specific issue encountered, the fault tolerance 
process is triggered based on the above two dimensions. In other words, the 
execution logic of fault tolerance does not depend on the nature of the problem 
itself but only on the dimension in which it occurs.

However, as production scales expand, algorithm or engineering framework 
implementations at the upper layer require more proactive control over fault 
tolerance logic, especially when certain errors are already identifiable. 
For example, if a specific error code clearly indicates an unrecoverable 
exception, the system should be able to halt further fault tolerance efforts 
based on the context.

This capability offers the following advantages:
1. Faster problem exposure, aiding in post-failure diagnostics.
2. Reduced meaningless fault tolerance attempts, conserving resources.


## Restriction

1. [For K8s Job Mode] To obtain detailed stack information after an exception 
   occurs in a torch process, users need to configure the record decorator in 
   the training entry script.

```python
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
  ...
```

2. This approach may not cover exceptional situations such as hangs caused by 
   non-application-level errors, as it currently heavily relies on error stack 
   information for diagnosis.


## Usage

### User SDK

- Supported Strategy:

| Strategy           | Description                                  |
|--------------------|----------------------------------------------|
| NORMAL_FAILOVER    | Normal fault tolerance processing.           |
| NODE_FAILOVER      | Trigger node-level fault tolerance directly. |
| GLOBAL_FAILOVER    | Trigger job-level restarting directly.       |
| ABORTION_FAILOVER  | Abort fault tolerance, fail job directly.    |

- User Extension Implementation:

  The user-side implementation is empty by default, meaning it returns: 
  NORMAL_FAILOVER. This implementation can either be a simple 
  rule definition based on error codes or involve complex logic calls to 
  external services or model inference.

```python
from abc import ABC, abstractmethod

from dlrover.python.common.enums import FailoverStrategy


class DynamicFailoverExtension(ABC):

    @abstractmethod
    def get_user_failover_strategy(self, failure_info) -> FailoverStrategy:
        return FailoverStrategy.NORMAL_FAILOVER
```


### For K8s Job Mode


1. For k8s job, implement the abstraction for dlrover-agent:

   ```python
   class DynamicAgentFailoverExtension(DynamicFailoverExtension):
       """
       Dynamic extension for agent(elastic agent) fault-tolerance execution.
       """
   
       @abstractmethod
       def get_user_failover_strategy(self, failure_info: AgentFailureInfo) -> FailoverStrategy:
           return FailoverStrategy.NORMAL_FAILOVER
   ```
   
   A simple example:
   
   ```python
   from dlrover.python.elastic_agent.torch.dynamic_failover import (
       DynamicAgentFailoverExtension, AgentFailureInfo,
   )
   from dlrover.python.common.enums import FailoverStrategy
   
   
   class TestDynamicFailoverExtension(DynamicAgentFailoverExtension):
   
       def get_user_failover_strategy(self,
                                      failure_info: AgentFailureInfo) -> FailoverStrategy:
           # stop job whatever the failure is
           return FailoverStrategy.ABORTION_FAILOVER
   ```

2. Pass the above implementation class through the 'dynamic_failover_extension' 
   parameter in the job arguments. Note the format: must be ${module}::${class}.
   
   `For example: test.user_defined::UserExtensionTest. `

   For specific configurations of job arguments, please refer to: [doc](../deployment/argument.md)

   example:

   ```python
   --dynamic_failover_extension="test.user_def_failover::TestDynamicFailoverExtension"
   ```

### For Ray Job Mode

TODO
