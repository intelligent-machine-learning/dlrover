# Training Pre Check Procedure Introduction

## Background

As the scale of training production expands, the factors that may cause 
runtime exceptions related to the environment during training are gradually 
increasing. Although the current DLRover adds a so-called node-check before 
training (which runs a small-scale training task to detect any abnormal 
scenarios), it primarily observes availability from the training dimension 
perspective. Therefore, it is not easy to specifically diagnose problematic 
nodes and classify issues. 

So we are seeking a more advanced and precise check, 
conducting certain related inspections right when all nodes are in place 
(or not yet in place) to identify issues as early as possible and implement 
fault tolerance or blockages in advance. The checks mentioned earlier rely more 
on external services, as they require specialized and more precise validation 
of components such as containers and networks. Traditional training frameworks 
do not and should not involve these components. 

Considering that the actual checks are highly related to the specific deployment 
environment and heavily rely on external capabilities, implementations added here 
on the open-source side will be updated in the future (in a more generic way), 
only the exposure of interfaces and the implementation of processes for now. 
This article only discusses the core design of this aspect.


## Target

- Support configurable multiple types of synchronous check before 
  training starts.
- Support limited checks(some check may not be supported during runtime) 
  during training execution.

## Design

- The existing training start process:

<img src="../figures/current_start_process.png" alt="Current Training Start Process">


- The training start process combined with pre-check:

<img src="../figures/new_start_process.png" alt="New Training Start Process">

  The pre-check process involves the following key states:

  - DISABLED: The check is disabled, and the training worker will ignore the 
              check results and the training will continue.
  - CHECKING: The check is in progress, continue to wait.
  - FAIL: The check has failed, and the training will be interrupted.
  - PASS: The check has passed, and the training will continue.


- The pre-check operators will be executed as follows:

<img src="../figures/pre_check_flow.png" alt="Pre Check Flow">


## Interface

### PreCheckOperator
The PreCheckOperator is the core component for executing pre-checks. Multiple 
implemented operators are driven by the DiagnosisManager to run before each 
training session starts, as illustrated in the code below:

```python
class PreCheckOperator(ABC):

    @classmethod
    def get_retry_interval_secs(cls) -> int:
        """The retry interval seconds, can be overridden in subclasses."""
        return 5

    @abstractmethod
    def check(self, *args, **kwargs) -> PreCheckResult:
        """The abstraction of the main check procedure."""
        pass

    @abstractmethod
    def failed_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        """The abstraction of the actions when operator check failed."""
        pass
```

The execution process of each PreCheckOperator is roughly as follows:

<img src="../figures/pre_check_op.png" alt="Pre Check Operator Process">

### PreCheckResult

The PreCheckResult represents the result of the Operator's check. It includes 
a numeric result to indicate the general outcome of the check 
(similar to an error code), a string description, and a set to represent the 
abnormal nodes. It is defined as follows:

```python
@dataclass
class PreCheckResult(object):

    # The default success result is 0. The other result code(>0) should be
    # defined by different pre-check operator it's self.
    result: int = 0

    # The simple description info for the result.
    result_msg: str = ""

    # Abnormal nodes.
    abnormal_nodes: List[Node] = field(default_factory=list)

    def is_success(self):
        return self.result == 0
```

Considering that custom extensions of the Operator may involve more complex 
logic for determining whether a result is successful, users can fully extend 
the PreCheckResult and override the is_success method to define their own 
logic for result evaluation.
