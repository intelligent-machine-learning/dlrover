# DLRover Diagnosis System Design

A training job may encounter various issues during runtime. While some issues lead
to immediate failure, others may not manifest right away, posing potential or 
imminent risks to the training's progress. Many systems resort to resuming training on new machines to diagnose errors.
However, such approach significantly prolongs the job's training time. For example, one faulty machine
will halt the training process until the the machine is replaced and resuming the training.
A more efficient method is to conduct a pre-check before starting the training, which can help 
avoid the fault-tolerance overhead. Furthermore, certain latent issues can have a greater 
negative impact on training performance. For example, a job may hang for an unknown reason, 
only triggering timeout errors after 30 minutes or even longer.

Therefore, DLRover requires a diagnosis system that can quickly identify and diagnose failures 
before, during, and even after the training process. There are three main challenges in failure 
diagnosis:

1. Diversity of faults: The root causes of failures may stem from various sources, such as hardware, 
network issues, the training environment, or user code. Each type of fault requires specific 
diagnosis methods tailored to its nature. 
2. Rapid evolution of diagnosis approaches: As our understanding of failures improves, we continue 
to refine our diagnosis techniques. For example, initially, we relied on timeouts to identify faulty 
machines. Over time, we found it more efficient to check logs and perform hardware diagnostics instead. 
3. Complexity of root causes: Some failures involve multiple factors and require extensive checks 
to pinpoint the root cause. For example, training hangs may result from issues with the chip or 
network, but could also be caused by user code or the training framework. Those checks are independent
The diagnosis system must provide an easy way to integrate these various checks and reach an accurate conclusion.

## Overview

The general diagnosis process consists of four phases: collecting runtime data, identifying faults, 
generating solutions, and executing those solutions. The DLRover diagnosis system has corresponding 
components for each of these functions.

To handle the diversity of failures and the rapid evolution of diagnostic approaches, 
these components are designed as plugins. Additionally, we have developed the Inference Chain, 
which allows for multiple inference steps to systematically identify the root causes and generate 
solutions for complex problems.

<div align="center">
<img src="../figures/smart_ft_250123.jpg" alt="Editor" width="500">
</div>

As shown in the figure, the **DataCollector** is to collect various of runtime data for diagnosis process.
The **Diagnostician** is responsible to observe and resolve failures. Usually one particular failure
has a corresponding diagnostician to handle the failure. The diagnostician will generate and pass the 
solutions (**DiagnosisAction**) to the WorkerAgent (worker-level failure) or JobManager (job-level failure) will execute
those solutions to diagnose failures.

### DataCollector

Collector is to collect necessary data to resolve the issues during training.

```python
class DataCollector(metaclass=ABCMeta):
    """
    DataCollector collects certain type of data and report to master.
    Those data is used to diagnosis the faults of training.
    """

    def __init__(self):
        pass

    @abstractmethod
    def collect_data(self) -> object:
        """The implementation of data collector."""
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether the collector is enabled."""

        return True
```


### DataManager

For job level failures, the master has to collect runtime data from workers. The DataManager is to 
collect and store runtime data collected by DataCollectors on workers. The data obtained by DataManager
will be used in job-level failure diagnosis.

### Diagnostician

Diagnostician is responsible for identifying failures and propose final solutions. It provides 
**observe** to identify the failure and **resolve** to explore solution. 

```python
class Diagnostician(metaclass=ABCMeta):
    """
    Diagnostician is to observe problems and resolve those problems
    """

    def __init__(self):
        pass

    @abstractmethod
    def observe(self, **kwargs) -> DiagnosisAction:
        # observe if particular problem happened
        return NoAction()

    @abstractmethod
    def resolve(self, problem: DiagnosisAction, **kwargs) -> DiagnosisAction:
        # explore the solution to resolve the problem
        return NoAction()
```

### InferenceChain

The Diagnostician could employ **Rule-based Inference Chains** for reasoning.
Each chain comprises multiple inference steps, with each step leveraging a
particular inference to identify
the immediate cause. Subsequently, each inference step builds upon the previous
step's output, progressively moving towards the root cause.

For an **Inference**, there is an **InferenceOperator** to infer from this inference to a new
one. This chain continues in this way until reach the final conclusion.

```python
@dataclass
class Inference(object):
    """
    Inference object reflects problems and failures during training.
    """

    name: str = ""
    attribution: str = ""
    description: str = ""
    configs: Dict[str, str] = field(default_factory=dict)

class InferenceOperator(metaclass=ABCMeta):
    """
    InferenceOperator is used to infer the root cause of problems.
    """

    def __init__(self, data_manager):
        self._data_manager = data_manager

    @abstractmethod
    def infer(self, inferences: List[Inference]) -> List[Inference]:
        pass

    # check if the InferenceOperator can be used to infer a given inference
    @abstractmethod
    def is_compatible(self, inference: Inference) -> bool:
        pass
```

The generation of an inference chain is tailored to the nature of the problem and
the available data. For example, in the case of a hang fault, the system
prioritizes CUDA events, initiating inference steps based on them. However,
in the absence of CUDA events, the system adapts the inference chain to
utilize chip metrics for inference instead.

## Example: Failure Node Identification During Training

Based on our experience, over 80% of training failures are caused by temporary errors, 
which can usually be resolved by simply restarting the training process on the same machine. 
However, when the failure is due to a malfunctioning machine, the correct solution is to 
relaunch the worker on a new machine. 
Therefore, we need to detect machine failures and take the appropriate action to recover 
the training process.

To address this, we have implemented the FailureNodeDiagnostician, 
which monitors for failed machines and triggers the necessary recovery actions.

<div align="center">
<img src="../figures/failure_node_observer.jpg" alt="Editor" width="500">
</div>

## Example: Identify Hang Error

Since training hangs can be caused by various factors, 
we need to perform multiple independent checks to reach a final conclusion. 
For instance, we use a tool called XPU_Timer, which collects CUDA kernel stack traces from each worker. 
In some cases, the training hang can be easily detected with this tool. However, 
relying solely on XPU_Timer is not sufficient to guarantee accuracy. 
Therefore, additional checks are necessary to make a more precise determination.

<div align="center">
<img src="../figures/hang_inference_chain.jpg" alt="Editor" width="500">
</div>

The figure shows the inference chain to check if a training hang has happened. Here we have check the 
CUDA kernel stacktrace in **XPUTimerHangChecker**. After that we check some other runtime metrics (e.g., chip)
in **AntMonitorHangChecker**. Based on the output of AntMonitorHangChecker, we will check
if there is message mismatch in send/recv in **SendRecvHangChecker** or check XCCL communication in
**XCCLHangChecker**.
