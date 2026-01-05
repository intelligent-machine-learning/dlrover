### User Guide with Node Group Using

> "Node groups are primarily used in scenarios where there are high-performance 
> computing nodes with highly cohesive environmental resources, or in 
> situations involving other infrastructure practices that require group-based 
> scheduling. DLRover implements specific logic for fault tolerance when 
> handling nodes within the same group."

#### Usage: Kubernetes Scheduler Support Following Labels 

To let DLRover know the group info, the k8s scheduler needs to add the following labels to pod:

| label name                 | description                                                                   | format | example  |
|----------------------------|-------------------------------------------------------------------------------|--------|----------|
| scheduling/rack-group      | Index of group, start from 0. Such as: 0, 1, 2, 3...                          | int    | 3        |
| scheduling/rack-id         | The name                                                                      | string | "xxx.xx" |
| scheduling/rack-group-size | The total node group size(How many groups), start from 1. Such as: 1, 2, 4... | int    | 4        |

DLRover ensures high affinity within node groups and maintains a continuous 
rank distribution when handling training jobs involving node groups. So:
1. If all nodes are scheduled normally, the node's fault tolerance logic 
   operates as usual. 
2. However, if a node's fault tolerance process is blocked due to objective 
   factors such as lack of resource, the fault tolerance mechanism will 
   escalate from a single node to the entire group of nodes. The default 
   timeout for this blockage is 300 seconds.
