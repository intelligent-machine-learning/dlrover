# Argument Reference

This article primarily introduces the various arguments that users might use
when training with DLRover.

## 1. DLRover Master Arguments

- Usage

- Arguments

| name      | description                                                                 | mandatory | format                | default | options                                                                           |
|-----------|-----------------------------------------------------------------------------|-----------|-----------------------|-------|-----------------------------------------------------------------------------------|
| job_name  | <div style="width: 300pt"> The name of the job defined by user.             | Yes       | string                | n/a   | n/a                                                                               |
| namespace | The name of the Kubernetes namespace where ElasticJob pods will be created. | No        | string                | default | n/a                                                                               |
| platform  | The name of platform.                                                       | No       | string                | pyk8s   | pyk8s, k8s, ray or local                                                          |
| pending_timeout  | The timeout value of pending.                                               | No       | integer(unit: second) | 900   | '>=0'                                                                              |
| pending_fail_strategy  | The fail strategy for pending case.                                         | No       | integer               | 1     | -1: disabled <br/>0: skip <br/>1: verify necessary parts <br/>2: verify all parts |


## 2. Training Arguments

- Usage

- Arguments
