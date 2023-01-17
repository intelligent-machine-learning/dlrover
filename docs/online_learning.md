
# Introduction to Develop Online Learnin Model with DLRover
The document describes pre-requisites for online learning and how to develop online learning model with DLRover.

## A Stable Running Environment
Online Learning is a long running job since it needs to learn from the lastest data and export model for serving as soon as possible.
Thus, a stable running environment in which the online learning job won't be killed accidently is preferred. 
Refer(tutorial)[https://github.com/intelligent-machine-learning/dlrover/blob/master/docs/dev.md] for how to set up a K8s running environment and submit a deepfm training model.

## Fault Tolerance
For a safe distributed training loop, special attention should be paid to creating/restoring checkpoint files and recovering from failures.
For parameter server training, both ps and worker would fail due to reasons like physical node is removed. It's unnecessary to resubmit the job because of overhead.
For asynchronous training, a worker should join the training loop after it fails while other ps/workers doesn't need stop and handle the situation.
If a ps fails, other healthy ps doesn't need to restart and chief should restore the checkpoint once the ps is recovered.
In addition, upwards streaming job may also fail and restore from the latest checkpoints. Streaming job and training job should work together to ensure that data is delivered exactly once.

## Dynamic Embedding 
Since online learning is a long running job, there will be numerous brand new items in the training data and embedding for the new items will be learned.
As the embedding size grows, memory used will also grow. As a result, it is likely that an oom would occur even if a large memory is preserved at the beginning.
There should be a critirion to keep or remove items' embeddings. In addition, training framework needs to support dynamic embedding.

## Auto Resource Configuration and Scaling 
The amount of training data is varing. It's likely that there are more training data at noon and fewer at mid-night. The demand for the number worker should ensures that training data should be consumed once it arrives. Thus worker number should adjusted dymanically for the purpose of saving resource and keeping the model fresh.
In addition, if parameter servers are at a risk of failing to store the model parameters, extra parameter servers should be added to the training loop. Training framework needs to support resharding model parameters among different parameter servers.

## High Performance States Storage Backend
Training data for online learning are processed by streaming engine like Flink or a message queue like Kafka for the sake of timeliness. In the field of recommendation, 
page view stream data and click stream data should be joined together and feature engineering for the streaming data is also needed before fed to training. In order to avoid backpressure, high performace states storage backend is needed.


Onling learning provided by DLrover is decoupled from specific training framework and streaming engine by its design.

Currently, DLrover supports following features for tensorflow estimator model

- [x] Auto Resource Configuration for Training job
- [x] Worker/PS Failover
- [x] Worker AutoScaling 
- [x] PS Migration from a Slow Machine to a Fast Machine

Other kinds of model will be supported if needed.


In addition, following features is under developing
- [ ] High Performance States Storage Backend
- [ ] Model Parameters Resharding and PS AutoScaling  
- [ ] An End-to-End Online Learning Example 
- [ ] BufferService to decouple Training with Streaming








