# Streaming DataShardManger and Splitter

The design describes the architecture of the Streaming DataShardManger.
The Streaming DataShardManger is responsible for dispatching data and keep data checkpoints.

## An Intro to Online learning

Online learning represents a family of machine learning methods, where a learner attempts
to tackle some predictive task by learning from a sequence of data instances one by one
at each time. In contrast, offline/batch learner
learns from static shuffled data samples and are not sensitive to the data sequence.

Online learning has become a promising technique for learning from continuous
streams of data in many real-world applications.

Thus, the key point for online learning the data should be dispatched sequentially and consumed at least once.

## PartitionOffsets

Stream processing is the processing of data in motion, or in other words,
computing on data directly as it is produced or received.
In addition, we would never know how many training samples are in advance and when they would arrive.
As a result, the worker and ps keeps running and waiting for the upstream sample.

PartitionOffsets is responsible for holding consuming status of streaming data.

```Python
class PartitionOffsets(object):

    def __init__(self, partition_offsets):
        self.partition_offsets = partition_offsets
        self.start_partition = 0
        self.partitions = []
        self.partition_num = 0
        self.update_partitions()
```

## Streaming Data Splitter

The streaming data splitter assumes that streaming samples are stored in different partition and every sample is marked
with an offset which indicates the sample's sequence. Streaming data splitter is responsible for creating shards.
The shard contains offset ranges [start, end) and partition of records.

## Streaming DataShardManger Checkpoints

- When doing checkpoints, Streaming DataShardManger saves not only current doing tasks and
undo tasks but also the splitter info.
- When restoring from checkpoints, Streaming DataShardManger loads not only current doing
tasks and undo tasks but also the splitter info.

## Streaming Reader

As for getting training data,there are two kind of modes of online learning:

- The training data is stored in the log store or kafka topic, the reader reads data from the log store or topic.
- The training data is processed by a streaming job and sink of the job sends the data to a buffer. The reader
reads data from the buffer. By this means, the worker is decoupled with data source.

In conclusion, the worker is stateless in both online learning and offline learning.
