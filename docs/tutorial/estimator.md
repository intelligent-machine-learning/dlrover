# Introduction to Develop Tensorflow Estimator Model with DLRover trainer

The document describes how to develop tensorflow estimator model with DLRover trainer.

## Develop model with tensorflow estimator

[Tensorflow Estimator](https://www.tensorflow.org/guide/estimator)
encapsulate Training, Evaluation, Prediction and Export for serving actions.
In DLrover, both custome estimators and pre-made estimators are supported.

A DLrover program with Estimator typically consists of the following four steps:

### Define the features and label column in the conf

Each `Column` identifies a feature name, its type and whether it is label.
The following snippet defines two feature columns in the
[example](../../examples/tensorflow/criteo_deeprec/train_conf.py).

```python
train_set = {
    "reader": FileReader("test.data"),
    "columns": (
        Column.create(  # type: ignore
            name="x",
            dtype="float32",
            is_label=False,
        ),
        Column.create(  # type: ignore
            name="y",
            dtype="float32",
            is_label=True,
        ),
    ),
}
```

The first feature is `x` and its type is `float32`.
The second feature is `y` and is label. Its type is `float32`.
`dlrover.trainer` helps build `input_fn` for train set and test set with those columns.

### Add Custom Reader for TF Estimator in DLrover

In some case, the reader provided by DLrover trainer doesn't satisfy user's need.
User need to develop custom reader and set it in the conf.

#### Add Custom Elastic Reader for TF Estimator in DLrover

One necessary arguments in the `__init__` method is path.
The key funcion is `read_data_by_index_range` and `count_data`. `count_data` is used for
konwing how many dataset are there before training. During training, `read_data_by_index_range`
will be called to get train data.

```python
from dlrover.trainer.tensorflow.reader.base_reader import ElasticReader
class FakeReader(ElasticReader):
    def __init__(self, path=None):
        self.count = 1
        super().__init__(path=path)

    def count_data(self):
        self._data_nums = 10

    def read_data_by_index_range(self, start_index, end_index):
        data = []
        for i in range(start_index, end_index):
            x = np.random.randint(1, 1000)
            y = 2 * x + np.random.randint(1, 5)
            d = "{},{}".format(x, y)
            data.append(d)
        return data
```

you need to initial you reader and set it in the conf. Here is an example  

```python
eval_set = {"reader": FakeReader("./eval.data"), "columns": train_set["columns"]}
```

#### Add Custom Non Elastic Reader for TF Estimator in DLrover

The key funcion is `iterator`.  During training, `iterator` will be called to get train data.

```python
class Reader:
    def __init__(
        self,
        path=None,
        batch_size=None
    ):
        pass

    def get_data(self):
        # you custom code
        while True:
            yield "1,1"

    def iterator(self):
        while True:
            for d in self.get_data():
                yield d
```

you need to initial you reader and set it in the conf. Here is an example  

```python
eval_set = {"reader": Reader("./eval.data"), "columns": train_set["columns"]}
```

### Instantiate the Estimator

The heart of every Estimator—whether pre-made or custom—is its model function, model_fn,
which is a method that builds graphs for training, evaluation, and prediction.  
In `dlrover.trainer`, we assume the Estimator is a custom estimator.
And pre-made estimators should be converted to custom estimator with little overhead.

#### Train a model from custome estimators

When relying on a custom Estimator, you must write the model function yourself.
Refer the [tutorial](https://www.tensorflow.org/guide/estimator).

#### Train a model from pre-made estimators

You can convert an existing pre-made estimators by writing an Adaptor to fit with `dlrover.trainer`.
As we can see, the model_fn is the key part of estimator.
When training and evaluating, the model_fn is called with different mode and the graph is returned.
Thus, you can define a custom estimator in which model_fn function acts as a wrapper for pre-made estimator model_fn.
In the example of [DeepFMAdaptor](../../dlrover/trainer/examples/deepfm/DeepFMAdaptor.py),
`DeepFMEstimator` in [`deepctr.estimator.models`](https://github.com/shenweichen/DeepCTR/tree/master/deepctr/estimator/models)
is a pre-made estimator.

```python
from deepctr.estimator.models.deepfm import DeepFMEstimator

class DeepFMAdaptor(tf.estimator.Estimator):
    """Adaptor"""

    def model_fn(self, features, labels, mode, params):
        '''
            featurs: type dict, key is the feature name and value is tensor.
            labels: type tensor, corresponding to the colum which `is_label` equals True.
        '''
        x =  features["x"]
        x_buckets = feature_column.bucketized_column(x, boundaries=[1, 3, 5])
        linear_feature_columns = [x_buckets]
        dnn_feature_columns = [x]
        self.estimator = DeepFMEstimator(
            linear_feature_columns,
            dnn_feature_columns,
            task=params["task"],
        )
        return self.estimator._model_fn(
            features, labels, mode, self.run_config
        )

```

### Saving object-based checkpoints with Estimator

Estimators by default save checkpoints with variable names rather than the
object graph described in the Checkpoint guide.
The checkpoint hook is added by `dlrover.trainer.estimator_executor`.

### SavedModels from Estimators

Estimators export SavedModels through tf.Estimator.export_saved_model.
The exporter hook is added by `dlrover.trainer.estimator_executor`.

When the job is launched, `dlrover.trainer.estimator_executor` parses the conf and builds input_fn,
estimator and related hooks.

## Submit a Job to Train the Estimator model

### Build an Image with Models

You can install dlrover in your image.

```bash
pip install dlrover[tensorflow] - U
```

Or you also can build your image from the dlrover base image.

```dockerfile
FROM registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:deeprec_criteo_v1
COPY model_zoo /home/model_zoo
```

```bash
docker build -t ${IMAGE_NAME} -f ${DockerFile} .
docker push ${IMAGE_NAME} 
```

### Set the Command to Train the Model

We need to set the command of ps and worker to train the model like the
[DeepCTR example](../../examples/tensorflow/criteo_deeprec/autoscale_job.yaml)

```yaml
command:
    - /bin/bash
    - -c
    - " cd ./examples/tensorflow/criteo_deeprec \
        && python -m dlrover.trainer.entry.local_entry \
        --platform=Kubernetes --conf=train_conf.TrainConf \
        --enable_auto_scaling=True"
```

Then, we can submit the job by `kubectl`.

```bash
kubectl -n dlrover apply -f ${JOB_YAML_FILE}
```
