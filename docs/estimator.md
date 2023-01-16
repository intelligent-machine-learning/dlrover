# Introduction to Develop Tensorflow Estimator Model with DLRover trainer

The document describes how to develop tensorflow estimator model with DLRover trainer.

## Develop model with tensorflow estimator
[Tensorflow Estimator](https://www.tensorflow.org/guide/estimator)
encapsulate Training, Evaluation, Prediction and Export for serving actions.
In DLrover, both custome estimators and pre-made estimators are supported.

A DLrover program with Estimator typically consists of the following four steps:
### Define the features and label column in the conf

Each `Column` identifies a feature name, its type and whether it is label.
(For example)[https://github.com/intelligent-machine-learning/dlrover/tree/master/dlrover/trainer/examples/estimator_executor], the following snippet defines two feature columns. 
```
train_set = {
    "path": "fake://test.data",
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
   
### Instantiate the Estimator.
The heart of every Estimator—whether pre-made or custom—is its model function, model_fn, which is a method that builds graphs for training, evaluation, and prediction.  
In `dlrover.trainer`, we assume the Estimator is a custom estimator. And pre-made estimators should be converted to custom estimator with little overhead.
#### Train a model from custome estimators
When relying on a custom Estimator, you must write the model function yourself. Refer the [tutorial](https://www.tensorflow.org/guide/estimator).
#### Train a model from pre-made estimators 
You can convert an existing pre-made estimators by writing an Adaptor to fit with `dlrover.trainer`.
As we can see, the model_fn is the key part of estimator.
When training and evaluating, the model_fn is called with different mode and the graph is returned.
Thus, you can define a custom estimator in which model_fn function acts as a wrapper for pre-made estimator model_fn.
(For example)[https://github.com/intelligent-machine-learning/dlrover/tree/master/dlrover/trainer/examples/deepfm], `DeepFMEstimator` in [`deepctr.estimator.models`](https://github.com/shenweichen/DeepCTR/tree/master/deepctr/estimator/models) is a pre-made estimator. 

```
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
Estimators by default save checkpoints with variable names rather than the object graph described in the Checkpoint guide. 
The checkpoint hook is added by `dlrover.trainer.estimator_executor`.

### SavedModels from Estimators
Estimators export SavedModels through tf.Estimator.export_saved_model.
The exporter hook is added by `dlrover.trainer.estimator_executor`.

When the job is launched, `dlrover.trainer.estimator_executor` parses the conf and builds input_fn, estimator and related hooks.


 