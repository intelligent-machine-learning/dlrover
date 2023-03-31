# Add Custom Reader for TF Estimator in DLrover
In some case, the reader provided by DLrover trainer doesn't satisfy user's need.
User need to develop custom reader and set it in the conf.


## Add Custom Elastic Reader for TF Estimator in DLrover
### Define Elastic Reader Class
One necessary arguments in the `__init__` method is path.
The key funcion is `read_data_by_index_range` and `count_data`. `count_data` is used for konwing how many dataset are there before training. During training, `read_data_by_index_range` will be called to get train data.

    
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

### Set Reader Conf file
you need to initial you reader and set it in the conf. Here is an example  

```python
eval_set = {"reader": FakeReader("./eval.data"), "columns": train_set["columns"]}
```

## Add Custom Non Elastic Reader for TF Estimator in DLrover

### Define Reader Class

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

    def iterator(self):
        while True:
            data = self.get_data()
            for d in data:
                yield d
```

### Set Reader Conf file
you need to initial you reader and set it in the conf. Here is an example  
```python
eval_set = {"reader": FakeReader("./eval.data"), "columns": train_set["columns"]}
```