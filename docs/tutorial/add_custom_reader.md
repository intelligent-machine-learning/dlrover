# Add Custom Reader for TF Estimator in DLrover
In some case, the reader provided by DLrover trainer doesn't satisfy user's need.
User need to develop custom and register it to the 'reader_registery'.capitalize

Here is an example, 

## Define Reader Class
Arguments in the `__init__` method are path, num_epochs, batch_size  and enable_dynamic_sharding.
As for some specific parameter, it can be defined as `python class attribute`.
The key funcion is `iterator`. During training, `iterator` will be called to get train data.


    
```python
from dlrover.trainer.tensorflow.reader.base_reader import (  # noqa: F401
    ElasticReader,
)
class CustomReader(BaseReader):
    sepeicif_param = "1"

    def __init__(
        self,
        path=None,
        num_epochs=1,
        batch_size=64,
        enable_dynamic_sharding=True,
    ):
        
        # your custome code 
        # your custome code
        super().__init__(path=path,
                         num_epochs=num_epochs,
                         batch_size=batch_size, 
                         enable_dynamic_sharding=enable_dynamic_sharding)
    def count_data(self):
        '''
        Count how many training datas are there
        '''
        # your custome code 
        self._data_nums = 1000

    def _read_data(self):
        '''
        Read data from data source.
        '''
        data = None
        # your custom code
        if self.data_shard_client is not None:
            shard = self.data_shard_client.fetch_shard()
            start_index, end_index = shard.start, shard.end
            # your custom code, read data from start_index to end_index
            # data = read_data(start_index, end_index)
            data =  
        else:
            # you can sequentially read data without getting shard from dlrover master
            data = 
        return data 

    def iterator(self):
        while True:
            data = self._read_data()
            if data is None:
                break
            yield data
```

## Register The Reader Class 
Register the Reader to the registery, `custome_reader` is the key and `CustomReader` is the reader Class.
```python
from dlrover.trainer.tensorflow.util.reader_util import reader_registery
reader_registery.register_reader("custome_reader", CustomReader)
```


## Set reader type and path in Conf file
you need to set path for dataset. Here is an example `custome_reader://eval.data`. `custome_reader` is the key to get reader class and `//eval.data` is the path of the custom file system.
```python
eval_set = {"path": "custome_reader://eval.data", "columns": train_set["columns"]}
```