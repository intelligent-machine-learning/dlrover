### User's manual

1. Init instance with args.
2. Record spike loss with save_loss func.
3. Decode loss with decode_loss_spike func.

### Example code

```python
from atorch.utils.loss_spike_utils import TokenLossSpike


loss_spike_ins = TokenLossSpike(
    loss_spike_save_dir="loss_spike_save_dir", 
    sample_data_paths=[("wikipedia", "corpus/base"), ("zhihu", "/dataset/fd5061f6/data/tokenize_data/zhihu.lazy")], 
    each_sample_len=4, 
    min_iter=2000, 
    min_loss=10,
    loss_info_splitter='\t', 
    loss_sample_str_splitter=','
)

loss_spike_ins.save_loss(file_name="", 
                         cur_loss=4, 
                         cur_iter=100, 
                         losses_str="2.44,2.33,4.05", 
                         sample_infos_str="20-1-1385697-14158189-2,20-1-1385697-14158189-2,20-1-1385697-14158189-2")

loss_spike_ins.decode_loss_spike(result_file_path="", 
                                 tokenizer=None, 
                                 min_iter=None, 
                                 min_loss=None)
```

### Parameter interpretation

```python
loss_spike_ins = TokenLossSpike(
    loss_spike_save_dir="loss_spike_save_dir", 
    sample_data_paths=[("wikipedia", "corpus/base"), ("zhihu", "/dataset/fd5061f6/data/tokenize_data/zhihu.lazy")], 
    each_sample_len=4, 
    min_iter=2000, 
    min_loss=10,
    loss_info_splitter='\t', 
    loss_sample_str_splitter=','
)
```

1. loss_spike_save_dir, the directory storing loss files, make sure it is an **existing directory**.
2. sample_data_paths, each_sample_len, using in decode_loss_spike func: **run save_loss with None input**.
   1. sample_data_paths: the map for sample data and their file path.
   2. each_sample_len: single sample len for decoding sample data.
3. min_iter, min_loss
   1. min_iter, Iterations greater than min iter are recorded.
   2. min_loss, Loss greater than min loss are recorded.
4. loss_info_splitter, f"{ctime}\t{iteration}\t{loss}\t{loss_str}\t{sample_infos_str}\n"
5. loss_sample_str_splitter, default value is “,”, **information is passed in by the user, so the user needs to match it**.

```python
loss_spike_ins.save_loss(file_name="", 
                         cur_loss=4, 
                         cur_iter=100, 
                         losses_str="2.44,2.33,4.05", 
                         sample_infos_str="20-1-1385697-14158189-2,20-1-1385697-14158189-2,20-1-1385697-14158189-2")
```

1. file_name, the file for saving loss.
2. cur_loss, the mean loss.
3. cur_iter, current iteration.
4. losses_str, the batch loss connected with splitter.
5. sample_infos_str, the batch sample info(**This structure is fixed in the current implementation of the inheritance class, and then if there are other formats, we use other inheritance class implementation, or abstract parameters**).

```python
loss_spike_ins.decode_loss_spike(result_file_path="", 
                                 tokenizer=None, 
                                 min_iter=None, 
                                 min_loss=None)
```

1. result_file_path, the file saving decoded sample.
2. tokenizer, default value is None, if exist, the sample will decode by tokenizer.decode func.
3. min_iter, min_loss.
