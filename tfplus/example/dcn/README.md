# DCN
This directory's train.py showcases an example of training a Deep & Cross Network (DCN) through Kv Variable and custom optimizer (including Adam, Adagrad, Group Adam, and Group Adagrad) from TFPlus to perform CTR prediction tasks.
>The implementation of this example is referenced from the DeepRec repository, see: https://github.com/DeepRec-AI/DeepRec/tree/main/modelzoo/dcn
## Model Architecture
[Deep & Cross Network](https://arxiv.org/abs/1708.05123)(DCN) is proposed by Google by 2017.  
## Usage
### Stand-alone Training
1.  Install TFPlus
Firstly, you need to install TFPlus according to the instructions in [README](../../README.md). You can manually build TFPlus locally or use Docker image of TFPlus (recommended).
2.  Download Criteo Dataset
In this example, the Train & eval dataset uses Kaggle Display Advertising Challenge Dataset (Criteo Dataset). Due to the large size of the Criteo Dataset, you need to manually download the dataset.  
Download the train/eval dataset(in csv format) from
  - [train dataset download link](https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv) 
  - [eval dataset download link](https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv)  
    After download, put data file **train.csv** & **eval.csv** into ./data/
3. Training
```python
python train.py
```
You can use the `--tf` argument to disable TFPlus features and use the community tensorflow to perform regular training.
```python
python train.py --tf
```  
Use arguments to set up a custom configuation:
  - TFPlus Features:
      - `--optimizer`: Choose the optimizer for deep model from ['adam', 'adagrad', 'group_adam', 'sparse_group_ftrl']. Use adam by default. These optimizers are custom versions from TFPlus, and are compatible with Kv Variable.
  - Basic Settings:
      - `--data_location`: Full path of train & eval data, default to `./data`.
      - `--steps`: Set the number of steps on train dataset. Default will be set to 1 epoch.
      - `--no_eval`: Do not evaluate trained model by eval dataset.
      - `--batch_size`: Batch size to train. Default to 2048.
      - `--output_dir`: Full path to output directory for logs and saved model, default to `./result`.
      - `--checkpoint`: Full path to checkpoints input/output directory, default to `$(OUTPUT_DIR)/model_$(MODEL_NAME)_$(TIMESTAMPS)`
      - `--save_steps`: Set the number of steps on saving checkpoints, zero to close. Default will be set to 0.
      - `--seed`: Set the random seed for tensorflow.
      - `--timeline`: Save steps of profile hooks to record timeline, zero to close, defualt to 0.
      - `--keep_checkpoint_max`: Maximum number of recent checkpoint to keep. Default to 1.
      - `--learning_rate`: Learning rate for network. Default to 0.1.
      - `--interaction_op`: Choose interaction op before top MLP layer('dot', 'cat'). Default to cat.
      - `--inter`: Set inter op parallelism threads. Default to 0.
      - `--intra`: Set intra op parallelism threads. Default to 0.
      - `--tf`: Use TF 2.13.0 embedding API and disable TFPlus features.

## Benchmark
The results below utilize the default parameters of train.py, with only the learning rate and optimizer changed.
<table>
    <tr>
        <td colspan="1"></td>
        <td>Optimizer</td>
        <td>learning rate</td>        
        <td>Accuracy</td>
        <td>AUC</td>
    </tr>
    <tr>
        <td rowspan="5">DCN</td>
        <td>TFPlus w/ Adam</td>
        <td>0.001</td>        
        <td>0.77356</td>
        <td>0.76201</td>
    </tr>
    <tr>
        <td>TFPlus w/ Adagard</td>
        <td>0.001</td>            
        <td>0.74652</td>
        <td>0.69305</td>
    </tr>
    <tr>
        <td>TFPlus w/ Adagard</td>
        <td>0.1</td>            
        <td>0.72445</td>
        <td>0.64390</td>
    </tr>
    <tr>
        <td>TFPlus w/ Group Adam</td>
        <td>0.001</td>             
        <td>0.76340</td>
        <td>0.76197</td>
    </tr>
    <tr>
        <td>TFPlus w/ Group Adagrad</td>
        <td>0.1</td>             
        <td>0.75896</td>
        <td>0.75370</td>
    </tr>
</table>

## Dataset Introduction
Train & eval dataset using ***Kaggle Display Advertising Challenge Dataset (Criteo Dataset)***.
### Fields
Total 40 columns:  
**[0]:Label** - Target variable that indicates if an ad was clicked or not(1 or 0)  
**[1-13]:I1-I13** - A total 13 columns of integer continuous features(mostly count features)  
**[14-39]:C1-C26** - A total 26 columns of categorical features. The values have been hashed onto 32 bits for anonymization purposes.

Integer column's distribution is as follow:
| Column | 1    | 2     | 3     | 4   | 5       | 6      | 7     | 8    | 9     | 10  | 11  | 12   | 13   |
| ------ | ---- | ----- | ----- | --- | ------- | ------ | ----- | ---- | ----- | --- | --- | ---- | ---- |
| Min    | 0    | -3    | 0     | 0   | 0       | 0      | 0     | 0    | 0     | 0   | 0   | 0    | 0    |
| Max    | 1539 | 22066 | 65535 | 561 | 2655388 | 233523 | 26279 | 5106 | 24376 | 9   | 181 | 1807 | 6879 |

Categorical column's numbers of types is as follow:
| column | C1   | C2  | C3      | C4     | C5  | C6  | C7    | C8  | C9  | C10   | C11  | C12     | C13  | C14 | C15   | C16     | C17 | C18  | C19  | C20 | C21     | C22 | C23 | C24    | C25 | C26   |
| ------ | ---- | --- | ------- | ------ | --- | --- | ----- | --- | --- | ----- | ---- | ------- | ---- | --- | ----- | ------- | --- | ---- | ---- | --- | ------- | --- | --- | ------ | --- | ----- |
| nums   | 1396 | 553 | 2594031 | 698469 | 290 | 23  | 12048 | 608 | 3   | 65156 | 5309 | 2186509 | 3128 | 26  | 12750 | 1537323 | 10  | 5002 | 2118 | 4   | 1902327 | 17  | 15  | 135790 | 94  | 84305 |

### Processing
- Interger columns **I[1-13]** is processed with `tf.feature_column.numeric_column()` function, and the data is normalized.  
    In order to save time, the data required for normalization has been calculated in advance.
- Categorical columns **C[1-26]** is processed with `tfplus.kv_variable.python.ops.variable_scope.get_kv_variable()` function after using `tf.strings.to_hash_bucket_fast()` function.