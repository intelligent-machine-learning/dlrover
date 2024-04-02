# Config file items

Config contains three sections: model, tokenizer and train. 
Below are items can be defined in these sections. More will be added.

## train

```
epochs: <int> # default 1, train epoch for prompt data
eg_batch_size: <int> # experience generation  batchsize
rl_epochs: <int> # default 1, rl training epoch number
rl_batch_size: <int> # rl training batchsize
experience_buffer_size: <int> # the number of experience to collect before rl training.
mode: <str> # sequential(default) or concurrent. If sequential, experience_generation followed by rl_training on same devices. If concurrent, experience_generation and rl_training running concurrently on different devices.
trainer: <str> # default PPOTrainer
strategy_file: <str> a python file defines optimization strategies for all models.

```


## model

Model section contains a list of models, which defines 4 models, actor, critic, ref\_model and reward\_model, their optimization strategies, optimizers. Also supports weight sharing, so the model list is a subset of (actor, critic, ref_model, reward\_model, actor\_critic, actor\_criti\c_ref, reward\_critic, actor\_critic\_ref\_reward). If weight shared, the model class should provide corresponding forward funtions for each model forward, and multiple outputs in one forward function.


```
model:
  actor:
      model_path: <str> # Name of hf model to load or a directory of hf model, or a python file including the model definition.
      model_cls: <str> # optional, if set, a class/function in the python file specified by model_path. Init/call this class/function would return a nn.Module.
      model_params: # list of (key, value) for model params.
      train_strategy: a function in train.strategy_file for training optimization strategy.
      inference_strategy: a function in train.strategy_file for inference optimization strategy.
      optimizer: 
         name: <str>
         kwargs: # list of 

      
```


## tokenizer

```
tokenizer_path: <str> # path to tokenizer
params: #list of (key, value) for tokenizer params. such as
#  truncation_side: right
      
```