代码修改自huggingface的[问答任务](https://github.com/huggingface/transformers/blob/v4.31.0/examples/pytorch/question-answering/README.md)

squad_v2数据集  
https://huggingface.co/datasets/squad_v2


bert-large-cased预训练模型  
https://huggingface.co/google-bert/bert-large-cased

启动训练：
```bash
bash run.sh
```

启动前，需要把`run.sh`中`DATASET_DIR`和`PRETRAINED_MODEL_DIR`设置为实际的数据和预训练模型路径。