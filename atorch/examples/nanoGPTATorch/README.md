# nanoGPTATorch

NanoGPT is the simplest, fastest repository for training/finetuning medium-sized GPTs. This example is modified from nanoGPT for the adaptation of atorch, see [source repo](https://github.com/karpathy/nanoGPT) for detail.

## requirements

```
pip install torch numpy transformers datasets tiktoken tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `tqdm` for progress bars <3


## Usage

The example uses the default config values designed to train a gpt2 (124M) on OpenWebText,  you can check and change config values in train_atorch.py. Several default config values:

- backend: "nccl"  # 'nccl', 'gloo', etc.
- device: "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
- dtype: "float16" 
- train_type: "fsdp"

### Scripts
The script will download openwebtext dataset firstly, then launch atorch training.

```
bash train_atorch_entry.sh output_dir
```