# Master the Training of NanoGPT with DLRover

Welcome to an exhaustive guide on how to train the `NanoGPT` model using DLRover.

## What's NanoGPT?

NanoGPT is a specialized version of the famous GPT (Generative Pretrained Transformer) model.
What makes it unique is its role in evaluating the scalability and elasticity of the DLRover job controller.
It provides the ability to tweak hyperparameters like _n_layer_, _n_head_, and _n_embedding_,
making it possible to conduct tests on GPT models of varying sizes.

For a more in-depth dive into the fascinating world of NanoGPT, don't hesitate to visit [NanoGPT](https://github.com/karpathy/nanoGPT)
for the source code and a plethora of other valuable resources.

## Local GPT Training

- Pull the image with the model and data.

```bash
docker pull registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:pytorch-example
docker run -it registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:pytorch-example bash
cd /dlrover/examples/pytorch/nanogpt/
```

- Local run the training by `dlrover-run`

```bash
dlrover-run --nnodes=1 --max_restarts=2 --nproc_per_node=2 \
    train.py --n_layer 48 --n_head 16 --n_embd 1600 \
    --data_dir './' --epochs 50 --save_memory_interval 50 \
    --save_storage_interval 500
```

You also can run the FSDP and DeepSpeed example by using the `fsdp_train.py` and `ds_train.py`.

## Distributed GPT Training on k8s - Let's Dive In

### Setting Up the DLRover Job Controller

Follow the comprehensive guide in the [Controller Deployment](dlrover/docs/deployment/controller.md)
document to get your DLRover job controller up and running.

### Getting Started with a Sample YAML

Starting off with your journey to evaluating the performance of DLRover, you'll be submitting multiple training jobs.
This will be done using NanoGPT with a variety of parameter settings to gauge performance under different conditions.

Kick off the process with the following command:

```bash
kubectl -n dlrover apply -f  examples/pytorch/nanogpt/elastic_job.yaml
```

Upon successful application of the job configuration,
you can monitor the status of the training nodes using the command below:

```bash
kubectl -n dlrover get pods
```

Expect an output that resembles this:

```bash
NAME                                              READY   STATUS    RESTARTS   AGE
dlrover-controller-manager-7dccdf6c4d-grmks       2/2     Running   0          12h
elasticjob-torch-nanogpt-dlrover-master.          1/1     Running   0          20s
torch-nanogpt-edljob-worker-0                     1/1     Running   0          11s
torch-nanogpt-edljob-worker-1                     1/1     Running   0          11s
```

### Examine the results obtained from two different parameter settings

parameter settings 1:

```bash
# parameter settings in examples/pytorch/nanogpt/ddp_elastic_job.yaml
--n_layer 6 \
--n_head 6 \
--n_embd 384
```

parameter settings 2:

```bash
# parameter settings in examples/pytorch/nanogpt/ddp_elastic_job.yaml
--n_layer 12 \
--n_head 12 \
--n_embd 768
```

#### More detailed description of the pods

Worker-0 Logs

```bash
kubectl logs -n dlrover torch-nanogpt-edljob-worker-0
```

results with parameter settings 1:

```text
iter 0: loss 4.2279, time 4542.46ms, mfu -100.00%, lr 6.00e-04, total time 4.54s
iter 1: loss 3.5641, time 4439.20ms, mfu -100.00%, lr 6.00e-04, total time 8.98s
iter 2: loss 4.2329, time 4477.08ms, mfu -100.00%, lr 6.00e-04, total time 13.46s
iter 3: loss 3.6564, time 4579.50ms, mfu -100.00%, lr 6.00e-04, total time 18.04s
iter 4: loss 3.5026, time 4494.54ms, mfu -100.00%, lr 6.00e-04, total time 22.53s
iter 5: loss 3.2993, time 4451.15ms, mfu 0.33%, lr 6.00e-04, total time 26.98s
iter 6: loss 3.3318, time 4391.21ms, mfu 0.33%, lr 6.00e-04, total time 31.38s
```

results with parameter settings 2:

```text
iter 0: loss 4.4201, time 31329.07ms, mfu -100.00%, lr 6.00e-04, total time 31.33s
iter 1: loss 4.6237, time 30611.01ms, mfu -100.00%, lr 6.00e-04, total time 61.94s
iter 2: loss 6.7593, time 30294.34ms, mfu -100.00%, lr 6.00e-04, total time 92.23s
iter 3: loss 4.2238, time 30203.78ms, mfu -100.00%, lr 6.00e-04, total time 122.44s
iter 4: loss 6.1183, time 30100.29ms, mfu -100.00%, lr 6.00e-04, total time 152.54s
iter 5: loss 5.0796, time 30182.75ms, mfu 0.33%, lr 6.00e-04, total time 182.72s
iter 6: loss 4.5217, time 30303.39ms, mfu 0.33%, lr 6.00e-04, total time 213.02s
```

Worker-1 Logs

```bash
kubectl logs -n dlrover torch-nanogpt-edljob-worker-1
```

results with parameter settings 1:

```text
iter 0: loss 4.2382, time 4479.40ms, mfu -100.00%, lr 6.00e-04, total time 4.48s
iter 1: loss 3.5604, time 4557.53ms, mfu -100.00%, lr 6.00e-04, total time 9.04s
iter 2: loss 4.3411, time 4408.12ms, mfu -100.00%, lr 6.00e-04, total time 13.45s
iter 3: loss 3.7863, time 4537.51ms, mfu -100.00%, lr 6.00e-04, total time 17.98s
iter 4: loss 3.5153, time 4489.47ms, mfu -100.00%, lr 6.00e-04, total time 22.47s
iter 5: loss 3.3428, time 4567.38ms, mfu 0.32%, lr 6.00e-04, total time 27.04s
iter 6: loss 3.3700, time 4334.36ms, mfu 0.32%, lr 6.00e-04, total time 31.37s
```

results with parameter settings 2:

```text
iter 0: loss 4.4402, time 31209.29ms, mfu -100.00%, lr 6.00e-04, total time 31.21s
iter 1: loss 4.5574, time 30688.11ms, mfu -100.00%, lr 6.00e-04, total time 61.90s
iter 2: loss 6.7668, time 30233.15ms, mfu -100.00%, lr 6.00e-04, total time 92.13s
iter 3: loss 4.2619, time 30400.66ms, mfu -100.00%, lr 6.00e-04, total time 122.53s
iter 4: loss 6.2001, time 29960.20ms, mfu -100.00%, lr 6.00e-04, total time 152.49s
iter 5: loss 5.0426, time 30222.85ms, mfu 0.32%, lr 6.00e-04, total time 182.71s
iter 6: loss 4.5057, time 30200.79ms, mfu 0.32%, lr 6.00e-04, total time 212.92s
```

### Building from Docker - Step by Step

### Preparing Your Data

To begin, you need a text document which can be a novel, drama, or any textual content.
For instance, you can name this document as data.txt.

Here's an example of a Shakespearean dialogue:p

```text
BUCKINGHAM:
Welcome, sweet prince, to London, to your chamber.

GLOUCESTER:
Welcome, dear cousin, my thoughts' sovereign
The weary way hath made you melancholy.

PRINCE EDWARD:
No, uncle; but our crosses on the way
Have made it tedious, wearisome, and heavy
I want more uncles here to welcome me.
```

Alternatively, you can use our provided data, which is available in the [examples/pytorch/nanogpt/data.txt](examples/pytorch/nanogpt/data.txt).
This data has already been prepared for use.

### Time to Run the Preparation Script

Now that you have your data, let's run the preparation script as follows:

```bash
python examples/pytorch/nanogpt/prepare.py --src_data_path data.txt
This command generates a train.bin and val.bin file in the data directory.
```

### Building the Training Image for PyTorch Models

Having prepared the data, the final step involves building the training image of PyTorch models. Here's how you do it:

```bash
docker build -t easydl/dlrover-train-nanogpt:test -f docker/pytorch/nanogpt.dockerfile .
```

And voila! You're all set to run the model and dive into the world of Natural Language Processing.
I hope this adds more life and detail to your README document. Let me know if there's anything else you need help with!

## References

This eaxmple is built upon and significantly influenced by the [NanoGPT](https://github.com/karpathy/nanoGPT) project.
Several scripts from the project, including but not limited to `prepare.py`, `train.py`, and `model.py`,
have been adapted to our specific requirements.

The original scripts can be found in the NanoGPT repository: [NanoGPT](https://github.com/karpathy/nanoGPT)

## Acknowledgments

We would like to express our sincere gratitude to the authors and contributors of the NanoGPT project.
Their work has provided us with a strong foundation for our example,
and their insights have been invaluable for our development process. Thank you!
