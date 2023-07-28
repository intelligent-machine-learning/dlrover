# Master the Training of NanoGPT with DLRover

Welcome to an exhaustive guide on how to train the `NanoGPT` model using DLRover. 

## What's NanoGPT?

NanoGPT is a specialized version of the famous GPT (Generative Pretrained Transformer) model. What makes it unique is its role in evaluating the scalability and elasticity of the DLRover job controller. It provides the ability to tweak hyperparameters like _n_layer_, _n_head_, and _n_embedding_, making it possible to conduct tests on GPT models of varying sizes.

For a more in-depth dive into the fascinating world of NanoGPT, don't hesitate to visit [NanoGPT](https://github.com/karpathy/nanoGPT) for the source code and a plethora of other valuable resources.

## Setting Up the DLRover Job Controller

Follow the comprehensive guide in the [Controller Deployment](dlrover/docs/deployment/controller.md) document to get your DLRover job controller up and running.

## GPT Training - Let's Dive In

### Getting Started with a Sample YAML 

Starting off with your journey to evaluating the performance of DLRover, you'll be submitting multiple training jobs. This will be done using NanoGPT with a variety of parameter settings to gauge performance under different conditions.

Kick off the process with the following command:

```bash
$ kubectl -n dlrover apply -f dlrover/examples/torch_nanogpt_job.yaml
```

Upon successful application of the job configuration, you can monitor the status of the training nodes using the command below:

```bash
$ kubectl -n dlrover get pods
```

Expect an output that resembles this:

```bash
NAME                                              READY   STATUS    RESTARTS   AGE
dlrover-controller-manager-7dccdf6c4d-grmks       2/2     Running   0          12h
elasticjob-torch-nanogpt-dlrover-master.          1/1     Running   0          20s
torch-nanogpt-edljob-worker-0                     1/1     Running   0          11s
torch-nanogpt-edljob-worker-1                     1/1     Running   0          11s
```

### Building from Docker - Step by Step

**Preparing Your Data**

To begin, you need a text document which can be a novel, drama, or any textual content. For instance, you can name this document as data.txt.

Here's an example of a Shakespearean dialogue:

```
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

**Time to Run the Preparation Script**

Now that you have your data, let's run the preparation script as follows:

```bash
python dlrover/model_zoo/pytorch/nanogpt/prepare.py --src_data_path data.txt
This command generates a train.bin and val.bin file in the data directory.
```

**Building the Training Image for PyTorch Models**

Having prepared the data, the final step involves building the training image of PyTorch models. Here's how you do it:

```bash
docker build -t easydl/dlrover-train-nanogpt:test -f docker/pytorch/nanogpt.dockerfile .
```

And voila! You're all set to run the model and dive into the world of Natural Language Processing.
I hope this adds more life and detail to your README document. Let me know if there's anything else you need help with!

# References

This eaxmple is built upon and significantly influenced by the [NanoGPT](https://github.com/karpathy/nanoGPT) project. Several scripts from the project, including but not limited to `prepare.py`, `train.py`, and `model.py`, have been adapted to our specific requirements. 

The original scripts can be found in the NanoGPT repository: [NanoGPT](https://github.com/karpathy/nanoGPT)

# Acknowledgments

We would like to express our sincere gratitude to the authors and contributors of the NanoGPT project. Their work has provided us with a strong foundation for our example, and their insights have been invaluable for our development process. Thank you!
