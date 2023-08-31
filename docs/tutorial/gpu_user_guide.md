### GPU User Guide

> "The first four steps in this document need to be run on each bare-metal machine that will use a GPU.
If you've already set up each node that requires GPU usage,
or you're working in a well-maintained cloud-based Kubernetes environment,
you can directly start from step five."

#### Step 1: Prepare the system for NVIDIA GPU support

To enable NVIDIA GPU support, follow these steps to configure the system:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

#### Step 2: Install NVIDIA Container Toolkit

Use the following command to install the NVIDIA Container Toolkit:

```bash
sudo apt-get update && \
sudo apt-get install -y nvidia-container-toolkit && \
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
```

This will install the necessary components for NVIDIA GPU support in Docker,
enabling you to utilize GPU resources within Docker containers.

#### Step 3: Set NVIDIA as the default runtime for Docker

To configure Docker to use NVIDIA as the default runtime, follow these steps:

1. Edit the Docker daemon configuration file:

```bash
sudo vim /etc/docker/daemon.json
```

2. Add the following content to the file:

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

3. Save and exit the editor.

#### Step 4: Restart Docker

After making the changes, restart the Docker service for the new configuration to take effect:

```bash
sudo systemctl restart docker
```

Now, NVIDIA will be set as the default runtime for Docker,
allowing you to use NVIDIA GPU support seamlessly with Docker containers.

#### Step 5: Deploy the NVIDIA Device Plugin for Kubernetes

Use the following command to deploy the NVIDIA Device Plugin for Kubernetes:

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v1.11/nvidia-device-plugin.yml
```

This plugin enables Kubernetes to recognize and manage NVIDIA GPUs on the worker nodes,
ensuring efficient allocation and utilization of GPU resources for container workloads.

#### Step 6: Create a test Pod with GPU resources

Create a test Pod with GPU resources using the provided YAML configuration:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: tf-pod
spec:
  containers:
    - name: tf-container
      image: tensorflow/tensorflow:latest-gpu
      resources:
        limits:
          nvidia.com/gpu: 1 # requesting 1 GPU
```

The above YAML configuration requests one GPU for the Pod.
Replace the image with your desired GPU-accelerated application image if needed.

#### Step 7: Deploy the test Pod

Use the following command to deploy the test Pod to Kubernetes:

```bash
kubectl apply -f <your-yaml-file-name>.yaml
```

This will create the Pod on your Kubernetes cluster,
and the GPU resource will be allocated to the Pod based on the NVIDIA Device Plugin's capabilities.

Now, you have successfully enabled GPU support in your Kubernetes cluster
and deployed a test Pod with GPU resources for running GPU-accelerated workloads.
