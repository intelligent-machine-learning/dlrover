# Introduction how to develop DLRover by Helm

The document describes how to deploy DLRover via Helm.

## Preparations
### Install Helm
```shell
xxxx
```

### Prepare Images
Prepare images for amd64/arm64 architectures according to the cluster architecture:
- kube-rbac-proxy image, e.g., kube-rbac-proxy:v0.13.1.
- dlrover-controller-manager image, e.g., dlrover-controller-manager:master.
- dlrover-master image, e.g., dlrover-master:v0.7.0.

Then push them to the corresponding repository or local registry.
## Installation and Deployment
### Uninstall Historical Versions (Optional)
If DLRover has been installed via Helm before, uninstall it first with the following commands:
```shell
helm list -n dlrover
helm uninstall dlrover-0-123456 -n dlrover
```

### Install the New Version
#### Replace Images
Pull the latest DLRover code, enter the dlrover/helm_install/templates directory, and replace the kube-rbac-proxy, dlrover-controller-manager, and dlrover-master images in the deployment.yaml file.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    control-plane: controller-manager
  name: dlrover-controller-manager
  namespace: {{ .Release.Namespace }}
spec:
  replicas: 1
  selector:
    matchLabels:
      control-plane: controller-manager
  template:
    metadata:
      annotations:
        kubectl.kubernetes.io/default-container: manager
      labels:
        control-plane: controller-manager
    spec:
      containers:
      - args:
        - --secure-listen-address=0.0.0.0:8443
        - --upstream=http://127.0.0.1:8080/
        - --logtostderr=true
        - --v=0
        image: docker.io/kubebuilder/kube-rbac-proxy:v0.13.1 # replace image
        name: kube-rbac-proxy
        ports:
        - containerPort: 8443
          name: https
          protocol: TCP
        resources:
          limits:
            cpu: 500m
            memory: 128Mi
          requests:
            cpu: 5m
            memory: 64Mi
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
      - args:
        - --health-probe-bind-address=:8081
        - --metrics-bind-address=127.0.0.1:8080
        - --leader-elect
        - --master-image=dlrover-master:v0.7.0 # # replace image
        command:
        - /manager
        image: "dlrover-controller-manager:master" # replace image
        imagePullPolicy: IfNotPresent
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8081
          initialDelaySeconds: 15
          periodSeconds: 20
        name: manager
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          limits:
            cpu: 500m
            memory: 128Mi
          requests:
            cpu: 10m
            memory: 64Mi
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
      securityContext:
        runAsNonRoot: true
      serviceAccountName: dlrover-controller-manager
      terminationGracePeriodSeconds: 10

```

#### Run the installation command in the dlrover/helm_install/ directory
When installing the new version, use --create-namespace to specify if the namespace does not exist; it can be omitted if the namespace already exists.
```shell
helm install dlrover . --create-namespace -n dlrover --generate-name --wait
```


## Check the Deployment Result
