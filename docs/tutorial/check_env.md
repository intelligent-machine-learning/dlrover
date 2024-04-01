# Environment Test before Start

Before you start installing this project, you need to perform the following
tests to ensure that your current computer environment meets the requirements,
so as to avoid some possible errors.

`grpcio >= 1.58.0`:

```python
import grpc

assert grpc.__version__ >= '1.58.0', f"Expected grpcio version >= 1.58.0 but found {grpc.__version__}"
```

`grpcio-tools >= 1.58.0`:

```python
import pkg_resources

grpc_tools_version = pkg_resources.get_distribution("grpcio-tools").version

assert grpc_tools_version >= '1.58.0', f"Expected grpcio-tools version >= 1.58.0 but found {grpc_tools_version}"
```

`protobuf >= 3.15.3, < 4.0`:

```python
import protobuf

assert '3.15.3' <= protobuf.__version__ < '4.0', f"Expected protobuf version >=3.15.3,<4.0 but found {protobuf.__version__}"
```

`psutil` should be installed:

```python
import importlib

assert importlib.util.find_spec("psutil") is not None, "psutil module is not installed"
```

`urllib3 < 1.27, >= 1.21.1`:

```python
import urllib3

assert '1.21.1' <= urllib3.__version__ < '1.27', f"Expected urllib3 version >=1.21.1,<1.27 but found {urllib3.__version__}"
```

`kubernetes` should be installed:

```python
import importlib

assert importlib.util.find_spec("kubernetes") is not None, "kubernetes module is not installed"
```

`ray` should be installed：

```python
import importlib

assert importlib.util.find_spec("ray") is not None, "ray module is not installed"
```
