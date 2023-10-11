import builtins
import collections
import functools
import inspect
import math
import operator
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

import torch
from torch import nn
from torch.fx.node import Node
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from atorch.utils.sharding_spec import _extract_sharding_spec

try:
    from torch.fx import Graph, Proxy, Tracer
    from torch.fx.proxy import ParameterProxy
except ImportError:
    raise RuntimeError("torch version too low, no fx tool")

from atorch.common.log_utils import default_logger as logger
from atorch.utils.graph_transform_utils import map_aggregate
from atorch.utils.meta_overrides import _DEVICE, _MANUAL_META_OVERRIDES
from atorch.utils.version import torch_version


class MetaProxy(Proxy):
    """
    Proxy that uses metadata to handle data-dependent control-flow.

    In the current implementation, we choose to not return any metadata in shape/size/.. calls,
    and rely on the capability of __iter__ to correctly handle iter calls.
    Keep the original implementations in the comment for comparison
    """

    def install_metadata(self, metadata):
        self._metadata = metadata

    @property
    def shape(self):
        # if hasattr(self, "_metadata") and self._metadata is not None:
        #     return self._metadata.shape
        return self.tracer.create_proxy("call_function", builtins.getattr, (self, "shape"), {})

    def size(self, dim=None):
        """We choose not to return metadata data here, as this would fix important parameters of this module,
        which could be undesirable.
        To further support this, we overrides the __iter__ method on MetaProxy
        """
        # if hasattr(self, "_metadata") and self._metadata is not None:
        #     return self._metadata.size(*[dim] if dim else [])
        return self.tracer.create_proxy("call_method", "size", (self, dim) if dim else (self,), {})

    def dim(self):
        # if hasattr(self, "_metadata") and self._metadata is not None:
        #     return self._metadata.dim()
        return self.tracer.create_proxy("call_method", "dim", (self,), {})

    @property
    def dtype(self):
        # if hasattr(self, "_metadata") and self._metadata is not None:
        #     return self._metadata.dtype
        return self.tracer.create_proxy("call_function", builtins.getattr, (self, "dtype"), {})

    @property
    def device(self):
        # Hack so we can track when devices are used. During meta-tensor propagation,
        # replace these values with a constant 'meta'
        return MetaDeviceAttribute(self, "device")

    def __len__(self):
        if hasattr(self, "_metadata") and self._metadata is not None:
            return len(self._metadata)
        return super().__len__()

    def __bool__(self):
        if hasattr(self, "_metadata"):
            return bool(self._metadata)
        return super().__bool__()

    def __iter__(self):
        def is_iterable(obj):
            try:
                iter(obj)
                return True
            except TypeError:
                return False

        if hasattr(self, "_metadata") and is_iterable(self._metadata):
            num_elements = len(self._metadata)
            for i in range(num_elements):
                # Create a 'getitem' node in the graph for each index
                getitem_node = self.tracer.create_proxy("call_function", operator.getitem, (self, i), {})
                # Yield the proxy corresponding to the 'getitem' node
                yield getitem_node
        else:
            return super().__iter__()

    def __getattr__(self, k):
        if k == "_metadata":
            return self.__getattribute__(k)
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return MetaAttribute(self, k)

    def __setitem__(self, indices, values):
        return self.tracer.create_proxy("call_function", operator.setitem, (self, indices, values), {})

    def __contains__(self, key):
        # To handle cases such as :
        # `"some_key" in kwargs`
        if self.node.op == "placeholder":
            return False
        return super().__contains__(key)


class MetaAttribute(MetaProxy):
    def __init__(self, root, attr):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy("call_function", getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)


class MetaDeviceAttribute(MetaAttribute):
    pass


def _proxies_to_metas(v):
    """Returns the underlying metadata for MetaProxies, and behaves like the identity for the others."""
    if isinstance(v, MetaDeviceAttribute):
        return _DEVICE
    if isinstance(v, torch.fx.Proxy):
        if not (isinstance(v, MetaProxy) and hasattr(v, "_metadata")):
            raise RuntimeError(f"No metadata was found for {v}")
        return v._metadata
    return v


def _gen_constructor_wrapper(target):
    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        def check_has_proxy(v):
            if isinstance(v, Proxy):
                nonlocal proxy
                proxy = v

        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            return target(*args, **kwargs)

    return wrapper, target


class MetaTracer(Tracer):
    """
    Tracer that is able to symbolically trace models given train data.
    To do that, it uses the MetaProxy instead of the regular PyTorch torch.fx.Proxy.
    Metadata are computed on meta
    Adapted from HFTracer (transformers library)
    """

    # Feature flag for proxying accesses to buffer values
    proxy_buffer_attributes: bool = True
    allow_insert_stateless_mods: bool = True
    _TORCH_METHODS_TO_PATCH = [
        "arange",
        "zeros",
        "ones",
        "full",
        "full_like",
        "eye",
        "empty",
    ]

    def __init__(self, autowrap_modules=(math,), autowrap_functions=()):
        self.registered_leaf_module = list()
        self.meta_overrides = _MANUAL_META_OVERRIDES
        version_tuple = torch_version()
        self.major, self.minor = int(version_tuple[0]), int(version_tuple[1])
        self.trace_asserts = True
        if self.major == 1 and self.minor < 10:
            super().__init__(autowrap_modules=autowrap_modules)
        else:
            super().__init__(autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions)

    def register_leaf_modules(self, modules: List[nn.Module]):
        if modules is not None:
            self.registered_leaf_module = self.registered_leaf_module + modules

    def register_meta_overrides(self, overrides):
        self.meta_overrides.update(overrides)

    def _remove_concrete_arg_nodes(self, graph, input_names):
        for node in graph.nodes:
            if node.op == "placeholder":
                # Removing default values for inputs as the forward pass will fail with them.
                if node.target in input_names:
                    node.args = ()
                    # Without this, torch.jit.script fails because the inputs type is Optional[torch.Tensor].
                    # It cannot infer on the attributes and methods the input should have, and fails.
                    node.type = torch.Tensor
                # It is a concrete arg so it is not used and should be removed.
                else:
                    if self.major == 1 and self.minor < 10:
                        graph.erase_node(node)
                    else:
                        to_visit = [node]
                        to_delete = collections.OrderedDict()
                        while to_visit:
                            n = to_visit.pop(0)
                            to_delete[n] = None
                            to_visit += list(n.users.keys())

                        for user in reversed(to_delete.keys()):
                            self.graph.erase_node(user)

            # TODO: solves GraphModule creation.
            # Without this, return type annotation "Tuple" is causing code execution failure.
            if node.op == "output":
                node.type = None

        return graph

    def _remove_isolated_subgraphs(self, graph):
        to_keep = dict()
        backward_stack = list()
        tracked_nodes = dict()
        for node in reversed(graph.nodes):
            if node.op == "output":

                def _extract_tensormeta(input_):
                    if isinstance(input_, torch.fx.Node):
                        return input_.meta.get("tensor_meta", None)
                    else:
                        return None

                # output nodes only has args, with args[0] being the exact output
                args_tensormeta = map_aggregate(node.args, _extract_tensormeta)
                node.meta["tensor_meta"] = args_tensormeta[0]

                def _extract_output_spec(input_):
                    if isinstance(input_, torch.fx.Node):
                        return input_.meta["output_sharding_spec"]
                    else:
                        return None

                output_spec = map_aggregate(node.args, _extract_output_spec)
                node.meta["output_sharding_spec"] = output_spec[0]
                input_spec = {input_node.name: _extract_output_spec(input_node) for input_node in node.all_input_nodes}
                node.meta["input_sharding_spec"] = input_spec
                backward_stack.append(node)
                tracked_nodes[node] = True

        while len(backward_stack) != 0:
            cur_node = backward_stack.pop()
            to_keep[cur_node] = True
            for node in cur_node.all_input_nodes:
                if not tracked_nodes.get(node, False):
                    backward_stack.append(node)
                    tracked_nodes[node] = True

        for node in reversed(graph.nodes):
            if not to_keep.get(node, False) and node.op != "placeholder":
                graph.erase_node(node)

        return graph

    def create_proxy(
        self,
        kind,
        target,
        args,
        kwargs,
        name=None,
        type_expr=None,
        proxy_factory_fn=None,
    ):
        if self.major == 1 and self.minor < 10:
            rv = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        else:
            rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

        metadata = None
        args_metas = ()
        kwargs_metas = dict()

        if kind == "placeholder" and target in self.meta_args:
            metadata = self.meta_args[target]
            args_metas = ()
            kwargs_metas = self.meta_args
            # rv.install_metadata(self.meta_args[target])
            # return rv
        else:
            if target in self.orig_fns:
                # NOTE: tensor constructors in PyTorch define the `device` argument as
                # *kwargs-only*. That is why this works. If you add methods to
                # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
                # this will break and you will likely see issues where we cannot infer
                # the size of the output.
                if "device" in kwargs:
                    kwargs["device"] = _DEVICE

            try:
                args_metas = torch.fx.node.map_aggregate(args, _proxies_to_metas)
                kwargs_metas = torch.fx.node.map_aggregate(kwargs, _proxies_to_metas)

                if kind == "call_function":
                    meta_target = _MANUAL_META_OVERRIDES.get(target, target)
                    meta_out = meta_target(*args_metas, **kwargs_metas)
                    # meta_out = target(*args_metas, **kwargs_metas)
                    if isinstance(meta_out, torch.Tensor):
                        meta_out = meta_out.to(device=_DEVICE)
                elif kind == "call_method":
                    method = getattr(args_metas[0].__class__, target)
                    meta_target = _MANUAL_META_OVERRIDES.get(method, method)
                    meta_out = meta_target(*args_metas, **kwargs_metas)
                    if isinstance(meta_out, torch.Tensor):
                        meta_out = meta_out.to(device=_DEVICE)
                    # meta_out = method(*args_metas, **kwargs_metas)
                elif kind == "call_module":
                    if not hasattr(self, "orig_forward"):
                        raise AttributeError(f"{self} does not have an attribute called orig_forward")
                    self._disable_module_getattr = True
                    try:
                        # since the inner layers are patched, a call to higher level's (even non-patched)
                        # forward function will results in call_module called in inner layers' forward pass
                        mod = self.root.get_submodule(target)
                        mod_type = type(mod)
                        if mod_type in self.meta_overrides:
                            meta_out = self.meta_overrides[mod_type](mod, *args_metas, **kwargs_metas)
                        else:
                            meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                        # meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                    finally:
                        self._disable_module_getattr = False
                elif kind == "get_attr":
                    self._disable_module_getattr = True
                    try:
                        attr_itr = self.root
                        atoms = target.split(".")
                        for atom in atoms:
                            attr_itr = getattr(attr_itr, atom)
                        if isinstance(attr_itr, torch.Tensor):
                            meta_out = attr_itr.to(device=_DEVICE)
                        else:
                            meta_out = attr_itr
                    finally:
                        self._disable_module_getattr = False

                    if not isinstance(rv, Proxy):
                        raise ValueError("Don't support composite output yet")
                    # rv.install_metadata(meta_out)

                metadata = meta_out

            except Exception as e:
                logger.debug(f"Could not compute metadata for {kind} target {target} and node: {rv.node.name}: {e}")

        rv.install_metadata(metadata)

        # Install tensor_meta info into the node directly, avoid doing graph interpretation

        # shape inference utility
        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(metadata, extract_tensor_meta)
        if found_tensor:
            rv.node.meta["tensor_meta"] = meta

        # sharding spec utility
        inputs = list(args_metas) + list(kwargs_metas.values())
        output_sharding_spec = map_aggregate(metadata, _extract_sharding_spec)
        rv.node.meta["output_sharding_spec"] = output_sharding_spec
        input_sharding_spec = map_aggregate(inputs, _extract_sharding_spec)
        keys = list(rv.node.args) + list(rv.node.kwargs.values())
        # example:
        # code: (self.layer(inp), ) + (1, )
        # graph: add, args: ((layer, ), (1, ))
        # We need to unwrap the sharding spec here
        input_specs = dict()

        def _unwrap_sharding_spec(node_arg, sharding_spec):
            if isinstance(node_arg, Node):
                input_specs[node_arg.name] = sharding_spec
            elif isinstance(node_arg, (list, tuple)):
                for n_arg, s_spec in zip(node_arg, sharding_spec):
                    _unwrap_sharding_spec(n_arg, s_spec)
            elif isinstance(node_arg, dict):
                for n_arg, s_spec in zip(node_arg.values(), sharding_spec.values()):
                    _unwrap_sharding_spec(n_arg, s_spec)

        for i, key in enumerate(keys):
            _unwrap_sharding_spec(key, input_sharding_spec[i])

        rv.node.meta["input_sharding_spec"] = input_specs
        return rv

    # fit for torch 1.13+
    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, "_disable_module_getattr", False):
            return attr_val
        else:

            def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
                for n, p in collection_to_search:
                    if attr_val is p:
                        if n not in parameter_proxy_cache:
                            kwargs = {}
                            if "proxy_factory_fn" in inspect.signature(self.create_proxy).parameters:
                                kwargs["proxy_factory_fn"] = (
                                    None
                                    if not self.param_shapes_constant
                                    else lambda node: ParameterProxy(self, node, n, attr_val)
                                )
                            val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                            parameter_proxy_cache[n] = val_proxy
                        return parameter_proxy_cache[n]
                return None

            if isinstance(attr_val, torch.nn.Parameter):
                maybe_parameter_proxy = maybe_get_proxy_for_attr(
                    attr_val, self.root.named_parameters(), parameter_proxy_cache
                )
                if maybe_parameter_proxy is not None:
                    return maybe_parameter_proxy

            if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
                maybe_buffer_proxy = maybe_get_proxy_for_attr(
                    attr_val, self.root.named_buffers(), parameter_proxy_cache
                )
                if maybe_buffer_proxy is not None:
                    return maybe_buffer_proxy

            return attr_val

    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        return self._module_getattr(attr, attr_val, parameter_proxy_cache)

    def call_module(self, m, forward, args, kwargs):
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    def proxy(self, node):
        return MetaProxy(node, self)

    def trace(
        self,
        root: torch.nn.Module,
        input_args: Optional[Dict[str, Any]] = None,
        method_names: Optional[Iterable[str]] = None,
    ) -> Graph:
        """Trace the module root, given a batch of data input_args

        Args:
            root: the root module to be traced
            input_args (dict): a batch of data as  dict. Keys are the argument names, values
                are the corresponding input

        Returns:
            a graph
        """
        if input_args is None:
            input_args = {}
        sig = inspect.signature(root.forward)
        input_names = sig.parameters.keys() & input_args.keys()
        default_names = sig.parameters.keys() - input_args.keys()

        # FIXME move all inputs back to cpu/meta
        # values of input_args are not necessarily tensors
        def _lower_tensor_to_device(input_):
            return input_.to(_DEVICE) if isinstance(input_, torch.Tensor) else input_

        input_args = {k: map_aggregate(v, _lower_tensor_to_device) for k, v in input_args.items()}

        concrete_metas = {name: input_ for name, input_ in input_args.items() if name in input_names}
        self.meta_args = concrete_metas

        # force the unset input_args to be concrete_args with default value
        # to avoid Proxies created for these args in submodules
        default_args = {p.name: p.default for p in sig.parameters.values() if p.name in default_names}

        self.patched_torch_methods = {
            target: _gen_constructor_wrapper(getattr(torch, target)) for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns: Set[Callable] = set()
        # self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            self.graph = super().trace(root, concrete_args=default_args)
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)

        # This is necessary because concrete args are added as input to the traced module since
        # https://github.com/pytorch/pytorch/pull/55888.
        self.graph = self._remove_concrete_arg_nodes(self.graph, input_names)

        # This aims to remove all nodes created during meta analysis
        self.graph = self._remove_isolated_subgraphs(self.graph)

        return self.graph

    def _stateless_mod_instanciation_depends_on_proxies(self, mod: nn.Module) -> bool:
        """
        Whether the module was instantiated with Proxies.
        If that is the case, such module cannot be a leaf module
        because its attributes are input-dependent.
        """
        return any(isinstance(attr, Proxy) for attr in mod.__dict__.values())

    def _insert_module_as_submodule(self, mod: nn.Module) -> str:
        """
        Helper method which tries to insert a module that was not declared as submodule.
        """
        # If one of the module attributes is a Proxy, it means that its instantiation is input-dependent.
        # It is not possible to insert such modules, those should be traced through.
        if self._stateless_mod_instanciation_depends_on_proxies(mod):
            return ""
        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        already_inserted = False
        while hasattr(self.root, path):
            if getattr(self.root, path) is mod:
                already_inserted = True
                break
            path = f"{mod_name}_{idx}"
            idx += 1

        # No need to add multiple instances of the same module.
        if not already_inserted:
            self.root.add_module(path, mod)
        return path

    def path_of_module(self, mod: nn.Module) -> str:
        """
        Helper method to find the qualified name of `mod` in the Module hierarchy of `root`. For example, if `root` has
        a submodule named `foo`, which has a submodule named `bar`, passing `bar` into this function will return the
        string "foo.bar".

        Args:
            mod (str): The `Module` to retrieve the qualified name for.
        """
        try:
            return super().path_of_module(mod)
        except NameError as e:
            if self.allow_insert_stateless_mods and len(list(mod.parameters())) == 0 and len(list(mod.buffers())) == 0:
                path = self._insert_module_as_submodule(mod)
                return path
            raise e

    def keys(self, obj):
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an iterator if ** is supposed to work in
        your custom tracer.
        """
        attribute = MetaAttribute(obj, "keys")()
        if obj.node.target == "**kwargs":
            return attribute._metadata
        return attribute

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (
            (not self._stateless_mod_instanciation_depends_on_proxies(m))
            and super().is_leaf_module(m, module_qualified_name)
        ) or type(m) in self.registered_leaf_module
