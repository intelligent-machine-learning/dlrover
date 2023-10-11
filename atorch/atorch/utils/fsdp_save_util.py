import glob
import json
import pickle
import struct
from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import safetensors
import torch
import torch.distributed as dist
from packaging.version import Version
from safetensors.torch import _SIZE, _TYPES, _tobytes, safe_open
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from atorch.utils.version import torch_version

if torch_version() <= (1, 13, 1):
    from torch.distributed.fsdp.fully_sharded_data_parallel import clean_tensor_name
elif torch_version() >= (2, 0, 0):
    from torch.distributed.fsdp._common_utils import clean_tensor_name
else:
    raise ValueError(f"Unsupported version is {torch.__version__}")

from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import local_rank, nproc_per_node, parallel_group

TYPES_TO_STR = {v: k for k, v in _TYPES.items()}


def version_check():
    if Version(torch.__version__) <= Version("2.0.9"):
        raise ValueError(f"fsdp save util only support torch>=2.1.0, your version is {torch.__version__}")


def parse_safetensors_head(path):
    """Parse header of safetensors, format refers to
    https://huggingface.co/docs/safetensors/index#format
    """
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        buf = f.read(length)
        meta = json.loads(buf)
        return meta


def safetensors_dump(tensors, path):
    """safetensors is not iterable, flush all tensor together, if all
    process dumps simultaneously, may cause OOM.
    """
    meta_info = {
        k: {
            "dtype": v.dtype,
            "shape": v.shape,
        }
        for k, v in tensors.items()
    }
    # sort with [sizeof(dtype), name]
    sorted_meta_info = {k: v for k, v in sorted(meta_info.items(), key=lambda x: (-_SIZE[x[1]["dtype"]], x[0]))}
    meta_buf = {}
    offset = 0
    for k, v in sorted_meta_info.items():
        meta_buf[k] = {}
        data = meta_buf[k]
        data["dtype"] = TYPES_TO_STR[v["dtype"]]
        data["shape"] = list(v["shape"])
        end = offset + _SIZE[v["dtype"]] * v["shape"].numel()
        data["data_offsets"] = [offset, end]
        offset = end
    meta_to_flush = json.dumps(meta_buf, separators=(",", ":")).encode()
    # Force alignment to 8 bytes.
    extra = (8 - len(meta_to_flush) % 8) % 8
    if extra:
        meta_to_flush = meta_to_flush + b" " * extra

    head = struct.pack("<Q", len(meta_to_flush))
    with open(path, "wb") as f:
        f.write(head)
        f.write(meta_to_flush)
        for k in sorted_meta_info:
            data = _tobytes(tensors[k], k)
            f.write(data)


def get_handle(fsdp_unit):
    if hasattr(fsdp_unit, "_handles"):
        if len(fsdp_unit._handles) != 1:
            # currently, num of handles is 1
            raise ValueError("unknown error, currently, length of handles in FSDP is 1, check version of pytorch")
        return fsdp_unit._handles[0]
    elif hasattr(fsdp_unit, "_handle"):
        return fsdp_unit._handle
    else:
        raise ValueError("unknown error, pytorch version should in 20230431-20230731 check version of pytorch")


def save_fsdp_flat_param(model, path):
    """Save flat param and meta info of each fsdp units.
    Structure of save path describe as below, it includes 3 parts,
        1. flat_meta.<rank>-<world size>, meta file of each flat param, it saved by pickle.
        2. flat_param.<rank>-<world size>, flat param of each FSDP unit, it saved by safetensors.
        3. buffers, all buffer in model, it saved by safetensors
    .
    ├── buffers
    ├── flat_meta.00000-00002
    ├── flat_meta.00001-00002
    ├── flat_param.00000-00002
    └── flat_param.00001-00002

    Flat param is 1d flatten tensor, maybe with zero pad at last.
    Meta data has those field:
        param_names: name of original parameter in FSDP unit.
        param_shapes: shape of original parameter.
        param_numels: num of elements of original parameter.
        param_offsets: tuple of int, [start, end], record position of original parameter in flatten parameter.
        pad: pad num of flatten parameter, only in last rank.
        rank: rank number.
        striped_fsdp_name: name of FSDP unit.
        flat_numel: num of elements of flatten parameter.
        For example
        {
            'param_names': 'glm.word_embeddings.weight',
            'param_shapes': torch.Size([50304, 1024]),
            'param_numels': 51511296,
            'param_offsets': (0, 26806271),
            'pad': 0,
            'rank': 0,
            'striped_fsdp_name': '',
            'flat_numel': 26806272
        }
    """
    fsdp_units = [
        m
        for m in model.named_modules()
        if isinstance(m[1], torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel)
    ]
    metas = {}
    params = {}
    buffers = {clean_tensor_name(k): v for k, v in model.named_buffers()}
    data_group = parallel_group("data")
    for fsdp_name, fsdp_unit in fsdp_units:
        fsdp_flat_handle = get_handle(fsdp_unit)
        meta = fsdp_flat_handle.shard_metadata()
        flat_param = fsdp_flat_handle.flat_param
        pad = flat_param._shard_numel_padded

        name = f"{clean_tensor_name(fsdp_name)}" if fsdp_name else ""
        keys, items = list(zip(*meta._asdict().items()))
        items = list(zip(*items))
        data = list(dict(zip(keys, i)) for i in items)
        shard_info_in_this_rank = [i for i in flat_param._shard_param_infos if i.in_shard]
        if len(shard_info_in_this_rank) != len(data):
            raise ValueError("flat param mismatch")
        for shard_info, each in zip(shard_info_in_this_rank, data):

            global_name = (
                [name, clean_tensor_name(each["param_names"])] if name else [clean_tensor_name(each["param_names"])]
            )
            global_name = ".".join(global_name)
            each["pad"] = pad
            each["rank"] = dist.get_rank(data_group)
            each["striped_fsdp_name"] = name
            each["flat_numel"] = flat_param.numel()
            each["param_offsets"] = (shard_info.offset_in_shard, shard_info.offset_in_shard + shard_info.numel_in_shard)
            metas[global_name] = each
        params[name] = flat_param

    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    suffix = f"{str(dist.get_rank(data_group)).zfill(5)}-{str(dist.get_world_size(data_group)).zfill(5)}"
    safetensors_dump(params, f"{path}/flat_param.{suffix}")
    if dist.get_rank(data_group) == 0:
        safetensors_dump(buffers, f"{path}/buffers")
    with open(f"{path}/flat_meta.{suffix}", "wb") as f:
        pickle.dump(metas, f)


def save_fsdp_optim_param(model, optimizer, path):
    """Save state of optimizer for each FSDP unit. This func is only support FSDP with `use_orig_params`
    Structure of save path describe as below, it includes 2 parts:
        1. optim_meta, which saves param_groups and others hyperparameters, saved by pickle
        2. optim_param.<rank>-<world_size>, which saves optimizer's state or each FSDP unit, saved by safetensors.
    ├── optim_meta
    ├── optim_param.00000-00002
    └── optim_param.00001-00002

    """
    param_mappings = {}
    start_index = 0
    param_to_names = {v: clean_tensor_name(k) for k, v in model.named_parameters()}
    optim_state = optimizer.state_dict()

    def pack_group(group):
        nonlocal start_index
        packed = {k: v for k, v in group.items() if k != "params"}
        param_mappings.update(
            {id(p): i for i, p in enumerate(group["params"], start_index) if id(p) not in param_mappings}
        )
        packed["params"] = [param_to_names[p] for p in group["params"]]
        packed["params_names"] = {param_mappings[id(p)]: param_to_names[p] for p in group["params"]}
        start_index += len(packed["params"])
        return packed

    param_groups = [pack_group(g) for g in optimizer.param_groups]
    optim_param_idx_to_name = {}
    _ = [optim_param_idx_to_name.update(i["params_names"]) for i in param_groups]
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    flat_state = {optim_param_idx_to_name[k]: v for k, v in optim_state["state"].items()}
    save_state = {}
    for name, state in flat_state.items():
        save_state.update({f"{name}-{k}": v for k, v in state.items()})
    data_group = parallel_group("data")
    suffix = f"{str(dist.get_rank(data_group)).zfill(5)}-{str(dist.get_world_size(data_group)).zfill(5)}"
    safetensors_dump(save_state, f"{path}/optim_param.{suffix}")
    if dist.get_rank(data_group) == 0:
        with open(f"{path}/optim_meta", "wb") as f:
            pickle.dump(param_groups, f)


class ShardOptim:
    class ShardFlatManager:
        def __init__(self, param_name, fds, optim_slice):
            """This class is mocking dict behavior, and it called in function `FSDP.shard_full_optim_state_dict`."""
            self.fds = fds
            self.param_name = param_name
            self.optim_slice = optim_slice

        def items(self):
            """Compatible with dict class."""
            return self.optim_slice.items()

        def check_1d(self, state_name):
            """All params must be flattend, if not, the param is scalar value."""
            for each_slice in self.optim_slice[state_name]:
                if len(each_slice.get_shape()) != 1:
                    return False
            return True

        def load_tensor_by_param_and_state_name(self, state_name, target_start=None, target_end=None):
            """Load optim state by name and range. if range is None, we load whole parameter."""
            if target_start is None and target_end is None:
                if self.check_1d(state_name):
                    raise ValueError(f"{self.param_name}, {state_name} should be scaler")
                return self.fds[state_name].get_tensor(f"{self.param_name}-{state_name}")
            if target_start is None or target_end is None:
                raise ValueError("Both `target_start` and `target_end` must be None or number")
            optim_slice = self.optim_slice[state_name]
            shards = []
            for s in optim_slice:
                seg_start, seg_end = 0, s.get_shape()[0]
                if seg_start <= target_start and target_end <= seg_end:
                    shards.append(s[target_start:target_end])
                    break
                if target_start <= seg_end and target_end >= seg_start:
                    shards.append(s[max(target_start, seg_start) : min(target_end, seg_end)])
                target_start -= seg_end
                target_end -= seg_end

            return torch.cat(shards)

    def __init__(self, path):
        """This class load flatten optim state, mock state dict, and call `FSDP.shard_full_optim_state_dict` to
        reshard optim state to new FSDP model.
        """
        osd_param_files = sorted(glob.glob(f"{path}/optim_param*"))
        self.osd_param = []
        self.fds = defaultdict(lambda: defaultdict(dict))

        with open(f"{path}/optim_meta", "rb") as f:
            self.osd_meta = pickle.load(f)
        for i in osd_param_files:
            with open(i, "rb") as f:
                self.osd_param.append(safetensors.safe_open(i, framework="pt"))
        self.state_dict = defaultdict(lambda: defaultdict(list))
        for fd in self.osd_param:
            for k in fd.keys():
                param_name, state_name = k.split("-")
                self.state_dict[param_name][state_name].append(fd.get_slice(k))
                self.fds[param_name][state_name] = fd

        self.state_dict["param_groups"] = []
        for meta in self.osd_meta:
            self.state_dict["param_groups"].append({k: v for k, v in meta.items() if k != "params_names"})

        state = self.state_dict["state"]
        for p in self.osd_meta:
            for param_name in p["params"]:
                state[param_name] = ShardOptim.ShardFlatManager(
                    param_name, self.fds[param_name], self.state_dict[param_name]
                )

    def reshard_optim_state_dict(self, model):
        """Reshard flatten optim state by FSDP model"""
        from torch.distributed.fsdp import _optim_utils

        orig_fn = _optim_utils._shard_orig_param_state
        _optim_utils._shard_orig_param_state = self._shard_orig_param_state
        shard_dict = FSDP.shard_full_optim_state_dict(self.state_dict, model)
        _optim_utils._shard_orig_param_state = orig_fn
        return shard_dict

    def load_tensor_by_param_and_state_name(self, param_name, state_name, target_start=None, target_end=None):
        """Load param, it calls `ShardFlatManager.load_tensor_by_param_and_state_name` to reshard flat param"""
        return self.state_dict["state"][param_name].load_tensor_by_param_and_state_name(
            state_name, target_start, target_end
        )

    @staticmethod
    def _shard_orig_param_state(
        fsdp_param_info,
        fqn,
        optim_state,
    ):
        """
        Shard the optimizer state for the original parameter with the name ``fqn``.
        This API should only be used when ``use_orig_params`` is True.

        Layout of optimize state is described as below.
        Each optim state belones to one flat param, if we do reshard optimizer state, we recalculate
        intra_param_start_idx and intra_param_end_idx from model which is wrapped by FSDP.
        if [intra_param_start_idx, intra_param_end_idx] is cross multiple state, we concats them together.

        NOTE, we don't reshape optimizer state, because the model has been transformated to FSDP, optimizer
        state should also be flatten param.

               intra_param_start_idx   intra_param_end_idx
                     +                +
                     |                |
                     v                v
        +------------+------+---------+-----------+
        |  optim state0     |  optim state1       |
        +--------+----------+---------+-----------+
                 ^                    ^
                 |                    |
            flat param0          flat param1

        """
        if not optim_state:
            return {}
        flat_param = fsdp_param_info.handle.flat_param
        param_idx = fsdp_param_info.param_indices[fqn]
        shard_param_info = flat_param._shard_param_infos[param_idx]  # type: ignore[attr-defined]
        # we don't have ShardedTensor, skip _gather_state_dict
        # optim_state = _gather_state_dict(optim_state, fsdp_param_info.state.process_group)
        if not shard_param_info.in_shard:
            return {}
        # Flatten and shard the state.
        new_optim_state: Dict[str, Any] = {}
        intra_param_start_idx = shard_param_info.intra_param_start_idx
        intra_param_end_idx = shard_param_info.intra_param_end_idx
        for state_name, _ in optim_state.items():
            if optim_state.check_1d(state_name):
                value = optim_state.load_tensor_by_param_and_state_name(
                    state_name, intra_param_start_idx, intra_param_end_idx + 1
                )
            else:
                value = optim_state.load_tensor_by_param_and_state_name(state_name)
            new_optim_state[state_name] = value
        return new_optim_state


class ShardTensorUtil:
    """Parse flatten parameter, get reshard flatten parameter or original parameter."""

    def __init__(self, path, rank, world_size, device=None, name_mapping_func_or_dict=None, create_param_cb=None):
        """
        param_meta: record all meta information of original parameter, maybe original parameter
            will across multiple flatten parameters. Key is name of top module, value is meta info
        rank_fds: record all safe open fds, key is rank.
        flat_param_segments: Length of each original parameter in flatten parameter, exclude the paddings.
        flat_param_length: Length of each flatten parameter, exclude the paddings.
        """
        self.buffers = {}
        self.load_local_rank = local_rank()
        self.device = device or f"cuda:{self.load_local_rank}"
        self.path: str = path
        self.rank: int = rank
        self.world_size: int = world_size
        self.load_buffer = None
        self.local_gpus_groups = []
        self.load_parallelism = nproc_per_node()
        self.param_meta: DefaultDict[str, List]
        self.rank_fds: Dict[int, safetensors._safetensors_rust.safe_open]
        self.flat_param_segments: Dict[Tuple[int, str], List[int]] = {}
        self.flat_param_length: DefaultDict[str, int] = defaultdict(int)
        self.param_meta, self.rank_fds = self._prepare_meta_and_fds()
        self.name_mapping_func = None
        if create_param_cb is not None and not callable(create_param_cb):
            raise ValueError("create_param_cb shoule be callable")
        self.create_param_cb = create_param_cb
        if name_mapping_func_or_dict is not None:
            if isinstance(name_mapping_func_or_dict, Mapping):

                def mapping_to_func(name):
                    if name not in name_mapping_func_or_dict:
                        readable_keys = "\n".join(name_mapping_func_or_dict.keys())
                        raise KeyError(
                            f"{name} is not in mapping, check content of `name_mapping_func_or_dict`"
                            f" keys are {readable_keys}"
                        )
                    return name_mapping_func_or_dict[name]

                self.name_mapping_func = mapping_to_func
            elif callable(name_mapping_func_or_dict):
                self.name_mapping_func = name_mapping_func_or_dict
            else:
                raise ValueError("`name_mapping_func` must be callable or dict, mapping str to str")
        if self.device.startswith("cuda"):
            self.init_local_ranks()

    def init_local_ranks(self):
        """Create local process group"""
        self.local_gpus_group, self.local_gpus_groups = dist.new_subgroups(group_size=self.load_parallelism)
        self.load_count = defaultdict(int)

    def change_to_gpu(self):
        """Change device to gpu and create intra process group"""
        if not self.device.startswith("cuda"):
            self.device = f"cuda:{self.load_local_rank}"
            self.init_local_ranks()

    def get_fsdp_load_map(self):
        """Get fsdp init order and dispatch order to each local rank.
        eg. If we have 6 parameters 0..5, and we have 3 gpus
        gpu0 load param [0, 3]
        gpu1 load param [1, 4]
        gpu2 load param [2, 5]
        """
        data_group = parallel_group("data")
        rank = dist.get_rank(data_group)
        node_rank = rank // self.load_parallelism
        rank_offset = node_rank * self.load_parallelism
        self.load_order = deque(self.fsdp_init_order[self.load_local_rank :: self.load_parallelism])
        self.param_name_to_rank = {
            name: i % self.load_parallelism + rank_offset for i, name in enumerate(self.fsdp_init_order)
        }
        prefetch = self.load_order.popleft()
        self.load_buffer = self.load_tensor_by_name(prefetch)

    def get_fsdp_init_order(self, model, wrap_cls):
        """Get parameters' init order of every FSDP unit."""
        all_names = OrderedDict()
        default_fsdp_names = []
        tied_weights = set()

        def inner(model, name_prefix):
            if type(model) in wrap_cls:
                # record all parameters in module
                n = []
                for name, p in model.named_parameters():
                    if id(p) in tied_weights:
                        continue
                    n.append(name)
                    tied_weights.add(id(p))
                all_names[".".join(name_prefix)] = n
                return
            n = []
            base = ".".join(name_prefix)
            for name, p in model.named_parameters(recurse=False):
                # record all parameters in module
                if id(p) in tied_weights:
                    continue
                default_fsdp_names.append(f"{base}.{name}")
                tied_weights.add(id(p))
            for name, child in model.named_children():
                name_prefix.append(name)
                inner(child, name_prefix)
                name_prefix.pop()

        inner(model, [])
        # empty string `""` is default FSDP wrap unit.
        all_names[""] = default_fsdp_names
        self.fsdp_init_order = []
        for k, v in all_names.items():
            prefix = f"{k}." if k else ""
            for i in v:
                self.fsdp_init_order.append(f"{prefix}{i}")

        self.get_fsdp_load_map()

    def load_with_intra_nodes(self, name):
        """Load parameters from intra node via broadcasting. For nvidia gpu, we use nvlink.
            eg. If we have 6 parameters 0..5, and we have 3 gpus
            gpu0 broadcasting [0, 3]
            gpu1 broadcasting [1, 4]
            gpu2 broadcasting [2, 5]
        For buffers, it requires tiny storage, so we load it from disk directly.
        """
        if name in self.buffers.keys():
            return self.buffers.get_tensor(name)
        src_rank = self.param_name_to_rank[name]
        self.load_count[name] += 1
        if self.load_count[name] != 1:
            # maybe tie weights
            shape = self.load_tensor_by_name(name, return_shape=True)
            return torch.empty(shape, device=self.device)

        if self.rank != src_rank:
            shape = self.load_tensor_by_name(name, return_shape=True)
            dst = torch.empty(shape, device=self.device)
            dist.broadcast(dst, src=src_rank, group=self.local_gpus_group)
            return dst
        else:
            if self.load_buffer is None:
                raise ValueError("load buffer is not prefetched")
            src = self.load_buffer.to(self.device, non_blocking=True)
            work = dist.broadcast(src, src=src_rank, group=self.local_gpus_group, async_op=True)
            if self.load_order:
                prefetch = self.load_order.popleft()
                self.load_buffer = self.load_tensor_by_name(prefetch)
            work.wait()
            return src

    def _get_flat_param_shard_range(self, name, target):
        """Calculate range in multiple flat parameters, return slice of each flat parameter."""
        s = []
        target_start, target_end = target
        for rank in range(self.shard_nums):
            seg_start, seg_end = self.flat_param_segments[(rank, name)]
            if target_start >= seg_end or target_end <= seg_start:
                continue
            if seg_start <= target_start and target_end <= seg_end:
                s.append((rank, target_start, target_end))
            else:
                s.append((rank, max(target_start, seg_start), min(target_end, seg_end)))
        return s

    def _prepare_meta_and_fds(self):
        """Parse flat param and flat meta files, open safe tensor files."""
        path = self.path
        buffers_path = Path(f"{path}/buffers")
        if buffers_path.exists():
            self.buffers = safe_open(buffers_path, framework="pt")
        rank_fds = {}
        param_meta = defaultdict(list)
        meta_files = sorted(glob.glob(f"{path}/flat_meta*"))
        safe_files = sorted(glob.glob(f"{path}/flat_param*"))
        if len(meta_files) != len(safe_files):
            raise ValueError(f"Num of meta is not equal to num of param, {len(meta_files)} vs {len(safe_files)}")
        suffix = safe_files[0].split(".")[-1]
        total_num = int(suffix.split("-")[-1])
        if total_num != len(safe_files):
            raise ValueError(f"Miss ckpt shards, num is {len(safe_files)}, needs {total_num}")
        self.need_reshard = len(meta_files) != self.world_size
        self.shard_nums = len(meta_files)
        for rank, f in enumerate(safe_files):
            fd = safe_open(f, framework="pt")
            for key in fd.keys():
                tensor_length = fd.get_slice(key).get_shape()[0]
                start = self.flat_param_length[key]
                self.flat_param_segments[(rank, key)] = [start, start + tensor_length]
                self.flat_param_length[key] += tensor_length
            rank_fds[rank] = fd
        for f in meta_files:
            with open(f, "rb") as f:
                data = pickle.load(f)
            for k, v in data.items():
                pad = v["pad"]
                rank = v["rank"]
                fsdp_unit_name = v["striped_fsdp_name"]
                # index -1 means end offset
                self.flat_param_segments[(rank, fsdp_unit_name)][-1] -= pad
                self.flat_param_length[fsdp_unit_name] -= pad
                param_meta[k].append(v)
        return param_meta, rank_fds

    def load_flat_param_by_name(self, name):
        """Load flat parameter, if shards number is not equals to world size,
        we do reshard flat parameters. Only support same wrap class in fsdp transformation.
        """
        fd = self.rank_fds[0]
        shards = []
        total_length = self.flat_param_length[name]
        flat_length = fd.get_slice(name).get_shape()[0]
        pad_num = (self.world_size - total_length % self.world_size) % self.world_size
        new_total_length = total_length + pad_num
        each_num = new_total_length // self.world_size
        request_start = self.rank * each_num
        request_end = self.rank * each_num + each_num
        if self.rank + 1 == self.world_size:
            # only last rank padding zeros.
            request_end -= pad_num
        # given range of parameters, we calculate cusor range of each flat parameter.
        shard_slice = self._get_flat_param_shard_range(name, (request_start, request_end))
        for rank, start, end in shard_slice:
            fd = self.rank_fds[rank]
            tensor = fd.get_slice(name)
            rank_offset = rank * flat_length
            local_start = start - rank_offset
            local_end = end - rank_offset  # [start, end] is include, so we add 1
            shards.append(tensor[local_start:local_end])
        if self.rank + 1 == self.world_size and pad_num:
            pad_tensor = torch.zeros(pad_num)
            shards.append(pad_tensor)

        return torch.cat(shards)

    def check_tensor_not_in_ckpt(self, name):
        return name not in self.buffers.keys() and name not in self.param_meta

    def load_tensor_by_name(self, name: str, strict=True, return_shape=False, sync_module_states=False):
        """Load origin tensors in flatten parameters. If tensor is cross multiple flat parameters,
        concats them together, reshape flatten parameters at last.

        Parameter layout is described as below, shard0 and shard1 is part of flat parameter.
        flat param0 is saved by rank0, flat param1 is saved by rank1.
        offset is meta data which is saved by each rank.

                     offset0    offset1
        +----------+----------+----------+--------------+
        |          | shard0   | shard1   |              |
        +----------+---------------------+--------------+
        ^                     ^
        |                     |
        +-----+ flat param0   +------+ flat param1

        """
        if self.name_mapping_func is not None:
            old_name = name
            name = self.name_mapping_func(name)
            logger.info(f"[{self.rank}]: shard util replace {old_name} -> {name}")
        miss = self.check_tensor_not_in_ckpt(name)
        if miss:
            if not strict:
                logger.warning(f"miss key {name}, maybe tie weights")
                return None
            if self.create_param_cb is None:
                param_keys = "\n".join(self.param_meta.keys())
                buffer_keys = "\n".join(self.buffers.keys())
                raise ValueError(
                    f"Miss param/buffer, key {name}, maybe you can pass"
                    f"`create_param_cb` to specify how {name} is created"
                    f"\nparam keys is:\n{param_keys}"
                    f"\nbuffer keys is:\n{buffer_keys}"
                )
            else:
                logger.info(f"[{self.rank}]: shard util use create_param_cb to create {name}")
                return self.create_param_cb(name)

        if name in self.buffers.keys():
            return self.buffers.get_tensor(name)

        meta = self.param_meta[name]
        ori_shape = meta[0]["param_shapes"]  # we use first element in meta list
        if return_shape:
            return ori_shape
        # if sync_module_states, rank0 broadcast parameters
        if sync_module_states and self.rank != 0:
            return torch.empty(ori_shape, device=self.device)

        shards = []
        for m in meta:
            slice_ = m["param_offsets"]
            rank = m["rank"]
            orig_fsdp_unit_name = m["striped_fsdp_name"]
            fd = self.rank_fds[rank]
            tensor_slice = fd.get_slice(orig_fsdp_unit_name)
            shards.append(tensor_slice[slice(*slice_)].to(self.device, non_blocking=True))
        torch.cuda.synchronize(torch.cuda.current_device())
        tensor = torch.cat(shards)
        return tensor.reshape(ori_shape)

    def __del__(self):
        for g in self.local_gpus_groups:
            dist.destroy_process_group(g)
