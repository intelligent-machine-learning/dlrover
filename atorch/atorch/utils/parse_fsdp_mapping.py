import math
from collections import OrderedDict, defaultdict
from operator import attrgetter
from string import Template
from typing import Dict, List, Tuple

import torch.distributed as dist
from torch.distributed.fsdp.flat_param import FlatParamShardMetadata

from atorch.common.log_utils import default_logger as logger

FSDP_MAPPING_HTML = Template(
    r"""
<html>
<head>
<style>
h1 {text-align: center;}
div {text-align: center;}
p {text-align: center;}
</style>
<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
<script type="text/javascript" sync="" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script>
</head>
<body>

<h1>endpoint format </h1>
<p>{rank}.{name of fsdp module} -> {name of module in origin model}</p>

<h1>hover format </h1>
<p><span class="math display">\[\frac{\mbox{fsdp size}}{\mbox{fsdp total size}}\% \rightarrow \frac{\mbox{origin shard size}}{\mbox{total size}}\%\]</span></p>

<div id="sankey_multiple" style="width: 100%; height: 100%;"></div>

<script type="text/javascript">
  google.charts.load("current", {packages:["sankey"]});
  google.charts.setOnLoadCallback(drawChart);
   function drawChart() {
    var data = new google.visualization.DataTable();
    data.addColumn('string', 'From');
    data.addColumn('string', 'To');
    data.addColumn('number', 'Weight');
    data.addColumn({type: 'string', role: 'tooltip'});
    data.addRows($rows);

    // Set chart options
    var options = {
      width: 1500,
      height: $height,
      sankey: {
          node: {
            interactivity: true,
          },
          link: {
            interactivity: true
          },
      },
    };

    // Instantiate and draw our chart, passing in some options.
    var chart = new google.visualization.Sankey(document.getElementById('sankey_multiple'));
    chart.draw(data, options);
   }
</script>
</body>
</html>
"""  # noqa: E501
)


class ParseFSDP:
    @staticmethod
    def _gen_name_getattr():
        class DictWithFactoryCallable(dict):
            def __init__(self, factory):
                self._factory = factory

            def __missing__(self, key):
                v = self._factory(key)
                self[key] = v
                return v

        cache_ = DictWithFactoryCallable(lambda x: attrgetter(x))

        def inner(fsdp_name):
            return cache_[fsdp_name]

        return inner

    def __init__(self, func, enable=True, report_callback=None):
        self.fsdp_flat_handle_getattr = ParseFSDP._gen_name_getattr()
        self.func = func
        self.enable = enable
        self.report_callback = report_callback

    def get_origin_name_from_fsdp(self, fsdp_wrap_name, fsdp_name):
        """strip name '_fsdp_wrapped_module', '_fpw_module', 'flat_param',"""
        fsdp_attr = {"_fsdp_wrapped_module", "_fpw_module", "flat_param"}
        origin_attr = ".".join(i for i in fsdp_name.split(".") if i not in fsdp_attr)
        if origin_attr == "":
            return fsdp_wrap_name
        origin_attr = [".".join((origin_attr, name)) for name in fsdp_wrap_name]
        return origin_attr

    def get_meta(self, flat_param):
        """copy from distributed/fsdp/flat_param.py"""
        assert hasattr(flat_param, "_shard_indices") and hasattr(
            flat_param, "_shard_param_offsets"
        ), "Shard metadata has not been initialized"
        shard_param_start_index = flat_param._shard_indices[0]  # type: ignore[attr-defined]
        shard_param_end_index = flat_param._shard_indices[1]  # type: ignore[attr-defined]
        sl = (
            slice(shard_param_start_index, shard_param_end_index + 1)
            if shard_param_start_index <= shard_param_end_index
            else slice(0, 0)
        )
        return FlatParamShardMetadata(
            flat_param._prefixed_param_names[sl],
            flat_param._shapes[sl],
            flat_param._numels[sl],
            flat_param._shard_param_offsets[:],  # type: ignore[attr-defined]
        )

    def fsdp_mapping(self, model, origin_name_num) -> Dict[str, List[Tuple[int, float, str, int, float]]]:
        """parsing fsdp mapping
        {"fsdp-name:fsdp-byte-size" -> [
            [
                fsdp shard size in bytes
                fsdp shard size / fsdp-byte-size
                name of wrapped parameters
                size of wrapped parameters
                size of wrapped parameters / size of wrapped param_names
            ]
        ]}
        """
        fsdp_name_num = OrderedDict((k, v.numel()) for k, v in model.named_parameters())
        mapping = {}
        dist_names = []
        for name, fsdp_total_num in fsdp_name_num.items():
            dist_names.append(name.split("."))
            flat_handle = self.fsdp_flat_handle_getattr(name)(model)
            elm_size = flat_handle.element_size()
            meta = self.get_meta(flat_handle)
            origin_names = self.get_origin_name_from_fsdp(meta.param_names, name)
            fsdp_total_size = fsdp_total_num * elm_size

            each_fsdp_size = [(i[1] - i[0] + 1) * elm_size for i in meta.param_offsets]
            each_origin_size = [origin_name_num[ori_name] * elm_size for ori_name in origin_names]

            fsdp_self_ratio = [round(n / fsdp_total_size, 3) for n in each_fsdp_size]
            origin_ratio = [round(fsdp / origin, 3) for fsdp, origin in zip(each_fsdp_size, each_origin_size)]
            mapping[f"{name}:{fsdp_total_size}"] = list(
                zip(each_fsdp_size, fsdp_self_ratio, origin_names, each_origin_size, origin_ratio)
            )
        return mapping

    def format_bytes(self, size):
        power = 2**10
        n = 0
        power_labels = {0: "B", 1: "KiB", 2: "MiB", 3: "GiB", 4: "TiB"}
        while size >= power:
            size /= power
            n += 1
        return f"{round(size, 1)}{power_labels[n]}"

    def __call__(self, model, process_group, **kwargs):
        # only enabled in FSDP
        # get meta before transform model
        origin_name_num = OrderedDict((k, v.numel()) for k, v in model.named_parameters())
        min_value = min(v.numel() * v.element_size() for _, v in model.named_parameters())
        model = self.func(model, process_group, **kwargs)
        all_ranks_in_group = dist.get_process_group_ranks(process_group)
        global_rank = dist.get_rank()
        # In some case, dp process group is splited by other process group,
        # for each dp rank group has all meta information of fsdp, so we only `first group`.
        # suppose we have 2tp 2dp. rank of dp is [0,2],[1,3], we use rank[0,2] to parse meta
        # information of fsdp.
        if 0 not in all_ranks_in_group:
            logger.info(f"local rank {global_rank} is not main group")
            return model
        if not self.enable:
            logger.info("parse fsdp only support fsdp")
            return model
        group_rank = dist.get_group_rank(process_group, global_rank)
        mapping = self.fsdp_mapping(model, origin_name_num)
        output = [None for _ in range(process_group.size())]
        dist.gather_object(
            mapping,
            output if group_rank == 0 else None,
            dst=0,
            group=process_group,
        )
        if group_rank == 0:
            sankey = []
            total_sum = 0

            def process_rank(rank, data, result):
                nonlocal total_sum
                strip_digit_name_counter = defaultdict(int)
                for k, v in data.items():
                    name, total_size = k.split(":")
                    total_size = self.format_bytes(int(total_size))
                    strip_digit_name = tuple(i for i in name.split(".") if not i.isdigit())
                    if strip_digit_name_counter[strip_digit_name] != 0:
                        continue
                    strip_digit_name_counter[strip_digit_name] += 1
                    name = f"{rank}.{name}"
                    for each_fsdp_size, fsdp_ratio, origin_name, each_origin_size, origin_ratio in v:
                        each_fsdp_size_format = self.format_bytes(each_fsdp_size)
                        each_origin_size_format = self.format_bytes(each_origin_size)
                        shard_origin_size_format = self.format_bytes(each_origin_size * origin_ratio)
                        weight = int(math.log((each_fsdp_size / min_value) * 2, 2))
                        total_sum += weight
                        result.append(
                            f"['{name}', '{origin_name}', {weight}, "
                            f"'{each_fsdp_size_format}/{total_size}({round(fsdp_ratio*100, 1)}%) -> "
                            f"{shard_origin_size_format}/{each_origin_size_format}"
                            f"({round(origin_ratio * 100, 1)}%)']"
                        )

            for i, d in zip(all_ranks_in_group, output):
                process_rank(i, d, sankey)
            sub_data = {}
            sub_data["rows"] = "[\n" + ",\n".join(sankey) + "\n]"
            sub_data["height"] = 10 * total_sum
            mapping_str = FSDP_MAPPING_HTML.substitute(sub_data)
            with open("fsdp_mapping.html", "w") as f:
                f.write(mapping_str)
            if self.report_callback is not None:
                try:
                    self.report_callback(mapping_str)
                except Exception:
                    logger.warning("fsdp mapping report callback error")
        return model
