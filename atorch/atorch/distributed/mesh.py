from atorch.common.log_utils import default_logger as logger

try:
    from torch.distributed.device_mesh import init_device_mesh
except ImportError:
    init_device_mesh = None


def build_mesh(slicing_dim, pg_name_prefix="", device_type="cuda"):
    dims = []
    names = []
    for item in slicing_dim:
        name = pg_name_prefix + item[0]
        d = item[1]
        if d > 1:
            dims.append(d)
            names.append(name)
    logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
    names = tuple(names)
    return init_device_mesh(device_type, dims, mesh_dim_names=names)
