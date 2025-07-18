from dlrover.python.unified.common.workload_base import WorkerStage


def status() -> WorkerStage:
    """Get the status of the elastic training job."""
    ...


def get_master_addr() -> str:
    """Get the master address."""
    ...


def setup_torch_process_group(
    master_addr: str, world_size: int, rank: int
) -> None:
    """Setup the torch process group."""
    ...


def destroy_torch_process_group() -> None:
    """Destroy the torch process group."""
    ...


def get_ray_node_id() -> str:
    """Get the Ray node ID."""
    ...


def run_network_check() -> float:
    """Run network check before starting the job.
    Returns the time taken for the check."""
    ...


def start_elastic_job() -> None:
    """Start the elastic training job."""
    ...
