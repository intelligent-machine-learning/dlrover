from dlrover.python.unified.common.workload_base import WorkerStage


def status() -> WorkerStage:  # pragma: no cover
    """Get the status of the elastic training job."""
    ...


def get_master_addr() -> str:  # pragma: no cover
    """Get the master address."""
    ...


def setup_torch_process_group(
    master_addr: str, world_size: int, rank: int
) -> None:  # pragma: no cover
    """Setup the torch process group."""
    ...


def destroy_torch_process_group() -> None:  # pragma: no cover
    """Destroy the torch process group."""
    ...


def get_ray_node_id() -> str:  # pragma: no cover
    """Get the Ray node ID."""
    ...


def run_network_check() -> float:  # pragma: no cover
    """Run network check before starting the job.
    Returns the time taken for the check."""
    ...


def start_elastic_job() -> None:  # pragma: no cover
    """Start the elastic training job."""
    ...
