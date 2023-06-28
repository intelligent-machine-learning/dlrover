import os
import sys

try:
    from easydl.python.runtime.metric_collector import MetricCollector
except ImportError:
    MetricCollector = None


def get_cluster_name():
    service_host = os.getenv("KUBERNETES_SERVICE_HOST", "")
    if service_host == "":
        return ""
    try:
        # the format of KUBERNETES_SERVICE_HOST is
        # apiserver.sigma-{cluster}.svc.{cluster}.alipay.com
        cluster_name = service_host.split(".")[3]
    except Exception:
        cluster_name = ""
    return cluster_name


def get_job_name():
    job_name = os.getenv("APP_ID")
    if job_name is None:
        job_name = os.getenv("AISTUDIO_JOB_NAME", "atorch_job")
    return job_name


# `EXECUTIONRECORD_ID` is an env variable only existed in online environment. Each job has its unique
# `EXECUTIONRECORD_ID`. If `unique_id != ""`, it means current process is in the online env.
# `hasattr(sys, "ps1") is False` means current process is not in the interactive mode,
_UNIQUE_JOB_ID = os.getenv("EXECUTIONRECORD_ID", "")
_REPORTER_ENABLED = MetricCollector is not None and hasattr(sys, "ps1") is False and _UNIQUE_JOB_ID != ""
_USER_ID = os.getenv("USERNUMBER", "")
_CLUSTER_NAME = get_cluster_name()
_JOB_NAME = get_job_name()
if _REPORTER_ENABLED:
    try:
        _EASYDL_COLLECTOR = MetricCollector()
    except Exception:
        _EASYDL_COLLECTOR = None
        _REPORTER_ENABLED = False
else:
    _EASYDL_COLLECTOR = None


def report_import():
    global _REPORTER_ENABLED
    if _REPORTER_ENABLED is False:
        return
    try:
        _EASYDL_COLLECTOR.report_job_meta(
            _UNIQUE_JOB_ID,
            _JOB_NAME,
            _USER_ID,
            namespace="kubemaker",
            cluster=_CLUSTER_NAME,
        )
        _EASYDL_COLLECTOR.report_job_type(_UNIQUE_JOB_ID, "atorch")
    except Exception:
        _REPORTER_ENABLED = False
