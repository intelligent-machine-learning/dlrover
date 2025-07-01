import os


def init_coverage():
    if "COVERAGE_PROCESS_START" not in os.environ:
        return
    try:
        import coverage

        coverage.process_startup()
    except ImportError:
        pass


def coverage_enabled():
    """Check if coverage is enabled."""
    try:
        import coverage

        return coverage.Coverage.current() is not None
    except ImportError:
        return False
