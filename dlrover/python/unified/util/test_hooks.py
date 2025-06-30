import os


def init_coverage():
    if "COVERAGE_PROCESS_START" not in os.environ:
        return
    try:
        import coverage

        coverage.process_startup()
    except ImportError:
        pass
