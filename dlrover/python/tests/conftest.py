import os


def pytest_configure():
    # disable default event exporter for unit tests
    os.environ["DLROVER_EVENT_ENABLE"] = "false"
