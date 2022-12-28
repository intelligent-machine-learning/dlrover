import os
from penrose.platform import starter
from penrose.util.log_util import default_logger as logger

if __name__ == "__main__":
    logger.info(
        "WORKFLOW_ID: %s, USERNUMBER: %s",
        os.environ.get("WORKFLOW_ID", None),
        os.environ.get("USERNUMBER", None),
    )

    logger.info("local entry is running")
    starter.run()