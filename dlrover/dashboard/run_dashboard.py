#  Copyright 2026 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys

from dlrover.python.common.log import default_logger as logger
from dlrover.dashboard.app import start_dashboard_server


def main():
    parser = argparse.ArgumentParser(description="DLRover Dashboard Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the server (default: 8080)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp/dlrover",
        help="Directory for log files (default: /tmp/dlrover)",
    )

    args = parser.parse_args()

    # Set log directory
    os.environ["DLROVER_LOG_DIR"] = args.log_dir

    logger.info(f"Starting DLRover Dashboard on {args.host}:{args.port}")
    logger.info(f"Log directory: {args.log_dir}")

    try:
        start_dashboard_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Dashboard server stopped by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start dashboard server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
