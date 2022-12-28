import argparse
from penrose.util.log_util import default_logger as logger

def add_platform_args(parser):
    parser.add_argument(
        "--platform",
        help="execute platform",
        type=lambda x: x.upper(),
        default="KUBERNETES"
    )
    parser.add_argument(
        "--worker_action",
        help="worker's action",
        default="run"
    )

    parser.add_argument(
        "--ps_num",
        help="ps number",
        type=int,
        default=1
    )

    parser.add_argument(
        "--worker_num",
        help="worker number",
        type=int,
        default=3
    )

    parser.add_argument(
        "--evaluator_num",
        help="evaluator number",
        type=int,
        default=1
    )

    parser.add_argument(
        "--conf",
        help="configuration for training",
        default=None
    )

    parser.add_argument(
        "--enable_easydl",
        help="configuration for elastic training",
        type=bool,
        default=False
    )

    


def build_parser():
    """Build a parser for penrose"""
    parser = argparse.ArgumentParser()
    add_platform_args(parser)
    return parser


def get_parsed_args():
    """Get parsed arguments"""
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if _:
        logger.info("Unknown arguments: %s", _)
    return args