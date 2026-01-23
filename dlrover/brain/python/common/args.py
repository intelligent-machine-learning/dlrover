import argparse
from dlrover.brain.python.common.log import default_logger as logger


def add_args(parser):
    parser.add_argument(
        "--port",
        help="port",
        type=int,
        default=8000,
    )


def build_parser():
    """Build a parser for dlrover trainer"""
    parser = argparse.ArgumentParser()
    add_args(parser)
    return parser


def get_parsed_args():
    """Get parsed arguments"""
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if _:
        logger.info("Unknown arguments: %s", _)
    return args
