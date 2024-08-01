import argparse
import logging
import sys
from typing import Optional, Union


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["solvation", "density"])
    parser.add_argument("--model-path")
    parser.add_argument("--log-level", default=logging.INFO)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-path", required=False)
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--n-gpu", type=int, default=1)
    return parser.parse_args()


def init_logging(level: Union[int, str] = logging.INFO):

    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
