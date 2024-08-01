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


def init_logging(level: Union[int, str] = logging.INFO) -> None:

    logger = logging.getLogger("BIOBENCH")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)




def which(program: str) -> Optional[str]:
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

