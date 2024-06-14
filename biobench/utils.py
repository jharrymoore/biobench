import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()
