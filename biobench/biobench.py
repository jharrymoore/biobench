import argparse
import os

from biobench.config import Configuration
from biobench.experiment import DensityExperiment, Experiment, SolvationExperiment
from .utils import get_args, init_logging
from enum import Enum
import logging


class ValidationDataPaths(Enum):
    density = "https://drive.google.com/uc?id=18RurJW7n7dWYx3vKSZcuN9ejf_FuwdPo"
    solvation = None


def main():
    args = get_args()
    log_level = logging.getLevelName(args.log_level)
    init_logging(log_level)
    default_data_path = ValidationDataPaths[args.experiment].value

    if args.experiment == "density":
        config = Configuration(
            data_path=default_data_path if args.data_path is None else args.data_path,
            output_dir=os.path.join(os.getcwd(), "density_calcs"),
            partition="ampere",
            timelimit="36:0:0",
            account="csanyi-sl2-gpu",
            overwrite=args.overwrite,
            model_path=args.model_path,
            steps=500000,
        )
        experiment = DensityExperiment(
            data_path=config.data_path,
            output_dir=config.output_dir,
            partition=config.partition,
            model_path=config.model_path,
            timelimit=config.timelimit,
            account=config.account,
            overwrite=config.overwrite,
            steps=config.steps,
        )
    elif args.experiment == "solvation":
        config = Configuration(
            data_path=default_data_path if args.data_path is None else args.data_path,
            output_dir=os.path.join(os.getcwd(), "solvation_calcs"),
            partition="ampere",
            timelimit="36:0:0",
            account="csanyi-sl2-gpu",
            overwrite=args.overwrite,
            model_path=args.model_path,
            steps=5000,
        )
        experiment = SolvationExperiment(
            data_path=config.data_path,
            output_dir=config.output_dir,
            partition=config.partition,
            model_path=config.model_path,
            timelimit=config.timelimit,
            account=config.account,
            overwrite=config.overwrite,
            steps=config.steps,
            replicas=args.replicas,
        )
    else:
        raise NotImplementedError(f"Experiment {args.experiment} not implemented.")

    if args.analyse:
        experiment.analyse()
    else:
        experiment.execute()
