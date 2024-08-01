import subprocess
import logging
import os
from pathlib import Path
from typing import Optional

from biobench.config import ExecutionEnvironment, SlurmParams


class Job:
    def __init__(
        self,
        name: str,
        work_dir: Path,
        command: str,
        execution_environment: ExecutionEnvironment,
        slurm_params: Optional[SlurmParams],
    ):
        self.name = name
        self.work_dir =work_dir
        self.execution_environment = execution_environment
        self.slurm_params = slurm_params
        self.command = command
        self.job_id = None

    def create_gpu_bind_string(self):
        idx_str = ""
        replicas_per_gpu = int(self.tasks / self.gpus)
        for i in range(self.n_gpu):
            idx_str += (str(i) + ",") * replicas_per_gpu
        return idx_str[:-1]

    def create_slurm_header(self):

        return f"""#!/bin/bash -l
#SBATCH --time={self.slurm_params.timelimit}
#SBATCH --partition={self.slurm_params.partition}
#SBATCH --gres=gpu:{self.slurm_params.gpus}
#SBATCH --account={self.slurm_params.account}
#SBATCH --ntasks={self.slurm_params.tasks}
#SBATCH --nodes={self.slurm_params.nodes}

source ~/.bashrc
conda activate mace-mlmm\n"""

    def write_job_script(self):
        with open(f"{self.job_dir}/{self.name}.sh", "w") as f:
            f.write(self.create_slurm_header())
            f.write("\n")
            if self.ntasks > 1:
                gpu_bind_str = self.create_gpu_bind_string()
                f.write(
                    f"srun -n {self.ntasks} --gpu-bind=map_gpu:{gpu_bind_str} {self.command}"
                )
            else:
                f.write(self.command)

    def submit_job(self):
        logging.info(f"Submitting job {self.name}...")
        current_cwd = os.getcwd()
        os.chdir(self.job_dir)
        output = subprocess.run(
            f"sbatch {self.name}.sh", shell=True, check=True, capture_output=True
        )
        os.chdir(current_cwd)
        logging.info(f"Job {self.name} submitted with job id {output.stdout}")
        self.job_id = output.stdout.strip()
