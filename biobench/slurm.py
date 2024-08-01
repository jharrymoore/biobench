import subprocess
import logging
import os

from biobench.config import SlurmParams


class SlurmJob:
    def __init__(
        self,
        name: str,
        work_dir: Path,
        command: str,
        slurm_params: SlurmParams,
    ):
        self.name = name
        self.work_dir = workdir
        self.job_id = None
        self.timelimit=slurm_params.timelimit
        self.account = slurm_params.account
        self.partition = slurm_params.partition
        self.gpus =slurm_params.gpus
        self.tasks = slurm_params.tasks
        self.nodes = slurm_params.nodes
        self.command = command

    def create_gpu_bind_string(self):
        idx_str = ""
        replicas_per_gpu = int(self.tasks / self.gpus)
        for i in range(self.n_gpu):
            idx_str += (str(i) + ",") * replicas_per_gpu
        return idx_str[:-1]

    def create_slurm_header(self):

        return f"""#!/bin/bash -l
#SBATCH --time={self.time}
#SBATCH --partition={self.partition}
#SBATCH --gres=gpu:{self.gpus}
#SBATCH --account={self.account}
#SBATCH --ntasks={self.tasks}
#SBATCH --nodes={self.nodes}

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
