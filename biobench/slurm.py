import subprocess
import logging
import os


class SlurmJob:
    def __init__(
        self,
        name,
        job_dir,
        time,
        account,
        partition,
        gres,
        ntasks,
        nodes,
        mem,
        command,
    ):
        self.name = name
        self.job_dir = job_dir
        self.job_id = None
        self.time = time
        self.account = account
        self.partition = partition
        self.gres = gres
        self.ntasks = ntasks
        self.nodes = nodes
        self.mem = mem
        self.command = command

    def create_gpu_bind_string(self):
        idx_str = ""
        n_gpu = int(self.gres.split(":")[1])
        replicas_per_gpu = int(self.ntasks / n_gpu)
        for i in range(n_gpu):
            idx_str += (str(i) + ",") * replicas_per_gpu
        return idx_str[:-1]

    def create_slurm_header(self):

        return f"""#!/bin/bash -l
#SBATCH --time={self.time}
#SBATCH --partition={self.partition}
#SBATCH --gres={self.gres}
#SBATCH --account={self.account}
#SBATCH --ntasks={self.ntasks}
#SBATCH -N={self.nodes}
#SBATCH --mem={self.mem}

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
        output = subprocess.run(f"sbatch {self.name}.sh", shell=True, check=True)
        os.chdir(current_cwd)
        logging.info(f"Job {self.name} submitted with job id {output.stdout}")
        self.job_id = output.stdout
