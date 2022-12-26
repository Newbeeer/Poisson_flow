import os
import sys
import time
import subprocess


def main():
    while(True):
        # get output of squeue
        result = subprocess.run(["squeue"], stdout=subprocess.PIPE)
        out = result.stdout.decode('utf-8')
        lines = out.split('\n')
        njobs = len(lines) - 2
        print(f"Currently running {njobs} jobs.")
        l1 = lines[1].split() # remove multiple whitespaces
        # check if only the autotrainer is running
        if njobs == 1 and l1[2] == 'wrap':
            print("Continuing training")
            subprocess.run([
                "sbatch",
                "-J",
                "autotrain"
                "job_slurm_training_audio.sh"])
        # check every 10 minutes
        time.sleep(60*10)


if __name__ == "__main__":
    main()