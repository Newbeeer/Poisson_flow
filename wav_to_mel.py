import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    parser.add_argument('-t', '--target')
    args = parser.parse_args()

    SC09 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    folders = [f for f in os.listdir(args.dir) if f in SC09]

    for folder in folders:
        folder = os.path.join(args.dir, folder)
        if not os.path.isdir(folder): continue
        print(f"Processing {folder}")
        txt = f"python3 convert_folder.py -d {folder} -t {args.target}"
        shfile = f"tmp_folder.sh"
        with open(shfile, "w") as f:
            f.writelines("#!/bin/bash\n#SBATCH -n 2\n" + txt)
        subprocess.run({
            "sbatch",
            f"{shfile}",
        })
        os.remove(shfile)


if __name__ == "__main__":
    main()
