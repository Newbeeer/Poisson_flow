import argparse
from utils.convert_folder import convert_folder
from configs import audio_ddpmpp_128_deep, audio_ddpmpp_64_deep, audio_diffwave_128, audio_sd_128, audio_sd_64
import os
from multiprocessing import Pool
from multiprocessing import cpu_count
from configs.get_configs import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder with audio data')
    parser.add_argument('--target_dir', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--conf', type=str, help='Name of the config')
    args = parser.parse_args()

    # convert all class folders in the input directory
    folder_list = [dir for dir in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, dir))]

    config = get_config(args).data.spec
    with Pool(cpu_count()) as p:
        p.starmap(convert_folder,
                  [(os.path.join(args.input_dir, folder), os.path.join(args.target_dir), config) for folder in
                   folder_list])


if __name__ == "__main__":
    main()
