import argparse
from configs.get_configs import get_config
import numpy as np
from datasets import get_loader_mel
import torch
from tqdm import tqdm
# import the config you need here so that it is registered
from configs import audio_ddpmpp_128_tiny


# factor to correct to original PFGM authors norm
NORM_CONST = 0.671


def calculate_datanorm(args):
    # set batch size to 1
    args.config.training.batch_size = 1
    args.DDP = False
    loader = get_loader_mel(mode="training", args=args)

    norms = torch.tensor([])
    print("Calculating data norm...")
    for item in tqdm(loader):
        vec = item[0].flatten()
        norms = torch.cat((norms, vec.norm(dim=0, keepdim=True)))
    mean_norm = norms.mean() * NORM_CONST
    return mean_norm.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="config file")
    args = parser.parse_args()
    args.test = False

    config = get_config(args)
    args.config = config

    DATA_NORM = calculate_datanorm(args)
    N = config.data.image_height*config.data.image_height*config.data.num_channels + 1
    sigma = config.model.sigma_min
    tau = config.training.tau

    M = int(0.75 * np.log(DATA_NORM ** 2/(2 * np.sqrt(N) * sigma ** 2)) / np.log(1+tau)) + 1
    EXP = (sigma**(3/2) * (1+tau)**M)**(4/3)
    ZMAX = np.sqrt(2/np.pi) * sigma * (1 + tau)**M
    UPPERNORM = np.sqrt(N) * sigma * (1 + tau)**M
    
    print("Data norm: ", DATA_NORM)
    print("M: ", M)
    print("E: ", EXP)
    print("zmax: ", ZMAX)
    print("Upper norm: ", UPPERNORM)

    
if __name__ == "__main__":
    main()