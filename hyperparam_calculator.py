import argparse
from configs.get_configs import get_config
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default='128_deep')
    args = parser.parse_args()
    args.test = False

    config = get_config(args)
    N = config.data.image_height*config.data.image_height*config.data.num_channels + 1
    sigma = config.model.sigma_min
    tau = config.training.tau
    M = config.training.M

    EXP = (sigma**(3/2) * (1+tau)**M)**(4/3)
    ZMAX = np.sqrt(2/np.pi) * sigma * (1 + tau)**M
    UPPERNORM = np.sqrt(N) * sigma * (1 + tau)**M
    
    print("E: ", EXP)
    print("zmax: ", ZMAX)
    print("Upper norm: ", UPPERNORM)
if __name__ == "__main__":
    main()