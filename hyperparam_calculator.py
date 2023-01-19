import argparse
from configs.get_configs import get_config
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default='diffwave')
    args = parser.parse_args()
    args.test = False

    config = get_config(args)

    DATA_NORM = 28 # is 32 for cifar | 22 for 64 mels | 51 for 128 mels | 28 for pure audio
    N = config.data.image_height*config.data.image_height*config.data.num_channels + 1
    #N = 3072 #3072 for cifar
    sigma = config.model.sigma_min
    tau = config.training.tau

    M = int(0.75 * np.log(DATA_NORM ** 2/(2 * np.sqrt(N) * sigma ** 2)) / np.log(1+tau)) + 1
    EXP = (sigma**(3/2) * (1+tau)**M)**(4/3)
    ZMAX = np.sqrt(2/np.pi) * sigma * (1 + tau)**M
    UPPERNORM = np.sqrt(N) * sigma * (1 + tau)**M
    
    print("M: ", M)
    print("E: ", EXP)
    print("zmax: ", ZMAX)
    print("Upper norm: ", UPPERNORM)

    
if __name__ == "__main__":
    main()