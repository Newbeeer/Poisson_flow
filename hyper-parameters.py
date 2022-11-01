# Calculating the hyper-parameters of PFGM by data norm and dimension
# Details in Appendix B.1 and B.2 in https://arxiv.org/pdf/2209.11178.pdf

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PFGM hyper-parameters')
parser.add_argument('--data_norm', type=float, default=30., help='Average norm of data')
parser.add_argument('--data_dim', type=float, default=3072, help='Data dimension')
parser.add_argument('--sigma', type=float, default=0.01, help='config.model.sigma_end')
parser.add_argument('--tau', type=float, default=0.03, help='config.training.tau')

args = parser.parse_args()

print("Recommended hyper-parameters for your dataset:")

M = int(0.75 * np.log(args.data_norm ** 2/(2 * np.sqrt(args.data_dim) * args.sigma ** 2))
                                / np.log(1+args.tau)) + 1
print("config.training.M:", M)

print("config.sampling.z_max:", np.sqrt(2/np.pi) * args.sigma * (1+args.tau) ** M)

print("config.sampling.upper_norm:", np.sqrt(args.data_dim) * args.sigma * (1+args.tau) ** M)