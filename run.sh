CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode eval \
--workdir ../Poisson_flow/test --config.eval.enable_sampling --config.sampling.N 10

CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode eval \
--workdir ../Poisson_flow/test --config.eval.enable_sampling --config.sampling.N 50

CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode eval \
--workdir ../Poisson_flow/test --config.eval.enable_sampling --config.sampling.N 20

CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode eval \
--workdir ../Poisson_flow/test --config.eval.enable_sampling --config.sampling.N 100