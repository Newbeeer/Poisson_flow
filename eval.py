import json
import argparse
from configs.get_configs import get_config
from evaluation.metrics import compute_metrics
from evaluation import evaluate


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--checkpoint_dir", default=None, required=True)
    parser.add_argument("--ckpt_number", type=int, required=True)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--DDP", action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--input_mel", default="128")
    parser.add_argument("--save_audio", action='store_true', default=False)
    parser.add_argument("--enable_benchmarking", action='store_true', default=False)
    parser.add_argument("--save_mels", action='store_true', default=False)
    parser.add_argument("--gt_metrics", action='store_true', default=False)
    parser.add_argument("--ode_solver", choices=['rk45', 'torchdiffeq', 'improved_euler', 'forward_euler'],
                        default='rk45')
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--z_max", type=float, default=45)
    parser.add_argument("--z_min", type=float, default=1e-3)
    parser.add_argument("--upper_norm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading configuration ...")
    args.test = True
    args.config = get_config(args)

    args.config.eval.batch_size = args.batch_size
    args.config.eval.num_samples = args.num_samples
    args.config.eval.input_mel = args.input_mel
    args.config.eval.save_audio = args.save_audio
    args.config.eval.enable_benchmarking = args.enable_benchmarking
    args.config.eval.save_mels = args.save_mels

    #  Sampling params
    args.config.sampling.ode_solver = args.ode_solver
    args.config.sampling.ckpt_number = args.ckpt_number
    args.config.sampling.N = args.steps
    args.config.sampling.z_max = args.z_max
    args.config.sampling.z_min = args.z_min
    args.config.sampling.upper_norm = args.upper_norm
    args.config.seed = args.seed

    args.eval_folder = f"os_{args.config.sampling.ode_solver}_N_{args.config.sampling.N}_zmax_" \
                       f"{args.config.sampling.z_max}_zmin_{args.config.
    sampling.z_min}_un_{args.config.sampling.upper_norm}_seed_{args.config.seed}"

    print("Generate samples... ")
    evaluate.run(args)

    print("Computing metrics... ")
    metrics = compute_metrics(f"{args.workdir}/ckpt_{args.config.sampling.ckpt_number}/{args.eval_folder}/audio",
                              gt_metrics=args.gt_metrics)

    #  Log metrics
    with open(f'{args.workdir}/ckpt_{args.config.sampling.ckpt_number}/{args.eval_folder}/metrics.txt',
              'w+') as metric_file:
        metric_file.write(json.dumps(metrics))


if __name__ == "__main__":
    eval()
