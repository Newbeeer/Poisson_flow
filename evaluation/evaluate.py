"Generate samples of a pretrained model and calculate metrics."
import os
import time
import copy
import logging
import gc
import numpy as np
import torch
import torchaudio
from torchvision.utils import make_grid, save_image
import soundfile as sf

# Keep the import below for registering all model definitions
from models import ncsnpp_audio, stablediff, diffwave
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import losses
from configs.default_audio_configs import get_mels_64, get_mels_128
from evaluation import sampling
from evaluation.utils.mel_to_wav import convert
import datasets
import methods
from utils.checkpoint import restore_checkpoint


def run(args):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    # Create directory to eval_folder
    config = args.config
    workdir = args.workdir
    checkpoint_dir = args.checkpoint_dir
    eval_folder = args.eval_folder

    os.makedirs(workdir, exist_ok=True)

    # Setup logger
    gfile_stream = open(os.path.join(args.workdir, 'stdout_eval.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')

    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    # Build data pipeline
    if config.eval.enable_loss:
        train_ds, eval_ds, _ = datasets.get_dataset(args, evaluation=True)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    net = mutils.create_model(args)
    print("Created Model")
    optimizer, scheduler = losses.get_optimizer(config, net.parameters())
    ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=net, ema=ema, scheduler=scheduler, step=0)

    torch.cuda.empty_cache()
    gc.collect()

    # Setup methods
    if config.training.sde.lower() == 'poisson':
        sde = methods.Poisson(args=args)
        sampling_eps = config.sampling.z_min
        print("--- sampling eps:", sampling_eps)
    else:
        raise NotImplementedError(f"Method {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn, reduce_mean=reduce_mean,
                                       method_name=config.training.sde.lower())

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size,
                          config.data.num_channels,
                          config.data.image_height, config.data.image_width)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, net)

    # Wait if the target checkpoint doesn't exist yet
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if config.training.sde == 'poisson':
        if config.sampling.ckpt_number > 0:
            ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(config.sampling.ckpt_number))
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{config.sampling.ckpt_number}.pth')
        else:
            raise ValueError("Please provide a ckpt_number!")

    if not os.path.exists(ckpt_filename):
        print(f"{ckpt_filename} does not exist! Loading from meta-checkpoint")
        ckpt_filename = os.path.join(checkpoint_dir, os.pardir, 'checkpoints-meta', 'checkpoint.pth')
        if not os.path.exists(ckpt_filename):
            logging.info("No checkpoints-meta")
        return

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    print("Loading from ", ckpt_path)
    try:
        state = restore_checkpoint(ckpt_path, state, map_location=config.device)
    except Exception as e:
        print("Loading Failed!")
        print(e)
        time.sleep(60)
        try:
            state = restore_checkpoint(ckpt_path, state, map_location=config.device)
        except Exception as e:
            time.sleep(120)
            state = restore_checkpoint(ckpt_path, state, map_location=config.device)

    ckpt = config.sampling.ckpt_number
    ema.copy_to(net.parameters())

    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
        logging.info("please don't set the config.eval.save_images flag, or the datasets wouldn't be loaded.")
        all_losses = []
        eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
        for i, batch in enumerate(eval_iter):
            eval_batch = batch.to(config.device).float()
            eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step(state, eval_batch)
            all_losses.append(eval_loss.item())
            if (i + 1) % 1000 == 0:
                logging.info("Finished %dth step loss evaluation" % (i + 1))

        all_losses = np.asarray(all_losses)
        np.savez_compressed(os.path.join(workdir, f"ckpt_{ckpt}_loss.npz"), all_losses=all_losses,
                            mean_loss=all_losses.mean())

    # Generate samples
    if config.eval.enable_sampling:
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size
        # Directory to save samples. Different for each host to avoid writing conflicts
        sample_dir = os.path.join(workdir, f"ckpt_{ckpt}/{eval_folder}/mels")
        audio_dir = os.path.join(workdir, f"ckpt_{ckpt}/{eval_folder}/audio")
        img_dir = os.path.join(workdir, f"ckpt_{ckpt}/{eval_folder}/img")
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        print(f"Sampling for {num_sampling_rounds} rounds...")
        for r in range(num_sampling_rounds):
            print("sampling -- ckpt: %d, round: %d" % (ckpt, r))

            t1 = time.time()
            samples, n = sampling_fn(net)
            result = time.time() - t1

            if config.eval.enable_benchmarking:
                b_file = open(f"{workdir}/ckpt_{ckpt}/{eval_folder}/benchmarking.txt", "a+")
                b_file.write(f"{result}\n")
                b_file.close()

                nfe_file = open(f"{workdir}/ckpt_{ckpt}/{eval_folder}/nfe.txt", "a+")
                nfe_file.write(f"{n}\n")
                nfe_file.close()

            print(f"nfe: {n}")
            print(f"sample shape: {samples.shape}")
            samples_torch = copy.deepcopy(samples)
            samples_torch = samples_torch.view(-1, config.data.num_channels, config.data.image_height,
                                               config.data.image_width)

            # sample the output matrices differently for pictures vs mel spectograms
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()

            if config.data.category == 'mel':
                print("Saving images as raw mel specs.")
                samples = samples.reshape(
                    (-1, config.data.image_height, config.data.image_width, config.data.num_channels))
                if config.eval.save_mels == True:
                    np.savez_compressed(os.path.join(sample_dir, f"samples_{r}.npz"), samples=samples)

                if config.eval.save_audio:
                    mel_cfg = None
                    if "128" in config.eval.input_mel:
                        mel_cfg = get_mels_128()
                    elif "64" in config.eval.input_mel:
                        mel_cfg = get_mels_64()
                    else:
                        raise ValueError("Could not find mel size name!")

                    sample_rate = mel_cfg.sample_rate
                    nfft = mel_cfg.nfft
                    hop_length = mel_cfg.hop_length

                    for i, mel in enumerate(samples):
                        audio = convert(mel.squeeze(), sample_rate, nfft, hop_length, clipping=False)
                        sf.write(f'{audio_dir}/sample_{r}_{i}.wav', audio, sample_rate, 'PCM_24')

                if config.eval.save_images:
                    # Saving a few generated images for debugging / visualization
                    image_grid = make_grid(samples_torch, nrow=int(np.sqrt(len(samples_torch))))
                    save_image(image_grid, f"{img_dir}/ode_images_{r}.png")

            elif config.data.category == 'audio':
                samples = samples_torch.reshape((-1, config.data.image_width)).cpu()
                for si, sample in enumerate(samples):
                    sample = torch.clamp(sample, -1.0, 1.0).unsqueeze(0)
                    torchaudio.save(os.path.join(audio_dir, f"sample_{r}_{si}.wav"), sample, 16000)

        # Free allocated net after evaluation
        del net
