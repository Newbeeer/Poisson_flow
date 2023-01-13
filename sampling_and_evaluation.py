import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from datasets import get_dataset
from configs.get_configs import get_config
import torchaudio

from librosa.feature.inverse import db_to_power, mel_to_audio
import soundfile as sf

from configs.default_audio_configs import get_mels_64, get_mels_128
from resnext.resnext_utils import generate_embeddings, generate_label_distribution

from compute_scores import compute_metrics

import json

import run_lib

class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val

print("Loading configuration ... ")
args = DotDict()
args.conf = "128_deep"
args.test = False
args.config = get_config(args)
args.workdir = "evaluation/128_deep"
args.checkpoint_dir = "checkpoints/pfgm"
args.eval_folder = "samples"
args.config.eval.batch_size = 1
args.DDP = False
args.config.eval.num_samples = 100
args.config.sampling.ckpt_number = 270000

print("Generate samples ... ")
run_lib.evaluate(args)

print("Convert mels to wav")
mel_cfg = get_mels_128()

sample_rate = mel_cfg.sample_rate
nfft = mel_cfg.nfft
hop_length = mel_cfg.hop_length

samples_path = f"{args.workdir}/ckpt_{args.config.sampling.ckpt_number}/mels/"
audio_path = f"{args.workdir}/ckpt_{args.config.sampling.ckpt_number}/audio/"
sample_files = os.listdir(samples_path)

for sample in sample_files:
    sample_name = sample.split(".")[0]
    mel_dat = np.load(f"{samples_path}{sample_name}.npz")["samples"]
    mel_data = mel_dat.squeeze()
    # reshape to -80 to 0 db range from librosa standard
    #mel_data /= mel_data.max()
    mel_data = np.clip(mel_data, 0.0, 1.0)
    mel_data *= 80
    mel_data -= 80

    mel_data = db_to_power(mel_data)
    audio = mel_to_audio(
        M=mel_data,
        sr=sample_rate,
        n_fft=nfft,
        hop_length=hop_length,
        win_length=hop_length * 4,
        center=True,
        power=1,
        n_iter=32,
        fmin=20.0,
        fmax=sample_rate / 2.0,
        pad_mode="reflect",
        norm='slaney',
        htk=True
    )
    
    sf.write(f'{audio_path}{sample_name}.wav', audio, sample_rate, 'PCM_24')

print("Compute embeddings and distrubution")
resnext_checkpoint = "checkpoints/resnext/1673112409482-resnext29_8_64_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02-best-loss.pth"
embeddings_train = np.load("checkpoints/resnext/embeddings.npy")
embeddings_samples = generate_embeddings(resnext_checkpoint, audio_path)

np.save("checkpoints/resnext/sample_embeddings.npy", embeddings_samples)

label_distribution_train = generate_label_distribution(resnext_checkpoint, audio_path)
np.save("checkpoints/resnext/sample_label_dist.npy", label_distribution_train)

print("Computing scores")
metrics = compute_metrics(embeddings_train, embeddings_samples) # , label_distribution_train)

with open('metrics.txt', 'w') as metric_file:
     metric_file.write(json.dumps(metrics))

