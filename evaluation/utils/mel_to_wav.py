import numpy as np
from librosa.feature.inverse import mel_to_audio
from librosa.feature.inverse import db_to_power
import argparse
import soundfile as sf
import torchaudio
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

sys.path.append('..')

from configs.default_audio_configs import get_mels_64, get_mels_128

plotting = False
clipping = False


def convert(mel, sample_rate, nfft, hop_length, clipping=False):
    mel_data = mel.copy()
    # reshape to -80 to 0 db range from librosa standard
    if not clipping:
        mel_data /= mel_data.max()
    else:
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
        fmin=20,
        fmax=sample_rate / 2.0,
        pad_mode="reflect",
        norm='slaney',
        htk=True
    )
    audio /= max(audio.max(), -audio.min())

    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    if "128" in args.dir:
        mel_cfg = get_mels_128()
    elif "64" in args.dir:
        mel_cfg = get_mels_64()
    else:
        raise ValueError("Could not find the mel size in dir name!")

    sample_rate = mel_cfg.sample_rate
    nfft = mel_cfg.nfft
    hop_length = mel_cfg.hop_length

    files = os.listdir(os.path.join(args.dir, args.ckpt))

    for fnum, file in enumerate(files):
        if file.split('.')[-1] not in ['np', 'npz', 'npy']:
            continue
        file_path = os.path.join(args.dir, args.ckpt, file)
        try:
            zfile = np.load(file_path)
            data = zfile.f.samples
        except:
            file = np.load(file_path)
            data = file

        os.makedirs(os.path.join(args.dir, 'audio', args.ckpt), exist_ok=True)
        print(f"Data in range [{data.min()},{data.max()}]")

        for i, im in tqdm(enumerate(data), total=data.shape[0]):
            mel_dat = torch.from_numpy(im).type(torch.float)
            if plotting and i < 5:
                plt.figure()
                plt.imshow(mel_dat)
                plt.show()
            mel_data = mel_dat.squeeze().numpy()
            # reshape to -80 to 0 db range from librosa standard
            if not clipping:
                mel_data /= mel_data.max()
            else:
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
                fmin=20,
                fmax=sample_rate / 2.0,
                pad_mode="reflect",
                norm='slaney',
                htk=True
            )
            audio /= max(audio.max(), -audio.min())
            if plotting:
                plt.figure()
                plt.plot(audio)
                plt.show()

            sf.write(f"{args.dir}/audio/{args.ckpt}/sample_{i + fnum * len(data)}.ogg", audio, 16_000)


if __name__ == "__main__":
    main()
