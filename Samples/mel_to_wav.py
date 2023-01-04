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

mel_cfg = get_mels_128()

sample_rate = mel_cfg.sample_rate
nfft = mel_cfg.nfft
hop_length = mel_cfg.hop_length

torch_inverse = False
plotting = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    files = os.listdir(os.path.join(args.dir, args.ckpt))

    for file in files:
        if file.split('.')[-1] not in ['np','npz']:
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
            data = torch.from_numpy(im).type(torch.float)
            if plotting and i < 5:
                plt.figure()
                plt.imshow(data)
                plt.show()
            if torch_inverse:
                # change to channel first
                data = data.permute(-1, 0, 1)
                # inverse mel scales to spectogram
                inverse_data = torchaudio.transforms.InverseMelScale(
                    sample_rate=sample_rate,
                    n_stft=1024,
                    n_mels=64,
                    f_min=20,
                    f_max=8000
                )(data)
                # inverse the spectogram
                audio = torchaudio.transforms.GriffinLim(n_fft=1024)(inverse_data[:, nfft // 2:, :]).squeeze().numpy()
            else:
                mel_data = data.squeeze().numpy()
                # reshape to -80 to 0 db range from librosa standard
                mel_data /= mel_data.max()
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
            if plotting:
                plt.figure()
                plt.plot(audio)
                plt.show()

            sf.write(f"{args.dir}/audio/{args.ckpt}/sample_{i}.ogg", audio, 16_000)


if __name__ == "__main__":
    main()
