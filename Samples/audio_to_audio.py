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

from configs.mel_configs import get_mels_64, get_mels_128

mel_cfg = get_mels_128()

sample_rate = mel_cfg.sample_rate
nfft = mel_cfg.nfft
hop_length = mel_cfg.hop_length

torch_inverse = False
plotting = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    files = os.listdir(os.path.join(args.dir, args.ckpt))
    for file in files:
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
            data = torch.from_numpy(im).type(torch.float).squeeze()
            if plotting and i < 5:
                plt.figure()
                try:
                    plt.imshow(data)
                except:
                    plt.plot(data)
                plt.show()
            audio = data

            sf.write(f"{args.dir}/audio/{args.ckpt}/sample_{i}.ogg", audio, sample_rate)


if __name__ == "__main__":
    main()
