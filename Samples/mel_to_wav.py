import numpy as np
import librosa.display
from librosa.feature.inverse import mel_to_audio
from librosa.feature.inverse import db_to_power
import argparse
import soundfile as sf
import torchaudio
import torch
import matplotlib.pyplot as plt

sample_rate = 16_000
nfft = 1024
torch_inverse = False
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    zfile = np.load(f"{args.dir}/samples_0.npz")
    data = zfile.f.samples

    for i, im in enumerate(data):
        data = torch.from_numpy(im).type(torch.float)
        plt.figure()
        plt.imshow(data)
        plt.show()
        if torch_inverse:
        # change to channel first
            data = data.permute(2, 0, 1)
            # inverse mel scales to spectogram
            inverse_data = torchaudio.transforms.InverseMelScale(
                sample_rate=sample_rate,
                n_stft=64,
                n_mels=64,
                f_min=20,
                f_max=8000
                )(data)
            # inverse the spectogram
            audio = torchaudio.transforms.GriffinLim(n_fft=1024)(inverse_data[:,nfft//2:,:]).squeeze().numpy()
        else:
            mel_data = data.squeeze().numpy()
            # reshape to -80 to 0 db range from librosa standard
            mel_data = mel_data * 80
            mel_data -= mel_data.max()
            mel_data = db_to_power(mel_data)
            audio = mel_to_audio(M=mel_data, sr=sample_rate,n_fft=1024, hop_length=256, center=True, power=1, n_iter=16)
            
        print(audio.shape)
        plt.figure()
        plt.plot(audio)
        plt.show()
        sf.write(f"{args.dir}/audio/sample_{i}.wav", audio, 16_000, 'PCM_24')

if __name__ == "__main__":
    main()