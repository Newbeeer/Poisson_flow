import os
import argparse
import librosa
import numpy as np
from configs.default_audio_configs import get_mels_128, get_mels_64
SC09 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


config = get_mels_128()
FMIN = 20.0

def librosa_melspec(y):
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_fft=config.nfft,
        win_length=config.hop_length * 4,
        hop_length=config.hop_length,
        n_mels=config.num_mels, 
        fmin=FMIN, 
        fmax=config.sample_rate / 2.0,
        center=True,
        pad_mode="reflect",
        power=1.0,
        norm='slaney',
        htk=True,
    )
    return melspec

def convert_folder(folder, target):
    number = folder.split('/')[-1]
    os.makedirs(os.path.join(target, number))
    files = os.listdir(folder)
    for f in files:
        f_path = os.path.join(folder, f)
        waveform, sr = librosa.load(f_path, sr=config.sample_rate, res_type='kaiser_fast')
        mel = librosa_melspec(waveform)
        mel = librosa.power_to_db(mel, top_db=None).astype(np.float32)
        mel = mel - mel.min()
        mel = mel / mel.max()
        # pad with zeros
        pad_len = config.image_size - mel.shape[1]
        if pad_len > 0:
            mel = np.hstack((mel, np.zeros((mel.shape[0], pad_len))))
        np.save(os.path.join(target, number, f.split('.')[0]), mel)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    parser.add_argument('-t', '--target')
    args = parser.parse_args()

    folders = [f for f in os.listdir(args.dir) if f in SC09]

    for folder in folders:
        folder = os.path.join(args.dir, folder)
        if not os.path.isdir(folder): continue
        print(f"Processing {folder}")
        convert_folder(folder, args.target)

if __name__ == "__main__":
    main()