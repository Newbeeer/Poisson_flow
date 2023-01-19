import os
import argparse
import librosa
import numpy as np
import warnings

# filter warnings for mel bin errors, be aware of it!
warnings.filterwarnings("ignore")
from configs.get_configs import get_config


def librosa_melspec(y, config):
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_fft=config.nfft,
        win_length=config.hop_length * 4,
        hop_length=config.hop_length,
        n_mels=config.num_mels,
        fmin=config.fmin,
        fmax=config.sample_rate / 2.0,
        center=True,
        pad_mode="reflect",
        power=1.0,
        norm='slaney',
        htk=True,
    )
    return melspec


def convert_folder(folder, target, config):
    """
    It takes a folder of audio files, converts them to mel spectrograms, and saves them as numpy arrays
    
    :param folder: the folder containing the audio files
    :param target: the folder where the converted files will be saved
    :param config: the config file with mel informations
    """
    number = folder.split('/')[-1]
    os.makedirs(os.path.join(target, number), exist_ok=True)
    files = os.listdir(folder)
    for f in files:
        f_path = os.path.join(folder, f)
        waveform, sr = librosa.load(f_path, sr=config.sample_rate, res_type='kaiser_fast')
        mel = librosa_melspec(waveform, config)
        mel = librosa.power_to_db(mel, top_db=80).astype(np.float32)
        # scale from 0 to 1
        mel = mel - mel.min()
        mel = mel / mel.max()
        # pad with zeros
        pad_len = config.image_size - mel.shape[1]
        if pad_len > 0:
            mel = np.hstack((mel, np.zeros((mel.shape[0], pad_len))))
        np.save(os.path.join(target, number, f.split('.')[0]), mel)
    print("Converted {folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', required=True)
    parser.add_argument('-t', '--target', required=True)
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()
    config = get_config(args.config).data.spec
    convert_folder(args.dir, args.target, config)


if __name__ == "__main__":
    main()
