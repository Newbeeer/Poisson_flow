"""
Converts the SpeechCommands/speech_commands_v0.02/ folder to tfrecord dataset.
The wav is converted to a melspectrogram using librosa or optionally pytorch (inverting librosa is easier)
The folders are filtered to only contain the numbers from 0 to 9 automatically
"""

import os
import argparse
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import librosa
from configs.poisson.audio_ddpmpp import get_config
from torchaudio import load as torch_load
import torch 
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm

SC09 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
config = get_config()
FMIN = 20.0

meltransform = MelSpectrogram(
            sample_rate=config.data.sample_rate,
            n_fft=config.data.nfft,
            n_mels=config.data.num_mels,
            win_length=config.data.hop_length * 4,
            hop_length=config.data.hop_length,
            center=True,
            pad_mode="reflect",
            norm='slaney',
            onesided=True,
            mel_scale="htk",
            f_min=FMIN,
            f_max = config.data.sample_rate / 2.0,
            power = 1.0, # coherent source
            normalized=True
            )


ptodb = AmplitudeToDB(stype='magnitude')


def librosa_melspec(y):
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=config.data.sample_rate,
        n_fft=config.data.nfft,
        win_length=config.data.hop_length * 4,
        hop_length=config.data.hop_length,
        n_mels=config.data.num_mels, 
        fmin=FMIN, 
        fmax=config.data.sample_rate / 2.0,
        center=True,
        pad_mode="reflect",
        power=1.0,
        norm='slaney',
        htk=True,
    )
    return melspec


def convert_folder(folder, args, file_writer):
    target = args.target
    use_torch = args.torch

    files = os.listdir(folder)
    for f in files:
        f_path = os.path.join(folder, f)
        waveform, sr = librosa.load(f_path, sr=config.data.sample_rate, res_type=)
        if use_torch:
            mel = meltransform(waveform) # 1 MEL_BINS LENGTH = 1x64x64
            mel = ptodb(mel) # amp to db
            mel = mel - mel.min()
            mel = mel / mel.max()
            # pad with zeros
            pad_len = config.data.image_size - mel.shape[2]
            if pad_len > 0:
                mel = torch.cat((mel, torch.zeros((mel.shape[0], mel.shape[1], pad_len))), axis=-1)
            mel_vec = mel.numpy().ravel()
        else:
            mel = librosa_melspec(waveform)
            mel = librosa.power_to_db(mel, top_db=None).astype(np.float32)
            mel = mel - mel.min()
            mel = mel / mel.max()
            # pad with zeros
            pad_len = config.data.image_size - mel.shape[1]
            if pad_len > 0:
                mel = np.hstack((mel, torch.zeros((mel.shape[0], pad_len))))
            mel_vec = mel.ravel()

        # save the array
        record_bytes = tf.train.Example(
            features=tf.train.Features(feature={
                "mel": tf.train.Feature(float_list=tf.train.FloatList(value=mel_vec))
            })
        ).SerializeToString()
        file_writer.write(record_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    parser.add_argument('-t', '--target')
    parser.add_argument('--torch', action='store_true')
    args = parser.parse_args()

    folders = [f for f in os.listdir(args.dir) if f in SC09]
    print(f"Generating {config.data.num_mels}x{config.data.image_size} specs.")

    with tf.io.TFRecordWriter(args.target) as file_writer:
        for folder in tqdm(folders):
            folder = os.path.join(args.dir, folder)
            if not os.path.isdir(folder): continue
            print(f"Processing {folder}")
            convert_folder(folder, args, file_writer)

if __name__ == "__main__":
    main()