import os
import argparse
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import librosa
from configs.poisson.audio_ddpmpp import get_config
from torchaudio import load as torch_load
import torch 
import numpy as np
import tensorflow as tf

SC09 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
config = get_config()

meltransform = MelSpectrogram(
            sample_rate=config.data.sample_rate,
            n_fft=config.data.nfft,
            n_mels=config.data.num_mels,
            win_length=config.data.hop_length * 4,
            hop_length=config.data.hop_length,
            f_min=20.0,
            f_max = config.data.sample_rate / 2.0,
            power = 1.0, # coherent source
            normalized=True
            )

ptodb = AmplitudeToDB(stype='magnitude', top_db=80)

def convert_folder(folder, target, file_writer):

    files = os.listdir(folder)
    for f in files:
        f_path = os.path.join(folder, f)
        waveform, sr = torch_load(f_path)
        mel = meltransform(waveform) # 1 MEL_BINS LENGTH = 1x64x64
        mel = ptodb(mel) # amp to db
        mel = mel - mel.min()
        mel = mel / mel.max()
        # pad with zeros
        pad_len = config.data.image_size - mel.shape[2]
        if pad_len > 0:
            mel = torch.cat((mel, torch.zeros((mel.shape[0], mel.shape[1], pad_len))), axis=-1)
        # save the array
        record_bytes = tf.train.Example(
            features=tf.train.Features(feature={
                "mel": tf.train.Feature(float_list=tf.train.FloatList(value=mel.numpy().ravel()))
            })
        ).SerializeToString()
        file_writer.write(record_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    parser.add_argument('-t', '--target')
    args = parser.parse_args()

    folders = [f for f in os.listdir(args.dir) if f in SC09]

    with tf.io.TFRecordWriter(args.target) as file_writer:
        for folder in folders:
            folder = os.path.join(args.dir, folder)
            if not os.path.isdir(folder): continue
            print(f"Processing {folder}")
            convert_folder(folder, args.target, file_writer)

if __name__ == "__main__":
    main()