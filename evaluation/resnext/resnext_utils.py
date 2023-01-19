import time
import csv
import os

from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.transforms import *
import torchnet

from evaluation.resnext.resnext_datasets import SamplesDataset
from evaluation.resnext.transforms import *
import models

import torch.nn.functional as F


def generate_embeddings(checkpoint_path, dataset_dir, batch_size=128, n_workers=4, input="mel32"):
    print("loading model...")
    model = torch.load(checkpoint_path).module
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    model.float()

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        model.cuda()

    n_mels = 32
    if input == 'mel40':
        n_mels = 40

    feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])
    dataset = SamplesDataset(dataset_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=None,
                            pin_memory=use_gpu, num_workers=n_workers)

    model.eval()

    embeddings_list = []
    pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)

        n = inputs.size(0)

        inputs = Variable(inputs, volatile=True)

        if use_gpu:
            inputs = inputs.cuda()

        # forward
        embeddings = F.avg_pool2d(model(inputs), 8, 1).detach().squeeze().cpu()
        embeddings_list += [embeddings]

    del model

    return torch.cat(embeddings_list, dim=0).numpy()


def generate_label_distribution(checkpoint_path, dataset_dir, batch_size=128, n_workers=4, input="mel32"):
    print("loading model...")
    model = torch.load(checkpoint_path).module

    model.float()

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        model.cuda()

    n_mels = 32
    if input == 'mel40':
        n_mels = 40

    feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])
    dataset = SamplesDataset(dataset_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=None,
                            pin_memory=use_gpu, num_workers=n_workers)

    model.eval()

    logits_list = []
    pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)

        n = inputs.size(0)

        inputs = Variable(inputs, volatile=True)

        if use_gpu:
            inputs = inputs.cuda()

        # forward
        logits = model(inputs).detach().squeeze().cpu()
        logits_list += [logits]

    del model

    return torch.cat(logits_list, dim=0).numpy()


if __name__ == '__main__':
    print("Generating train embeddings...")
    embeddings = generate_embeddings()
    logits = generate_label_distribution()
    np.save("embeddings.npy", embeddings)
    np.save("logits.npy", logits)
