"""Return training and evaluation/test datasets from config files."""
from datasets_torch import get_loader as get_torch_loader
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def get_dataset(args, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    config = args.config

    # Create dataset builders for each dataset.
    if config.data.dataset == "speech_commands":
        if config.data.category in ['audio', 'mel']:
            train_loader = get_loader_mel(mode="training", args=args)
            valid_loader = get_loader_mel(mode="validation", args=args)
    else:
        raise NotImplementedError(
            f'Dataset {config.data.dataset} not yet supported.')

    dataset_builder = None

    return train_loader, valid_loader, dataset_builder


class MelDataset(Dataset):
    def __init__(self, args):
        self.args = args
        # make list from all files in all subfolders in args.data.mel_root
        self.paths = [os.path.join(path, name) for path, subdirs, files in os.walk(args.data.mel_root) for name in
                      files]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        mel = np.load(self.paths[index])
        mel = torch.tensor(mel, dtype=torch.float)
        return mel.unsqueeze(0)


def get_loader_mel(mode="training", args=None):
    assert mode in ["training", "validation", "testing"], "Wrong type given!"

    config = args.config

    if mode in ["testing", "validation"] or args.DDP:
        shuffling = False
    else:
        shuffling = True

    data = MelDataset(config)

    # make a dataloader
    if args.DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=args.world_size,
                                                                        rank=args.rank)
    else:
        train_sampler = None

    loader = DataLoader(
        data,
        batch_size=config.training.batch_size,
        shuffle=shuffling,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        sampler=train_sampler
    )

    return loader
