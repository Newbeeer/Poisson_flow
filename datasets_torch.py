import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import extract_archive
from torchaudio import load as torch_load
from torch.utils.data import DataLoader
import torch 

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
SC09 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def get_loader(dataset="speech", mode="training", args=None):
    assert mode in ["training", "validation", "testing"], "Wrong type given!"

    config = args.config

    if mode in ["testing", "validation"] or args.DDP:
        shuffling=False
    else:
        shuffling=True
    
    if dataset=="speech":
        if config.data.category == 'mel':
            data = SPEECHCOMMANDS_MEL(subset=mode, config=config)
        else:
            exit("Wrong data category for speech dataset!")
    
    # make a dataloader
    if args.DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=args.world_size, rank=args.rank)
    else:
        train_sampler = data
    
    loader = DataLoader(
        data,
        batch_size=config.training.batch_size,
        shuffle=shuffling,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    return loader
    
# added filtering for SC09 equivalence
def _load_list(root, *filenames, number_filter=False):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        # do filtering of SC09 dataset
        with open(filepath) as fileobj:
            if number_filter:
                output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj if line.split('/')[0] in SC09]
            else:
                output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output


def _get_speechcommands_metadata(filepath: str, path: str) -> Tuple[str, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    reldir, filename = os.path.split(relpath)
    _, label = os.path.split(reldir)
    # Besides the officially supported split method for datasets defined by "validation_list.txt"
    # and "testing_list.txt" over "speech_commands_v0.0x.tar.gz" archives, an alternative split
    # method referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of the original
    # paper, and the checksums file from the tensorflow_datasets package [1] is also supported.
    # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives have the form
    # "xxx.wav.wav", so file extensions twice needs to be stripped twice.
    # [1] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/speech_commands.txt
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    return relpath, 16_000, label, speaker_id, utterance_number


class SPEECHCOMMANDS_MEL(Dataset):

    def __init__(
        self,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        subset: Optional[str] = None,
        config = None,
        filter_numbers=True,
    ) -> None:

        if subset is not None and subset not in ["training", "validation", "testing"]:
            raise ValueError("When `subset` is not None, it must be one of ['training', 'validation', 'testing'].")


        # Get string representation of 'root' in case Path object is passed
        root = os.fspath('.')
        self._archive = os.path.join(root, folder_in_archive)
        self._mel_root = config.data.mel_root

        self._path = 'SpeechCommands/speech_commands_v0.02'
        
        if not os.path.exists(self._path):
            raise RuntimeError(
                f"The path {self._path} doesn't exist. "
                "Please check the ``root`` path or set `download=True` to download it"
            )

        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt", number_filter=filter_numbers)
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt", number_filter=filter_numbers)
        elif subset == "training":
            #DECIDED TO TRAIN ON ALL SINCE WE DON'T KNOW IF DIFFWAVE USED A SUBSET OR NOT
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            if filter_numbers:
                walker = [f for f in walker if f.split('/')[2] in SC09]
            self._walker = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w
            ]
        else:
            raise("Please specify the dataset subtype!")
        print(f"Initilaized MEL {subset} dataset.")
        

    def get_metadata(self, n: int) -> Tuple[str, int, str, str, int]:
        fileid = self._walker[n]
        return _get_speechcommands_metadata(fileid, self._archive)


    def __getitem__(self, n: int):
        metadata = self.get_metadata(n)
        path = metadata[0]
        splits = path.split('/')

        mel_path = os.path.join(self._mel_root, splits[1], splits[2].split('.')[0]+".npy")
        mel = np.load(mel_path)

        return torch.tensor(mel, dtype=torch.float).unsqueeze(0)


    def __len__(self) -> int:
        return len(self._walker)
