import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import extract_archive
from torchaudio import load as torch_load
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import DataLoader
import torch 
import logging

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
SC09 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

_CHECKSUMS = {
    "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz": "743935421bb51cccdb6bdd152e04c5c70274e935c82119ad7faeec31780d811d",  # noqa: E501
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",  # noqa: E501
}


def get_loader(dataset="speech", mode="training", config=None):
    assert mode in ["training", "validation", "testing"], "Wrong type given!"
    if mode in ["testing", "validation"]:
        shuffling=False
    else:
        shuffling=True
    if dataset=="speech":
        if config.data.category == 'mel':
            data = SPEECHCOMMANDS_MEL(root='.', download=True, subset=mode, config=config)
        elif config.data.category == 'audio':
            data = SPEECHCOMMANDS(root='.', download=True, subset=mode, config=config)
        else:
            exit("Wrong data category for speech dataset!")
    
    # make a dataloader
    loader = DataLoader(
        data,
        batch_size=config.training.batch_size,
        shuffle=shuffling,
        drop_last=True,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True
    )

    return loader

def _load_waveform(
    root: str,
    filename: str,
    exp_sample_rate: int,
):
    path = os.path.join(root, filename)
    waveform, sample_rate = torch_load(path)
    if exp_sample_rate != sample_rate:
        raise ValueError(f"sample rate should be {exp_sample_rate}, but got {sample_rate}")
    return waveform


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


class SPEECHCOMMANDS(Dataset):
    """*Speech Commands* :cite:`speechcommandsv2` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"speech_commands_v0.01"`` and ``"speech_commands_v0.02"``
            (default: ``"speech_commands_v0.02"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"SpeechCommands"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str or None, optional):
            Select a subset of the dataset [None, "training", "validation", "testing"]. None means
            the whole dataset. "validation" and "testing" are defined in "validation_list.txt" and
            "testing_list.txt", respectively, and "training" is the rest. Details for the files
            "validation_list.txt" and "testing_list.txt" are explained in the README of the dataset
            and in the introduction of Section 7 of the original paper and its reference 12. The
            original paper can be found `here <https://arxiv.org/pdf/1804.03209.pdf>`_. (Default: ``None``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
        config = None,
        filter_numbers=True
    ) -> None:

        if subset is not None and subset not in ["training", "validation", "testing"]:
            raise ValueError("When `subset` is not None, it must be one of ['training', 'validation', 'testing'].")

        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            base_url = "http://download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)
     
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                extract_archive(archive, self._path)
        else:
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
            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            if filter_numbers:
                walker = [f for f in walker if f.split('/')[2] in SC09]
            
            self._walker = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]
        

    def get_metadata(self, n: int) -> Tuple[str, int, str, str, int]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Path to the audio
            int:
                Sample rate
            str:
                Label
            str:
                Speaker ID
            int:
                Utterance number
        """
        fileid = self._walker[n]
        return _get_speechcommands_metadata(fileid, self._archive)


    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Label
            str:
                Speaker ID
            int:
                Utterance number
        """
        metadata = self.get_metadata(n)
        sr = metadata[1]
        waveform = _load_waveform(self._archive, metadata[0], sr)
        
        # TODO padding or cutting, flip phase randomly, normalize from -1 to 1
        waveform = torch.mean(waveform,dim=0, keepdim=True)
        len_wav = waveform.shape[-1]
        if len_wav < sr:
            waveform = torch.cat((waveform, torch.zeros(sr - len_wav).unsqueeze(0)), dim=-1)
            
        # normalize from -1 to 1
        waveform /= waveform.max()

        return waveform


    def __len__(self) -> int:
        return len(self._walker)


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
        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)
        self._mel_root = config.data.mel_root

        basename = os.path.basename(url)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)
        
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
            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            if filter_numbers:
                walker = [f for f in walker if f.split('/')[2] in SC09]
            self._walker = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
            ]
        else:
            raise("Please specify the dataset subtype!")
        print("INITALIZED MEL DATASET")
        

    def get_metadata(self, n: int) -> Tuple[str, int, str, str, int]:
        fileid = self._walker[n]
        return _get_speechcommands_metadata(fileid, self._archive)


    def __getitem__(self, n: int):
        metadata = self.get_metadata(n)
        path = metadata[0]
        splits = path.split('/')

        mel_path = os.path.join(self._mel_root, splits[1], splits[2].split('.')[0]+".npy")
        mel = np.load(mel_path)

        return torch.FloatTensor(mel).unsqueeze(0)


    def __len__(self) -> int:
        return len(self._walker)
