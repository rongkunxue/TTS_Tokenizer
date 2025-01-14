from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np
import lightning as L
from librosa.filters import mel as librosa_mel_fn
import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
import pathlib
from tqdm import tqdm
from typing import List, Tuple, Optional
from transformers import HubertModel,  Wav2Vec2FeatureExtractor

MAX_WAV_VALUE = 32767.0  # NOTE: 32768.0 -1 to prevent int16 overflow (results in popping sound in corner cases)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


mel_basis_cache = {}
hann_window_cache = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
        center (bool): Whether to pad the input to center the frames. Default is False.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


class SpeechTokenizerDataModule(L.LightningDataModule):
    def __init__(self, batch_size,train_metalst,val_metalst,segement_size, num_workers):
        super(SpeechTokenizerDataModule).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.segement_size=segement_size
        self.train_metalst = train_metalst
        self.val_metalst = val_metalst
        self.prepare_data_per_node = True
        self._log_hyperparams=False
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def pad_collate_fn(self,data):
        # return pad_sequence(data, batch_first=True)
        # return pad_sequence(*data)
        is_one_data = not isinstance(data[0], tuple)
        outputs = []
        if is_one_data:
            for datum in data:
                if isinstance(datum, torch.Tensor):
                    output = datum.unsqueeze(0)
                else:
                    output = torch.tensor([datum])
                outputs.append(output)
            return tuple(outputs)        
        for datum in zip(*data):
            if isinstance(datum[0], torch.Tensor):
                output = pad_sequence(datum, batch_first=True)
            else:
                output = torch.tensor(list(datum))
            outputs.append(output)
        return tuple(outputs)
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = audioDataset(self.train_metalst,self.segement_size,False)
        if stage == "test" or stage is None:
            self.test = audioDataset(self.val_metalst,self.segement_size,True)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,collate_fn=self.pad_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False, collate_fn=self.pad_collate_fn)

class audioDataset(Dataset):
    def __init__(self,
                 file_path,
                 segment_size,
                 if_val):
        super().__init__()
        f = open(file_path)
        lines = f.readlines()
        self.data = [line.strip().split('|') for line in lines]

        self.segment_size = segment_size
        self.sample_rate = 24000
        self.downsample_rate = 320
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960",cache_dir="/checkpoint")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        data_path = self.data[i][4]
        audio, sr = torchaudio.load(data_path)
        audio = audio.mean(axis=0)
        if sr != 24000:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        if sr != 16000:
            audio_16k = torchaudio.functional.resample(audio, sr, self.sample_rate)
        if audio.size(-1) > self.segment_size:
            max_audio_start = audio.size(-1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_size]
            audio_16k = audio_16k[audio_start:audio_start+self.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(-1)), 'constant')
            audio_16k = torch.nn.functional.pad(audio_16k, (0, self.segment_size - audio.size(-1)), 'constant')
        audio = torch.FloatTensor(audio)
        audio_16k = torch.FloatTensor(audio_16k)
        feature_value = self.feature_extractor(audio_16k.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values
        audio = audio.unsqueeze(0)  # [B(1), self.segment_size]
        mel = mel_spectrogram(
                    audio,
                    1024,
                    100,
                    24000,
                    256,
                    1024,
                    0,
                    None,
                    center=False,
                )
        assert (
                audio.shape[1] == mel.shape[2] * 256
            ), f"Audio length must be mel frame length * hop_size. Got audio shape {audio.shape} mel shape {mel.shape} "
        return audio.squeeze(0),mel.squeeze(),audio_16k,feature_value.squeeze(0)
