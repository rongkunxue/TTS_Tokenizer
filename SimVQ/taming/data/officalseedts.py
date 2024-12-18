import os, tarfile, glob, shutil
import yaml
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import librosa
import torch
import random
import torchaudio


class speechttsBase(Dataset):
    def __init__(self, date_root, transform=None):
        self.data_root = date_root
        self.sample_rate = 24000
        self.channels = 1
        self.clip_seconds = -1
        self.transform = transform
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data_path = self.data[i]
        waveform, sample_rate = librosa.load(
            os.path.join(self.data_root, data_path), 
            sr=self.sample_rate,
            mono=self.channels == 1
        )
        waveform = torch.as_tensor(waveform)
        
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            waveform = waveform.expand(self.channels, -1)
            
        #gain = np.random.uniform(-1, -6) if self.NAME == "train" else -3
        #waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, self.sample_rate, [["norm", f"{gain:.2f}"]])
            
        if self.transform:
            waveform = self.transform(waveform)
            
        if self.clip_seconds < 0:
            pass
        elif waveform.shape[1] > self.clip_seconds * sample_rate:
            start = random.randint(0, waveform.shape[1] - self.clip_seconds * sample_rate - 1)
            waveform = waveform[:, start:start + self.clip_seconds * sample_rate] # cut tensor
        else:
            pad_length = self.clip_seconds * sample_rate - waveform.size(1)
            padding_tensor = waveform.repeat(1, 1 + pad_length // waveform.size(1))
            waveform = torch.cat((waveform, padding_tensor[:, :pad_length]), dim=1)
        
        return {
                "waveform": waveform,
                "audio_path": data_path
            }
               
            
class speechttsTest_zh(speechttsBase):
    NAME = "zh"
    def _load(self):
        txt_filelist = os.path.join(self.data_root, self.NAME + ".txt")
        with open(txt_filelist, "r") as f:
            self.data = f.read().splitlines()

class speechttsTest_en(speechttsBase):
    NAME = "en"
    def _load(self):
        f = open(self.metalst)
        self.lines = f.readlines()
        f.close()
        
