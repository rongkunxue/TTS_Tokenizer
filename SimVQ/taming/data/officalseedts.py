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
    def __init__(self, metalst, transform=None):
        self.metalst=metalst
        self.sample_rate = 16000
        self.channels = 1
        self.clip_seconds = -1
        self.transform = transform
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data_path = self.data[i][4]
        waveform, sample_rate = librosa.load(
            data_path, 
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
   
        return {
                "waveform": waveform,
                "prompt_text": self.data[i][1],
                "infer_text": self.data[i][3],
                "utt": self.data[i][0],
                "audio_path": self.data[i][4]
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
        lines = f.readlines()
        self.data = [line.strip().split('|') for line in lines]
        
if __name__ == "__main__":
    data_module = speechttsTest_en(metalst="/root/Github/TTS_Tokenizer/data/ref.txt")
    train_loader = torch.utils.data.DataLoader(data_module, batch_size=10, num_workers=4, shuffle=True)
    for batch in train_loader:
        a=1
        break