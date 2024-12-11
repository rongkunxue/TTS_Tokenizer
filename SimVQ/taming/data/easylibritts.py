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
import lightning as L
from torch.utils.data import Dataset, DataLoader


class LibriTTSDataModule(L.LightningDataModule):
    def __init__(self, batch_size=20, num_workers=8,dataset_path="/mnt/nfs3/zhangjinouwen/dataset/LibriTTS"):
        super(LibriTTSDataModule).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = dataset_path
        self.prepare_data_per_node = True
        self._log_hyperparams=False
        self.allow_zero_length_dataloader_with_multiple_devices = False 

    def pad_collate_fn(self,batch):
        """Collate function for padding sequences."""
        return {
            "waveform": torch.nn.utils.rnn.pad_sequence(
                [x["waveform"].transpose(0, 1) for x in batch], 
                batch_first=True, 
                padding_value=0.
            ).permute(0, 2, 1),
            "audio_path": [x["audio_path"] for x in batch]
        }
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = LibriTTSTrain(self.dataset_path,1)
            self.dev = LibriTTSDev(self.dataset_path,1)
        if stage == "test" or stage is None:
            self.test_other = LibriTTSTestOther(self.dataset_path,-1)
            self.test_clean = LibriTTSTestClean(self.dataset_path,-1)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,collate_fn=self.pad_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.pad_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_other, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.pad_collate_fn)


class LibriTTSBase(Dataset):
    def __init__(self, data_root,clip_seconds=-1, transform=None):
        self.data_root = data_root
        self.sample_rate = 16000
        self.channels = 1
        self.clip_seconds = clip_seconds
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
                
class LibriTTSTrain(LibriTTSBase):
    NAME = "train"
    
    def _load(self):
        txt_filelist = os.path.join(self.data_root, self.NAME + ".txt")
        with open(txt_filelist, "r") as f:
            self.data = f.read().splitlines()


class LibriTTSDev(LibriTTSBase):
    NAME = "dev"
    
    def _load(self):
        txt_filelist = os.path.join(self.data_root, self.NAME + ".txt")
        with open(txt_filelist, "r") as f:
            self.data = f.read().splitlines()
            

class LibriTTSTestOther(LibriTTSBase):
    NAME = "test-other"
    
    def _load(self):
        txt_filelist = os.path.join(self.data_root, self.NAME + ".txt")
        with open(txt_filelist, "r") as f:
            self.data = f.read().splitlines()
            
class LibriTTSTestClean(LibriTTSBase):
    NAME = "test-clean"
    
    def _load(self):
        txt_filelist = os.path.join(self.data_root, self.NAME + ".txt")
        with open(txt_filelist, "r") as f:
            self.data = f.read().splitlines()