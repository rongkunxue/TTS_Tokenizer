from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import random
from io import BytesIO
import lightning as L
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torchaudio.transforms import Resample

class EmiliaDataset(IterableDataset):
    def __init__(self,clip_seconds=-1,transform=None):
        path = "ZH/*.tar"
        self.dataset = load_dataset("amphion/Emilia-Dataset", data_files={"zh": path}, split="zh", streaming=True)
        self.transform=transform
        self.clip_seconds = clip_seconds
        self.sample_rate = 24000
        self.resampler = Resample(orig_freq=32000, new_freq=self.sample_rate)
        self.channels = 1

    def __iter__(self):
        for item in self.dataset:
            mp3_data = item["mp3"]
            waveform, sample_rate = torchaudio.load(BytesIO(mp3_data))
            waveform = torch.as_tensor(waveform)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
                waveform = waveform.expand(self.channels, -1)
            if self.transform:
                waveform = self.transform(waveform)
            waveform = self.resampler(waveform)
            if self.clip_seconds < 0:
                pass
            elif waveform.shape[1] > self.clip_seconds * self.sample_rate:
                start = random.randint(0, waveform.shape[1] - self.clip_seconds * self.sample_rate - 1)
                waveform = waveform[:, start:start + self.clip_seconds * self.sample_rate] # cut tensor
            else:
                pad_length = self.clip_seconds * sample_rate - waveform.size(1)
                padding_tensor = waveform.repeat(1, 1 + pad_length // waveform.size(1))
                waveform = torch.cat((waveform, padding_tensor[:, :pad_length]), dim=1)
            yield {
                "waveform": waveform,
            }

# Lightning DataModule
class EmiliaDataModule(L.LightningDataModule):
    def __init__(self,batch_size=32, num_workers=4):
        super(EmiliaDataModule).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        }
    
    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train = EmiliaDataset(1)
        if stage == "test" or stage is None:
            self.test_other = EmiliaDataset(-1)
            self.test_clean = EmiliaDataset(-1)
    
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.pad_collate_fn,
        )

# Example usage
if __name__ == "__main__":
    data_module = EmiliaDataModule(batch_size=16)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        a=1
        break
