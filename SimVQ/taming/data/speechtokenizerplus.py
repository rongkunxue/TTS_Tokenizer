from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np
import lightning as L
import soundfile

class SpeechTokenizerDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers,train_path,val_path):
        super(SpeechTokenizerDataModule).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset_path = train_path 
        self.val_dataset_path = val_path    
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
            self.train = audioDataset(self.train_dataset_path,48000,False)
        if stage == "test" or stage is None:
            self.test = audioDataset(self.train_dataset_path,48000,False)

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
        with open(file_path, 'r') as f:
            self.file_list = f.readlines()
        self.segment_size = segment_size
        self.sample_rate = 16000
        self.downsample_rate = 320
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        file = self.file_list[index].strip()
        audio_file, feature_file = file.split('\t')
        y, sr = soundfile.read(audio_file)
        feature = torch.from_numpy(np.load(feature_file)).squeeze(0)
        y = torch.tensor(y).float().unsqueeze(0)
        gain = np.random.uniform(-1, -6) 
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        y = y.squeeze(0)
        if sr != self.sample_rate:
           audio = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)
        if audio.size(-1) > self.segment_size:
            max_audio_start = audio.size(-1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_size]
            feature_start = min(int(audio_start / self.downsample_rate), feature.size(0) - self.segment_size // self.downsample_rate)
            feature = feature[feature_start:feature_start + self.segment_size // self.downsample_rate, :]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(-1)), 'constant')
        return audio, feature
    

if __name__ == "__main__":
    data_module = SpeechTokenizerDataModule(batch_size=6, num_workers=4,train_path="/mnt/nfs3/zhangjinouwen/dataset/rep/rep_small_hubert_train.txt",val_path = "/mnt/nfs3/zhangjinouwen/dataset/rep/ep_small_hubert_eval.txt")
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        a=1
        break
