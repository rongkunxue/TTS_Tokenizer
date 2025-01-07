from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np


def pad_collate_fn(data):
    # PyTorch Lightning DataModule arguments
    #self.prepare_data_per_node = True
    #self._log_hyperparams=False
    #self.allow_zero_length_dataloader_with_multiple_devices = False

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


def create_dataloader(args):
    train_dataset = AudioDataset(args["train_dataset_path"], 48000, False)
    val_dataset = AudioDataset(args["val_dataset_path"], 48000, False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=True,
        collate_fn=pad_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=False,
        collate_fn=pad_collate_fn
    )
    return train_loader, val_loader


class AudioDataset(Dataset):
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
        audio, sr = torchaudio.load(audio_file)
        feature = torch.from_numpy(np.load(feature_file)).squeeze(0)
        audio = audio.mean(axis=0)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
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
    args = {
        "batch_size": 6,
        "num_workers": 4,
        "train_dataset_path": "/mnt/afs/niuyazhe/data/speech_tokenizer/small_hubert/rep_small_hubert_eval.txt",
        "val_dataset_path": "/mnt/afs/niuyazhe/data/speech_tokenizer/small_hubert/rep_small_hubert_eval.txt",
    }
    train_loader, val_loader = create_dataloader(args)
    for batch in train_loader:
        audio, feature = batch
        print(audio.shape)
        print(feature.shape)
        break
