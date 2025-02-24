import torch
import torchaudio
from vocos import Vocos
vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to("cuda")
wav, sr = torchaudio.load('/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/train-clean-100/1098/133695/1098_133695_000003_000004.wav').to("cuda")
bandwidth_id = torch.tensor([2]).to("cuda")
y_hat = vocos(wav, bandwidth_id=bandwidth_id)
a=1