import tqdm
import argparse
import os
import sys

import os
import torch.nn.functional as F
import fairseq
import pytorch_lightning as pl
import requests
import torch
import torch.nn as nn
from tqdm import tqdm
sys.path.append("/root/Github/TTS_Tokenizer/thirdPartyLibrary/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification")
from verification import verification,init_model


class SimScore:
    def __init__(self, device, ckpt_path="/mnt/nfs3/zhangjinouwen/dataset/wavlm_large_finetune.pth"):
        self.device = device
        self.model=init_model('wavlm_large',ckpt_path).eval().to(device)

    def score(self, wav1: torch.tensor,wav2:torch.tensor) -> torch.tensor:
        with torch.no_grad():
            emb1 = self.model(wav1)
            emb2 = self.model(wav2)
        sim = F.cosine_similarity(emb1, emb2)
        return sim

