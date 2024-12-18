import tqdm
import argparse
import os
import sys
import sys, os
from tqdm import tqdm
import multiprocessing

import string
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel
import os
import torch.nn.functional as F
import fairseq
import pytorch_lightning as pl
import requests
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
from funasr import AutoModel
from jiwer import compute_measures
from zhon.hanzi import punctuation



class WERScore:
    def __init__(self, device):
        self.device = device
        model_id = "openai/whisper-large-v3"
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        # self.model_2 = AutoModel(model="paraformer-zh").to(device)

    def score_en(self, wav: torch.tensor,text:torch.tensor) -> torch.tensor:
        input_features = self.processor(wav.cpu(), sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.device)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
        predicted_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        raw_truth, raw_hypo, wer, subs, dele, inse = self.process_one(transcription, text)
        return wer

    def process_one(self,hypo, truth):
        raw_truth = truth
        raw_hypo = hypo
        punctuation_all = punctuation + string.punctuation
        for x in punctuation_all:
            if x == '\'':
                continue
            truth = truth.replace(x, '')
            hypo = hypo.replace(x, '')

        truth = truth.replace('  ', ' ')
        hypo = hypo.replace('  ', ' ')
        truth = truth.lower()
        hypo = hypo.lower()
        measures = compute_measures(truth, hypo)
        ref_list = truth.split(" ")
        wer = measures["wer"]
        subs = measures["substitutions"] / len(ref_list)
        dele = measures["deletions"] / len(ref_list)
        inse = measures["insertions"] / len(ref_list)
        return (raw_truth, raw_hypo, wer, subs, dele, inse)
