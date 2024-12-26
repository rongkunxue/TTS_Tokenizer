import sys, os
from tqdm import tqdm
from pathlib import Path
import torchaudio
import torch
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    wav_res_ref_text="/root/Github/TTS_Tokenizer/data/test_clean.txt"
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='wav,mp3')
    path1 = Path("/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/test-clean")
    args = parser.parse_args()
    exts = args.exts.split(',')
    file_list = [
        str(file) for ext in exts 
        for path in [path1] 
        for file in path.glob(f'**/*.{ext}')
    ]
    f_w = open(wav_res_ref_text, 'w')
    for i, audio_file in tqdm(enumerate(file_list)):
        file_name = os.path.basename(audio_file)
        utt = os.path.splitext(file_name)[0]
        prompt_text="0"
        prompt_wav="0"
        infer_text=audio_file.replace(".wav", ".normalized.txt")
        with open(infer_text, 'r', encoding='utf-8') as file:
            content = file.read()
        out_line = '|'.join([utt, prompt_text, prompt_wav,content,audio_file])
        f_w.write(out_line + '\n')
    f_w.close()