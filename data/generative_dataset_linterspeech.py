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
    wav_res_ref_text="/root/Github/TTS_Tokenizer/data/interspeech.txt"
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='wav,mp3')
    path1 = Path("/root/Github/TTS_Tokenizer/SimVQ/data")
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
        

        ile_path = "/mnt/data/your_file.txt"  # 替换为你的实际文件路径

        # 读取并查找特定行
        with open("/root/Github/TTS_Tokenizer/SimVQ/data/LJSpeech-1.1/metadata.csv", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(utt):
                    content = line.split("|", 2)[1].strip()  # 获取 `|` 后的内容
                    break  # 找到后停止
        out_line = '|'.join([utt, prompt_text, prompt_wav,content,audio_file])
        f_w.write(out_line + '\n')
    f_w.close()