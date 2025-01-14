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
size="middle" #middle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='wav,mp3')
    path1 = Path("/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/train-clean-100")
    path2 = Path("/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/train-clean-360")

    args = parser.parse_args()
    exts = args.exts.split(',')


    if size=="small":
        wav_res_ref_text="/root/Github/TTS_Tokenizer/data/train_small.txt"
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
    
    elif size=="middle":
        wav_res_ref_text="/root/Github/TTS_Tokenizer/data/train_middle.txt"
        file_list = [
            str(file) for ext in exts 
            for path in [path1,path2] 
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

    #todo add big size
    # elif size=="big":
    #     wav_res_ref_text="/root/Github/TTS_Tokenizer/data/train_big.txt"
    #     file_list = [
    #         str(file) for ext in exts 
    #         for path in [path1,path2,path3] 
    #         for file in path.glob(f'**/*.{ext}')
    #     ]
    #     f_w = open(wav_res_ref_text, 'w')
    #     for i, audio_file in tqdm(enumerate(file_list)):
    #         file_name = os.path.basename(audio_file)
    #         utt = os.path.splitext(file_name)[0]
    #         prompt_text="0"
    #         prompt_wav="0"
    #         infer_text=audio_file.replace(".wav", ".normalized.txt")
    #         with open(infer_text, 'r', encoding='utf-8') as file:
    #             content = file.read()
    #         out_line = '|'.join([utt, prompt_text, prompt_wav,content,audio_file])
    #         f_w.write(out_line + '\n')
    #     f_w.close()

    else:
        raise ValueError("size should be small, middle or big")