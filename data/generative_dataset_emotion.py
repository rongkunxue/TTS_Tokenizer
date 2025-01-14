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
type="test"

category_mapping = {
    "angry": 1,
    "animal": 2,
    "animaldir": 3,
    "awe": 4,
    "bored": 5,
    "calm": 6,
    "child": 7,
    "childdir": 8,
    "confused": 9,
    "default": 10,
    "desire": 11,
    "disgusted": 12,
    "enunciated": 13,
    "fast": 14,
    "fearful": 15,
    "happy": 16,
    "laughing": 17,
    "narration": 18,
    "nonverbal": 19,
    "projected": 20,
    "sad": 21,
    "sarcastic": 22,
    "singing": 23,
    "sleepy": 24,
    "sympathetic": 25,
    "whisper": 26
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    wav_res_ref_text="/root/Github/TTS_Tokenizer/data/emotion_test.txt"
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='wav,mp3')
    path1 = Path("/mnt/nfs3/zhangjinouwen/dataset/emotion/audio/train")
    path2 = Path("/mnt/nfs3/zhangjinouwen/dataset/emotion/audio/test")
    path3 = Path("/mnt/nfs3/zhangjinouwen/dataset/emotion/audio/val")
    args = parser.parse_args()
    exts = args.exts.split(',') 

    if type=="train":
        file_list = [
            str(file) for ext in exts 
            for path in [path1] 
            for file in path.glob(f'**/*.{ext}')]
        f_w = open(wav_res_ref_text, 'w')
        for i, audio_file in tqdm(enumerate(file_list)):
            file_name = os.path.basename(audio_file)
            utt = os.path.splitext(file_name)[0]
            prompt_text="0"
            prompt_wav="0"
            content= Path(audio_file).parent.name[5:]
            numeric_category = (category_mapping.get(content, "Unknown"))
            if numeric_category == "Unknown":
                print(f"Unknown category: {content}")
            out_line = '|'.join([utt, prompt_text, prompt_wav,str(numeric_category),audio_file])
            f_w.write(out_line + '\n')
        f_w.close()
    
    elif type=="test":
        file_list = [
            str(file) for ext in exts 
            for path in [path2] 
            for file in path.glob(f'**/*.{ext}')]
        f_w = open(wav_res_ref_text, 'w')
        for i, audio_file in tqdm(enumerate(file_list)):
            file_name = os.path.basename(audio_file)
            utt = os.path.splitext(file_name)[0]
            prompt_text="0"
            prompt_wav="0"
            content= Path(audio_file).parent.name[5:]
            numeric_category = (category_mapping.get(content, "Unknown"))
            out_line = '|'.join([utt, prompt_text, prompt_wav,str(numeric_category),audio_file])
            f_w.write(out_line + '\n')
        f_w.close()