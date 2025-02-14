import os
import sys
sys.path.append(os.getcwd())
import glob
import torchaudio
from pesq import pesq
import numpy as np
import torch
import math
from pystoi import stoi
from pathlib import Path
import importlib
import argparse
from torch import utils
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_obj_from_str(string, reload=False):
        print(string)
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)


def create_model(config_path, ckpt_path=None,is_gumbel=False):
     from omegaconf import OmegaConf
     config = OmegaConf.load(config_path)
     config= config.model
     model = get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))
     sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
     missing, unexpected = model.load_state_dict(sd, strict=False)
     return model



def main():
    config_path="/root/Github/TTS_Tokenizer/SimVQ/vq_audio_simvq_bert_mel/8k_ration_20_loss/config.yaml"
    ckpt_path="/root/Github/TTS_Tokenizer/SimVQ/vq_audio_simvq_bert_mel/8k_ration_20_loss/epoch=49-step=156000.ckpt"
    model = create_model(config_path,ckpt_path).to(DEVICE)
    import librosa
    waveform, sample_rate = librosa.load(
            "/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/test-other/8461/278226/8461_278226_000023_000000.wav", 
            sr=24000,
            mono=1 == 1
        )
    waveform = torch.as_tensor(waveform)

    audio = waveform.to(DEVICE).unsqueeze(0).unsqueeze(0)
    with model.ema_scope():
        quant, diff, indices, loss_break,first_quant,second_quant,first_index = model.encode(audio)
        
        #请注意，我们只使用first_quant和first_index

        # 如果你需要使用 token 生成量化结果
        first_quant = model.quantize.indices_to_codes(first_index)

        # 如果你已经有量化结果，需要将其转换回音频
        mel, reconstructed_audios = model.decode(first_quant)

        # 保存重建的音频文件，裁剪振幅以避免溢出
        torchaudio.save("/root/Github/TTS_Tokenizer/SimVQ/vq_audio_simvq_bert_mel/exam/a.wav", 
                        reconstructed_audios[0].cpu().clip(min=-0.99, max=0.99), 
                        sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

if __name__=="__main__":
    main()