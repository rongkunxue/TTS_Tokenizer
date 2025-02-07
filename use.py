import os
import sys
sys.path.append(os.getcwd())
import glob
from metrics.UTMOS import UTMOSScore
# from metrics.ss import SimScore
from metrics.periodicity import calculate_periodicity_metrics
from metrics.wer import WERScore
import torchaudio
from pesq import pesq
import numpy as np
import torch
import math
from pystoi import stoi
from pathlib import Path
from tqdm import tqdm
from taming.data.speech_24k import speechttsTest_en
import importlib
from omegaconf import OmegaConf
import argparse
from torch import utils
metalst="/root/Github/TTS_Tokenizer/data/test_other.txt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
    model = instantiate_from_config(config.model)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

def main(args):
    from omegaconf import DictConfig
    config_data = OmegaConf.load(args.config_file)
    config_model = load_config(args.config_file, display=False)
    model = load_vqgan_new(config_model, ckpt_path=args.ckpt_path).to(DEVICE)
    
    #audio=
    with model.ema_scope():
        quant, diff, indices, loss_break,first_quant,second_quant = model.encode(audio)
        mel,reconstructed_audios = model.decode(first_quant)

