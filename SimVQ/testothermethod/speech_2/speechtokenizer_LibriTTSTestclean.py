import os
import sys
sys.path.append(os.getcwd())
import glob
from metrics.UTMOS import UTMOSScore
from metrics.ss import SimScore
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
from taming.data.speech import speechttsTest_en
import importlib
from omegaconf import OmegaConf
import argparse
from torch import utils
metalst="/root/Github/TTS_Tokenizer/data/test_clean.txt"
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
from speechtokenizer import SpeechTokenizer

def print_and_save(message, file):
    print(message)  
    file.write(message + '\n') 



def main(args):
    config_path = '/root/Github/TTS_Tokenizer/SimVQ/test/z_speech/config.py'
    ckpt_path = '/root/Github/TTS_Tokenizer/SimVQ/test/z_speech/SpeechTokenizer.pt'
    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).to(DEVICE)
    model.eval()
    codebook_size = 1024
    usage = {}
    for i in range(codebook_size):
        usage[i] = 0
    if not os.path.exists(f"{args.ckpt_path.parent}/recons/test_clean.txt"):
        def pad_collate_fn(batch):
            """Collate function for padding sequences."""
            return {
                "waveform": torch.nn.utils.rnn.pad_sequence(
                    [x["waveform"].transpose(0, 1) for x in batch], 
                    batch_first=True, 
                    padding_value=0.
                ).permute(0, 2, 1),
                "prompt_text": [x["prompt_text"] for x in batch],
                "infer_text": [x["infer_text"] for x in batch],
                "utt": [x["utt"] for x in batch],
                "audio_path": [x["audio_path"] for x in batch],
                "prompt_wav_path": [x["prompt_wav_path"] for x in batch]    
            }
        speechdataset = speechttsTest_en(metalst)
        test_loader = utils.data.DataLoader(speechdataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=pad_collate_fn)
        paths=[]
        with torch.no_grad():
            for batch in tqdm(test_loader):
                assert batch["waveform"].shape[0] == 1
                utt = batch["utt"][0]
                prompt_text = batch["prompt_text"][0]
                infer_text = batch["infer_text"][0]
                prompt_wav_path = batch["prompt_wav_path"][0]
                orgin_wav_path = batch["audio_path"][0].replace("infer","wavs")
                audio = batch["waveform"].to(DEVICE)
                codes = model.encode(audio)
                reconstructed_audios = model.decode(codes[0: (3 + 1)], st=0) 
                generative_audio_path = os.path.join(f"{args.ckpt_path.parent}/recons/test_clean/{utt}.wav")
                directory = os.path.dirname(generative_audio_path)
                os.makedirs(directory, exist_ok=True)
                torchaudio.save(generative_audio_path, reconstructed_audios[0].cpu().clip(min=-0.99, max=0.99), sample_rate=16000, encoding='PCM_S', bits_per_sample=16)
                out_line = '|'.join([utt, prompt_text, prompt_wav_path,infer_text,orgin_wav_path,generative_audio_path])
                paths.append(out_line)
            num_count = sum([1 for key, value in usage.items() if value > 0])
            utilization = num_count / codebook_size
            with open(f"{args.ckpt_path.parent}/recons/test_clean.txt", "w") as f:
                for path in paths:
                    f.write(path + "\n")
    else:
        paths = []
        f = open(f"{args.ckpt_path.parent}/recons/test_clean.txt")
        lines = f.readlines()
        paths = [line.strip() for line in lines]
                        
    
    UTMOS=UTMOSScore(device=DEVICE)
    Sim=SimScore(device=DEVICE)
    wer=WERScore(device=DEVICE)
    utmos_sumgt=0
    utmos_sumencodec=0
    pesq_sumpre=0
    f1score_sumpre=0
    stoi_sumpre=[]
    f1score_filt=0

    sim_rec_all=0

    wer_score=0

    for i in tqdm(range(len(paths))):
        rawwav,rawwav_sr=torchaudio.load(paths[i].split("|")[4])
        prewav,prewav_sr=torchaudio.load(paths[i].split("|")[5])
        
        rawwav=rawwav.to(DEVICE)
        prewav=prewav.to(DEVICE)
   
        rawwav_16k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=16000)  #测试UTMOS的时候必须重采样
        prewav_16k=torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=16000)

       

        sim_rec =Sim.score(rawwav_16k,prewav_16k)
        sim_rec_all+=sim_rec
        print("****similarity_rec",sim_rec)
        
        
    with open(Path(args.ckpt_path).parent / "test_clean_result.txt", 'w') as f:
        print_and_save(f"similarity_rec: {sim_rec_all/len(paths)}", f)
    
    
def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--ckpt_path", default="/root/Github/TTS_Tokenizer/SimVQ/test/speech_2/a.pt", type=Path)
    parser.add_argument("--batch_size", default=1, type=int)

    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    main(args)