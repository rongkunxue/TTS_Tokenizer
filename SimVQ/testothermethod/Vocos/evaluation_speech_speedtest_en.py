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
from taming.data.speech_24k import speechttsTest_en
import importlib
from omegaconf import OmegaConf
import argparse
from torch import utils
metalst="/root/Github/TTS_Tokenizer/data/interspeech.txt"
DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

from speechtokenizer import SpeechTokenizer
import torch


from vocos import Vocos
def print_and_save(message, file):
    print(message)  
    file.write(message + '\n') 


def main(args):
    
    bandwidth_id = torch.tensor([1]).to(DEVICE)
    codebook_size = 1024
    usage = {}
    vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(DEVICE)
    for i in range(codebook_size):
        usage[i] = 0
    if not os.path.exists(f"{args.ckpt_path.parent}/recons/seedtest_paths.txt"):
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
                audio = batch["waveform"].squeeze(0).to(DEVICE)
                y_audio=vocos(audio, bandwidth_id=bandwidth_id)
         
                generative_audio_path = os.path.join(f"{args.ckpt_path.parent}/recons/seed_tts/{utt}.wav")
                directory = os.path.dirname(generative_audio_path)
                os.makedirs(directory, exist_ok=True)
                torchaudio.save(generative_audio_path, y_audio.cpu().clip(min=-0.99, max=0.99), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
                out_line = '|'.join([utt, prompt_text, prompt_wav_path,infer_text,orgin_wav_path,generative_audio_path])
                paths.append(out_line)
            num_count = sum([1 for key, value in usage.items() if value > 0])
            utilization = num_count / codebook_size
            with open(f"{args.ckpt_path.parent}/recons/seedtest_paths.txt", "w") as f:
                for path in paths:
                    f.write(path + "\n")
    else:
        paths = []
        f = open(f"{args.ckpt_path.parent}/recons/seedtest_paths.txt")
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

    sim_pro_wav_all=0
    sim_rec_all=0

    wer_score=0

    for i in tqdm(range(len(paths))):
        rawwav,rawwav_sr=torchaudio.load(paths[i].split("|")[4])
        prewav,prewav_sr=torchaudio.load(paths[i].split("|")[5])
        
        rawwav=rawwav.to(DEVICE)
        prewav=prewav.to(DEVICE)
   
        rawwav_16k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=16000)  #测试UTMOS的时候必须重采样
        prewav_16k=torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=16000)

        # # 1.UTMOS
        # print("****UTMOS_raw",i,UTMOS.score(rawwav_16k.unsqueeze(1))[0].item())
        # print("****UTMOS_encodec",i,UTMOS.score(prewav_16k.unsqueeze(1))[0].item())
        # utmos_sumgt+=UTMOS.score(rawwav_16k.unsqueeze(1))[0].item()
        # utmos_sumencodec+=UTMOS.score(prewav_16k.unsqueeze(1))[0].item()
    
        # # breakpoint()

        # ## 2.PESQ  
        # min_len=min(rawwav_16k.size()[1],prewav_16k.size()[1])
        # rawwav_16k_pesq=rawwav_16k[:,:min_len].squeeze(0)
        # prewav_16k_pesq=prewav_16k[:,:min_len].squeeze(0)
        # pesq_score = pesq(16000, rawwav_16k_pesq.cpu().numpy(), prewav_16k_pesq.cpu().numpy(), "wb", on_error=1)
        # print("****PESQ",i,pesq_score)
        # pesq_sumpre+=pesq_score
        # # breakpoint()

        # ## 3.F1-score
        # min_len=min(rawwav_16k.size()[1],prewav_16k.size()[1])
        # rawwav_16k_f1score=rawwav_16k[:,:min_len]
        # prewav_16k_f1score=prewav_16k[:,:min_len]
        # periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(rawwav_16k_f1score,prewav_16k_f1score)
        # print("****f1",periodicity_loss, pitch_loss, f1_score,f1score_sumpre)
        # if(math.isnan(f1_score)):
        #     f1score_filt+=1
        #     print("*****",f1score_filt)
        # else:
        #     f1score_sumpre+=f1_score
        # # breakpoint()


        ##4.similarity
        sim_rec =Sim.score(rawwav_16k,prewav_16k)
        # sim_pro_wav =Sim.score(prowav_16k,prewav_16k)
        sim_rec_all+=sim_rec
        # sim_pro_wav_all+=sim_pro_wav
        print("****similarity_rec",sim_rec)
        # print("****similarity_pro_wav",sim_pro_wav)

        ##5.wer
        # text=paths[i].split("|")[3]
        # wer_s = wer.score_en(prewav_16k,text)
        # wer_score+=wer_s
        # print("****wer",wer_s)
        
        ## 4.STOI
        # for ljspeech
        # rawwav_24k=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=24000)
        # min_len=min(rawwav_24k.size()[1],prewav.size()[1])
        # rawwav_stoi=rawwav_24k[:,:min_len].squeeze(0)
        # prewav_stoi=prewav[:,:min_len].squeeze(0)
        # tmp_stoi=stoi(rawwav_stoi.cpu(),prewav_stoi.cpu(),24000,extended=False)
        # print("****stoi",tmp_stoi)
        # stoi_sumpre.append(tmp_stoi)
        # # breakpoint()

        # min_len=min(rawwav.size()[1],prewav.size()[1])
        # rawwav_stoi=rawwav[:,:min_len].squeeze(0)
        # prewav_stoi=prewav[:,:min_len].squeeze(0)
        # tmp_stoi=stoi(rawwav_stoi.cpu(),prewav_stoi.cpu(),rawwav_sr,extended=False)
        # print("****stoi",tmp_stoi)
        # stoi_sumpre.append(tmp_stoi)
        
        
    with open(Path(args.ckpt_path).parent / "speedtest_en_result.txt", 'w') as f:
        # print_and_save(f"UTMOS_raw: {utmos_sumgt}, {utmos_sumgt/len(paths)}", f)
        # print_and_save(f"UTMOS_encodec: {utmos_sumgt}, {utmos_sumencodec/len(paths)}", f)
        # print_and_save(f"PESQ: {pesq_sumpre}, {pesq_sumpre/len(paths)}", f)
        # print_and_save(f"F1_score: {f1score_sumpre}, {f1score_sumpre/(len(paths)-f1score_filt)}, {f1score_filt}", f)
        # print_and_save(f"STOI: {np.mean(stoi_sumpre)}", f)
        print_and_save(f"similarity_rec: {sim_rec_all/len(paths)}", f)
        # print_and_save(f"similarity_pro_wav: {sim_pro_wav_all/len(paths)}", f)
        # print_and_save(f"WER: {wer_score/len(paths)}", f)
    
    
def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--ckpt_path", default="/root/Github/TTS_Tokenizer/SimVQ/test/Vocos/a.pt", type=Path)
    parser.add_argument("--batch_size", default=1, type=int)

    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    main(args)