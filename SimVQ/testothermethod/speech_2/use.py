from speechtokenizer import SpeechTokenizer

config_path = '/root/Github/TTS_Tokenizer/SimVQ/test/speech/config.py'
ckpt_path = '/root/Github/TTS_Tokenizer/SimVQ/test/speech/SpeechTokenizer.pt'
model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
model.eval()
import torchaudio
import torch

# Load and pre-process speech waveform
wav, sr = torchaudio.load('/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/train-clean-100/1098/133695/1098_133695_000003_000004.wav')

# monophonic checking
if wav.shape[0] > 1:
    wav = wav[:1,:]

if sr != model.sample_rate:
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)

wav = wav.unsqueeze(0)

# Extract discrete codes from SpeechTokenizer
with torch.no_grad():
    codes = model.encode(wav) # codes: (n_q, B, T)


# Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
wav = model.decode(codes[0: (3 + 1)], st=0) 
a=1

