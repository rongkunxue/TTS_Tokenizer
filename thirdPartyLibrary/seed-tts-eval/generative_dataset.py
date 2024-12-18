import sys, os
from tqdm import tqdm

metalst = "/mnt/nfs3/zhangjinouwen/dataset/seedtts_testset/en/meta.lst"
wav_dir = None
wav_res_ref_text = "/root/Github/TTS_Tokenizer/thirdPartyLibrary/seed-tts-eval/ref.txt"

f = open(metalst)
lines = f.readlines()
f.close()

f_w = open(wav_res_ref_text, 'w')
for line in tqdm(lines):
    if len(line.strip().split('|')) == 4:
        utt, prompt_text, prompt_wav, infer_text = line.strip().split('|')
    else:
        a=1
    # if not os.path.exists(os.path.join(wav_dir, utt + '.wav')):
    #     continue
    # tmp
    #prompt_wav = infer_wav

    if not os.path.isabs(prompt_wav):
        prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        orgin_wav = os.path.join(os.path.dirname(metalst), utt+ '.wav')
        out_line = '|'.join([utt, prompt_text, prompt_wav,infer_text,orgin_wav])
    f_w.write(out_line + '\n')
f_w.close()
