import torch
import torch.nn.functional as F
import lightning as L
import math
import sys
sys.path.insert(0,'/root/Github/TTS_Tokenizer/thirdPartyLibrary/3D-Speaker')
from speakerlab.utils.builder import dynamic_import
import os
import re
import pathlib
import numpy as np
import torch


from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path
from main import instantiate_from_config
from contextlib import contextmanager

from taming.modules.diffusionmodules.seanet import SEANetEncoder as Encoder
from taming.modules.diffusionmodules.seanet import SEANetDecoder as Decoder
from taming.modules.diffusionmodules.fourierhead import ISTFTHead as FourierHead
from taming.modules.diffusionmodules.vocosbackbone import VocosBackbone as Backbone

from taming.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from taming.modules.util import requires_grad
from collections import OrderedDict
from taming.modules.ema import LitEma

import torch.nn as nn
class VideoMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VideoMLP, self).__init__()
        self.mlp = nn.Linear(input_dim, output_dim)  

        
    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.mlp(x) 
        x = x.transpose(1, 2) 
        return x

class VideoConv(nn.Module):
    def __init__(self, input_channels):
        super(VideoConv, self).__init__()
        self.transconv = nn.ConvTranspose1d(
            in_channels=input_channels, 
            out_channels=input_channels,  
            kernel_size=6,                
            stride=8,  
            padding=2,
            output_padding=1,
        )
    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.transconv(x) 
        x = x.transpose(1, 2) 
        return x
    
class VQModel(L.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ### Quantize Related
                 quantconfig,
                 sample_rate,
                 target_bandwidths,
                 audio_normalize,
                 segment,
                 ckpt_path=None,
                 ignore_keys=[],
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate=None,
                 ### scheduler config
                 warmup_epochs=1.0, #warmup epochs
                 scheduler_type = "linear-warmup_cosine-decay",
                 min_learning_rate = 0,
                 use_ema = False,
                 stage = None,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.backbone = Backbone(
            input_channels=512, 
            dim=768,
            intermediate_dim=2304,
            num_layers=12,
            adanorm_num_embeddings=None # only one quantizer layer so no adanorm required
        )
        self.head = FourierHead(
            dim=768,
            n_fft=2048,
            hop_length=800,
            padding="same"
        )
        self.loss = instantiate_from_config(lossconfig)
        
        self.audio_normalize = audio_normalize
        
        self.quantize = instantiate_from_config(quantconfig)
        self.use_ema = use_ema
        self.stage = stage
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.automatic_optimization = False
        self.embeding_model_init()
        if self.use_ema and stage is None:  # no need to construct EMA when training transformer
            self.model_ema = LitEma(self)

    def embeding_model_init(self):
        self.strict_loading = False
        CAMPPLUS_VOX = {
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {
                'feat_dim': 80,
                'embedding_size': 512,
            },
        }

        CAMPPLUS_COMMON = {
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        }
        supports = {
            # CAM++ trained on 200k labeled speakers
            'iic/speech_campplus_sv_zh-cn_16k-common': {
                'revision': 'v1.0.0', 
                'model': CAMPPLUS_COMMON,
                'model_pt': 'campplus_cn_common.bin',
            },
            # CAM++ trained on a large-scale Chinese-English corpus
            'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
                'revision': 'v1.0.0', 
                'model': CAMPPLUS_COMMON,
                'model_pt': 'campplus_cn_en_common.pt',
            },
            # CAM++ trained on VoxCeleb
            'iic/speech_campplus_sv_en_voxceleb_16k': {
                'revision': 'v1.0.2', 
                'model': CAMPPLUS_VOX, 
                'model_pt': 'campplus_voxceleb.bin', 
            },
        }

        model_id='iic/speech_campplus_sv_zh-cn_16k-common'
        save_dir = os.path.join("pretrained", "pretrained/speech_campplus_sv_zh-cn_16k-common")
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        conf = supports["iic/speech_campplus_sv_zh-cn_16k-common"]
        cache_dir = snapshot_download(
                    model_id,
                    revision=conf['revision'],
            )
        cache_dir = pathlib.Path(cache_dir)
        embedding_dir = save_dir / 'embeddings'
        embedding_dir.mkdir(exist_ok=True, parents=True)
        download_files = ['examples', conf['model_pt']]
        for src in cache_dir.glob('*'):
            if re.search('|'.join(download_files), src.name):
                dst = save_dir / src.name
                try:
                    dst.unlink()
                except FileNotFoundError:
                    pass
                dst.symlink_to(src)
        pretrained_model = save_dir / conf['model_pt']
        pretrained_state = torch.load(pretrained_model, map_location='cpu')
        model = conf['model']
        self.embedding_model = dynamic_import(model['obj'])(**model['args'])
        self.embedding_model.load_state_dict(pretrained_state)
        self.embedding_model.eval()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        save the state_dict and filter out the 
        '''
        def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
            """
            Save the state_dict and filter out specific parameters
            """
            return {
                k: v
                for k, v in super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars).items()
                if (
                    "inception_model" not in k and
                    "lpips_vgg" not in k and
                    "lpips_alex" not in k and
                    "embedding_model" not in k  # Exclude embedding_model parameters
                )
            }
        
    def init_from_ckpt(self, path, ignore_keys=list(), stage=None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue 
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
            missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)
        else: ## simple resume
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    
    def embedding(self, wav):
        from taming.modules.processor import FBank
        feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        feat = feature_extractor(wav)
        # compute embedding
        with torch.no_grad():
            embedding = self.embedding_model(feat)
        return embedding
    
    def embedding_loss(self, x, xrec):
        x_embeding=self.embedding(x)
        xrec_embeding=self.embedding(xrec)
        embedingloss = F.mse_loss(x_embeding, xrec_embeding)
        return embedingloss

    def encode(self, x):
        if self.audio_normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        
        h = self.encoder(x)
        (quant, emb_loss, info), loss_breakdown = self.quantize(h)
        return (quant, scale), emb_loss, info, loss_breakdown

    def decode(self, quant_tuple):
        quant, scale = quant_tuple
        #dec = self.decoder(quant)
        
        dec = self.backbone(quant)
        # dec = self.VideoConv(dec)
        dec = self.head(dec).unsqueeze(1)
        
        if scale is not None:
            dec = dec * scale.view(-1, 1, 1)
        return dec

    def forward(self, input):
        quant, diff, indices, loss_break = self.encode(input)
        dec = self.decode(quant)
        for ind in indices.unique():
            self.codebook_count[ind] = 1
        return dec, diff, loss_break

    def get_input(self, batch):
        x = batch["waveform"].to(memory_format=torch.contiguous_format)
        return x.float()

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, eloss, loss_break = self(x)
        embedingloss = self.embedding_loss(x,xrec)
        opt_gen, opt_disc = self.optimizers()
        # scheduler_gen, scheduler_disc = self.lr_schedulers()
        if self.scheduler_type != "None":
            scheduler_gen,scheduler_disc = self.lr_schedulers()
        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        # opt_gen._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        # opt_gen._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################
        
        # optimize generator
        aeloss, log_dict_ae = self.loss(embedingloss,eloss, loss_break, x, xrec, 0, self.global_step,
                                        split="train")
        opt_gen.zero_grad()
        self.manual_backward(aeloss)
        opt_gen.step()
        # scheduler_gen.step()
        if self.scheduler_type != "None":
            scheduler_gen.step()
        log_dict_ae["train/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
        
        # optimize discriminator
        discloss, log_dict_disc = self.loss(embedingloss,eloss, loss_break, x, xrec, 1, self.global_step,
                                            split="train")
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()
        # scheduler_disc.step()
        if self.scheduler_type != "None":
            scheduler_disc.step()
        if torch.distributed.get_rank() == 0:
            print(log_dict_ae, log_dict_disc)
        
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)
            
    def on_train_epoch_start(self):
        self.codebook_count = [0] * self.quantize.n_e
        
    def on_validation_epoch_start(self):
        self.codebook_count = [0] * self.quantize.n_e

    def validation_step(self, batch, batch_idx): 
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch)
        quant, eloss, indices, loss_break = self.encode(x)
        x_rec = self.decode(quant)
        embedingloss = self.embedding_loss(x,x_rec)
        aeloss, log_dict_ae = self.loss(embedingloss,eloss, loss_break, x, x_rec, 0, self.global_step,
                                        split="val"+ suffix)

        discloss, log_dict_disc = self.loss(embedingloss,eloss, loss_break, x, x_rec, 1, self.global_step,
                                            split="val" + suffix)
        
        for ind in indices.unique():
            self.codebook_count[ind] = 1
        log_dict_ae[f"val{suffix}/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
    
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_gen = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                # list(self.VideoConv.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.backbone.parameters())+
                                  list(self.head.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.loss.multiperioddisc.parameters())+
                                     list(self.loss.multiresddisc.parameters())+
                                     list(self.loss.dac.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        # if self.trainer.is_global_zero:
        #     print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))
        # step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        # warmup_steps = step_per_epoch * self.warmup_epochs
        # training_steps = step_per_epoch * self.trainer.max_epochs

        # if self.scheduler_type == "None":
        return ({"optimizer": opt_gen}, {"optimizer": opt_disc})
    
        # if self.scheduler_type == "linear-warmup":
        #     scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
        #     scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        # elif self.scheduler_type == "linear-warmup_cosine-decay":
        #     multipler_min = self.min_learning_rate / self.learning_rate
        #     scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
        #     scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
        # else:
        #     raise NotImplementedError()
        # return {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}
