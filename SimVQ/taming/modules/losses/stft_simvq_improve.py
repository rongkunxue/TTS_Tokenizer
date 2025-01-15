import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from taming.modules.discriminator.dac import DACDiscriminator
from .speech_loss import DACGANLoss
from taming.modules.losses.improve_loss import (
    feature_loss,
    generator_loss,
    discriminator_loss,
    MultiScaleMelSpectrogramLoss,
)
from taming.modules.discriminator.discrim import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiBandDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)

class VQSTFTWithDiscriminator(nn.Module):
    def __init__(self, 
                 commit_weight , gen_loss_weight, mrd_loss_coeff,sample_rate):
        super().__init__()
        self.fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
            sampling_rate=sample_rate
        ) 
        self.commit_weight = commit_weight
        self.gen_loss_weight = gen_loss_weight
        self.distill_loss_coeff = 10.0
        self.mrd_loss_coeff=mrd_loss_coeff
        self.dac = DACDiscriminator()
        self.dacdiscriminator = DACGANLoss(self.dac)
        self.mpd=MultiPeriodDiscriminator()
        self.mrd=MultiScaleSubbandCQTDiscriminator(sample_rate)
        self.lambda_melloss=15.0
                    
    def forward(self,loss_distill, codebook_loss, loss_break, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        
        if optimizer_idx == 0:
            # generator update
            loss_dac_1, loss_dac_2 = self.dacdiscriminator.generator_loss(reconstructions, inputs)
            loss_mel = self.fn_mel_loss_singlescale(inputs, reconstructions) * self.lambda_melloss
            _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.mpd(inputs,reconstructions)

            _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.mrd(inputs,reconstructions)

            loss_fm_mp = feature_loss(fmap_rs_mp, fmap_gs_mp)
            loss_fm_mrd = feature_loss(fmap_rs_mrd, fmap_gs_mrd)
            loss_gen_mp, losses_gen_f = generator_loss(gen_score_mp)
            loss_gen_mrd, losses_gen_s = generator_loss(gen_score_mrd)
            
            loss = (
                self.gen_loss_weight * (loss_gen_mp
                + self.mrd_loss_coeff * loss_gen_mrd
                + self.mrd_loss_coeff * loss_fm_mrd
                + loss_fm_mp
                + loss_dac_1
                + loss_dac_2)
                + self.mel_loss_coeff * loss_mel
                + self.distill_loss_coeff * loss_distill
                + self.commit_weight * loss_break
            )
            log = {
                "{}/total_loss".format(split): loss.detach(),
                "{}/mel_loss".format(split): loss_mel.detach(),
                "{}/multi_res_loss".format(split): loss_gen_mrd.detach(),
                "{}/multi_period_loss".format(split): loss_gen_mp.detach(),
                "{}/dac1".format(split): loss_dac_1.detach(),
                "{}/dac2".format(split): loss_dac_2.detach(),
                "{}/distill_loss".format(split): loss_distill.detach(),
                "{}/commit_loss".format(split): loss_break.detach(),
            }
            return loss, log

        if optimizer_idx == 1:
            loss_dac = self.dacdiscriminator.discriminator_loss(reconstructions.contiguous().detach(), inputs.contiguous().detach())
            
            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(inputs, reconstructions.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            y_ds_hat_r, y_ds_hat_g, _, _ = self.mrd(inputs, reconstructions.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )
            loss = loss_disc_f + self.mrd_loss_coeff * loss_disc_s + loss_dac
            
            log = {"{}/total_loss".format(split): loss.detach(),
                     "{}/multi_res_loss".format(split): loss_disc_s.detach(),
                     "{}/multi_period_loss".format(split): loss_disc_f.detach(),
                     "{}/dac".format(split): loss_dac.detach(),
                     }
            return loss, log