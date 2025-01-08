#!/usr/bin/env python3
import argparse
import logging
import os
import yaml
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import wandb
from tqdm import tqdm

from timm.utils import setup_default_logging, CheckpointSaver, reduce_tensor
from utils import seed_everything, to_device
from taming.models.vq_audio_simvq import VQModel
from taming.data.speechtokenizer import create_dataloader

_logger = logging.getLogger('train')
# torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='/root/Github/TTS_Tokenizer/SimVQ/configs/simvq/libritts_8khz_20_bert.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
    
    parser = argparse.ArgumentParser(description='Audio VQ Training')
    # Keep core training/validation loop args
    parser.add_argument('--epochs', type=int, default=50,
                       help='number of epochs to train')
    parser.add_argument('--save-freq', type=int, default=5,
                       help='save checkpoint frequency')
    parser.add_argument('--output', default='output/train',
                       help='path to output folder')
    parser.add_argument('--experiment', default='',
                       help='name of train experiment')
    parser.add_argument('--local_rank', default=0, type=int)
    
    # Add other necessary args from original yaml
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='learning rate')
    parser.add_argument('--num-workers', type=int, default=3,
                       help='number of workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    args = parser.parse_args(remaining)
    return args


def main():
    setup_default_logging()
    args = parse_args()
    seed_everything(args.seed)
    wandb.init(project="audio-vq", name="text", config=args)
    
    # Initialize distributed training first
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False
    
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # Define rank before using it

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    # Initialize wandb after rank is defined
    if args.rank == 0:
        # Load Wandb API token from environment variable
        wandb_token = os.getenv('WANDB_API_KEY')
        if wandb_token is None:
            _logger.warning('WANDB_API_KEY not found in environment variables. WandB logging disabled.')
        else:
            wandb.login(key=wandb_token)
            wandb.init(
                project="audio-vq",
                name=args.experiment,
                config=args
            )
    # Create data loaders
    _logger.info(f'Creating data loaders')
    train_loader, eval_loader = create_dataloader(args.data["init_args"])
    step_per_epoch  = len(train_loader) // args.world_size
    _logger.info(f'step_per_epoch: {step_per_epoch}')

    # Create model
    _logger.info(f'Creating model')
    model = VQModel(
        ddconfig=args.model['init_args']['ddconfig'],
        lossconfig=args.model['init_args']['lossconfig'],
        step_per_epoch=step_per_epoch,
        max_epochs=args.trainer['max_epochs'],
        sample_rate=args.model['init_args']['sample_rate'],
        audio_normalize=args.model['init_args']['audio_normalize'],
        learning_rate=args.model['init_args']['learning_rate'],
        scheduler_type=args.model['init_args']['scheduler_type'],
        use_ema=args.model['init_args']['use_ema'],
        distrillmodel=args.model['init_args']['distrillmodel']
    )
    
    # Move model to GPU
    model.to(args.device)
    
    if args.distributed:
        model = NativeDDP(model, device_ids=[args.local_rank])
    
    
    # Setup checkpoint saver
    saver = None
    if args.rank == 0:
        saver = CheckpointSaver(
            model=model,
            optimizer=model.optimizers,
            args=args,
            checkpoint_dir=args.output,
            recovery_dir=args.output,
            max_history=10
        )

    # Training loop
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        train_metrics = train_one_epoch(
            epoch=epoch,
            model=model, 
            loader=train_loader,
            args=args,
            saver=saver
        )
        
        if eval_loader is not None:
            eval_metrics = validate(
                model=model,
                loader=eval_loader,
                args=args
            )
        
        # Save checkpoint
        if saver is not None and (epoch + 1) % args.save_freq == 0:
            save_metric = eval_metrics['val_ema/total_loss'] if eval_loader is not None else train_metrics['train/total_loss']
            saver.save_checkpoint(epoch, metric=save_metric)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(epoch, model, loader, args, saver=None):
    model.train()
    model.on_train_epoch_start()
    # Create multiple average meters for different metrics
    meters = {
        'train/total_loss': AverageMeter(),
        'train/commit_loss': AverageMeter(),
        'train/reconstruct_loss': AverageMeter(),
        'train/multi_period_loss': AverageMeter(),
        'train/multi_res_loss': AverageMeter(),
        'train/feature_matching_mp': AverageMeter(),
        'train/feature_matching_mrd': AverageMeter(),
        'train/loss_dac_1': AverageMeter(),
        'train/loss_dac_2': AverageMeter(),
        'train/loss_distill': AverageMeter(),
        'train/codebook_util': AverageMeter(),
        'train/dac': AverageMeter()
    }
    
    pbar = tqdm(enumerate(loader), total=len(loader), 
                desc=f'Epoch {epoch}/{args.epochs}',
                disable=args.rank != 0)
    
    for batch_idx, batch in pbar:
        # Move entire batch to device
        batch = to_device(batch, args.device)
        
        # Forward pass
        log_dict = model.training_step(batch, batch_idx)
        model.on_train_batch_end()
        
        # Update all metrics
        batch_size = batch[0].size(0)
        for key, meter in meters.items():
            if key in log_dict:
                meters[key].update(log_dict[key].item(), batch_size)
        
        # Update progress bar with main loss
        pbar.set_postfix({
            'loss': f'{meters["train/total_loss"].val:.4f} ({meters["train/total_loss"].avg:.4f})',
        })
        
        # Log to wandb periodically
        if batch_idx % 100 == 0 and args.rank == 0:
            wandb_dict = {
                'epoch': epoch,
                'step': epoch * len(loader) + batch_idx,
            }
            # Add all metrics to wandb dict
            for key, meter in meters.items():
                wandb_dict[f'{key}'] = meter.val
                wandb_dict[f'{key}_avg'] = meter.avg
            
            wandb.log(wandb_dict)
            
    # Return average of all metrics
    return OrderedDict([(key, meter.avg) for key, meter in meters.items()])


def validate(model, loader, args):
    model.eval()
    model.on_validation_epoch_start()
    meters = {
        'val_ema/total_loss': AverageMeter(),
        'val_ema/commit_loss': AverageMeter(),
        'val_ema/reconstruct_loss': AverageMeter(),
        'val_ema/multi_period_loss': AverageMeter(),
        'val_ema/multi_res_loss': AverageMeter(),
        'val_ema/feature_matching_mp': AverageMeter(),
        'val_ema/feature_matching_mrd': AverageMeter(),
        'val_ema/loss_dac_1': AverageMeter(),
        'val_ema/loss_dac_2': AverageMeter(),
        'val_ema/loss_distill': AverageMeter(),
        'val_ema/codebook_util': AverageMeter(),
        'val_ema/dac': AverageMeter()
    }
    
    pbar = tqdm(enumerate(loader), total=len(loader),
                desc='Validation',
                disable=args.rank != 0)
    
    for batch_idx, batch in pbar:
        # Move data to GPU
        batch = to_device(batch, args.device)
        
        # Forward pass
        with torch.no_grad():
            log_dict = model.validation_step(batch, batch_idx)
        
        batch_size = batch[0].size(0)
        for key, meter in meters.items():
            if key in log_dict:
                meters[key].update(log_dict[key].item(), batch_size)
        
        pbar.set_postfix({
            'loss': f'{meters["val_ema/total_loss"].val:.4f} ({meters["val_ema/total_loss"].avg:.4f})',
        })

        if batch_idx % 100 == 0 and args.rank == 0:
            wandb_dict = {}
            # Add all metrics to wandb dict
            for key, meter in meters.items():
                wandb_dict[f'{key}'] = meter.val
                wandb_dict[f'{key}_avg'] = meter.avg
            
            wandb.log(wandb_dict)
        
    return OrderedDict([(key, meter.avg) for key, meter in meters.items()])


if __name__ == '__main__':
    main() 