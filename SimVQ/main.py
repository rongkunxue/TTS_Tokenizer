import argparse, os, sys, datetime, glob, importlib
from torch.utils.data import random_split, DataLoader, Dataset

import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from lightning import seed_everything

from torch.utils.data.dataloader import default_collate as custom_collate

import torch
# torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True #True
torch.backends.cudnn.benchmark = False #False

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def main():
    from taming.data.easylibritts import LibriTTSDataModule
    a=LibriTTSDataModule()
    

    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
