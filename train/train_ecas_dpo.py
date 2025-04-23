#!/usr/bin/env python3
# train_ecas_dpo.py

import sys
sys.path.append("src")

import os
import argparse
import yaml
import torch
import shutil
import logging
from tqdm import tqdm

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from audioldm_train.utilities.data.dataset import AudioDataset
from audioldm_train.utilities.tools import get_restore_step, copy_test_subset_data
from audioldm_train.utilities.model_util import instantiate_from_config

logging.basicConfig(level=logging.INFO)

def main(cfg, cfg_path, exp_group, exp_name, do_val):
    # 1. Reproducibility
    seed_everything(cfg.get("seed", 0))
    torch.set_float32_matmul_precision(cfg.get("precision", "high"))

    # 2. Prepare directories
    log_dir  = cfg["log_directory"]
    ckpt_dir = os.path.join(log_dir, exp_group, exp_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(log_dir, exp_group, exp_name))

    # 3. Data loaders
    batch_size = cfg["model"]["params"]["batchsize"]
    addons     = cfg["data"].get("dataloader_add_ons", [])
    train_ds   = AudioDataset(cfg, split="train", add_ons=addons)
    val_ds     = AudioDataset(cfg, split="test",  add_ons=addons)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=16, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=min(batch_size,8),
                              num_workers=4,  pin_memory=True)

    # Copy a small subset for testing inference
    subset_dir = os.path.join(os.path.dirname(log_dir),
                              "testset_data", val_ds.dataset_name)
    os.makedirs(subset_dir, exist_ok=True)
    copy_test_subset_data(val_ds.data, subset_dir)

    # 4. Checkpoint resume logic
    resume_ckpt = None
    existing = os.listdir(ckpt_dir)
    if existing:
        step, _ = get_restore_step(ckpt_dir)
        resume_ckpt = os.path.join(ckpt_dir, step)
    elif cfg.get("reload_from_ckpt"):
        resume_ckpt = cfg["reload_from_ckpt"]

    # 5. Instantiate model (with ECAS & DPO built in)
    model = instantiate_from_config(cfg["model"])
    model.set_log_dir(log_dir, exp_group, exp_name)

    # 6. Logger & checkpointing
    wandb_logger = WandbLogger(
        project=cfg["project"],
        save_dir=os.path.join(log_dir, exp_group, exp_name),
        name=f"{exp_group}/{exp_name}",
        config=cfg,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step={step}-FAD={val/frechet_inception_distance:.3f}",
        every_n_train_steps=cfg["step"]["save_checkpoint_every_n_steps"],
        save_top_k=cfg["step"]["save_top_k"],
        monitor="val/frechet_inception_distance",
        mode="min",
    )

    # 7. Trainer setup
    trainer = Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=True),
        max_steps=cfg["step"]["max_steps"],
        check_val_every_n_epoch=cfg["step"]["validation_every_n_epochs"],
        limit_val_batches=cfg["step"].get("limit_val_batches", 1.0),
        callbacks=[checkpoint_cb],
        logger=wandb_logger,
        num_sanity_val_steps=1,
    )

    # 8. Start training (or validation)
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=resume_ckpt if resume_ckpt else None,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SPEECH with ECAS & DPO")
    parser.add_argument("-c", "--config", required=True,
                        help="path to experiment config .yaml")
    parser.add_argument("--val", action="store_true", help="run only validation")
    args = parser.parse_args()

    cfg_path  = args.config
    cfg       = yaml.safe_load(open(cfg_path))
    exp_name  = os.path.splitext(os.path.basename(cfg_path))[0]
    exp_group = os.path.basename(os.path.dirname(cfg_path))

    main(cfg, cfg_path, exp_group, exp_name, args.val)

