from pathlib import Path
from functools import partial

import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig

import lightning as L
from huggingface_hub import snapshot_download
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from trainer import Trainer
from dataset import TrainDataset, collate_fn


@hydra.main(version_base=None, config_path='configs', config_name='train.yaml')
def main(cfg: DictConfig) -> None:
    # Save config
    log_dir = HydraConfig.get().run.dir
    OmegaConf.save(cfg, f'{log_dir}/hydra_config.yaml')

    # Fix seed
    L.seed_everything(cfg.seed)

    # Trainer
    pl_module = Trainer(cfg)
    
    # Train dataset
    base_dir = snapshot_download(cfg.dataset_repo, repo_type='dataset')
    train_dataset = TrainDataset(
        base_dir=Path(base_dir),
        phone_vocab_size=cfg.model.phone_vocab_size,
        audio_vocab_size=cfg.model.audio_vocab_size,
        num_codebooks=cfg.model.num_codebooks,
        audio_window_size=cfg.model.audio_window_size,
        **cfg.dataset
    )
    collate_func = partial(collate_fn, phone_pad_token=cfg.model.phone_vocab_size - 1)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_func
    )

    # Callbacks
    logger = TensorBoardLogger(save_dir=log_dir, default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval=cfg.logging_interval)
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename='{epoch}',
        save_weights_only=cfg.save_weights_only
    )

    trainer = L.Trainer(
        precision=cfg.precision,
        logger=logger,
        max_epochs=cfg.max_epochs,
        devices=cfg.gpus,
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=cfg.gradient_clip_val,
        callbacks=[lr_monitor, checkpoint_callback],
        strategy=cfg.strategy
    )
    trainer.fit(pl_module, train_dataloader)


if __name__ == '__main__':
    main()
