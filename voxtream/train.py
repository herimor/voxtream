import json
from pathlib import Path

import hydra
import lightning as L
import torch
from huggingface_hub import snapshot_download
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from voxtream.dataset import DATASET_POOL
from voxtream.trainer import Trainer

torch.set_float32_matmul_precision("medium")
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.fx_graph_cache = True


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # Save config
    log_dir = HydraConfig.get().run.dir
    OmegaConf.save(cfg, f"{log_dir}/hydra_config.yaml")
    # Model config for inference (`model_config_path` of the generator config)
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg.model, resolve=True), f, indent=4)

    # Fix seed
    L.seed_everything(cfg.seed)

    # Trainer
    pl_module = Trainer(cfg)

    # Train dataset
    if cfg.dataset_base_dir is not None:
        base_dir = cfg.dataset_base_dir
    else:
        base_dir = snapshot_download(cfg.dataset_repo, repo_type="dataset")

    dataset_params = OmegaConf.to_container(cfg.dataset, resolve=True)
    train_dataset = DATASET_POOL[dataset_params.pop("name")](
        base_dir=Path(base_dir),
        model=OmegaConf.to_container(cfg.model, resolve=True),
        **dataset_params,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
        shuffle=cfg.shuffle,
    )

    # Callbacks
    logger = TensorBoardLogger(save_dir=log_dir, default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval=cfg.logging_interval)
    # save_top_k -1 keeps every epoch; the default 1 deletes the previous epoch file
    checkpoint_kwargs = dict(filename="{epoch}", save_top_k=cfg.save_top_k_epochs)
    if cfg.get("checkpoint_every_n_steps"):
        checkpoint_kwargs = dict(
            filename="{epoch}-{step}",
            every_n_train_steps=cfg.checkpoint_every_n_steps,
            save_top_k=-1,
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        save_weights_only=cfg.save_weights_only,
        **checkpoint_kwargs,
    )

    trainer = L.Trainer(
        precision=cfg.precision,
        logger=logger,
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        devices=cfg.gpus,
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=cfg.gradient_clip_val,
        callbacks=[lr_monitor, checkpoint_callback],
        strategy=cfg.strategy,
    )
    trainer.fit(pl_module, train_dataloader)


if __name__ == "__main__":
    main()
