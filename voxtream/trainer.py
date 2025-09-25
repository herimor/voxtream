import lightning as L
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from safetensors.torch import load_file
from torch.nn import ModuleDict
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAccuracy
from utils.trainer import LinearWarmupDecayScheduler

from voxtream.model import Model, ModelConfig


class Trainer(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self._init_metrics(
            audio_vocab_size=config.model.audio_vocab_size,
            num_phone_states=config.model.num_phone_states,
        )

        # Init model
        model_config = ModelConfig(**config.model)
        self.model = Model(model_config)

        if config.dep_former_name is not None:
            dep_former_weight_path = hf_hub_download(
                config.model_repo, config.dep_former_name
            )
            state_dict = load_file(dep_former_weight_path)
            self.model.load_state_dict(state_dict, strict=False)

            # Freeze pre-trained layers
            self.model.audio_head.requires_grad = False
            for param in self.model.audio_embeddings.parameters():
                param.requires_grad = False

            for param in self.model.dep_former.parameters():
                param.requires_grad = False

    def _init_metrics(
        self, audio_vocab_size: int, num_phone_states: int, top_k: int = 10
    ):
        self.metrics_config = {
            "semantic_acc_top10": (audio_vocab_size * num_phone_states, top_k),
            "audio_acc_top10": (audio_vocab_size, top_k),
        }
        metrics = {}
        for key, (num_classes, top_k) in self.metrics_config.items():
            metrics[key] = MulticlassAccuracy(num_classes, top_k)
        self.metrics = ModuleDict(metrics)

    def _log_metric(self, logits, labels, key):
        self.log(f"train_{key}", self.metrics[key](logits, labels))

    def training_step(self, batch, batch_idx):
        (
            phone_seq,
            phone_emb_idx,
            audio_codes,
            semantic_labels,
            audio_labels,
            spk_templates,
        ) = batch
        logits, audio_logits, rand_idx = self.model(
            phone_seq, phone_emb_idx, audio_codes, spk_templates
        )

        semantic_loss = F.cross_entropy(logits, semantic_labels)
        self.log("semantic_loss", semantic_loss)

        # Re-arange audio labels after Depth transformer amortization
        if rand_idx is not None:
            audio_labels = audio_labels[..., rand_idx]

        audio_loss = F.cross_entropy(audio_logits, audio_labels)
        self.log("audio_loss", audio_loss)

        loss = semantic_loss + audio_loss
        self.log("train_loss", loss)

        # Top-10 accuracy
        self._log_metric(logits, semantic_labels, "semantic_acc_top10")
        self._log_metric(audio_logits, audio_labels, "audio_acc_top10")

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = LinearWarmupDecayScheduler(
            optimizer=optimizer,
            warmup_steps=total_steps
            * self.config.warmup_epochs
            // self.config.max_epochs,
            total_steps=total_steps,
            final_lr=self.config.lr,
            initial_lr=self.config.initial_lr,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
