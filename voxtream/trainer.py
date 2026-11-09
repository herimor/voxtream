import lightning as L
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from safetensors.torch import load_file
from torch.nn import ModuleDict
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAccuracy

from voxtream.model import Model, ModelConfig
from voxtream.utils.model import load_model_weights
from voxtream.utils.mog import load_mimi_acoustic_codebooks
from voxtream.utils.trainer import LinearWarmupDecayScheduler

IGNORE_INDEX = -100


class Trainer(L.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.config = config
        self._init_metrics(
            audio_vocab_size=config.model.audio_vocab_size,
            num_phone_states=config.model.num_phone_states,
            audio_pad_size=config.model.audio_pad_size,
        )

        # Init model
        model_config = ModelConfig(**config.model)
        self.model = Model(model_config, config.compile_forward)
        self.mog = model_config.dep_former_head == "mog"

        # The pre-trained depth transformer of CSM, with its audio embeddings and AR head
        dep_former_weight_path = config.dep_former_weight_path
        if dep_former_weight_path is None and config.dep_former_name is not None:
            dep_former_weight_path = hf_hub_download(
                config.model_repo, config.dep_former_name
            )
        if dep_former_weight_path is not None:
            self.model.load_state_dict(load_file(dep_former_weight_path), strict=False)

        if config.model_weight_path is not None:
            self.model.load_state_dict(
                load_model_weights(config.model_weight_path), strict=True
            )

        # Parameters kept as loaded, by name prefix
        frozen = tuple(config.freeze)
        for name, param in self.model.named_parameters():
            if name.startswith(frozen):
                param.requires_grad = False

        if self.mog and not self.model.mog_codebooks.any():
            self.model.set_mog_codebooks(
                load_mimi_acoustic_codebooks(
                    config.mimi_repo,
                    config.mimi_name,
                    num_levels=model_config.num_codebooks - 1,
                    mimi_weight_path=config.mimi_weight_path,
                )
            )

    def _init_metrics(
        self,
        audio_vocab_size: int,
        num_phone_states: int,
        audio_pad_size: int,
        top_k: int = 10,
    ):
        self.metrics_config = {
            "semantic_acc_top10": (
                (audio_vocab_size + audio_pad_size) * num_phone_states,
                top_k,
            ),
            "audio_acc_top10": (audio_vocab_size, top_k),
        }
        metrics = {}
        for key, (num_classes, top_k) in self.metrics_config.items():
            metrics[key] = MulticlassAccuracy(
                num_classes, top_k, ignore_index=IGNORE_INDEX
            )
        self.metrics = ModuleDict(metrics)

    def _log_metric(self, logits, labels, key):
        self.log(f"train_{key}", self.metrics[key](logits, labels))

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: the dict of a dataset's `collate_fn`: the inputs of `Model.forward`,
                `semantic_labels` and `audio_labels`, and optionally the `loss_mask` and
                `audio_mask` of the frames to train on
        """
        logits, audio_out, rand_idx = self.model(
            batch["phone_tokens"],
            batch["phone_emb_indices"],
            batch["audio_tokens"],
            prompt_lens=batch.get("prompt_lens"),
            spk_embeddings=batch.get("spk_embeddings"),
            depth_audio_tokens=batch.get("depth_audio_tokens"),
            punct_del_indices=batch.get("punct_del_indices"),
        )

        # a crop rarely fills its window, and the entry it came from is padded with
        # silence: both are masked out rather than trained on
        semantic_labels = batch["semantic_labels"]
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            semantic_labels = semantic_labels.masked_fill(~loss_mask, IGNORE_INDEX)
            self.log("frames_kept", loss_mask.float().mean())
        semantic_loss = F.cross_entropy(
            logits, semantic_labels, ignore_index=IGNORE_INDEX
        )
        self.log("semantic_loss", semantic_loss)

        # Re-arange audio labels after Depth transformer amortization
        audio_labels = batch["audio_labels"][..., rand_idx]
        audio_mask = batch.get("audio_mask")
        if audio_mask is not None:
            audio_mask = audio_mask[:, rand_idx].unsqueeze(1).expand_as(audio_labels)

        if self.mog:
            # the MoG loss skips frames with special tokens, PAD marks the masked ones
            if audio_mask is not None:
                audio_labels = audio_labels.masked_fill(
                    ~audio_mask, self.model.config.audio_vocab_size - 1
                )
            mog_losses = self.model.mog_loss(audio_out, audio_labels)
            for key, value in mog_losses.items():
                self.log(f"mog_{key}", value)
            audio_loss = mog_losses["loss"]
        else:
            if audio_mask is not None:
                audio_labels = audio_labels.masked_fill(~audio_mask, IGNORE_INDEX)
            audio_loss = F.cross_entropy(
                audio_out, audio_labels, ignore_index=IGNORE_INDEX
            )
            self.log("audio_loss", audio_loss)
            self._log_metric(audio_out, audio_labels, "audio_acc_top10")

        loss = semantic_loss + audio_loss
        self.log("train_loss", loss)

        # Top-10 accuracy
        self._log_metric(logits, semantic_labels, "semantic_acc_top10")

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
