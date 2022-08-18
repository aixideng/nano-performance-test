from abc import abstractmethod

import torch
from pytorch_lightning import LightningModule
from transformers import AutoModel
from torch.optim.lr_scheduler import OneCycleLR


class LitModel(LightningModule):
    def __init__(
        self,
        learning_rate,
        weight_decay,
        steps_per_epoch,
        model_name,
        freeze_encoder,
    ):
        super().__init__()

        self.save_hyperparameters(
            "learning_rate", "weight_decay", "steps_per_epoch"
        )
        self.encoder = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, return_dict=True
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @abstractmethod
    def forward(self, features):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def evaluate(self, batch, stage=None):
        pass

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            weight_decay=self.hparams.weight_decay,
        )
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer=optimizer,
                max_lr=self.hparams.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.hparams.steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
