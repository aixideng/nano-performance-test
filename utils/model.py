from pytorch_lightning import LightningModule
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import OneCycleLR


class MultilingualCrossEncoder(LightningModule):
    def __init__(
        self,
        learning_rate=1e-4,
        weight_decay=1e-2,
        steps_per_epoch=None,
        freeze_encoder=False,
        use_cls=True,
        model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        drop_prob=0.5,
        hidden_size=384,
        num_classes=4,
    ):
        super().__init__()

        self.save_hyperparameters(
            "learning_rate", "weight_decay", "steps_per_epoch"
        )
        self.cross_encoder = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, return_dict=True
        )
        self.use_cls = use_cls

        if freeze_encoder:
            for p in self.cross_encoder.parameters():
                p.requires_grad = False

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=drop_prob),
            nn.Linear(
                in_features=hidden_size if use_cls else hidden_size * 4,
                out_features=num_classes,
                bias=False,
            ),
        )

    def forward(self, features):
        token_ids = features["token_ids"]
        attn_masks = features["attn_masks"]
        outputs = self.cross_encoder(token_ids, attention_mask=attn_masks)
        if self.use_cls:
            cls_token = outputs.pooler_output
        else:
            hidden_states = torch.cat(
                tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1
            )
            cls_token = hidden_states[:, 0, :]
        logits = self.fc_layer(cls_token)
        logits = F.softmax(logits, dim=1)

        return logits

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

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
