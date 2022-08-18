import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from .base import LitModel


class CrossEncoder(LitModel):
    def __init__(
        self,
        learning_rate=1e-4,
        weight_decay=1e-2,
        steps_per_epoch=None,
        model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        freeze_encoder=False,
        use_cls=True,
        drop_prob=0.1,
        hidden_size=384,
        num_classes=4,
    ):
        super(CrossEncoder, self).__init__(
            learning_rate, weight_decay, steps_per_epoch, model_name, freeze_encoder
        )

        self.use_cls = use_cls
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=drop_prob),
            nn.Linear(
                in_features=hidden_size if use_cls else hidden_size * 4,
                out_features=num_classes,
                bias=False,
            ),
        )

    def forward(self, features):
        outputs = self.encoder(**features)

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
        score = accuracy(preds, labels)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_score", score, prog_bar=True)
