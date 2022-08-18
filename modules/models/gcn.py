import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import LitModel
from ..metrics import ndcg_at_k


class GCN(LitModel):
    def __init__(
        self,
        learning_rate=1e-4,
        weight_decay=1e-2,
        steps_per_epoch=None,
        model_name="bert-base-multilingual-uncased",
        freeze_encoder=False,
        hidden_size=768,
    ):
        super(GCN, self).__init__(
            learning_rate, weight_decay, steps_per_epoch, model_name, freeze_encoder
        )

        self.gcn_msg = nn.Sequential(
            nn.Linear(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=True,
            ),
            nn.ReLU(True),
        )

        self.gcn_updt = nn.Sequential(
            nn.Linear(
                in_features=hidden_size * 2,
                out_features=hidden_size,
                bias=True,
            ),
            nn.ReLU(True),
        )

    def forward(self, features):
        query_emb = self.encoder(**features["query"]).pooler_output
        product_emb = self.encoder(**features["text_all"]).pooler_output

        # TODO: Possible memory leak
        msg_list = []
        for i in range(len(features.keys())-2):
            neighbor_emb = self.encoder(**features[f"neighbor_{i}"]).pooler_output
            neighbor_msg = self.gcn_msg(neighbor_emb)
            msg_list.append(neighbor_msg.unsqueeze(1))
        concat_msg = torch.cat(msg_list, dim=1)
        agg_outs = torch.mean(concat_msg, dim=1)

        product_agg = torch.cat([product_emb, agg_outs], dim=-1)
        product_outs = self.gcn_updt(product_agg)
        # TODO: Scale similarity from [-1, 1] to [0, 1]
        similarity = F.cosine_similarity(product_outs, query_emb, dim=-1)

        return similarity

    def training_step(self, batch, batch_idx):
        features, labels = batch
        simi = self(features)
        loss = F.mse_loss(simi, labels)
        self.log("train_loss", loss)
        
        return loss

    def evaluate(self, batch, stage=None):
        features, labels = batch
        simi = self(features)
        loss = F.mse_loss(simi, labels)

        _, indices = torch.sort(simi, dim=-1, descending=True)
        ranks = torch.index_select(labels, dim=-1, index=indices)
        score = ndcg_at_k(ranks, 10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_score", score, prog_bar=True)
