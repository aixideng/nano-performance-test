import torch

from .base import Task1Dataset


class GCNDataset(Task1Dataset):
    def __init__(
        self,
        data_df,
        max_len=512,
        model_name="bert-base-multilingual-uncased",
    ):
        super(GCNDataset, self).__init__(data_df, max_len, model_name)

    def __getitem__(self, index):
        features = {}
        for col in ["query", "text_all"]:
            encoded_dict = self.encode_func(self.data_df.loc[index, col])
            for k in ["input_ids", "attention_mask"]:
                encoded_dict[k] = encoded_dict[k].squeeze(0)
            features[col] = encoded_dict

        for idx, neighbor in enumerate(self.data_df.loc[index, "neighbors"]):
            encoded_dict = self.encode_func(neighbor)
            for k in ["input_ids", "attention_mask"]:
                encoded_dict[k] = encoded_dict[k].squeeze(0)
            features[f"neighbor_{idx}"] = encoded_dict

        
        label = torch.tensor(self.data_df.loc[index, "label"], dtype=torch.float)

        return features, label
