import torch

from .base import Task1Dataset


class CrossEncoderDataset(Task1Dataset):
    def __init__(
        self,
        data_df,
        max_len=512,
        model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    ):
        super(CrossEncoderDataset, self).__init__(data_df, max_len, model_name)

    def __getitem__(self, index):
        query = self.data_df.loc[index, "query"]
        product = self.data_df.loc[index, "text_all"]

        # Tokenize the pair of sentences to get token ids and attention masks
        encoded_dict = self.encode_func([query, product])
        
        features = {}
        for k in ["input_ids", "attention_mask"]:
            features[k] = encoded_dict[k].squeeze(0)

        label = torch.tensor(self.data_df.loc[index, "label"], dtype=torch.int64)

        return features, label
