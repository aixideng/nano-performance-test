from abc import abstractmethod
import functools

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Task1Dataset(Dataset):
    def __init__(
        self,
        data_df,
        max_len,
        model_name,
    ):
        self.data_df = data_df

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encode_func = functools.partial(
            tokenizer.encode_plus,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.data_df)

    @abstractmethod
    def __getitem__(self, index):
        pass
