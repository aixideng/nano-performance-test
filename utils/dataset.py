import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from sklearn.utils import shuffle


def load_data(train_path, product_path, convert_to_gain=False):
    """ Init variables """
    col_query_id = "query_id"
    col_query = "query"
    col_query_locale = "query_locale"
    col_esci_label = "esci_label"
    col_product_id = "product_id"
    col_product_title = "product_title"
    col_product_description = "product_description"
    col_product_bullet = "product_bullet_point"
    col_product_brand = "product_brand"
    col_product_color = "product_color_name"
    col_product_locale = "product_locale"

    col_description_clean = "description_clean"
    col_bullet_clean = "bullet_point_clean"
    col_title_clean = "title_clean"
    col_color_clean = "color_clean"
    col_text_all = "text_all"

    esci2gain = {
        "exact": 1.0,
        "substitute": 0.1,
        "complement": 0.01,
        "irrelevant": 0.0,
    }
    esci2label = {
        "exact": 0,
        "substitute": 1,
        "complement": 2,
        "irrelevant": 3,
    }
    col_label = "label"

    """ Preprocess product data """
    print(f"Loading product file: {product_path}")
    df_p = pd.read_csv(product_path)
    df_p.fillna("", inplace=True)
    df_p[col_title_clean] = df_p[col_product_title]
    df_p[col_description_clean] = df_p[col_product_description] \
        .str.replace("<\w+>", " ", regex=True) \
        .str.replace("</\w+>", " ", regex=True) \
        .str.strip()
    df_p[col_bullet_clean] = df_p[col_product_bullet] \
        .str.replace("<\w+>", " ", regex=True) \
        .str.replace("</\w+>", " ", regex=True) \
        .str.strip()
    df_p[col_color_clean] = df_p[col_product_color]
    df_p[col_text_all] = df_p.apply(
        lambda x: x[col_title_clean] + " " + \
                  x[col_product_brand] + " " + \
                  x[col_color_clean] + " " + \
                  x[col_description_clean] + " " + \
                  x[col_bullet_clean],
        axis=1
    )
    df_p = df_p[[col_product_id, col_product_locale, col_text_all]]

    """ Preprocess train data """
    print(f"Loading training file: {train_path}")
    df = pd.read_csv(train_path)
    qid_lookup = df.groupby(
        [col_query, col_query_locale]) \
        .head(1)[[col_query_id, col_query, col_query_locale]] \
        .set_index([col_query_locale, col_query],
        drop=True
    )
    df[col_query_id] = df.apply(
        lambda x: qid_lookup.loc[x[col_query_locale], x[col_query]][col_query_id],
        axis=1
    )

    df = pd.merge(
        df,
        df_p,
        how="left",
        left_on=[col_product_id, col_query_locale],
        right_on=[col_product_id, col_product_locale],
    )
    df = df[df[col_text_all].notna()]

    if convert_to_gain:
        df[col_label] = df[col_esci_label].apply(lambda label: esci2gain[label])
    else:
        df[col_label] = df[col_esci_label].apply(lambda label: esci2label[label])

    return df


def train_test_split(data_df, test_size=0.1, reset=True, dev_ratio=None, random_state=None):
    if reset:
        data_df = shuffle(data_df, random_state=random_state)

    if dev_ratio:
        data_df = data_df[:int(len(data_df) * dev_ratio)].reset_index(drop=True)

    train_df = data_df[int(len(data_df) * test_size):].reset_index(drop=True)
    test_df = data_df[:int(len(data_df) * test_size)].reset_index(drop=True)

    return train_df, test_df


class Task1Dataset(Dataset):
    def __init__(
        self,
        data_df,
        max_len=512,
        model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    ):
        self.data_df = data_df

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        query = self.data_df.loc[index, "query"]
        product = self.data_df.loc[index, "text_all"]

        # Tokenize the pair of sentences to get token ids and attention masks
        encoded_dict = self.tokenizer.encode_plus(
            [query, product],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        token_ids = encoded_dict["input_ids"].squeeze(0)
        attn_masks = encoded_dict["attention_mask"].squeeze(0)

        features = {"token_ids": token_ids, "attn_masks": attn_masks}
        label = torch.tensor(self.data_df.loc[index, "label"], dtype=torch.int64)

        return features, label
