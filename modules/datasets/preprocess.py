import random
import os

import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def set_seed(seed):
    """Set all seeds to make results reproducible"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(train_path, product_path, model_type, num_neighbors=1):
    """Preprocess raw files, return a dataframe"""

    # Init variables
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
    col_neighbor = "neighbors"

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

    # Preprocess product data
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

    # Preprocess training data
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

    # Add neighbor queries for GCN
    if model_type == "GCN":
        df_p.insert(df_p.shape[1], col_esci_label, "exact")
        df_p = pd.merge(
            df,
            df_p,
            how="right",
            left_on=[col_product_id, col_query_locale, col_esci_label],
            right_on=[col_product_id, col_product_locale, col_esci_label],
        )
        df_p.fillna("", inplace=True)

        def neighbor_sample(series):
            k = len(series)
            if k == num_neighbors:
                return series.values.tolist()
            elif k > num_neighbors:
                indices = series.index.tolist()
                sampled_indices = random.sample(indices, num_neighbors)
                return series.loc[sampled_indices].tolist()
            elif k < num_neighbors:
                indices = series.index.tolist()
                sampled_indices = random.sample(indices, num_neighbors-k)
                sampled_data = series.loc[sampled_indices]
                return pd.concat([series, sampled_data]).tolist()

        print("Sampling neighbor queries...")
        df_p = df_p.groupby(
            [col_product_id, col_product_locale, col_text_all]
        )[col_query].apply(neighbor_sample)
        df_p = df_p.reset_index().rename(columns={col_query: col_neighbor})

    df = pd.merge(
        df,
        df_p,
        how="left",
        left_on=[col_product_id, col_query_locale],
        right_on=[col_product_id, col_product_locale],
    )
    df = df[df[col_text_all].notna()]

    if model_type == "GCN":
        df[col_label] = df[col_esci_label].apply(lambda label: esci2gain[label])
    else:
        df[col_label] = df[col_esci_label].apply(lambda label: esci2label[label])

    return df


def train_test_split(data_df, test_size=0.1, reset=True, dev_ratio=None, random_state=42):
    """Split and/or down-sample dataframe into random train and test subsets"""
    
    if reset:
        data_df = shuffle(data_df, random_state=random_state)

    if dev_ratio:
        data_df = data_df[:int(len(data_df) * dev_ratio)].reset_index(drop=True)

    train_df = data_df[int(len(data_df) * test_size):].reset_index(drop=True)
    test_df = data_df[:int(len(data_df) * test_size)].reset_index(drop=True)

    return train_df, test_df
