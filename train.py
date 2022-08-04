import numpy as np
import random
import os
from time import time
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from bigdl.nano.pytorch import Trainer

from utils import Task1Dataset, load_data, train_test_split
from utils import MultilingualCrossEncoder


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main(args):
    seed_everything(args.random_state)
    model_list = {
        "cross_encoder": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "gcn": "bert-base-multilingual-uncased",
    }
    model_name = model_list[args.model_type]

    data_df = load_data(args.train_path, args.product_path)
    train_df, val_df = train_test_split(
        data_df,
        test_size=args.test_set_size,
        reset=True,
        dev_ratio=args.dev_ratio,
        random_state=args.random_state,
    )

    print("Reading training data...")
    train_set = Task1Dataset(train_df, max_len=512, model_name=model_name)
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )

    print("Reading validation data...")
    val_set = Task1Dataset(val_df, max_len=512, model_name=model_name)
    val_loader = DataLoader(
        val_set,
        batch_size=args.test_batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=True,
    )

    # Adjust the training steps by the number of distributed processes
    training_steps_per_epoch = train_df.shape[0] // (args.train_batch_size * args.num_processes)
    if args.model_type == "cross_encoder":
        model = MultilingualCrossEncoder(
            learning_rate=1e-4,
            steps_per_epoch=training_steps_per_epoch,
            drop_prob=0.1,
        )
    elif args.model_type == "gcn":
        raise NotImplementedError

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/cross_multi_ipex_bf16/",
        save_top_k=1,
        monitor="val_loss",
        filename="{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}",
        every_n_train_steps=max(1, int(training_steps_per_epoch * args.val_check_interval)) + 1,
    )
    if args.trainer_mode == "nano":
        trainer = Trainer(
            use_ipex=True,
            enable_bf16=True,
            channels_last=False,
            num_processes=args.num_processes,
            max_epochs=args.num_epochs,
            val_check_interval=args.val_check_interval,
            log_every_n_steps=5,
            logger=TensorBoardLogger(save_dir="lightning_logs/", name="cross_multi_ipex_bf16"),
            callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback],
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            val_check_interval=args.val_check_interval,
            log_every_n_steps=5,
            logger=TensorBoardLogger(save_dir="lightning_logs/", name="cross_raw"),
            callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback],
        )

    start_time = time()
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    fit_time = time() - start_time
    outputs = trainer.test(model, dataloaders=val_loader)
    test_acc = outputs[0]["test_acc"] * 100
    print(f"Time: {fit_time}, Accuracy: {test_acc}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_type", type=str, help="Switch between 'cross_encoder' and 'gcn'.")
    parser.add_argument("trainer_mode", type=str, help="Switch between 'nano' and 'raw'.")
    parser.add_argument("train_path", type=str, help="Input training CSV.")
    parser.add_argument("product_path",
                        type=str,
                        help="Input product catalogue CSV.")
    parser.add_argument("--num_processes",
                        type=int,
                        default=1,
                        help="Number of processes in distributed training.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=1,
                        help="Stop training once this number of epochs is reached.")
    parser.add_argument("--val_check_interval",
                        type=float,
                        default=1,
                        help="How often within one training epoch to check the validation set.")
    parser.add_argument("--random_state",
                        type=int,
                        default=42,
                        help="Random seed.")
    parser.add_argument("--dev_ratio",
                        type=float,
                        default=None,
                        help="Use part of datasets for development.")
    parser.add_argument("--test_set_size",
                        type=float,
                        default=0.1,
                        help="Batch size for training.")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=32,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=32,
                        help="Batch size for testing.")
    args = parser.parse_args()
    
    main(args)
