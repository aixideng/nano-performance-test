from time import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from bigdl.nano.pytorch import Trainer

from modules.datasets.preprocess import load_data, train_test_split
from modules import datasets
from modules import models


def main(args):
    seed_everything(args.random_state)

    data_df = load_data(args.train_path, args.product_path, args.model_type, args.num_neighbors)
    train_df, val_df = train_test_split(
        data_df,
        test_size=args.test_set_size,
        reset=True,
        dev_ratio=args.dev_ratio,
        random_state=args.random_state,
    )

    print("Reading training data...")
    train_set = getattr(datasets, f"{args.model_type}Dataset")(train_df)
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )

    print("Reading validation data...")
    val_set = getattr(datasets, f"{args.model_type}Dataset")(val_df)
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=True,
    )

    # Adjust the training steps by the number of distributed processes
    training_steps_per_epoch = train_df.shape[0] // (args.train_batch_size * args.num_processes)
    model = getattr(models, args.model_type)(
        learning_rate=1e-4,
        weight_decay=1e-2,
        steps_per_epoch=training_steps_per_epoch,
        freeze_encoder=False,
    )
    
    log_name = (
        f"{args.model_type}"
        f"_opt={int(args.use_ipex)},{int(args.enable_bf16)},{int(args.channels_last)}"
        f"_num={args.num_processes}_dev={args.dev_ratio}"
    ) if args.nano else f"{args.model_type}_raw_dev={args.dev_ratio}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{log_name}/",
        save_top_k=1,
        monitor="val_loss",
        filename="{epoch}-{step}-{val_loss:.2f}-{val_score:.2f}",
        every_n_train_steps=max(1, int(training_steps_per_epoch * args.val_check_interval)) + 1,
    )
    trainer_configs = {
        "max_epochs": args.num_epochs,
        "val_check_interval": args.val_check_interval,
        "log_every_n_steps": 5,
        "logger": TensorBoardLogger(save_dir="lightning_logs/", name=f"{log_name}"),
        "callbacks": [LearningRateMonitor(logging_interval="step"), checkpoint_callback],
    }

    if args.nano:
        trainer = Trainer(
            use_ipex=args.use_ipex,
            enable_bf16=args.enable_bf16,
            channels_last=args.channels_last,
            num_processes=args.num_processes,
            **trainer_configs,
        )
    else:
        trainer = pl.Trainer(**trainer_configs)

    start_time = time()
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    fit_time = time() - start_time
    outputs = trainer.test(model, dataloaders=val_loader)
    test_score = outputs[0]["test_score"]
    print(f"Time: {fit_time}, Score: {test_score}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_type", type=str, help="'CrossEncoder' or 'GCN'.")
    parser.add_argument("train_path",
                        type=str,
                        help="Input training CSV with pairs of queries and products.")
    parser.add_argument("product_path",
                        type=str,
                        help="Input product catalogue CSV.")
    parser.add_argument('--nano', action='store_true')
    parser.add_argument('--use_ipex', action='store_true')
    parser.add_argument('--enable_bf16', action='store_true')
    parser.add_argument('--channels_last', action='store_true')
    parser.add_argument("--num_processes",
                        type=int,
                        default=1,
                        help="Number of processes in distributed training.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=1,
                        help="Stop training once this number of epochs is reached.")
    parser.add_argument("--dev_ratio",
                        type=float,
                        default=None,
                        help="Use part of datasets for development.")
    parser.add_argument("--test_set_size",
                        type=float,
                        default=0.1,
                        help="Split datasets into random train and test subsets.")
    parser.add_argument("--val_check_interval",
                        type=float,
                        default=1,
                        help="How often within one training epoch to check the validation set.")
    parser.add_argument("--random_state",
                        type=int,
                        default=42,
                        help="Random seed.")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=32,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_neighbors",
                        type=int,
                        default=1,
                        help="[GCN] Number of neighbor queries for each product.")
    args = parser.parse_args()
    
    main(args)
