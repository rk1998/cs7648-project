from argparse import ArgumentParser
from tokenizers import BertWordPieceTokenizer
from models import SentimentLSTM
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

import pytorch_lightning as pl

NUM_CLASSES = 2


def preprocess(example, tokenizer):
    example["text"] = tokenizer.encode(example["text"]).ids
    example["sentiment"] = min(example["sentiment"], 1)
    return example


def collate_fn(samples):
    inputs = torch.nn.utils.rnn.pad_sequence([sample["text"] for sample in samples])
    labels = torch.stack([sample["sentiment"] for sample in samples])
    return (inputs, labels)


def main(args):
    logger = TensorBoardLogger(save_dir="./experiment_logs")

    # load dataset and tokenize
    train_dataset = (
        load_dataset("sentiment140", split="train").shuffle().select(range(45000))
    )
    test_dataset = load_dataset("sentiment140", split="test").shuffle()
    tokenizer = BertWordPieceTokenizer(
        "bert-base-uncased-vocab.txt", lowercase=True
    )

    train_dataset = train_dataset.map(lambda e: preprocess(e, tokenizer), num_proc=4)
    test_dataset = test_dataset.map(lambda e: preprocess(e, tokenizer), num_proc=4)

    train_dataset.set_format("torch", columns=["text", "sentiment"])
    test_dataset.set_format("torch", columns=["text", "sentiment"])

    train_dataset, val_dataset = train_dataset.train_test_split(
        train_size=args.train_frac
    ).values()

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=20, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=20, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=20, collate_fn=collate_fn
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        # log_every_n_steps=10,
    )

    if args.action.lower() == "train":
        print("VOCAB SIZE")
        print(tokenizer.get_vocab_size())
        model = SentimentLSTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=NUM_CLASSES,
            vocab_size=tokenizer.get_vocab_size(),
        )
        trainer.fit(model, train_dataloader, val_dataloader)

    elif args.action.lower() == "eval":
        model = SentimentLSTM.load_from_checkpoint(args.model_path)
        model.eval()
        trainer.test(model, test_dataloaders=test_dataloader)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(ArgumentParser())
    parser.add_argument("action", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--model_path", type=str)
    main(parser.parse_args())
