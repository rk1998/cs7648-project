import pytorch_lightning as pl
import torch.nn as nn
import torch
from pytorch_lightning.metrics.functional import to_categorical


class SentimentLSTM(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, vocab_size, bidirectional=False):
        super(SentimentLSTM, self).__init__()
        self.save_hyperparameters()

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=False)
        self.decoder = nn.Linear(hidden_dim, output_dim)

        self.objective = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        return self.decoder(h.squeeze(0))

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        out = self(inputs)
        loss = self.objective(out, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        out = self(inputs)
        loss = self.objective(out, labels)
        self.log("val_loss", loss)

        correct = (to_categorical(out) == labels).sum().item()
        total = labels.shape[0]
        return {"correct": correct, "total": total}

    def validation_epoch_end(self, outputs):
        correct = 0
        total = 0
        for output in outputs:
            correct += output["correct"]
            total += output["total"]
        self.log("val_acc", correct / total, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        out = self(inputs)
        loss = self.objective(out, labels)

        correct = (to_categorical(out) == labels).sum()
        total = labels.shape[0]
        return {"correct": correct, "total": total}

    def test_epoch_end(self, outputs):
        correct = 0
        total = 0
        for output in outputs:
            correct += output["correct"]
            total += output["total"]
        print("Test accuracy:", correct / total)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
