import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, VOCAB_SIZE, DIM_EMB=300, NUM_CLASSES=2):
        super(CNN, self).__init__()
        self.NUM_CLASSES=NUM_CLASSES
        self.vocab_size = VOCAB_SIZE
        self.embedding = nn.Embedding(VOCAB_SIZE, DIM_EMB)
        self.conv1 = nn.Conv1d(DIM_EMB, 700, 3)
        self.conv2 = nn.Conv1d(DIM_EMB, 700, 4)
        self.conv3 = nn.Conv1d(DIM_EMB, 700, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.55)
        self.linear1 = nn.Linear(2100, NUM_CLASSES)
        # self.linear2 = nn.Linear(10, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)
        #TODO: Initialize parameters.

    def forward(self, X):
        out1  = self.embedding(X)
        out1 = self.dropout(out1)
        out1 = out1.reshape((out1.shape[0], out1.shape[2], out1.shape[1]))

        conv_out_1 = self.relu(self.conv1(out1))
        conv_out_2 = self.relu(self.conv2(out1))
        conv_out_3 = self.relu(self.conv3(out1))

        max_1 = torch.max(conv_out_1, dim=2)[0]
        max_2 = torch.max(conv_out_2, dim=2)[0]
        max_3 = torch.max(conv_out_3, dim=2)[0]
        lin_input = torch.cat((max_1, max_2, max_3), dim=1)
        drop_out = self.dropout(lin_input)
        lin_out_1 = self.linear1(drop_out)
        # lin_out_2 = self.linear2(lin_out_1)
        output = self.log_softmax(lin_out_1)
        return output
