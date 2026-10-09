import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class CNN(nn.Module):
    def __init__(self, vocab_size, classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(64, 64, k) for k in (3, 4, 5)])
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(192, classes))

    def forward(self, tokens, lengths):
        if tokens.shape[1] < 5:
            tokens = nn.functional.pad(tokens, (0, 5 - tokens.shape[1]))
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            values = torch.relu(conv(embedded))
            valid = (lengths.to(tokens.device) - conv.kernel_size[0] + 1).clamp_min(1)
            mask = torch.arange(values.shape[2], device=tokens.device)[None, :] >= valid[:, None]
            pooled.append(values.masked_fill(mask[:, None, :], float('-inf')).amax(2))
        return self.head(torch.cat(pooled, 1))


class LSTM(nn.Module):
    def __init__(self, vocab_size, classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.rnn = nn.LSTM(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(64, classes))

    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed)
        return self.head(hidden[-1])
