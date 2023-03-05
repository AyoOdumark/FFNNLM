import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNLM(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size):
        super(RNNLM, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.i2h = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(embedding_size + hidden_size, vocab_size)

    def forward(self, word_idx, hidden):
        embed = self.embeddings(word_idx)
        combined = torch.cat((embed, hidden), dim=1)
        hidden = torch.tanh(self.i2h(combined))
        output = F.log_softmax(self.i2o(combined), dim=1)

        return output, hidden

    def init_hidden(self):
        return torch.zeros((1, self.hidden_size))

