import torch.nn as nn
import torch.nn.functional as F
import torch


class FeedforwardNeuralNet(nn.Module):
    def __init__(self, embedding_dim, vocab_size, context_size, h):
        super(FeedforwardNeuralNet, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size*embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size)

    def forward(self, x):
        embeds = self.embeddings(x).view((-1, self.context_size * self.embedding_dim))
        out = torch.tanh(self.linear1(embeds.view(1, -1)))
        log_prob = F.log_softmax(self.linear2(out), dim=1)
        return log_prob


