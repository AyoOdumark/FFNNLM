from process_data import read_file, clean_data, word_ix_map, idx_to_word, create_vocab, word_frequency, tokenize
from process_data import train_test_split, get_id_of_word, create_context_and_labels
from RNNLM import RNNLM
import torch
import torch.nn as nn

PATH = ".\\Brown.txt"
TRAIN_SIZE = 0.8
EMBEDDING_SIZE = 100
LEARNING_RATE = 0.005
HIDDEN_SIZE = 100
NUM_EPOCHS = 2

corpus = read_file(PATH)
#  Clean data
corpus = [clean_data(sentence) for sentence in corpus]
# Tokenize data
corpus = [tokenize(sentence) for sentence in corpus]
# Split into train and test so the vocabulary does not contain the test words
train_set, test_set = train_test_split(corpus, TRAIN_SIZE)
# Create word frequency map
word_frequency_map = word_frequency(train_set)
# Create vocabulary without unknown_word
vocabulary = create_vocab(train_set, word_frequency_map, unk=False)

# Create a word_to_idx map
word_to_ix = word_ix_map(vocabulary)

VOCAB_SIZE = len(vocabulary)

train_corpus = [word for sentence in train_set for word in sentence]
train_corpus_length = len(train_corpus)
NUM_ITERATIONS = len(train_corpus)

SEQUENCE_LENGTH = 15


def get_sequence(corpus, seq_len, idx):
    return corpus[idx:idx+seq_len]


