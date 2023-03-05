from process_data import read_file, clean_data, word_ix_map, idx_to_word, create_vocab, word_frequency, tokenize
from process_data import train_test_split, get_id_of_word
from RNNLM import RNNLM
import torch
import torch.nn as nn

PATH = ".\\Brown.txt"
TRAIN_SIZE = 0.8
EMBEDDING_SIZE = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 100

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
vocabulary = create_vocab(train_set, word_frequency_map, unk=True)

# Create a word_to_idx map
word_to_ix = word_ix_map(vocabulary)

VOCAB_SIZE = len(vocabulary)

# Building the model
# 1. Instantiate Model
rnn = RNNLM(EMBEDDING_SIZE, VOCAB_SIZE, HIDDEN_SIZE)

# 2. Loss function and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)

losses = []
num_iterations = 0

# Training Loop
for i, sequence in enumerate(train_set):
    num_iterations += 1
    hidden = rnn.init_hidden()
    for j in range(len(sequence)):
        if j < len(sequence) - 1:
            rnn.zero_grad()

            input_word = sequence[j]
            target_word = sequence[j+1]
            # Get input word index
            input_word_ix = torch.LongTensor([get_id_of_word(input_word, word_to_ix)])

            # Forward pass
            output, hidden = rnn(input_word_ix, hidden)
            hidden.detach_()

            # Compute loss
            target_word_ix = torch.LongTensor([get_id_of_word(target_word, word_to_ix)])
            loss = criterion(output, target_word_ix)
            loss.backward()

            losses.append(loss)

            optimizer.step()

    if num_iterations % 100 == 0:
        print(f"Epoch: {num_iterations}, loss: {sum(losses)/len(losses) : 4f}")















