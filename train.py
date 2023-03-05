import torch
from FFNLM import FeedforwardNeuralNet
from process_data import read_file, clean_data, word_ix_map, create_vocab, create_context_and_labels
from process_data import tokenize, train_test_split, word_frequency, get_id_of_word, idx_to_word


EMBEDDING_DIM = 100
BATCH_SIZE = 100
EPOCHS = 2
H = 1000
N_GRAMS = 4
PATH = ".\\Brown.txt"

# DATA PREPARATION
# 1. Read data
corpus = read_file(".\\Brown.txt")
# 2. Clean data
corpus = [clean_data(sentence) for sentence in corpus]
# 3. Tokenize data
corpus = [tokenize(sentence) for sentence in corpus]
# 4. Split corpus into train and test corpus
train_corpus, test_corpus = train_test_split(corpus, train_size=0.7)
# 5. Create word_frequency_map to be used when creating vocabulary
word_freq_map = word_frequency(train_corpus)
# 6. Create vocabulary from training data
vocabulary = create_vocab(train_corpus, word_freq_map)

# 7. Create word-to-idx map
word_to_ix = word_ix_map(vocabulary)
# 8. create n_gram train data and test data
train_corpus = [word for sentence in train_corpus for word in sentence]
test_corpus = [word for sentence in test_corpus for word in sentence]
train_data = create_context_and_labels(train_corpus, N_GRAMS)
test_data = create_context_and_labels(test_corpus, N_GRAMS)

VOCAB_SIZE = len(vocabulary)

# BUILDING MODEL
# 1. Initialize model
model = FeedforwardNeuralNet(embedding_dim=EMBEDDING_DIM, vocab_size=VOCAB_SIZE, context_size=N_GRAMS-1, h=H)
# 2. Define Learning rate and optimizer
LEARNING_RATE = 0.002
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
losses = []
# 3. Define training loop
for epoch in range(EPOCHS):
    for i, (context, target) in enumerate(train_data):
        # 1. Prepare input to be passed into the model
        context_ids = list(map(lambda w: get_id_of_word(w, word_to_ix), context))
        context_vars = torch.LongTensor(context_ids)

        # 2. forward pass
        output = model(context_vars)
        loss = criterion(output, torch.LongTensor([get_id_of_word(target, word_to_ix)]))

        # 3. backward pass
        optimizer.zero_grad()
        loss.backward()

        # 4. Update weights
        optimizer.step()

        losses.append(loss)

        if (i+1) % 100 == 0:
            print(f"epoch {epoch+1}/{EPOCHS}, step {i+1}/{i+1}, loss = {loss.item():.4f}")







