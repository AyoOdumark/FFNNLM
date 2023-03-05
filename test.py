import torch
from FFNLM import FeedforwardNeuralNet
from process_data import read_file, clean_data, word_ix_map, create_vocab, create_context_and_labels
from process_data import tokenize, train_test_split, word_frequency, get_id_of_word, idx_to_word
from torch.utils.data import Dataset, DataLoader


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


test_loader = DataLoader(test_data, batch_size=10)
example = iter(test_loader)
sentence, target = example.next()

print(len(train_data))


VOCAB_SIZE = len(vocabulary)

# BUILDING MODEL
# 1. Initialize model
model = FeedforwardNeuralNet(embedding_dim=EMBEDDING_DIM, vocab_size=VOCAB_SIZE, context_size=N_GRAMS-1, h=H)
# 2. Define Learning rate and optimizer
LEARNING_RATE = 0.002
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

context, target = test_data[0]
context_var = list(map(lambda w: get_id_of_word(w, word_to_ix), context))
# with torch.no_grad():
    # m = len(test_corpus)
    # log_probs = []
    # for test in test_data:
        # context, target = test
        # context_idx = list(map(lambda w: get_id_of_word(w, word_to_ix), context))
        # context_vars = torch.LongTensor(context_idx)
        # output = model(torch.LongTensor(context_var))
        # log_probs.append(torch.max(output).item())
    # perplexity = -(sum(log_probs)/m)
   # print("Perplexity:", perplexity)

    # target_id = get_id_of_word(target, word_to_ix)
    # predict_id = torch.argmax(output)
    # print("Target word:", target)
    # print("Predicted word:", idx_to_word(predict_id.item(), word_to_ix))





