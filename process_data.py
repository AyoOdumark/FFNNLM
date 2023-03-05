# 1. Read the corpus
# 2. Clean the corpus by removing punctuations and irrelevant symbols or numbers
# 3. Tokenization
# 4. Remove stop words
# 5. Create a vocabulary and a word-to-index map which will be used for the embeddings
# 6. We are building a feedforward neural language model, hence we need to create a context window of n-grams

import string
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))


def read_file(path):
    with open(path, "r") as f:
        file = f.read()
    list_of_lines = file.splitlines()
    corpus = []
    for sentence in list_of_lines:
        if sentence != "":
            corpus.append(sentence)
    return corpus


def clean_data(sentence):
    punctuations = string.punctuation
    punctuations = punctuations.replace(".", "")  # retain full stops
    punctuations = punctuations.replace(",", "")  # retain commas
    # punctuations = punctuations.replace('"', "")  # retain the quotation marks
    punctuations = punctuations.replace("&", "")  # retain & to be replaced later by .
    # pattern = r"[{}]".format(punctuations)
    # sentence = re.sub(pattern, "", sentence)
    sentence = sentence.translate(str.maketrans("", "", punctuations))

    # replace & with .
    sentence = sentence.replace("&", ".")

    # change upper case to lower case
    sentence = sentence.casefold()

    return sentence


def word_ix_map(vocab):
    # create a word-to-index map
    word_to_ix = {}
    for idx, word in enumerate(vocab):
        word_to_ix[word] = idx
    return word_to_ix


def idx_to_word(idx, word_idx_map):
    for key, value in word_idx_map.items():
        if value == idx:
            return key


def create_vocab(data, word_frequency_map, unk=False):
    # create the corpus' vocabulary from a word-to-index map
    if unk:
        vocab = set(["<UNK>"])
        for sentence in data:
            for word in sentence:
                if word_frequency_map.get(word) > 1:
                    vocab.add(word)
    else:
        vocab = []
        for word in word_frequency_map:
            vocab.append(word)
    return vocab


def word_frequency(data):
    # calculate the frequency of all words in the corpus. Returns a dictionary
    # Apply after tokenization
    # Result could be used as input to a pandas dataframe
    word_freq_dict = {}
    for sentence in data:
        for word in sentence:
            if word in word_freq_dict:
                # if word is already in the dictionary, increase its count by 1
                word_freq_dict[word] = word_freq_dict.get(word) + 1
            elif word not in word_freq_dict:
                # if word is not in dictionary yet, add it to the dictionary
                word_freq_dict[word] = 1
    return word_freq_dict


def tokenize(sentence):
    # Tokenize a string
    # input is a string. And output is a list of tokenized words
    list_of_sentence = word_tokenize(sentence)
    return list_of_sentence


def train_test_split(corpus, train_size, shuffle=False):
    if shuffle:
        random.shuffle(corpus)
    # list_of_words = [word for sentence in corpus for word in sentence]
    train_size = int(train_size * len(corpus))
    train_arr = corpus[:train_size]
    test_arr = corpus[train_size + 1:]

    return train_arr, test_arr


def create_context_and_labels(data, n_grams):
    data_size = len(data)
    context = n_grams - 1
    dataset = []
    for i in range(context, data_size - context):
        counter = context
        context_words = []
        while counter >= 1:
            context_words.append(data[i - counter])
            counter -= 1
        dataset.append((context_words, data[i]))
    return dataset


def get_id_of_word(word, word_id_map):
    unknown_word_id = word_id_map["<UNK>"]
    return word_id_map.get(word, unknown_word_id)


# corpus_list = [word for sentence in tokenized_corpus for word in sentence]

# word_freq_map = word_frequency(tokenized_corpus)

# word_freq_df = pd.DataFrame.from_dict(data=word_freq_map, orient="index", columns=["frequency"])

# print(word_freq_df)




