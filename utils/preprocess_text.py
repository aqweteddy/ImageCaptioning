import csv
import nltk

# TODO: a dictionary include all words(not test)
# TODO: a tokenizer using nltk
# map word to id
# map id to word


class Dictionary:
    def __init__(self, tokenizer):
        self.word2id = dict()
        self.id2word = dict()
        self.idx = 0
        self.tokenizer = tokenizer if tokenizer else nltk.tokenize.word_tokenize

    def add_word(self, word: str) -> int:
        """
        return word idx
        """
        if self.word2id.has_key(word):
            return self.word2id[word]
        else:
            self.word2id[word] = self.idx
            self.id2word[self.idx] = word
            self.idx = self.idx + 1
            return self.idx - 1

    def add_sentence(self, sentence: str) -> list:
        return [self.add_word(word) for word in self.tokenize(sentence)]

    def __call__(self, word: str) -> int:
        """
        return id
        """
        if word in self.word2idx:
            return self.word2idx[word]
        return self.word2idx['<unk>']

    def __len__(self) -> int:
        return len(self.word2idx)

    def save_dict(self, filename: str) -> None:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for (key, value) in self.word2idx.items():
                writer.writerow([key, value])

    def load_dict(self, filename: str) -> None:
        with open(filename, 'f', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                self.word2idx[row[0]] = row[1]
                self.id2word[row[1]] = row[0]
                self.idx = len(self.word2idx)

def build_dct(file: str):
    dct = Dictionary()
    dct.add_word('<pad>')
    dct.add_word('<start>')
    dct.add_word('<end>')
    dct.add_word('<unk>')

    with open(file, 'r'):