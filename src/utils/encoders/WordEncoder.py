from collections import Counter
import re

class WordEncoder:
    """
    Fixed word-level encoder/tokenizer.
    Splits text into words/punctuation tokens, builds vocab, encodes to indices.
    """
    BOS = "<s>"
    EOS = "</s>"
    UNK = "<unk>"

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def build_vocab(self, texts):
        """
        Build vocabulary from training texts (list of sentences).
        """
        counter = Counter()
        for text in texts:
            # Simple tokenizer: words + punctuation
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            for word in words:
                counter[word] += 1

        # Add special tokens first
        vocab = [self.BOS, self.EOS, self.UNK] + sorted(counter.keys())

        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}

    def encode(self, text):
        """
        Encode text to list of token indices.
        """
        words = [self.BOS] + re.findall(r'\w+|[^\w\s]', text.lower()) + [self.EOS]
        return [self.word2idx.get(word, self.word2idx[self.UNK]) for word in words]

    def decode(self, tokens):
        """
        Decode indices to text (with spaces).
        """
        words = [self.idx2word[i] for i in tokens]
        # Skip BOS/EOS, join inner words
        return ' '.join(words[1:-1])

    def __len__(self):
        return len(self.word2idx)