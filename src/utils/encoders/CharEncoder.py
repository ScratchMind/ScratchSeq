from collections import Counter

class CharEncoder:
    """
    Character-level encoder.
    Treats all characters literally (including <unk> text).
    """

    BOS = "<s>"
    EOS = "</s>"

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def build_vocab(self, texts):
        """
        Build vocabulary from training texts only.
        """
        counter = Counter()
        for text in texts:
            for ch in text:
                counter[ch] += 1

        vocab = [self.BOS, self.EOS] + sorted(counter.keys())

        self.char2idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

    def encode(self, text):
        """
        Encode text into list of tokens (characters).
        """
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)

    def __len__(self):
        return len(self.char2idx)