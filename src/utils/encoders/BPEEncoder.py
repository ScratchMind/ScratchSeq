from collections import Counter
import re

class BPEEncoder:
    """
    Minimal Byte Pair Encoding (BPE) tokenizer.
    Didactic implementation for ScratchSeq.
    """

    BOS = "<s>"
    EOS = "</s>"
    UNK = "<unk>"

    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.bpe_merges = []
        self.vocab = set()
        self.token2idx = {}
        self.idx2token = {}

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    def build_vocab(self, texts):
        """
        Learn BPE merges from training texts.
        """
        # Step 1: build word frequency dictionary
        word_freqs = Counter()
        for text in texts:
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            for word in words:
                word_freqs[word] += 1

        # Step 2: initialize word symbols (characters + </w>)
        words = {
            tuple(list(word) + ["</w>"]): freq
            for word, freq in word_freqs.items()
        }

        # Step 3: learn merges
        for _ in range(self.num_merges):
            pair_freqs = Counter()

            for word, freq in words.items():
                for i in range(len(word) - 1):
                    pair_freqs[(word[i], word[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair = pair_freqs.most_common(1)[0][0]
            self.bpe_merges.append(best_pair)

            # merge best pair
            new_words = {}
            for word, freq in words.items():
                new_word = []
                i = 0
                while i < len(word):
                    if (
                        i < len(word) - 1
                        and word[i] == best_pair[0]
                        and word[i + 1] == best_pair[1]
                    ):
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words[tuple(new_word)] = freq

            words = new_words

        # Step 4: collect final vocabulary
        vocab = set()
        for word in words:
            for symbol in word:
                vocab.add(symbol)

        vocab = [self.BOS, self.EOS, self.UNK] + sorted(vocab)
        self.token2idx = {tok: i for i, tok in enumerate(vocab)}
        self.idx2token = {i: tok for tok, i in self.token2idx.items()}
        self.vocab = set(vocab)

    # -------------------------------------------------
    # Encoding
    # -------------------------------------------------

    def _apply_bpe(self, word):
        """
        Apply learned BPE merges to a single word.
        """
        symbols = list(word) + ["</w>"]

        for a, b in self.bpe_merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == a
                    and symbols[i + 1] == b
                ):
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text):
        """
        Encode text into BPE subword tokens.
        """
        tokens = []
        words = re.findall(r'\w+|[^\w\s]', text.lower())

        for word in words:
            subwords = self._apply_bpe(word)
            for sw in subwords:
                if sw in self.vocab:
                    tokens.append(sw)
                else:
                    tokens.append(self.UNK)

        return tokens

    # -------------------------------------------------
    # Decoding (approximate)
    # -------------------------------------------------

    def decode(self, tokens):
        """
        Decode subwords back into text (approximate).
        """
        text = ""
        for tok in tokens:
            if tok == "</w>":
                text += " "
            else:
                text += tok
        return text.strip()

    def __len__(self):
        return len(self.token2idx)