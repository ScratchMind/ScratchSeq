def pad_sequences(sequences, n, bos, eos):
    if n < 1:
        raise ValueError("n must be >= 1")

    padded = []
    for seq in sequences:
        padded.append([bos] * (n - 1) + seq + [eos])
    return padded

def unpack_results(results):
    return {
        "n": [r["n"] for r in results],
        "train_ppl": [r["train_ppl"] for r in results],
        "test_ppl": [r["test_ppl"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "avg_branching": [r["avg_branching"] for r in results],
    }