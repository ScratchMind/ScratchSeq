import matplotlib.pyplot as plt 
from src.utils.misc import unpack_results

def plot_ppl_vs_n(results, title):
    data = unpack_results(results)

    plt.figure()
    plt.plot(data["n"], data["train_ppl"], marker="o", label="Train PPL")
    plt.plot(data["n"], data["test_ppl"], marker="o", label="Test PPL")
    plt.xlabel("n-gram order (n)")
    plt.ylabel("Perplexity")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()