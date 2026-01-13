import matplotlib.pyplot as plt 
from src.utils.misc import unpack_results

def plot_generalization_gap(results, title):
    data = unpack_results(results)
    gap = [
        test - train
        for train, test in zip(data["train_ppl"], data["test_ppl"])
    ]

    plt.figure()
    plt.plot(data["n"], gap, marker="o")
    plt.xlabel("n-gram order (n)")
    plt.ylabel("Test PPL − Train PPL")
    plt.title(title)
    plt.grid(True)
    plt.show()