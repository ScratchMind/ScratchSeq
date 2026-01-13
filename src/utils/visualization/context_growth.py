import matplotlib.pyplot as plt 
from src.utils.misc import unpack_results

def plot_context_growth(results, title):
    data = unpack_results(results)

    plt.figure()
    plt.plot(data["n"], data["contexts"], marker="o")
    plt.yscale("log")
    plt.xlabel("n-gram order (n)")
    plt.ylabel("Number of unique contexts (log scale)")
    plt.title(title)
    plt.grid(True)
    plt.show()