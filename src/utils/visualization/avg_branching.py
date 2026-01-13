import matplotlib.pyplot as plt 
from src.utils.misc import unpack_results

def plot_avg_branching(results, title):
    data = unpack_results(results)

    plt.figure()
    plt.plot(data["n"], data["avg_branching"], marker="o")
    plt.xlabel("n-gram order (n)")
    plt.ylabel("Average branching factor")
    plt.title(title)
    plt.grid(True)
    plt.show()