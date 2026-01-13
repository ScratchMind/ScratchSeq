import matplotlib.pyplot as plt 
from src.utils.misc import unpack_results

def plot_test_ppl_comparison(results_dict, title):
    """
    results_dict: {
        "Add-k": results_addk,
        "Interpolation": results_interp,
        "Backoff": results_backoff,
        ...
    }
    """
    plt.figure()

    for label, results in results_dict.items():
        data = unpack_results(results)
        plt.plot(data["n"], data["test_ppl"], marker="o", label=label)

    plt.xlabel("n-gram order (n)")
    plt.ylabel("Test Perplexity (log-scale)")
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()