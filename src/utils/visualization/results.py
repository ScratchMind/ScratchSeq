import pandas as pd

def print_results_table(results, name):
    df = pd.DataFrame(results)
    
    # Core columns always
    core_cols = ['n', 'train_ppl', 'test_ppl', 'contexts', 'avg_branching']
    
    # Detect and include extras (lambdas, k, etc.) before core
    extra_cols = [col for col in df.columns if col not in core_cols]
    display_cols = extra_cols + core_cols
    
    df = df[display_cols]
    df.columns = [col.replace('_', ' ').title() for col in df.columns]  # 'Train Ppl', 'Test Ppl', etc.
    
    print(f"\n{name}")
    print("-" * (10 * len(df.columns) + len(df.columns) * 3))  # Dynamic separator
    print(df.to_string(index=False, float_format='%.3f', justify='right'))
    
    csv_path = f"{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path} (cols: {', '.join(df.columns)})")
    
def print_addk_results(results_addk, k_values):
    """Print and save separate tables for each k (n=1 to 5), without k column."""
    saved_files = []
    for k in k_values:
        # Collect flat results for this k across all n (exclude 'k' key for table compat)
        k_results = []
        for n in sorted(results_addk.keys()):
            if k in results_addk[n]:
                # Copy dict without 'k' to match print_results_table expected keys
                res = {key: results_addk[n][k][key] for key in ['n', 'train_ppl', 'test_ppl', 'contexts', 'avg_branching']}
                k_results.append(res)
        
        if k_results:
            title = f"MLE with Add-K Smoothing N-gram (k={k})"
            print_results_table(k_results, title)
            saved_files.append(f"{title}.csv")
    
    print(f"\nSaved tables: {', '.join(saved_files)}")

def get_k_results(results_addk, k):
    """Extract flat list of results for fixed k across all n (for unpack_results/plotting)."""
    k_results = []
    for n in sorted(results_addk.keys()):
        if k in results_addk[n]:
            k_results.append(results_addk[n][k])
    return k_results