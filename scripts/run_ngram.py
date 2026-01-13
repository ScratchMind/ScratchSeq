#!/usr/bin/env python3
"""
N-gram Language Model Experiments
Run: python scripts/run_ngram.py --config config/ngram_config.yaml
"""

import argparse
import yaml
import os
import pandas as pd
from pathlib import Path

from data.dataset import TextLineDataset
from src.utils.encoders import CharEncoder, WordEncoder, BPEEncoder
from src.models.ngram import NgramLM
from src.utils.misc import pad_sequences


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_encoder(encoder_type, **kwargs):
    if encoder_type == 'char':
        return CharEncoder()
    elif encoder_type == 'word':
        return WordEncoder()
    elif encoder_type == 'bpe':
        return BPEEncoder(num_merges=kwargs.get('bpe_merges', 1000))
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")


def main():
    parser = argparse.ArgumentParser(description='Run N-gram LM experiments')
    parser.add_argument('--config', type=str, default='config/ngram_config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Create output directory
    output_dir = config['output'].get('results_dir', './results')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {config['data']['train_path']}...")
    train_dataset = TextLineDataset(config['data']['train_path'], None)
    test_dataset = TextLineDataset(config['data']['test_path'], None)

    # Create encoder
    print(f"Creating {config['encoder']['type']} encoder...")
    encoder = create_encoder(
        config['encoder']['type'],
        bpe_merges=config['encoder'].get('bpe_merges', 1000)
    )
    encoder.build_vocab(train_dataset.lines)
    vocab_size = len(encoder)
    print(f"Vocabulary size: {vocab_size}\n")

    # Encode datasets
    train_seqs = [encoder.encode(line) for line in train_dataset.lines]
    test_seqs = [encoder.encode(line) for line in test_dataset.lines]

    results = []

    # Run experiments
    for n in config['model']['n_values']:
        print(f"{'='*60}")
        print(f"N-gram order: {n}")
        print('='*60)

        # Pad sequences
        train_padded = pad_sequences(train_seqs, n, encoder.BOS, encoder.EOS)
        test_padded = pad_sequences(test_seqs, n, encoder.BOS, encoder.EOS)

        # Train model
        lm = NgramLM(n)
        lm.fit(train_padded)

        # Evaluate each smoothing method
        for method_config in config['smoothing']['methods']:
            method_name = method_config['name']
            print(f"\n{method_name}:")

            if method_name == 'mle':
                train_ppl = lm.perplexity(train_padded, method='mle', vocab_size=vocab_size)
                test_ppl = lm.perplexity(test_padded, method='mle', vocab_size=vocab_size)
                results.append({'n': n, 'method': method_name, 'train_ppl': train_ppl, 'test_ppl': test_ppl})
                print(f"  Train: {train_ppl:.4f}, Test: {test_ppl:.4f}")

            elif method_name == 'laplace':
                train_ppl = lm.perplexity(train_padded, method='laplace', vocab_size=vocab_size)
                test_ppl = lm.perplexity(test_padded, method='laplace', vocab_size=vocab_size)
                results.append({'n': n, 'method': method_name, 'train_ppl': train_ppl, 'test_ppl': test_ppl})
                print(f"  Train: {train_ppl:.4f}, Test: {test_ppl:.4f}")

            elif method_name == 'add_k':
                for k in method_config.get('k_values', [0.01, 0.1, 1.0]):
                    train_ppl = lm.perplexity(train_padded, method='add_k', k=k, vocab_size=vocab_size)
                    test_ppl = lm.perplexity(test_padded, method='add_k', k=k, vocab_size=vocab_size)
                    results.append({'n': n, 'method': f'{method_name}_k{k}', 'train_ppl': train_ppl, 'test_ppl': test_ppl})
                    print(f"  k={k}: Train: {train_ppl:.4f}, Test: {test_ppl:.4f}")

            elif method_name == 'interpolation':
                lambdas = method_config.get('lambdas', [1/n] * n)
                train_ppl = lm.perplexity(train_padded, method='interpolation', lambdas=lambdas, vocab_size=vocab_size)
                test_ppl = lm.perplexity(test_padded, method='interpolation', lambdas=lambdas, vocab_size=vocab_size)
                results.append({'n': n, 'method': method_name, 'train_ppl': train_ppl, 'test_ppl': test_ppl})
                print(f"  Train: {train_ppl:.4f}, Test: {test_ppl:.4f}")

            elif method_name == 'backoff':
                alpha = method_config.get('alpha', 0.4)
                k = method_config.get('k', 0.01)
                train_ppl = lm.perplexity(train_padded, method='backoff', alpha=alpha, k=k, vocab_size=vocab_size)
                test_ppl = lm.perplexity(test_padded, method='backoff', alpha=alpha, k=k, vocab_size=vocab_size)
                results.append({'n': n, 'method': method_name, 'train_ppl': train_ppl, 'test_ppl': test_ppl})
                print(f"  Train: {train_ppl:.4f}, Test: {test_ppl:.4f}")

    # Save results
    if config['output'].get('save_results', True):
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'ngram_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")


if __name__ == '__main__':
    main()


"""
Run using: python scripts/run_ngram.py --config config/ngram_config.yaml
"""
