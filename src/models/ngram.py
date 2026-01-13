import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import defaultdict
import math

class NgramLM:
    def __init__(self, n):
        self.n = n
        self.counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        
    def fit(self, sequences):
        for sequence in sequences:
            for i in range(len(sequence) - self.n + 1):
                context = tuple(sequence[i:i+self.n-1])
                token = sequence[i+self.n-1]
                
                
                self.counts[context][token] += 1
                self.context_counts[context] += 1
    
    def log_prob_mle(self, context, token):
        full_seq_count = self.counts[context].get(token, 0)
        context_count = self.context_counts[context]
        if(full_seq_count==0):
            return float("-inf")
        return math.log(full_seq_count) - math.log(context_count)
    
    def log_prob_laplace(self, context, token, vocab_size):
        full_seq_count = self.counts[context].get(token, 0)
        context_count = self.context_counts.get(context, 0)
        return math.log(full_seq_count+1) - math.log(context_count+vocab_size)
    
    def log_prob_add_k(self, context, token, vocab_size, k):
        full_seq_count = self.counts[context].get(token, 0)
        context_count = self.context_counts.get(context, 0)
        return math.log(full_seq_count+k) - math.log(context_count+k*vocab_size)
    
    def log_prob_interpolated(self, context, token, lambdas, vocab_size, k=0.01):
        assert len(lambdas) == self.n
        assert abs(sum(lambdas) - 1) < 1e-6
        
        prob = 0.0
        for order in range(1, self.n+1):
            weight = lambdas[order-1]
            if(order==1):
                count_full_sequence = self.counts[()].get(token, 0)
                context_count = self.context_counts.get((), 0)
                prob += weight * ((count_full_sequence+k)/(context_count + k*vocab_size))
            else:
                sub_context = context[-(order-1):]
                count_full_sequence = self.counts[sub_context].get(token, 0)
                context_count = self.context_counts.get(sub_context, 0)
                if(count_full_sequence>0):
                    prob += weight * (count_full_sequence/context_count)
    
    def log_prob_backoff(self, context, token, alpha=0.4, vocab_size=None, k=0.01):
        log_alpha_factor = 0.0  # Accumulate log(alpha) for each backoff
        for order in range(self.n, 0, -1):  # Include unigram order=1
            sub_context = context[-(order-1):]
            count_full_sequence = self.counts[sub_context].get(token, 0)
            context_count = self.context_counts.get(sub_context, 0)
            if count_full_sequence > 0:  # Use MLE if seen
                return log_alpha_factor + math.log(count_full_sequence) - math.log(context_count)
            else:
                log_alpha_factor += math.log(alpha)  # Penalize for backoff
        # Unigram fallback (smoothed, no further backoff)
        count_full_sequence = self.counts[()].get(token, 0)
        context_count = self.context_counts.get((), 0)
        prob = (count_full_sequence + k) / (context_count + k * vocab_size)
        return log_alpha_factor + math.log(prob)
    
    def sequence_log_likelihood(self, sequence, log_prob_fn):
        total = 0.0
        for i in range(len(sequence) - self.n + 1):
            context = tuple(sequence[i:i + self.n - 1])
            token   = sequence[i + self.n - 1]

            lp = log_prob_fn(context, token)
            if lp == float("-inf"):
                return float("-inf")
            total += lp
        return total
    
    def perplexity(self, sequences, log_prob_fn):
        total_logp = 0.0
        total_tokens = 0

        for seq in sequences:
            ll = self.sequence_log_likelihood(seq, log_prob_fn)
            if ll == float("-inf"):
                return float("inf")

            total_logp += ll
            total_tokens += len(seq) - (self.n - 1)

        return math.exp(-total_logp / total_tokens)
