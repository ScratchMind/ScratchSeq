import torch
import torch.nn as nn 

class TextLineDataset:
    def __init__(self, path, encoder):
        self.path = path
        self.encoder = encoder
        self.lines = self._load_lines(path)
        self.token_sequences = self._encode_lines(self.lines)
        
    def _load_lines(self, path):
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip():
                    lines.append(line)
    
    def _encode_lines(self, lines):
        return [self.encoder.encode(line) for line in lines]
    
    def __len__(self):
        return len(self.token_sequences)

    def __getitem__(self, idx):
        return self.token_sequences[idx]    