import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer_path, context_length=256):
        self.context_length = context_length

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        # Read and tokenize all text
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
        self.tokens = self.sp.encode(data, out_type=int)
        self.vocab_size = self.sp.vocab_size()

    def __len__(self):
        # number of training samples
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.context_length], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + 1 + self.context_length], dtype=torch.long)
        return x, y
