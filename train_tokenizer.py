import sentencepiece as spm


spm.SentencePieceTrainer.train(
    input='data/raw.txt',         
    model_prefix='tokenizer/bpe',  # tokenizer will be saved as bpe.model & bpe.vocab
    vocab_size=5000                # number of tokens 
)

print("Tokenizer trained successfully! Files saved in tokenizer/ folder.")
