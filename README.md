TinyLLM — A Small Language Model Trained from Scratch
by Sarthak Goyal

A minimal Transformer-based language model (LM) trained completely from scratch, without using any pre-trained weights or APIs.
Built and trained using PyTorch and SentencePiece, this project demonstrates that even a small, self-trained model can learn grammar, sentence structure, and text generation.

Project Overview

TinyGPT is a simplified GPT-like model that learns to predict the next token in a sequence of text.
It’s trained on a small dataset (BookCorpus) and can generate text similar in tone and structure to the dataset it’s trained on.

This project shows end-to-end understanding of how LLMs actually work:

tokenization → embeddings → attention → training → text generation

Features

Custom SentencePiece tokenizer

Lightweight Transformer decoder (GPT-style)

Causal self-attention for next-token prediction

Training from scratch on small datasets

Text generation with temperature & sampling

Optional loss curve visualization

Runs on CPU or Apple M2 GPU

Architecture Overview
Text → Tokenizer → Token IDs → Embeddings → Positional Encoding
     → Multi-Head Self-Attention → Feed-Forward Layers → Softmax
     → Next Token Prediction → Generation Loop


Each block contains:

Attention: Learns relationships between tokens

Feed-forward: Transforms features

Residual connections + LayerNorm: Stability during training

Config example:

config = GPTConfig(
    vocab_size=5000,
    n_layers=6,
    n_heads=6,
    d_model=384,
    d_ff=1536,
    context_length=512,
    dropout=0.1
)

Project Structure
tiny-llm/
│
├── data/                # raw training data (.txt)
├── tokenizer/           # SentencePiece model + vocab
├── model/
│   └── tiny_gpt.py      # Transformer architecture
├── train_dataset.py     # Dataset loader
├── train_tokenizer.py   # Tokenizer training script
├── train.py             # Training loop
├── generate.py          # Text generation script
└── requirements.txt

Setup Instructions
Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

Install dependencies
pip install torch tqdm numpy sentencepiece matplotlib

Prepare dataset

Place a plain-text file at:

data/raw.txt
Examples:

BookCorpus

Mix of classic books (Pride and Prejudice, Sherlock Holmes, Alice in Wonderland, etc.)

Training Your Model

Train tokenizer:

python train_tokenizer.py


Train model:

python train.py


You’ll see progress bars and a loss graph:

Epoch 1/3: loss=3.9
Epoch 2/3: loss=2.8
Model training complete! Saved as checkpoint.pt


Loss curve is saved as training_loss.png.

Generating Text

Run your model:

python generate.py


Example:

Enter a prompt: The sun rose over the city


Output:

The sun rose over the city and the streets began to shimmer with quiet light...


Control creativity with TEMPERATURE (0.7–1.2) and output length with MAX_NEW_TOKENS.

Results
Dataset	Behavior	Notes
Meditations	Philosophical, reflective	Focused on Stoic tone
BookCorpus	Fluent, modern English	General writing style
TinyStories	Simple, structured	Ideal for small models

Training took ~2–4 hours on an Apple M2 MacBook Air.

How It Works (Short Explanation)

Tokenization — splits text into subword tokens

Embeddings — map tokens into high-dimensional vectors

Positional encodings — preserve token order

Self-attention — each token “attends” to previous ones to gather context

Feed-forward layers — refine learned features

Output logits — predict the probability of the next token

Sampling loop — generate new tokens autoregressively

Loss function: Cross Entropy
Training objective: Next-token prediction

Example Outputs

(Replace with your own model’s examples)

Prompt: The meaning of life is
Output: The meaning of life is found not in possessions but in the way we perceive each moment.

Prompt: Once upon a time
Output: Once upon a time there was a child who dreamed of building a world from words.

What You Learn From This Project

How tokenization, embeddings, and attention interact

How Transformers actually learn text relationships

How generation uses probabilistic sampling

How model architecture changes affect output tone and quality

Future Improvements

Add attention map visualization

Experiment with INT8 quantization

Create chat interface (persistent conversation context)

Fine-tune on custom writing style or dialogues

Export for mobile inference (CoreML / ONNX)

Author

Sarthak Goyal