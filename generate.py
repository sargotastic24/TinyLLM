import torch
import sentencepiece as spm
from model.tiny_gpt import TinyGPT, GPTConfig

#Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoint.pt"
TOKENIZER_PATH = "tokenizer/bpe.model"
CONTEXT_LENGTH = 256
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.5  #creativity: lower = more predictable, higher = more random

#Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

#Text generation function
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -CONTEXT_LENGTH:]  #crop to context length
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx

#Load model 
dummy_sp = spm.SentencePieceProcessor()
dummy_sp.load(TOKENIZER_PATH)
vocab_size = dummy_sp.vocab_size()

config = GPTConfig(
    vocab_size=vocab_size,
    n_layers=6,
    n_heads=6,
    d_model=384,
    d_ff=1536,
    context_length=512
)
model = TinyGPT(config).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded! Ready to generate.\n")

#Prompt
prompt = input("Enter a prompt: ")
tokens = sp.encode(prompt, out_type=int)
x = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

#Generate
y = generate(model, x, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
output_text = sp.decode(y[0].tolist())

print("\nGenerated text:")
print("-----------------------------------")
print(output_text)
print("-----------------------------------")

while True:
    prompt = input("\nYou: ")
    if prompt.lower() in ["quit", "exit"]:
        break
    tokens = sp.encode(prompt, out_type=int)
    x = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    y = generate(model, x, max_new_tokens=100, temperature=TEMPERATURE)
    output = sp.decode(y[0].tolist())
    print(f"tinyLLM: {output}")