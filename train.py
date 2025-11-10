import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

from model.tiny_gpt import TinyGPT, GPTConfig
from train_dataset import TextDataset

#Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-4
CONTEXT_LENGTH = 256

#Dataset
dataset = TextDataset("data/raw.txt", "tokenizer/bpe.model", context_length=CONTEXT_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
vocab_size = dataset.vocab_size

#Model
config = GPTConfig(
    vocab_size=vocab_size,
    n_layers=6,        
    n_heads=6,         
    d_model=384,       
    d_ff=1536,         
    context_length=512 
)
model = TinyGPT(config).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR)
losses = []

#Training Loop
for epoch in range(EPOCHS):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())

#Save the model (very heavy file so not included in respository has all that training data)
torch.save(model.state_dict(), "checkpoint.pt")
print("Model training complete! Saved as checkpoint.pt")

#Plot loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.savefig("training_loss.png")
plt.show()