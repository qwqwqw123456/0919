import torch
from torch.utils.data import DataLoader
from core.deepseek_v4_model import DeepSeekV4Multimodal

class SFTTrainer:
    def __init__(self, model, tokenizer, lr=1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    def train(self, dataloader, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                outputs = self.model(input_ids)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.model
