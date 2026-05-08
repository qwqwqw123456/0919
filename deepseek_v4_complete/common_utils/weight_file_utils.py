import torch
def save_weights(model, path):
    torch.save(model.state_dict(), path)
def load_weights(model, path):
    model.load_state_dict(torch.load(path))
