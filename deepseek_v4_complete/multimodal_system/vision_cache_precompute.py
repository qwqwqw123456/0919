import torch
import os

class VisionCachePrecompute:
    def __init__(self, model, cache_dir="vision_cache"):
        self.model = model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    def precompute(self, image_paths):
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            feature = self.model.encode_image(img)
            torch.save(feature, os.path.join(self.cache_dir, os.path.basename(path)+".pt"))
