import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

# ==================== 固定配置（已优化，无需修改）====================
DATA_DIR = "./train_data"
DATA_JSON = os.path.join(DATA_DIR, "train.json")
SAVE_DIR = "./lora_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# 训练超参（适配免费低配GPU）
BATCH_SIZE = 1
GRAD_ACCUM = 2
EPOCHS = 1       # 首轮训练：仅跑1轮，快速出结果
LEARNING_RATE = 1e-4
MAX_SEQ_LEN = 256
IGNORE_LABEL = -100

# LoRA轻量化配置
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["wq", "wk", "wv", "wo"]

# 设备自动识别
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"运行设备: {DEVICE}")

# ==================== 模型核心模块 ====================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class VisionProjector(nn.Module):
    def __init__(self, d_model: int, num_visual_tokens: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.num_tokens = num_visual_tokens
    def forward(self, x):
        return self.norm(self.proj(x))[:, :self.num_tokens, :]

class ModelConfig:
    vocab_size = 129280
    d_model = 7168
    n_heads = 32
    d_ff = 28672
    n_layers = 16
    num_experts = 32
    top_k = 2
    num_visual_tokens = 128

class SimpleMultiModalModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight

        self.vis_embed = nn.Conv2d(3, config.d_model, 14, 14)
        self.vis_projector = VisionProjector(config.d_model, config.num_visual_tokens)
        self.img_start = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.img_end = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

    def encode_image(self, img_path: str, device):
        img = Image.open(img_path).convert("RGB").resize((448, 448))
        img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(448,448,3).permute(2,0,1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        vis_feat = self.vis_embed(img_tensor).flatten(2).transpose(1,2)
        return self.vis_projector(vis_feat)

    def fuse_image_text(self, text_emb, vis_feat):
        B, L, D = text_emb.shape
        fuse = torch.cat([self.img_start.expand(B,-1,-1), vis_feat, self.img_end.expand(B,-1,-1), text_emb], dim=1)
        return fuse

    def forward(self, input_ids, img_path_list):
        B = input_ids.shape[0]
        text_emb = self.embed(input_ids)
        vis_feats = [self.encode_image(p, DEVICE) for p in img_path_list]
        fuse_emb = self.fuse_image_text(text_emb, torch.cat(vis_feats, dim=0))
        hidden = self.norm(fuse_emb)
        logits = self.head(hidden)
        return logits

# ==================== 数据集加载 ====================
class TrainDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len):
        self.data = json.load(open(json_path, "r", encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(DATA_DIR, item["image"])
        text = item["text"]

        tokens = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze()
        labels = input_ids.clone()
        labels[tokens["attention_mask"].squeeze() == 0] = IGNORE_LABEL
        return {"img_path": img_path, "input_ids": input_ids, "labels": labels}

def collate_fn(batch):
    img_paths = [x["img_path"] for x in batch]
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return img_paths, input_ids, labels

# ==================== 首轮训练主逻辑 ====================
def main():
    print("===== 启动多模态模型【首轮训练】=====")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.pad_token = tokenizer.eos_token

    # 初始化模型
    model_cfg = ModelConfig()
    model = SimpleMultiModalModel(model_cfg).to(DEVICE)

    # LoRA配置
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    dataset = TrainDataset(DATA_JSON, tokenizer, MAX_SEQ_LEN)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    print(f"数据集加载完成，总样本数: {len(dataset)}")

    # 优化器 & 损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    scaler = GradScaler()
    model.train()

    # 首轮训练（仅1轮）
    global_step = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        print(f"\n----- 第 {epoch+1} 轮训练开始 -----")
        for idx, (img_paths, input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            with autocast(dtype=torch.float16):
                logits = model(input_ids, img_paths)
                loss = loss_fn(logits.view(-1, model_cfg.vocab_size), labels.view(-1))
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (idx + 1) % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                avg_loss = total_loss / GRAD_ACCUM
                print(f"Step: {global_step} | 当前损失 Loss: {avg_loss:.4f}")
                total_loss = 0.0

    # 保存首轮训练权重
    save_path = os.path.join(SAVE_DIR, "lora_epoch_1")
    model.save_pretrained(save_path)
    print(f"\n✅ 【首轮训练完成】权重已保存至: {save_path}")
    print("🎉 首轮训练结束，可下载权重本地使用/继续下一轮训练")

if __name__ == "__main__":
    main()
