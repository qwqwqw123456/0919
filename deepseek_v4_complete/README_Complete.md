# DeepSeek V4 Complete

DeepSeek V4 Complete 是一个完整的、大型语言模型（LLM）训练与推理框架，支持多模态能力、超长上下文理解和高级推理能力。

## 📊 项目统计

| 指标 | 数量 |
|------|------|
| Python 文件总数 | 142 个 |
| 模块目录数 | 21 个 |
| 配置文件数 | 6+ 个 |

## 🏗️ 项目结构

```
deepseek_v4_complete/
├── core/                           # 核心模型架构
│   ├── deepseek_v4_model.py       # 主多模态模型（7168维，64层）
│   ├── attention_module.py         # 多头潜在注意力（MLA）
│   ├── moe_architecture.py        # MoE架构（384专家，Top-K=6）
│   ├── mtp_predictor.py           # 多Token预测（MTP）
│   ├── pos_encoding.py            # RoPE位置编码（131072上下文）
│   ├── hybrid_ssm_attn.py         # 混合SSM+Attention
│   ├── infini_memory.py           # 无限记忆注意力
│   └── ...
│
├── tokenizer/                      # 分词器模块
│   ├── base_tokenizer.py          # BPE分词器实现
│   ├── encode_decode.py           # 编码解码工具
│   └── special_tokens.py          # 特殊令牌管理
│
├── data_engine/                    # 数据处理引擎
│   ├── raw_data_clean.py          # 数据清洗（URL/邮箱/HTML）
│   ├── data_filter_rule.py        # 数据质量过滤
│   ├── data_augment.py            # 数据增强（回译/Mixup）
│   ├── distributed_dataloader.py  # 分布式数据加载
│   └── streaming_dataset.py       # 流式数据集
│
├── train_system/                   # 训练系统
│   ├── pretrain_train.py          # 预训练脚本
│   └── deepspeed_cfg/             # DeepSpeed Zero-3配置
│
├── human_alignment/                # 人类对齐训练
│   ├── sft_supervised_train.py    # SFT监督微调
│   ├── reward_model_train.py       # Reward Model训练
│   ├── dpo_optimizer_train.py     # DPO对齐训练
│   ├── ppo_align_train.py         # PPO强化学习对齐
│   └── ...
│
├── omega_alignment/               # Omega对齐（核心创新）
│   ├── omega_sampler.py           # 自适应采样器
│   ├── self_rewarding.py          # 自奖励机制
│   └── constitution_guard.py      # 宪法守卫
│
├── multimodal_system/              # 多模态系统
│   ├── vision_backbone.py         # ViT视觉骨干
│   ├── cross_modal_fusion.py      # 跨模态融合
│   ├── multimodal_processor.py     # 多模态处理器
│   └── ...
│
├── intelligent_agent/              # 智能代理
│   ├── core_agent.py              # 核心代理
│   ├── intent_recognize.py        # 意图识别
│   ├── task_planner.py            # 任务规划
│   ├── tool_manager.py            # 工具管理器
│   └── ...
│
├── search_enhance/                # 搜索增强
│   ├── search_engine_base.py      # 搜索引擎基类
│   ├── duckduckgo_api.py          # DuckDuckGo集成
│   ├── searxng_engine.py          # SearXNG集成
│   └── ...
│
├── conversation_memory/            # 对话记忆
│   ├── memory_manager.py          # 记忆管理器
│   ├── short_memory.py            # 短期记忆
│   ├── long_memory_db.py          # 长期记忆
│   └── ...
│
├── high_perf_infer/               # 高性能推理
│   ├── inference_engine.py         # 推理引擎
│   ├── vllm_speed_infer.py       # vLLM加速
│   ├── tensorrt_optimize.py       # TensorRT优化
│   └── ...
│
├── safe_guard_system/             # 安全防护
│   ├── safety_filter.py           # 内容安全过滤
│   ├── sensitive_word_detect.py   # 敏感词检测
│   ├── privacy_protect.py         # 隐私保护
│   └── ...
│
├── model_evaluation/              # 模型评估
│   ├── basic_metric_calc.py      # 基础指标（PPL/BLEU/ROUGE）
│   ├── common_bench_test.py       # 基准测试（MMLU/GSM8K）
│   └── ...
│
├── backend_server/                # 后端API服务
│   ├── api_server.py             # FastAPI服务
│   ├── user_auth_verify.py       # 用户认证
│   ├── api_rate_limit.py         # 限流
│   └── ...
│
├── frontend_webui/               # 前端界面
│   ├── streamlit_chat_ui.py      # Streamlit聊天界面
│   └── gradio_chat_app.py        # Gradio聊天应用
│
├── database_storage/              # 数据库存储
│   ├── mysql_connect.py          # MySQL连接
│   ├── redis_cache.py            # Redis缓存
│   ├── vector_database.py        # 向量数据库（FAISS）
│   └── ...
│
├── operation_monitor/             # 运维监控
│   ├── unified_logger.py          # 统一日志
│   ├── gpu_status_monitor.py     # GPU监控
│   ├── service_qps_delay.py      # QPS监控
│   └── ...
│
├── common_utils/                  # 通用工具
│   ├── config_yaml_parse.py      # YAML配置解析
│   ├── device_env_utils.py       # 设备检测
│   ├── weight_file_utils.py      # 权重文件操作
│   ├── encrypt_decrypt.py        # 加密解密
│   └── ...
│
├── deploy_env/                    # 部署环境
│   ├── Dockerfile.base           # 基础镜像
│   ├── Dockerfile.service         # 服务镜像
│   ├── docker-compose.yml        # Docker Compose
│   ├── k8s_deploy.yaml          # Kubernetes部署
│   └── nginx_proxy.conf          # Nginx配置
│
├── start_scripts/                 # 启动脚本
│   ├── run_all_onekey.sh         # 一键启动
│   ├── start_pretrain.sh         # 预训练启动
│   ├── start_api_server.sh       # API服务启动
│   └── ...
│
├── evaluation/                     # 评估模块
├── requirements.txt               # 依赖列表
└── README_Complete.md             # 本文档
```

## 🚀 核心特性

### 模型架构
- **7168维模型**，32头注意力，64层Transformer
- **384专家MoE架构**，Top-K=6路由，负载均衡
- **131072超长上下文**支持（RoPE位置编码）
- **多Token预测（MTP）**支持
- **混合SSM+Attention**（Mamba2可选）
- **无限记忆注意力**（InfiniAttention）
- **多模态支持**（图像/文本/音频）

### 训练系统
- **DeepSpeed ZeRO-3** 分布式训练
- **混合精度训练**（FP16/FP8）
- **梯度累积**与**梯度裁剪**
- **Cosine + Warmup** 学习率调度
- **检查点保存/恢复**
- **流式数据加载**

### 对齐训练
- **SFT** 监督微调
- **Reward Model** 奖励模型
- **DPO** 直接偏好优化
- **PPO** 近端策略优化
- **IPO** 身份偏好优化
- **在线样本收集**

### 推理部署
- **PyTorch** 原生推理
- **vLLM** PagedAttention加速
- **TensorRT** 优化推理
- **ONNX** 跨平台部署
- **量化**（INT4/INT8）
- **批处理**与**流式输出**

## 📦 安装

```bash
# 克隆项目
cd deepseek_v4_complete

# 安装依赖
pip install -r requirements.txt

# 或安装核心依赖（开发环境）
pip install torch transformers deepspeed fastapi streamlit
```

## 🎯 快速开始

### 1. 预训练

```bash
python train_system/pretrain_train.py \
    --data_paths /path/to/train_data.jsonl \
    --vocab_file /path/to/vocab.json \
    --output_dir ./output \
    --batch_size 8 \
    --max_steps 1000000 \
    --learning_rate 2e-5
```

### 2. DeepSpeed 分布式训练

```bash
deepspeed --num_gpus=8 train_system/pretrain_train.py \
    --data_paths /path/to/train_data.jsonl \
    --vocab_file /path/to/vocab.json \
    --use_deepspeed \
    --deepspeed_config train_system/deepspeed_cfg/ds_zero3.json
```

### 3. SFT 监督微调

```python
from human_alignment import SFTTrainer, SFTConfig
from core import DeepSeekV4Multimodal

model = DeepSeekV4Multimodal(V4Config())
trainer = SFTTrainer(model, tokenizer, SFTConfig(max_steps=10000))
trainer.train(train_dataset)
```

### 4. DPO 对齐训练

```python
from human_alignment import DPOTrainer

trainer = DPOTrainer(policy_model, ref_model, beta=0.1)
trainer.train(preference_dataset)
```

### 5. 推理

```python
from high_perf_infer import InferenceEngine, GenerateConfig

engine = InferenceEngine(model, tokenizer)
config = GenerateConfig(max_new_tokens=512, temperature=0.7)

response = engine.chat([
    {'role': 'user', 'content': 'Hello!'}
], config)
```

### 6. 启动 API 服务

```bash
# 使用 FastAPI
uvicorn backend_server.api_server:app --host 0.0.0.0 --port 8000

# 或使用启动脚本
bash start_scripts/start_api_server.sh
```

### 7. 启动 WebUI

```bash
# Streamlit
streamlit run frontend_webui/streamlit_chat_ui.py --server.port 8501

# Gradio
python frontend_webui/gradio_chat_app.py
```

### 8. Docker 部署

```bash
# 开发环境
docker-compose -f deploy_env/docker-compose.yml up -d

# 生产集群
docker stack deploy -c deploy_env/docker_cluster.yml deepseek

# Kubernetes
kubectl apply -f deploy_env/k8s_deploy.yaml
```

## 🔧 配置参数

### 模型配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 7168 | 模型维度 |
| n_heads | 32 | 注意力头数 |
| n_layers | 64 | Transformer层数 |
| num_experts | 384 | MoE专家数量 |
| top_k | 6 | Top-K路由 |
| max_seq_len | 131072 | 最大序列长度 |

### 训练配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 8 | 批次大小 |
| learning_rate | 2e-5 | 学习率 |
| weight_decay | 0.01 | 权重衰减 |
| warmup_steps | 10000 | 预热步数 |
| max_steps | 1000000 | 最大步数 |
| clip_grad_norm | 1.0 | 梯度裁剪 |

### 推理配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_new_tokens | 128 | 最大生成长度 |
| temperature | 0.7 | 采样温度 |
| top_p | 0.9 | Nucleus采样 |
| top_k | 50 | Top-K采样 |
| repetition_penalty | 1.0 | 重复惩罚 |

## 📁 模块说明

### core - 核心模型
核心Transformer架构组件，包括：
- `DeepSeekV4Multimodal`: 主模型类
- `MultiHeadLatentAttention`: MLA注意力
- `DeepSeekMoE`: MoE专家网络
- `MultiTokenPredictor`: MTP预测头
- `RMSNorm`: RMS归一化
- `SwiGLU`: 激活函数

### tokenizer - 分词器
- `DeepSeekTokenizer`: BPE分词器
- `TokenizerEncodeDecode`: 编解码工具
- `SpecialTokens`: 特殊令牌管理

### data_engine - 数据引擎
- `RawDataCleaner`: 数据清洗
- `TextFilter`: 质量过滤
- `DataAugmenter`: 数据增强
- `StreamingDataset`: 流式数据加载

### human_alignment - 人类对齐
- `SFTTrainer`: SFT微调
- `DPOTrainer`: DPO训练
- `PPOTrainer`: PPO训练
- `RewardScorer`: 奖励评分

### multimodal_system - 多模态
- `VisionBackbone`: ViT骨干
- `CrossModalFusion`: 跨模态融合
- `ImageProcessor`: 图像处理

### intelligent_agent - 智能代理
- `CoreAgent`: 核心代理
- `IntentRecognizer`: 意图识别
- `TaskPlanner`: 任务规划
- `ToolManager`: 工具管理

### high_perf_infer - 高性能推理
- `InferenceEngine`: 推理引擎
- `VLLMInference`: vLLM加速
- `TensorRTOptimizer`: TensorRT优化

### backend_server - 后端服务
- `APIServer`: FastAPI服务
- `RateLimiter`: 限流器
- `WebSocketHandler`: WebSocket支持

## 📈 评估指标

### 基础指标
- Perplexity (PPL)
- BLEU / ROUGE
- METEOR / chrF

### 基准测试
- MMLU (大规模多任务)
- GSM8K (数学推理)
- HumanEval (代码生成)
- HellaSwag (常识推理)

### 对齐评估
- 响应质量
- 安全性
- 帮助性

## 🛡️ 安全特性

- 内容安全过滤
- 敏感词检测
- Prompt注入防护
- 隐私信息保护
- 响应水印

## 📊 监控指标

- GPU 利用率/显存
- CPU/内存使用
- 服务 QPS/延迟
- 错误率统计
- 自定义告警

## 🔗 外部集成

### 搜索增强
- DuckDuckGo API
- SearXNG

### 数据库
- MySQL
- Redis
- FAISS 向量数据库
- Milvus

### 通知
- Email
- SMS (阿里云/钉钉)
- Webhook

## 📝 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请提交 GitHub Issue。