# DeepSeek V4 Complete

DeepSeek V4 Complete 是一个完整的大型语言模型训练和推理框架，支持多模态能力、超长上下文理解和高级推理能力。

## 项目结构

```
deepseek_v4_complete/
├── core/                    # 核心模型模块
│   ├── __init__.py          # 导出所有核心类
│   ├── deepseek_v4_model.py # 主多模态模型
│   ├── attention_module.py  # 多头潜在注意力机制
│   ├── moe_architecture.py  # MoE架构（384专家）
│   ├── moe_route_balance.py # MoE路由平衡
│   ├── mtp_predictor.py     # 多Token预测器
│   ├── pos_encoding.py      # RoPE位置编码（131072序列）
│   ├── layer_norm.py        # RMSNorm
│   ├── activation_func.py   # SwiGLU激活
│   ├── kv_cache_manager.py  # KV缓存管理
│   ├── loss_collection.py   # 损失函数集合
│   ├── hybrid_ssm_attn.py   # 混合SSM注意力
│   ├── infini_memory.py     # 无限记忆注意力
│   └── unified_tokenizer.py # 统一Tokenizer
├── tokenizer/               # 分词器模块
│   ├── __init__.py
│   ├── base_tokenizer.py    # BPE分词器实现
│   ├── encode_decode.py     # 编码解码功能
│   ├── special_tokens.py    # 特殊令牌管理
│   └── vocab_config.json    # 词汇表配置
├── data_engine/             # 数据引擎模块
│   ├── __init__.py
│   ├── raw_data_clean.py    # 原始数据清洗
│   ├── data_filter_rule.py  # 数据过滤规则
│   ├── data_format_convert.py # 数据格式转换
│   ├── sample_builder.py    # 样本构建器
│   ├── data_augment.py      # 数据增强
│   ├── distributed_dataloader.py # 分布式数据加载器
│   ├── streaming_dataset.py # 流式数据集
│   └── data_path_config.yaml # 数据路径配置
├── train_system/            # 训练系统
│   ├── __init__.py
│   ├── pretrain_train.py    # 预训练脚本
│   └── deepspeed_cfg/       # DeepSpeed配置
│       └── ds_zero3.json
├── human_alignment/         # 人类对齐
├── omega_alignment/         # Omega对齐
├── multimodal_system/       # 多模态系统
├── intelligent_agent/       # 智能代理
├── search_enhance/          # 搜索增强
├── conversation_memory/     # 对话记忆
├── high_perf_infer/         # 高性能推理
├── safe_guard_system/       # 安全防护系统
├── model_evaluation/        # 模型评估
├── backend_server/          # 后端服务
├── frontend_webui/          # 前端WebUI
├── database_storage/        # 数据库存储
├── operation_monitor/       # 运维监控
├── common_utils/            # 通用工具
├── deploy_env/              # 部署环境
├── start_scripts/           # 启动脚本
├── evaluation/              # 评估模块
├── requirements.txt         # 依赖列表
└── README_Complete.md       # 项目说明
```

## 核心特性

### 模型架构
- **7168维模型**，32头注意力，64层Transformer
- **384专家MoE架构**，Top-K=6路由
- **131072超长上下文**支持（RoPE）
- **多Token预测（MTP）**支持
- **混合SSM注意力**（可选Mamba2）
- **无限记忆注意力**（InfiniAttention）
- **多模态支持**（图像/文本）

### 训练特性
- **DeepSpeed Zero-3**分布式训练支持
- **混合精度训练**（FP16/FP8）
- **梯度累积**
- **学习率调度**（Cosine + Warmup）
- **梯度裁剪**
- **检查点管理**

### 数据处理
- **流式数据加载**
- **数据清洗**（URL/邮箱/电话移除）
- **数据过滤**（质量评估）
- **数据增强**（同义词替换、随机删除等）
- **多格式支持**（JSON/JSONL/TXT/CSV/YAML/XML）

## 安装依赖

```bash
cd deepseek_v4_complete
pip install -r requirements.txt
```

## 训练

### 单卡训练

```bash
python train_system/pretrain_train.py \
    --data_paths /path/to/train_data.jsonl \
    --vocab_file /path/to/vocab.json \
    --output_dir ./output \
    --batch_size 8 \
    --max_steps 1000000 \
    --learning_rate 2e-5
```

### DeepSpeed 分布式训练

```bash
deepspeed --num_gpus=8 train_system/pretrain_train.py \
    --data_paths /path/to/train_data.jsonl \
    --vocab_file /path/to/vocab.json \
    --output_dir ./output \
    --batch_size 8 \
    --max_steps 1000000 \
    --learning_rate 2e-5 \
    --use_deepspeed \
    --deepspeed_config train_system/deepspeed_cfg/ds_zero3.json
```

## 模块说明

### Tokenizer
- **DeepSeekTokenizer**: BPE分词器，支持字节级编码
- **TokenizerEncodeDecode**: 编码解码工具类
- **SpecialTokens**: 特殊令牌管理（图像、音频、对话等）

### Data Engine
- **RawDataCleaner**: 数据清洗（URL、邮箱、HTML标签等）
- **TextFilter**: 文本质量过滤（长度、多样性、重复率等）
- **DataFormatConverter**: 数据格式转换（JSON/CSV/YAML/XML）
- **SampleBuilder**: 训练样本构建（预训练、QA、对话等）
- **DataAugmenter**: 数据增强（同义词替换、随机插入等）
- **StreamingDataset**: 流式数据集加载

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| batch_size | int | 8 | 每GPU批次大小 |
| max_steps | int | 1000000 | 最大训练步数 |
| learning_rate | float | 2e-5 | 学习率 |
| weight_decay | float | 0.01 | 权重衰减 |
| warmup_steps | int | 10000 | 预热步数 |
| clip_grad_norm | float | 1.0 | 梯度裁剪 |
| log_interval | int | 10 | 日志间隔 |
| save_interval | int | 1000 | 保存间隔 |

## License

MIT License