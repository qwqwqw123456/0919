# 项目代码结构分析

## 概览
- 模块数量: 148
- 类数量: 510
- 函数/方法数量: 4341

## 文件结构
```
└── /
    └── workspace/
        └── deepseek_v4_complete/
            ├── backend_server/
            │   ├── __init__.py
            │   ├── api_rate_limit.py
            │   ├── api_router_manage.py
            │   ├── api_server.py
            │   ├── cors_allow.py
            │   ├── fastapi_main_server.py
            │   ├── service_health_check.py
            │   ├── user_auth_verify.py
            │   └── websocket_stream.py
            ├── codegraph/
            │   ├── __init__.py
            │   ├── code_analyzer.py
            │   ├── dependency_extractor.py
            │   ├── graph_generator.py
            │   └── visualization_api.py
            ├── common_utils/
            │   ├── __init__.py
            │   ├── config_yaml_parse.py
            │   ├── device_env_utils.py
            │   ├── encrypt_decrypt.py
            │   ├── file_dir_operate.py
            │   ├── multi_thread_pool.py
            │   ├── time_datetime_utils.py
            │   └── weight_file_utils.py
            ├── conversation_memory/
            │   ├── __init__.py
            │   ├── long_memory_db.py
            │   ├── memory_compress.py
            │   ├── memory_manager.py
            │   ├── memory_retrieve.py
            │   ├── session_manage.py
            │   └── short_memory.py
            ├── core/
            │   ├── __init__.py
            │   ├── activation_func.py
            │   ├── attention_module.py
            │   ├── deepseek_v4_model.py
            │   ├── hybrid_ssm_attn.py
            │   ├── infini_memory.py
            │   ├── kv_cache_manager.py
            │   ├── layer_norm.py
            │   ├── loss_collection.py
            │   ├── moe_architecture.py
            │   ├── moe_route_balance.py
            │   ├── mtp_predictor.py
            │   ├── pos_encoding.py
            │   └── unified_tokenizer.py
            ├── data_engine/
            │   ├── __init__.py
            │   ├── data_augment.py
            │   ├── data_filter_rule.py
            │   ├── data_format_convert.py
            │   ├── distributed_dataloader.py
            │   ├── raw_data_clean.py
            │   ├── sample_builder.py
            │   └── streaming_dataset.py
            ├── database_storage/
            │   ├── __init__.py
            │   ├── data_backup_restore.py
            │   ├── mysql_connect.py
            │   ├── redis_cache.py
            │   ├── table_struct.py
            │   └── vector_database.py
            ├── deploy_env/
            │   └── __init__.py
            ├── evaluation/
            │   ├── __init__.py
            │   ├── auto_eval.py
            │   ├── benchmark_test.py
            │   └── eval_metrics.py
            ├── frontend_webui/
            │   ├── __init__.py
            │   ├── gradio_chat_app.py
            │   └── streamlit_chat_ui.py
            ├── high_perf_infer/
            │   ├── __init__.py
            │   ├── base_inference.py
            │   ├── batch_queue_infer.py
            │   ├── gpu_memory_tune.py
            │   ├── hybrid_engine.py
            │   ├── inference_engine.py
            │   ├── onnx_export_infer.py
            │   ├── quant_engine.py
            │   ├── stream_response.py
            │   ├── tensorrt_optimize.py
            │   └── vllm_speed_infer.py
            ├── human_alignment/
            │   ├── __init__.py
            │   ├── dpo_optimizer_train.py
            │   ├── ipo_train.py
            │   ├── online_sample_collect.py
            │   ├── ppo_align_train.py
            │   ├── reward_model_train.py
            │   ├── reward_scoring.py
            │   ├── sft_data_formatter.py
            │   └── sft_supervised_train.py
            ├── intelligent_agent/
            │   ├── __init__.py
            │   ├── code_interpreter.py
            │   ├── core_agent.py
            │   ├── function_call_parse.py
            │   ├── intent_recognize.py
            │   ├── memory_think_pool.py
            │   ├── plugin_loader.py
            │   ├── task_planner.py
            │   ├── thought_reflect.py
            │   └── tool_manager.py
            ├── model_evaluation/
            │   ├── __init__.py
            │   ├── basic_metric_calc.py
            │   ├── chat_quality_eval.py
            │   ├── common_bench_test.py
            │   ├── reasoning_eval.py
            │   ├── report_generator.py
            │   └── safety_eval.py
            ├── multimodal_system/
            │   ├── __init__.py
            │   ├── cross_modal_fusion.py
            │   ├── modal_projection.py
            │   ├── multi_modal_infer.py
            │   ├── multimodal_processor.py
            │   ├── multimodal_unified_encoder.py
            │   ├── text_embed_encoder.py
            │   ├── vision_backbone.py
            │   ├── vision_cache_precompute.py
            │   └── vision_feature_extract.py
            ├── omega_alignment/
            │   ├── __init__.py
            │   ├── constitution_guard.py
            │   ├── omega_sampler.py
            │   └── self_rewarding.py
            ├── operation_monitor/
            │   ├── __init__.py
            │   ├── cpu_mem_monitor.py
            │   ├── error_catch_record.py
            │   ├── gpu_status_monitor.py
            │   ├── service_qps_delay.py
            │   ├── sms_email_alert.py
            │   └── unified_logger.py
            ├── safe_guard_system/
            │   ├── __init__.py
            │   ├── harmful_classify.py
            │   ├── illegal_content_judge.py
            │   ├── privacy_protect.py
            │   ├── refuse_reply_strategy.py
            │   ├── safety_filter.py
            │   ├── sensitive_word_detect.py
            │   └── watermark_embed.py
            ├── search_enhance/
            │   ├── __init__.py
            │   ├── content_summarize.py
            │   ├── duckduckgo_api.py
            │   ├── result_filter.py
            │   ├── search_answer_merge.py
            │   ├── search_engine_base.py
            │   ├── search_judge.py
            │   └── searxng_engine.py
            ├── start_scripts/
            │   └── __init__.py
            ├── test_imports.py
            ├── tokenizer/
            │   ├── __init__.py
            │   ├── base_tokenizer.py
            │   ├── encode_decode.py
            │   └── special_tokens.py
            └── train_system/
                ├── __init__.py
                ├── deepspeed_cfg/
                │   └── __init__.py
                └── pretrain_train.py
```

## 模块依赖图
```mermaid
graph TD
    subgraph "deepseek_v4_complete.test_imports"
    end
    subgraph "search_enhance.search_engine_base"
    end
    subgraph "search_engine_base.BaseSearchEngine"
    end
    subgraph "search_enhance.search_judge"
    end
    subgraph "search_judge.SearchType"
    end
    subgraph "search_judge.QueryIntent"
    end
    subgraph "search_judge.QueryDomain"
    end
    subgraph "search_enhance.searxng_engine"
    end
    subgraph "searxng_engine.SearXNGCategory"
    end
    subgraph "searxng_engine.SearXNGEngineType"
    end
    subgraph "searxng_engine.SearXNGEngine"
    end
    subgraph "search_enhance.duckduckgo_api"
    end
    subgraph "duckduckgo_api.SearchType"
    end
    subgraph "duckduckgo_api.DuckDuckGoAPI"
    end
    subgraph "deepseek_v4_complete.backend_server"
    end
    subgraph "backend_server.api_server"
    end
    subgraph "api_server.Message"
    end
    subgraph "api_server.ChatCompletionRequest"
    end
    subgraph "api_server.CompletionRequest"
    end
    subgraph "api_server.ChatCompletionResponse"
    end
    subgraph "api_server.CompletionResponse"
    end
    subgraph "api_server.APIError"
    end
    subgraph "backend_server.cors_allow"
    end
    subgraph "backend_server.api_router_manage"
    end
    subgraph "backend_server.fastapi_main_server"
    end
    subgraph "backend_server.websocket_stream"
    end
    subgraph "deepseek_v4_complete.data_engine"
    end
    subgraph "data_engine.streaming_dataset"
    end
    subgraph "streaming_dataset.StreamingDataset"
    end
    subgraph "streaming_dataset.ConcatDataset"
    end
    subgraph "streaming_dataset.DynamicDataset"
    end
    subgraph "streaming_dataset.TextDataset"
    end
    subgraph "data_engine.raw_data_clean"
    end
    subgraph "data_engine.data_augment"
    end
    subgraph "data_engine.sample_builder"
    end
    subgraph "data_engine.data_format_convert"
    end
    subgraph "data_engine.data_filter_rule"
    end
    subgraph "data_engine.distributed_dataloader"
    end
    subgraph "deepseek_v4_complete.core"
    end
    subgraph "core.infini_memory"
    end
    subgraph "infini_memory.InfiniAttention"
    end
    subgraph "core.attention_module"
    end
    subgraph "attention_module.MultiHeadLatentAttention"
    end
    subgraph "attention_module.StreamingAttention"
    end
    subgraph "attention_module.SparseAttention"
    end
    subgraph "attention_module.FlashAttentionWrapper"
    end
    subgraph "attention_module.LongContextAttention"
    end
    subgraph "core.layer_norm"
    end
    subgraph "layer_norm.RMSNorm"
    end
    subgraph "core.mtp_predictor"
    end
    subgraph "mtp_predictor.MultiTokenPredictor"
    end
    subgraph "core.hybrid_ssm_attn"
    end
    subgraph "hybrid_ssm_attn.HybridBlock"
    end
    subgraph "core.loss_collection"
    end
    subgraph "loss_collection.VLContrastiveLoss"
    end
    subgraph "loss_collection.TotalLoss"
    end
    subgraph "core.activation_func"
    end
    subgraph "activation_func.SwiGLU"
    end
    subgraph "core.deepseek_v4_model"
    end
    subgraph "deepseek_v4_model.VisionEncoder"
    end
    subgraph "deepseek_v4_model.VisionProjector"
    end
    subgraph "deepseek_v4_model.RMSNorm"
    end
    subgraph "deepseek_v4_model.SwiGLU"
    end
    subgraph "deepseek_v4_model.DeepSeekMoE"
    end
    subgraph "deepseek_v4_model.LongContextAttention"
    end
    subgraph "deepseek_v4_model.V4TransformerBlock"
    end
    subgraph "deepseek_v4_model.DeepSeekV4Multimodal"
    end
    subgraph "core.kv_cache_manager"
    end
    subgraph "core.moe_route_balance"
    end
    subgraph "moe_route_balance.MoEWithBalance"
    end
    subgraph "core.moe_architecture"
    end
    subgraph "moe_architecture.DeepSeekMoE"
    end
    subgraph "core.unified_tokenizer"
    end
    subgraph "core.pos_encoding"
    end
    subgraph "pos_encoding.StreamingRoPE"
    end
    subgraph "pos_encoding.YaRNRoPE"
    end
    subgraph "pos_encoding.LinearRoPE"
    end
    subgraph "deepseek_v4_complete.model_evaluation"
    end
    subgraph "model_evaluation.common_bench_test"
    end
    subgraph "common_bench_test.BaseBenchmark"
    end
    subgraph "common_bench_test.MMLUBenchmark"
    end
    subgraph "common_bench_test.GSM8KBenchmark"
    end
    subgraph "common_bench_test.HumanEvalBenchmark"
    end
    subgraph "common_bench_test.HellaSwagBenchmark"
    end
    subgraph "common_bench_test.TruthfulQABenchmark"
    end
    subgraph "model_evaluation.chat_quality_eval"
    end
    subgraph "model_evaluation.basic_metric_calc"
    end
    subgraph "model_evaluation.reasoning_eval"
    end
    subgraph "reasoning_eval.ReasoningType"
    end
    subgraph "model_evaluation.report_generator"
    end
    subgraph "model_evaluation.safety_eval"
    end
    subgraph "deepseek_v4_complete.safe_guard_system"
    end
    subgraph "safe_guard_system.safety_filter"
    end
    subgraph "deepseek_v4_complete.operation_monitor"
    end
    subgraph "operation_monitor.gpu_status_monitor"
    end
    subgraph "gpu_status_monitor.GPUBackend"
    end
    subgraph "gpu_status_monitor.NVMLBackend"
    end
    subgraph "gpu_status_monitor.SMIBackend"
    end
    subgraph "gpu_status_monitor.MockGPUBackend"
    end
    subgraph "operation_monitor.unified_logger"
    end
    subgraph "unified_logger.JsonFormatter"
    end
    subgraph "unified_logger.ColoredFormatter"
    end
    subgraph "operation_monitor.sms_email_alert"
    end
    subgraph "sms_email_alert.AlertLevel"
    end
    subgraph "sms_email_alert.AlertChannel"
    end
    subgraph "sms_email_alert.AlertBackend"
    end
    subgraph "sms_email_alert.EmailBackend"
    end
    subgraph "sms_email_alert.SMSBackend"
    end
    subgraph "sms_email_alert.WebhookBackend"
    end
    subgraph "sms_email_alert.DingTalkBackend"
    end
    subgraph "sms_email_alert.WeChatBackend"
    end
    subgraph "sms_email_alert.FeishuBackend"
    end
    subgraph "sms_email_alert.ConsoleBackend"
    end
    subgraph "operation_monitor.service_qps_delay"
    end
    subgraph "operation_monitor.error_catch_record"
    end
    subgraph "error_catch_record.ErrorLevel"
    end
    subgraph "error_catch_record.ErrorCategory"
    end
    subgraph "error_catch_record.ErrorBackend"
    end
    subgraph "error_catch_record.FileBackend"
    end
    subgraph "error_catch_record.MemoryBackend"
    end
    subgraph "operation_monitor.cpu_mem_monitor"
    end
    subgraph "cpu_mem_monitor.SystemMonitorBackend"
    end
    subgraph "cpu_mem_monitor.PSUtilBackend"
    end
    subgraph "cpu_mem_monitor.CommandBackend"
    end
    subgraph "cpu_mem_monitor.MockBackend"
    end
    subgraph "high_perf_infer.batch_queue_infer"
    end
    subgraph "deepseek_v4_complete.high_perf_infer"
    end
    subgraph "high_perf_infer.inference_engine"
    end
    subgraph "inference_engine.TensorRTInferenceEngine"
    end
    subgraph "high_perf_infer.onnx_export_infer"
    end
    subgraph "high_perf_infer.vllm_speed_infer"
    end
    subgraph "deepseek_v4_complete.omega_alignment"
    end
    subgraph "omega_alignment.omega_sampler"
    end
    subgraph "omega_sampler.OmegaSampler"
    end
    subgraph "omega_sampler.OmegaAligner"
    end
    subgraph "omega_sampler.AdaptiveOmegaLoss"
    end
    subgraph "deepseek_v4_complete.frontend_webui"
    end
    subgraph "frontend_webui.gradio_chat_app"
    end
    subgraph "frontend_webui.streamlit_chat_ui"
    end
    subgraph "human_alignment.online_sample_collect"
    end
    subgraph "online_sample_collect.OnlineSampleCollector"
    end
    subgraph "online_sample_collect.CollectedSampleDataset"
    end
    subgraph "human_alignment.ipo_train"
    end
    subgraph "ipo_train.IPODataset"
    end
    subgraph "ipo_train.IPOLoss"
    end
    subgraph "ipo_train.SimpleIPOLoss"
    end
    subgraph "ipo_train.RegularizedIPOLoss"
    end
    subgraph "ipo_train.SimpleIPOTrainer"
    end
    subgraph "ipo_train.RegularizedIPOTrainer"
    end
    subgraph "human_alignment.reward_model_train"
    end
    subgraph "reward_model_train.RewardModel"
    end
    subgraph "reward_model_train.RewardModelWithScalarHead"
    end
    subgraph "reward_model_train.PreferencePairDataset"
    end
    subgraph "reward_model_train.BradleyTerryLoss"
    end
    subgraph "reward_model_train.RewardModelLoss"
    end
    subgraph "human_alignment.ppo_align_train"
    end
    subgraph "ppo_align_train.ValueNetwork"
    end
    subgraph "ppo_align_train.ActorCriticModel"
    end
    subgraph "ppo_align_train.PPODataset"
    end
    subgraph "ppo_align_train.PPOLoss"
    end
    subgraph "human_alignment.sft_data_formatter"
    end
    subgraph "sft_data_formatter.ChatMLTemplate"
    end
    subgraph "sft_data_formatter.Llama2Template"
    end
    subgraph "sft_data_formatter.QwenTemplate"
    end
    subgraph "sft_data_formatter.SFTDataset"
    end
    subgraph "human_alignment.dpo_optimizer_train"
    end
    subgraph "dpo_optimizer_train.DPODataset"
    end
    subgraph "dpo_optimizer_train.DPOLoss"
    end
    subgraph "dpo_optimizer_train.AdaptiveDPO_loss"
    end
    subgraph "dpo_optimizer_train.DPOwithRewardModel"
    end
    subgraph "dpo_optimizer_train.UnlikelihoodDPO_loss"
    end
    subgraph "human_alignment.sft_supervised_train"
    end
    subgraph "sft_supervised_train.SFTDataset"
    end
    subgraph "sft_supervised_train.SFTLoss"
    end
    subgraph "human_alignment.reward_scoring"
    end
    subgraph "deepseek_v4_complete.conversation_memory"
    end
    subgraph "conversation_memory.memory_manager"
    end
    subgraph "deepseek_v4_complete.common_utils"
    end
    subgraph "common_utils.multi_thread_pool"
    end
    subgraph "multi_thread_pool.TaskStatus"
    end
    subgraph "multi_thread_pool.BaseExecutor"
    end
    subgraph "multi_thread_pool.ThreadPoolExecutorHelper"
    end
    subgraph "multi_thread_pool.ProcessPoolExecutorHelper"
    end
    subgraph "common_utils.encrypt_decrypt"
    end
    subgraph "encrypt_decrypt.HashAlgorithm"
    end
    subgraph "encrypt_decrypt.CipherMode"
    end
    subgraph "common_utils.device_env_utils"
    end
    subgraph "device_env_utils.DeviceType"
    end
    subgraph "common_utils.file_dir_operate"
    end
    subgraph "common_utils.time_datetime_utils"
    end
    subgraph "time_datetime_utils.TimeFormat"
    end
    subgraph "common_utils.weight_file_utils"
    end
    subgraph "common_utils.config_yaml_parse"
    end
    subgraph "deepseek_v4_complete.tokenizer"
    end
    subgraph "tokenizer.base_tokenizer"
    end
    subgraph "tokenizer.encode_decode"
    end
    subgraph "tokenizer.special_tokens"
    end
    subgraph "deepseek_v4_complete.train_system"
    end
    subgraph "train_system.pretrain_train"
    end
    subgraph "deepseek_v4_complete.codegraph"
    end
    subgraph "codegraph.graph_generator"
    end
    subgraph "codegraph.visualization_api"
    end
    subgraph "visualization_api.AnalysisResult"
    end
    subgraph "visualization_api.DependencyGraphResult"
    end
    subgraph "codegraph.code_analyzer"
    end
    subgraph "codegraph.dependency_extractor"
    end
    subgraph "deepseek_v4_complete.multimodal_system"
    end
    subgraph "multimodal_system.cross_modal_fusion"
    end
    subgraph "cross_modal_fusion.ModalityEmbedding"
    end
    subgraph "cross_modal_fusion.GatedFusion"
    end
    subgraph "cross_modal_fusion.CrossAttention"
    end
    subgraph "cross_modal_fusion.AttentionFusion"
    end
    subgraph "cross_modal_fusion.BilinearFusion"
    end
    subgraph "cross_modal_fusion.ConcatenationFusion"
    end
    subgraph "cross_modal_fusion.HierarchicalFusion"
    end
    subgraph "cross_modal_fusion.CrossModalFusion"
    end
    subgraph "cross_modal_fusion.MultiHeadCrossModalAttention"
    end
    subgraph "cross_modal_fusion.TensorFusionNetwork"
    end
    subgraph "cross_modal_fusion.FactorizedTensorFusion"
    end
    subgraph "cross_modal_fusion.LowRankModalFusion"
    end
    subgraph "multimodal_system.vision_feature_extract"
    end
    subgraph "vision_feature_extract.ConvBottleneck"
    end
    subgraph "vision_feature_extract.ResNetBackbone"
    end
    subgraph "vision_feature_extract.FeaturePyramid"
    end
    subgraph "vision_feature_extract.PathAggregationFPN"
    end
    subgraph "vision_feature_extract.AttentionPool2d"
    end
    subgraph "vision_feature_extract.VisionFeatureExtractor"
    end
    subgraph "vision_feature_extract.MultiScaleFeatureExtractor"
    end
    subgraph "vision_feature_extract.HierarchicalFeatureExtractor"
    end
    subgraph "vision_feature_extract.SpatialAttention"
    end
    subgraph "vision_feature_extract.ChannelAttention"
    end
    subgraph "vision_feature_extract.CBAM"
    end
    subgraph "vision_feature_extract.AttentionEnhancedExtractor"
    end
    subgraph "vision_feature_extract.SemanticFeatureExtractor"
    end
    subgraph "multimodal_system.vision_backbone"
    end
    subgraph "vision_backbone.PatchEmbedding"
    end
    subgraph "vision_backbone.RotaryPositionEmbedding"
    end
    subgraph "vision_backbone.ViTAttention"
    end
    subgraph "vision_backbone.ViTMLP"
    end
    subgraph "vision_backbone.ViTBlock"
    end
    subgraph "vision_backbone.ViTEncoder"
    end
    subgraph "vision_backbone.VisionBackbone"
    end
    subgraph "vision_backbone.ImageTokenizer"
    end
    subgraph "vision_backbone.ResidualBlock"
    end
    subgraph "vision_backbone.VectorQuantizer"
    end
    subgraph "vision_backbone.TileProcessor"
    end
    subgraph "multimodal_system.vision_cache_precompute"
    end
    subgraph "multimodal_system.multimodal_unified_encoder"
    end
    subgraph "multimodal_unified_encoder.UnifiedAttention"
    end
    subgraph "multimodal_unified_encoder.UnifiedTransformerBlock"
    end
    subgraph "multimodal_unified_encoder.ModalityTypeEmbedding"
    end
    subgraph "multimodal_unified_encoder.TextBranch"
    end
    subgraph "multimodal_unified_encoder.ImageBranch"
    end
    subgraph "multimodal_unified_encoder.AudioBranch"
    end
    subgraph "multimodal_unified_encoder.MultimodalUnifiedEncoder"
    end
    subgraph "multimodal_unified_encoder.CrossModalUnifiedEncoder"
    end
    subgraph "multimodal_unified_encoder.CrossAttentionLayer"
    end
    subgraph "multimodal_system.multi_modal_infer"
    end
    subgraph "multimodal_system.modal_projection"
    end
    subgraph "modal_projection.LinearProjection"
    end
    subgraph "modal_projection.MLPProjection"
    end
    subgraph "modal_projection.CrossAttentionProjection"
    end
    subgraph "modal_projection.ResidualProjection"
    end
    subgraph "modal_projection.GatedProjection"
    end
    subgraph "modal_projection.MultiplicativeProjection"
    end
    subgraph "modal_projection.BiDirectionalProjection"
    end
    subgraph "modal_projection.ModalProjection"
    end
    subgraph "modal_projection.MultimodalProjector"
    end
    subgraph "modal_projection.AdaptiveProjection"
    end
    subgraph "modal_projection.ComposedProjection"
    end
    subgraph "multimodal_system.multimodal_processor"
    end
    subgraph "multimodal_processor.ImageProcessor"
    end
    subgraph "multimodal_processor.VisionEncoder"
    end
    subgraph "multimodal_processor.TransformerLayer"
    end
    subgraph "multimodal_processor.VisionProjector"
    end
    subgraph "multimodal_processor.DynamicImageProcessor"
    end
    subgraph "multimodal_processor.AudioProcessor"
    end
    subgraph "multimodal_processor.MultimodalProjector"
    end
    subgraph "multimodal_processor.V4TransformerBlock"
    end
    subgraph "multimodal_processor.DeepSeekV4Multimodal"
    end
    subgraph "multimodal_system.text_embed_encoder"
    end
    subgraph "text_embed_encoder.LearnedPositionalEmbedding"
    end
    subgraph "text_embed_encoder.SinusoidalPositionalEmbedding"
    end
    subgraph "text_embed_encoder.RotaryPositionalEmbedding"
    end
    subgraph "text_embed_encoder.RelativePositionBias"
    end
    subgraph "text_embed_encoder.TextAttention"
    end
    subgraph "text_embed_encoder.TextMLP"
    end
    subgraph "text_embed_encoder.TransformerBlock"
    end
    subgraph "text_embed_encoder.TextEncoderLayer"
    end
    subgraph "text_embed_encoder.TextEmbedEncoder"
    end
    subgraph "text_embed_encoder.PositionalTextEncoder"
    end
    subgraph "text_embed_encoder.RotaryTextEncoder"
    end
    subgraph "text_embed_encoder.TextEncoderWithProjection"
    end
    subgraph "text_embed_encoder.GatedProjection"
    end
    subgraph "text_embed_encoder.MultiScaleTextEncoder"
    end
    subgraph "text_embed_encoder.ConditionalTextEncoder"
    end
    subgraph "text_embed_encoder.TextEncoderPooler"
    end
    subgraph "intelligent_agent.function_call_parse"
    end
    subgraph "intelligent_agent.task_planner"
    end
    subgraph "task_planner.StepType"
    end
    subgraph "task_planner.PlanStrategy"
    end
    subgraph "task_planner.StepStatus"
    end
    subgraph "task_planner.HierarchicalPlanner"
    end
    subgraph "task_planner.ReactivePlanner"
    end
    subgraph "intelligent_agent.core_agent"
    end
    subgraph "core_agent.AgentState"
    end
    subgraph "core_agent.ErrorType"
    end
    subgraph "core_agent.MultiTurnAgent"
    end
    subgraph "core_agent.ReActAgent"
    end
    subgraph "intelligent_agent.intent_recognize"
    end
    subgraph "intent_recognize.IntentType"
    end
    subgraph "intent_recognize.AdvancedIntentRecognizer"
    end
    subgraph "intelligent_agent.thought_reflect"
    end
    subgraph "thought_reflect.ReflectionLevel"
    end
    subgraph "thought_reflect.EvaluationDimension"
    end
    subgraph "thought_reflect.ImprovementType"
    end
    subgraph "thought_reflect.AdvancedThoughtReflection"
    end
    subgraph "thought_reflect.SelfCorrectionReflection"
    end
    subgraph "deepseek_v4_complete.database_storage"
    end
    subgraph "database_storage.mysql_connect"
    end
    subgraph "mysql_connect.Base"
    end
    subgraph "database_storage.table_struct"
    end
    subgraph "table_struct.UserStatus"
    end
    subgraph "table_struct.MessageRole"
    end
    subgraph "table_struct.SessionStatus"
    end
    subgraph "table_struct.TokenType"
    end
    subgraph "table_struct.AuditAction"
    end
    subgraph "table_struct.User"
    end
    subgraph "table_struct.Token"
    end
    subgraph "table_struct.Session"
    end
    subgraph "table_struct.Message"
    end
    subgraph "table_struct.VectorStore"
    end
    subgraph "table_struct.AuditLog"
    end
    subgraph "table_struct.Config"
    end
    subgraph "table_struct.Model"
    end
    subgraph "table_struct.FileStorage"
    end
    subgraph "table_struct.RateLimit"
    end
    subgraph "table_struct.Webhook"
    end
    subgraph "table_struct.WebhookDelivery"
    end
    subgraph "table_struct.Collection"
    end
    subgraph "table_struct.Tool"
    end
    subgraph "database_storage.vector_database"
    end
    subgraph "vector_database.VectorIndexType"
    end
    subgraph "vector_database.DistanceMetric"
    end
    subgraph "vector_database.BaseVectorDB"
    end
    subgraph "vector_database.FAISSVectorDB"
    end
    subgraph "vector_database.MilvusVectorDB"
    end
    subgraph "database_storage.data_backup_restore"
    end
    subgraph "data_backup_restore.BackupStatus"
    end
    subgraph "data_backup_restore.BackupType"
    end
    subgraph "data_backup_restore.CompressionType"
    end
    subgraph "database_storage.redis_cache"
    end
```

## 类继承图
```mermaid
classDiagram
    class SearchAnswerMerge {
        +merge(self, question, search_results, answer)
    }
    class ResultFilter {
        +filter(self, results)
    }
    class SearchResult {
        +title
        +url
        +snippet
        +source
        +score
        +published_date
        +author
        +language
        +metadata
        +__post_init__(self)
        +to_dict(self): Dict
        +is_valid(self): bool
    }
    class SearchResponse {
        +query
        +results
        +total_results
        +page
        +per_page
        +response_time
        +error
        +cached
        +__post_init__(self)
        +to_dict(self): Dict
        +filter_valid_results(self): Constant(value='SearchResponse', kind=None)
    }
    class BaseSearchEngine {
        +ENGINE_NAME
        +DEFAULT_MAX_RESULTS
        +MAX_RETRY_ATTEMPTS
        +REQUEST_TIMEOUT
        +__init__(self, timeout, max_retries, default_language, enable_cache, rate_limit_delay)
        +_execute_search(self, query, max_results): List
        +_parse_result(self, raw_result): SearchResult
        +_normalize_query(self, query): str
        +_rate_limit(self)
        +_get_from_cache(self, cache_key): Optional
        +_save_to_cache(self, cache_key, response)
        +_create_cache_key(self, query, max_results): str
        +search(self, query, max_results, page, language): SearchResponse
        +_execute_search_with_retry(self, query, max_results, page, language): List
        +get_engine_info(self): Dict
        +clear_cache(self)
        +validate_url(self, url): bool
    }
    ABC <|-- BaseSearchEngine
    class SearchEngineRegistry {
        +_engines
        +_default_engine
        +register(cls, name, engine, set_default)
        +get(cls, name): Optional
        +get_default(cls): Optional
        +list_engines(cls): List
        +unregister(cls, name)
    }
    class SearchEngineFactory {
        +create(engine_type): BaseSearchEngine
    }
    class SearchType {
    }
    Enum <|-- SearchType
    class QueryIntent {
    }
    Enum <|-- QueryIntent
    class QueryDomain {
    }
    Enum <|-- QueryDomain
    class JudgeResult {
        +need_search
        +search_type
        +confidence
        +intent
        +domain
        +keywords
        +reasons
        +suggestions
        +to_dict(self): Dict
    }
    class SearchJudge {
        +_RECENCY_KEYWORDS
        +_NEWS_KEYWORDS
        +_TECHNICAL_KEYWORDS
        +_PRODUCT_KEYWORDS
        +_ACADEMIC_KEYWORDS
        +_LOCAL_KEYWORDS
        +_IMAGE_KEYWORDS
        +_VIDEO_KEYWORDS
        +_TRIVIA_KEYWORDS
        +_NO_SEARCH_PATTERNS
        +__init__(self, custom_rules, enable_fuzzy_match, min_confidence_threshold)
        +add_custom_rules(self, rules): Constant(value=None, kind=None)
        +_normalize_query(self, query): str
        +_check_no_search_patterns(self, query): bool
        +_extract_keywords(self, query): List
        +_calculate_keyword_score(self, query, keyword_set): float
        +_analyze_intent(self, query, scores): QueryIntent
        +_analyze_domain(self, query, scores): QueryDomain
        +_determine_search_type(self, scores): SearchType
        +should_search(self, query): bool
        +get_search_strategy(self, query): SearchType
        +judge_query(self, query): JudgeResult
        +batch_judge(self, queries): List
        +get_statistics(self, results): Dict
    }
    class ContentSummarizer {
        +summarize(self, text)
    }
    class SearXNGCategory {
    }
    Enum <|-- SearXNGCategory
    class SearXNGEngineType {
    }
    Enum <|-- SearXNGEngineType
    class SearXNGConfig {
        +url
        +timeout
        +max_retries
        +retry_delay
        +verify_ssl
        +use_proxy
        +proxy_url
        +categories
        +engines
        +language
        +safe_search
        +time_range
    }
    class SearXNGResult {
        +title
        +url
        +snippet
        +engine
        +engine_type
        +thumbnail
        +img_src
        +video_src
        +iframe_src
        +published_date
        +author
        +latitude
        +longitude
        +template
        +to_search_result(self): SearchResult
    }
    class SearXNGEngine {
        +_SUPPORTED_CATEGORIES
        +_SUPPORTED_ENGINES
        +__init__(self, config, enable_cache, cache_size, enable_rate_limit)
        +_get_session(self): requests.Session
        +_check_rate_limit(self): Constant(value=None, kind=None)
        +_parse_results(self, raw_results, response_metadata): List
        +_validate_categories(self, categories): List
        +_validate_engines(self, engines): List
        +_build_search_params(self, query, categories, engines, language, safe_search, time_range, page): Dict
        +_execute_search_with_retry(self, query, max_results, categories, engines, language, safe_search, time_range, page): tuple
        +search(self, query, max_results, categories, engines, language, safe_search, time_range, page): SearchResponse
        +search_news(self, query, max_results, time_range): SearchResponse
        +search_images(self, query, max_results): SearchResponse
        +search_videos(self, query, max_results): SearchResponse
        +get_instance_info(self): Dict
        +get_available_engines(self): List
        +clear_cache(self): Constant(value=None, kind=None)
        +close(self): Constant(value=None, kind=None)
        +__enter__(self): Constant(value='SearXNGEngine', kind=None)
        +__exit__(self, exc_type, exc_val, exc_tb): Constant(value=None, kind=None)
        +__del__(self): Constant(value=None, kind=None)
    }
    BaseSearchEngine <|-- SearXNGEngine
    class SearchType {
    }
    Enum <|-- SearchType
    class DuckDuckGoConfig {
        +timeout
        +max_retries
        +retry_delay
        +safe_search
        +region
        +source
    }
    class DuckDuckGoResult {
        +title
        +url
        +snippet
        +image_url
        +video_url
        +source
        +published_date
        +to_search_result(self): SearchResult
    }
    class DuckDuckGoAPI {
        +__init__(self, config, enable_cache, cache_size, enable_rate_limit)
        +_check_rate_limit(self): Constant(value=None, kind=None)
        +_parse_results(self, raw_results, search_type): List
        +_execute_search_with_retry(self, query, max_results, search_type): List
        +search(self, query, max_results, search_type): SearchResponse
        +search_text(self, query, max_results): SearchResponse
        +search_news(self, query, max_results): SearchResponse
        +search_images(self, query, max_results): SearchResponse
        +search_videos(self, query, max_results): SearchResponse
        +search_answers(self, query): SearchResponse
        +get_trending_searches(self, region): List
        +get_suggestions(self, query): List
        +clear_cache(self): Constant(value=None, kind=None)
    }
    BaseSearchEngine <|-- DuckDuckGoAPI
    class Message {
        +role
        +content
    }
    BaseModel <|-- Message
    class ChatCompletionRequest {
        +model
        +messages
        +max_tokens
        +temperature
        +top_p
        +top_k
        +repetition_penalty
        +stream
    }
    BaseModel <|-- ChatCompletionRequest
    class CompletionRequest {
        +model
        +prompt
        +max_tokens
        +temperature
        +top_p
        +top_k
        +repetition_penalty
        +stream
    }
    BaseModel <|-- CompletionRequest
    class ChatCompletionResponse {
        +id
        +object
        +created
        +model
        +choices
        +usage
    }
    BaseModel <|-- ChatCompletionResponse
    class CompletionResponse {
        +id
        +object
        +created
        +model
        +choices
        +usage
    }
    BaseModel <|-- CompletionResponse
    class APIError {
        +error
    }
    BaseModel <|-- APIError
    class APIServer {
        +__init__(self, inference_engine, host, port)
        +_setup_cors(self)
        +_setup_routes(self)
        +run(self)
    }
    class RateLimiter {
        +__init__(self, max_per_minute)
        +is_allowed(self, user_id)
    }
    class UserAuth {
        +verify(self, token)
    }
    class StreamingDataset {
        +__init__(self, data_paths, transform, shuffle, buffer_size)
        +_read_file(self, file_path): Iterator
        +_stream_files(self): Iterator
        +__iter__(self)
    }
    IterableDataset <|-- StreamingDataset
    class ConcatDataset {
        +__init__(self, datasets, weights)
        +__iter__(self)
    }
    IterableDataset <|-- ConcatDataset
    class DynamicDataset {
        +__init__(self, base_dir, file_pattern, transform, shuffle)
        +refresh(self)
        +__iter__(self)
    }
    IterableDataset <|-- DynamicDataset
    class TextDataset {
        +__init__(self, text_files, block_size, tokenizer, shuffle)
        +_read_text(self, file_path): str
        +__iter__(self)
    }
    IterableDataset <|-- TextDataset
    class RawDataCleaner {
        +__init__(self)
        +clean_text(self, text): str
        +_decode_html_entities(self, text): str
        +_remove_urls(self, text): str
        +_remove_emails(self, text): str
        +_remove_phone_numbers(self, text): str
        +_remove_html_tags(self, text): str
        +_normalize_whitespace(self, text): str
        +_clean_special_chars(self, text): str
        +_trim_text(self, text): str
        +clean_json(self, json_str): Optional
        +clean_dict(self, data): Dict
        +clean_list(self, data): List
        +remove_empty_fields(self, data): Dict
        +validate_text_length(self, text, min_length, max_length): bool
        +detect_language(self, text): str
        +clean_batch(self, texts): List
        +filter_empty(self, texts): List
        +remove_duplicates(self, texts): List
    }
    class DataAugmenter {
        +__init__(self)
        +random_synonym_replacement(self, text, ratio): str
        +random_deletion(self, text, ratio): str
        +random_insertion(self, text, ratio): str
        +random_swap(self, text, ratio): str
        +back_translation(self, text, translator): str
        +random_case_change(self, text, ratio): str
        +add_noise(self, text, noise_level): str
        +shuffle_sentences(self, text): str
        +augment(self, text, methods): str
        +augment_batch(self, texts, methods): List
        +mixup(self, text1, text2, alpha): str
        +cutmix(self, text1, text2, cut_ratio): str
    }
    class SampleBuilder {
        +__init__(self, tokenizer, max_seq_length)
        +build_pretrain_sample(self, text, add_special_tokens): Dict
        +build_pair_sample(self, text_a, text_b, label): Dict
        +build_qa_sample(self, question, answer, context): Dict
        +build_conversation_sample(self, conversations, system_prompt): Dict
        +build_multimodal_sample(self, text, image_features, image_positions): Dict
        +pad_sample(self, sample, pad_token_id): Dict
        +build_batch(self, samples, padding): Dict
        +truncate_pair(self, tokens_a, tokens_b): Tuple
    }
    class DataFormatConverter {
        +__init__(self)
        +json_to_dict(self, json_str): Optional
        +dict_to_json(self, data, indent): str
        +csv_to_list(self, csv_str, delimiter): List
        +list_to_csv(self, data, delimiter): str
        +yaml_to_dict(self, yaml_str): Optional
        +dict_to_yaml(self, data): str
        +xml_to_dict(self, xml_str): Optional
        +_xml_element_to_dict(self, element): Dict
        +dict_to_xml(self, data, root_tag): str
        +_dict_to_xml_element(self, data, parent)
        +read_file(self, file_path): Optional
        +write_file(self, file_path, content)
        +convert_file(self, input_path, output_path, input_format, output_format)
        +format_to_huggingface(self, data, text_key, label_key): List
        +huggingface_to_format(self, data, text_key, label_key): List
    }
    class FilterResult {
        +passed
        +reason
        +score
    }
    class DataFilterRule {
        +__init__(self)
        +add_rule(self, name, func, weight)
        +filter(self, data): FilterResult
        +batch_filter(self, data_list): List
    }
    class TextFilter {
        +__init__(self)
        +check_length(self, text): FilterResult
        +check_char_diversity(self, text): FilterResult
        +check_repeat_ratio(self, text): FilterResult
        +check_special_chars(self, text): FilterResult
        +check_invalid_patterns(self, text): FilterResult
        +check_low_quality(self, text): FilterResult
        +check_language(self, text, allowed_langs): FilterResult
        +filter(self, text, allowed_langs): FilterResult
    }
    class DocumentFilter {
        +__init__(self)
        +filter_document(self, doc): FilterResult
    }
    class DistributedDataLoader {
        +__init__(self, dataset, batch_size, shuffle, num_workers, pin_memory, drop_last, collate_fn)
        +__iter__(self)
        +__len__(self)
        +set_epoch(self, epoch)
        +get_global_batch_size(self): int
        +get_local_batch_size(self): int
    }
    class DataPrefetcher {
        +__init__(self, loader, device)
        +preload(self)
        +_to_device(self, data)
        +__iter__(self)
        +__next__(self)
    }
    class BatchSampler {
        +__init__(self, dataset, batch_size, shuffle, seed)
        +set_epoch(self, epoch)
        +__iter__(self)
        +__len__(self)
    }
    class DataLoaderWrapper {
        +__init__(self, dataset, batch_size, shuffle, num_workers, pin_memory, drop_last, collate_fn, distributed)
        +__iter__(self)
        +__len__(self)
        +set_epoch(self, epoch)
        +to_prefetcher(self, device)
    }
    class InfiniAttention {
        +__init__(self, d_model, n_heads, d_memory)
        +forward(self, x, kv_cache, cache_pos, use_cache)
    }
    nn.Module <|-- InfiniAttention
    class AttentionConfig {
        +d_model
        +n_heads
        +max_seq_len
        +q_lora_rank
        +kv_lora_rank
        +chunk_size
        +use_flash_attn
        +use_streaming_attn
        +window_size
        +use_sparse_attn
        +sparse_ratio
    }
    class MultiHeadLatentAttention {
        +__init__(self, config)
        +forward(self, x, kv_cache, cache_pos, use_cache): Tuple
    }
    nn.Module <|-- MultiHeadLatentAttention
    class StreamingAttention {
        +__init__(self, config)
        +forward(self, x, kv_cache, cache_pos, use_cache): Tuple
    }
    nn.Module <|-- StreamingAttention
    class SparseAttention {
        +__init__(self, config)
        +forward(self, x, kv_cache, cache_pos, use_cache): Tuple
    }
    nn.Module <|-- SparseAttention
    class FlashAttentionWrapper {
        +__init__(self, config)
        +forward(self, x, mask): torch.Tensor
    }
    nn.Module <|-- FlashAttentionWrapper
    class LongContextAttention {
        +__init__(self, config)
        +forward(self, x, kv_cache, cache_pos, use_cache, attention_mode): Tuple
    }
    nn.Module <|-- LongContextAttention
    class RMSNorm {
        +__init__(self, dim, eps)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- RMSNorm
    class MultiTokenPredictor {
        +__init__(self, d_model, vocab_size, n_predict)
        +forward(self, hidden, labels)
    }
    nn.Module <|-- MultiTokenPredictor
    class HybridBlock {
        +__init__(self, d_model, n_heads, d_state, expand)
        +forward(self, x, kv_cache, cache_pos, use_cache)
    }
    nn.Module <|-- HybridBlock
    class VLContrastiveLoss {
        +__init__(self, temperature)
        +forward(self, image_features, text_features)
    }
    nn.Module <|-- VLContrastiveLoss
    class TotalLoss {
        +__init__(self, weight_mtp, weight_aux, weight_cl)
        +forward(self, logits_main, hidden, input_ids, aux_loss, mtp_losses, image_feat, text_feat)
    }
    nn.Module <|-- TotalLoss
    class SwiGLU {
        +__init__(self, d_model, d_ff)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- SwiGLU
    class V4Config {
        +vocab_size
        +d_model
        +n_heads
        +d_ff
        +n_layers
        +num_experts
        +top_k
        +image_size
        +max_tiles
        +max_seq_len
        +q_lora_rank
        +kv_lora_rank
        +use_streaming_rope
        +use_paged_cache
        +use_flash_attn
        +use_long_context_attn
        +attention_chunk_size
    }
    class DynamicImageProcessor {
        +__init__(self, image_size, patch_size, max_num_tiles)
        +process(self, image)
        +process_tiles(self, image, num_tiles)
    }
    class VisionEncoder {
        +__init__(self, d_model)
        +forward(self, tiles)
    }
    nn.Module <|-- VisionEncoder
    class VisionProjector {
        +__init__(self, d_model, num_visual_tokens)
        +forward(self, x)
    }
    nn.Module <|-- VisionProjector
    class RMSNorm {
        +__init__(self, dim, eps)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- RMSNorm
    class SwiGLU {
        +__init__(self, d_model, d_ff)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- SwiGLU
    class DeepSeekMoE {
        +__init__(self, d_model, d_ff, n_shared, n_routed, top_k)
        +forward(self, x)
    }
    nn.Module <|-- DeepSeekMoE
    class LongContextAttention {
        +__init__(self, config)
        +_init_rope(self, config)
        +forward(self, x, kv_cache, cache_pos, use_cache): Tuple
    }
    nn.Module <|-- LongContextAttention
    class V4TransformerBlock {
        +__init__(self, config)
        +forward(self, x, kv_cache, cache_pos, use_cache)
    }
    nn.Module <|-- V4TransformerBlock
    class DeepSeekV4Multimodal {
        +__init__(self, config)
        +encode_images(self, images)
        +get_multimodal_embeddings(self, input_ids, images, img_pos)
        +forward(self, input_ids, images, img_pos, return_hidden, kv_caches, use_cache)
        +generate(self, input_ids, max_new_tokens, temperature, top_p, kv_caches, use_cache)
    }
    nn.Module <|-- DeepSeekV4Multimodal
    class KVCacheConfig {
        +max_seq_len
        +max_batch_size
        +num_layers
        +num_heads
        +head_dim
        +dtype
        +device
        +use_paged_cache
        +page_size
        +prealloc_size
        +enable_cpu_offload
        +offload_threshold
    }
    class PagedKVCache {
        +__init__(self, config)
        +_allocate_block(self): Tuple
        +get_block(self, block_id): Optional
        +allocate_sequence(self, seq_id, max_len): List
        +update(self, seq_id, position, k, v)
        +get(self, seq_id, start, end): Tuple
        +free_sequence(self, seq_id)
    }
    class StreamingKVCache {
        +__init__(self, config)
        +update(self, k, v, position)
        +_expand_cache(self)
        +get(self, start, end): Tuple
        +reset(self)
    }
    class QuantizedKVCache {
        +__init__(self, config, quant_bits)
        +quantize(self, x): Tuple
        +dequantize(self, x, scale): torch.Tensor
        +update(self, k, v, position)
        +get(self, start, end): Tuple
    }
    class KVCacheManager {
        +__init__(self, config)
        +update(self, layer_idx, position, k, v)
        +get(self, layer_idx, start, end): Tuple
        +get_all_layers(self, start, end): List
        +reset(self)
        +allocate_sequence(self, seq_id, max_len)
        +free_sequence(self, seq_id)
        +get_memory_usage(self): Dict
    }
    class MoEWithBalance {
        +__init__(self, d_model, d_ff, n_shared, n_routed, top_k, capacity_factor)
        +forward(self, x)
    }
    nn.Module <|-- MoEWithBalance
    class DeepSeekMoE {
        +__init__(self, d_model, d_ff, n_shared, n_routed, top_k)
        +forward(self, x)
    }
    nn.Module <|-- DeepSeekMoE
    class UnifiedTokenizer {
        +__init__(self, text_tokenizer, image_encoder, audio_encoder)
        +encode(self, text, images, audio): List
        +decode(self, ids): str
    }
    class RoPEConfig {
        +head_dim
        +max_seq_len
        +base
        +factor
        +beta_fast
        +beta_slow
        +use_ntk_scaling
    }
    class StreamingRoPE {
        +__init__(self, config)
        +_compute_base_freqs(self)
        +_compute_freqs_cis(self, seq_len, start_pos): Tuple
        +_ntk_aware_scaling(self, seq_len): float
        +forward(self, seq_len, start_pos, device): Tuple
        +apply_rotary(q, k, cos, sin): Tuple
    }
    nn.Module <|-- StreamingRoPE
    class YaRNRoPE {
        +__init__(self, config)
        +forward(self, seq_len, start_pos, device): Tuple
    }
    nn.Module <|-- YaRNRoPE
    class LinearRoPE {
        +__init__(self, head_dim, max_seq_len)
        +forward(self, seq_len, start_pos, device): Tuple
        +apply_rotary(q, k, cos, sin): Tuple
    }
    nn.Module <|-- LinearRoPE
    class BenchmarkResult {
        +benchmark_name
        +score
        +total_samples
        +correct_samples
        +details
        +execution_time
        +metadata
        +to_dict(self): Dict
    }
    class BaseBenchmark {
        +__init__(self, name, description)
        +load_dataset(self): List
        +evaluate_sample(self, model, sample): Dict
        +evaluate(self, model, num_samples, show_progress): BenchmarkResult
    }
    ABC <|-- BaseBenchmark
    class MMLUBenchmark {
        +__init__(self, subjects, data_path)
        +load_dataset(self): List
        +_load_subject_samples(self, subject): List
        +evaluate_sample(self, model, sample): Dict
        +_extract_answer(self, response, choices): Optional
    }
    BaseBenchmark <|-- MMLUBenchmark
    class GSM8KBenchmark {
        +__init__(self, data_path)
        +load_dataset(self): List
        +evaluate_sample(self, model, sample): Dict
        +_extract_number(self, text): Optional
        +_check_answer(self, predicted, expected): bool
    }
    BaseBenchmark <|-- GSM8KBenchmark
    class HumanEvalBenchmark {
        +__init__(self, data_path)
        +load_dataset(self): List
        +evaluate_sample(self, model, sample): Dict
        +_execute_code(self, code, test): bool
    }
    BaseBenchmark <|-- HumanEvalBenchmark
    class HellaSwagBenchmark {
        +__init__(self, data_path)
        +load_dataset(self): List
        +evaluate_sample(self, model, sample): Dict
        +_parse_choice(self, response, num_choices): int
    }
    BaseBenchmark <|-- HellaSwagBenchmark
    class TruthfulQABenchmark {
        +__init__(self, data_path)
        +load_dataset(self): List
        +evaluate_sample(self, model, sample): Dict
        +_evaluate_truthfulness(self, response, sample): float
        +_get_truth_keywords(self, text): List
    }
    BaseBenchmark <|-- TruthfulQABenchmark
    class CommonBenchTest {
        +__init__(self, model)
        +set_model(self, model): Constant(value=None, kind=None)
        +run_benchmark(self, benchmark_name, num_samples): BenchmarkResult
        +run_all_benchmarks(self, benchmarks, num_samples): Dict
        +run_mmlu(self, num_samples): BenchmarkResult
        +run_gsm8k(self, num_samples): BenchmarkResult
        +run_humaneval(self, num_samples): BenchmarkResult
        +run_hellaswag(self, num_samples): BenchmarkResult
        +run_truthfulqa(self, num_samples): BenchmarkResult
        +get_summary(self): Dict
        +save_results(self, filepath): Constant(value=None, kind=None)
        +load_results(self, filepath): Constant(value=None, kind=None)
    }
    class MockModel {
        +__init__(self, accuracy)
        +generate(self, prompt): str
        +__call__(self, prompt): str
    }
    class Message {
        +role
        +content
        +timestamp
        +metadata
        +to_dict(self): Dict
    }
    class Conversation {
        +id
        +messages
        +metadata
        +add_message(self, role, content): Message
        +get_last_response(self): Optional
        +get_conversation_text(self): str
    }
    class QualityScore {
        +name
        +score
        +details
        +__repr__(self): str
    }
    class ConversationEvalResult {
        +conversation_id
        +overall_score
        +quality_scores
        +turn_results
        +metadata
        +to_dict(self): Dict
    }
    class TextAnalyzer {
        +__init__(self)
        +tokenize(self, text): List
        +get_word_count(self, text): int
        +get_sentence_count(self, text): int
        +get_avg_word_length(self, text): float
        +get_avg_sentence_length(self, text): float
        +compute_overlap(self, text1, text2): float
        +get_ngrams(self, text, n): List
        +compute_bleu_like(self, reference, candidate, n): float
        +has_question(self, text): bool
        +has_greeting(self, text): bool
        +has_acknowledgment(self, text): bool
    }
    class ResponseQualityEvaluator {
        +__init__(self)
        +evaluate_response(self, query, response, context): Dict
        +_evaluate_relevance(self, query, response, context): float
        +_evaluate_coherence(self, response): float
        +_evaluate_helpfulness(self, query, response): float
        +_evaluate_completeness(self, query, response): float
        +_evaluate_conciseness(self, response): float
        +_evaluate_safety(self, response): float
    }
    class CoherenceEvaluator {
        +__init__(self)
        +evaluate_turn_coherence(self, previous_turns, current_response): float
        +_check_topic_continuity(self, previous_turns, current_response): float
        +evaluate_dialogue_flow(self, conversation): Dict
    }
    class SentimentAnalyzer {
        +__init__(self)
        +analyze_sentiment(self, text): Dict
        +check_sentiment_match(self, user_sentiment, assistant_sentiment): float
    }
    class ChatQualityEval {
        +__init__(self)
        +evaluate_conversation(self, conversation, reference_responses): ConversationEvalResult
        +evaluate_single_response(self, query, response, context, previous_turns): Dict
        +_compute_overall_quality(self, turn_results): float
        +_compute_single_overall(self, quality_scores, coherence_score): float
        +_aggregate_quality_scores(self, turn_results): Dict
        +batch_evaluate(self, conversations, reference_responses): List
        +get_summary(self): Dict
        +save_results(self, filepath): Constant(value=None, kind=None)
    }
    class BasicMetricCalc {
        +__init__(self, n_workers)
        +perplexity(self, loss): float
        +sentence_perplexity(self, log_probs): float
        +calculate_bleu(self, reference, candidate, n_gram, smooth): float
        +_get_ngrams(self, tokens, n): Counter
        +_calculate_brevity_penalty(self, reference, candidate): float
        +calculate_bleu_multi_references(self, references, candidate, n_gram): float
        +calculate_rouge(self, reference, candidate, rouge_type): Dict
        +_rouge_n(self, reference, candidate, n): Dict
        +_rouge_l(self, reference, candidate): Dict
        +_lcs_length(self, seq1, seq2): int
        +calculate_rouge_batch(self, references, candidates, rouge_types): Dict
        +_f_beta_score(self, precision, recall, beta): float
        +calculate_meteor(self, reference, candidate, alpha, beta, gamma): float
        +_meteor_unigram_overlap(self, reference, candidate): float
        +_fragmentation_penalty(self, reference, candidate, beta, gamma): float
        +_count_chunk_matches(self, reference, candidate): int
        +_meteor_align(self, reference, candidate): List
        +calculate_chrf(self, reference, candidate, n_gram_order, beta): float
        +_get_char_ngrams(self, text, n): Counter
        +exact_match(self, reference, candidate): float
        +_normalize_text(self, text): str
        +f1_score(self, reference, candidate, average): float
        +calculate_accuracy(self, predictions, references): float
        +token_accuracy(self, pred_tokens, ref_tokens): float
        +calculate_all_metrics(self, reference, candidate, include_metrics): Dict
        +batch_evaluate(self, references, candidates, metrics): Dict
        +_standard_deviation(self, values): float
    }
    class ReasoningType {
    }
    Enum <|-- ReasoningType
    class ReasoningResult {
        +reasoning_type
        +question
        +model_response
        +ground_truth
        +is_correct
        +reasoning_steps
        +step_scores
        +overall_score
        +execution_time
        +metadata
        +to_dict(self): Dict
    }
    class ReasoningEvalResult {
        +reasoning_type
        +total_questions
        +correct_count
        +accuracy
        +avg_score
        +avg_execution_time
        +results
        +detailed_stats
    }
    class ReasoningDataset {
        +get_dataset(cls, reasoning_type): List
    }
    class ReasoningStepExtractor {
        +__init__(self)
        +extract_steps(self, text): List
        +evaluate_step_quality(self, steps, expected_steps): List
        +_evaluate_single_step(self, step): float
    }
    class ReasoningEval {
        +__init__(self)
        +evaluate(self, model, reasoning_types, num_samples, verbose): Dict
        +_evaluate_type(self, model, reasoning_type, num_samples, verbose): ReasoningEvalResult
        +_evaluate_sample(self, model, sample, reasoning_type): ReasoningResult
        +_build_prompt(self, question, reasoning_type, sample): str
        +_check_answer(self, response, ground_truth, sample): bool
        +_calculate_overall_score(self, response, ground_truth, steps, step_scores, sample): float
        +_compute_detailed_stats(self, results): Dict
        +evaluate_logical_reasoning(self, model): ReasoningEvalResult
        +evaluate_mathematical_reasoning(self, model): ReasoningEvalResult
        +evaluate_commonsense_reasoning(self, model): ReasoningEvalResult
        +evaluate_chain_of_thought(self, model): ReasoningEvalResult
        +evaluate_multi_step(self, model): ReasoningEvalResult
        +get_summary(self, results): Dict
        +save_results(self, results, filepath): Constant(value=None, kind=None)
        +load_results(self, filepath): Dict
    }
    class MockModel {
        +generate(self, prompt): str
        +__call__(self, prompt): str
    }
    class ReportMetadata {
        +model_name
        +model_version
        +evaluation_date
        +evaluator
        +total_samples
        +execution_time
        +additional_info
        +to_dict(self): Dict
    }
    class BenchmarkSection {
        +name
        +score
        +total
        +correct
        +details
        +recommendations
        +to_dict(self): Dict
    }
    class EvaluationReport {
        +metadata
        +summary
        +benchmark_results
        +reasoning_results
        +quality_results
        +safety_results
        +charts
        +recommendations
        +to_dict(self): Dict
    }
    class MarkdownReportGenerator {
        +__init__(self)
        +_get_template(self): str
        +generate(self, report): str
        +_generate_overall_score(self, summary): str
        +_generate_benchmark_section(self, results): str
        +_generate_reasoning_section(self, results): str
        +_generate_quality_section(self, results): str
        +_generate_safety_section(self, results): str
        +_generate_detailed_analysis(self, report): str
        +_generate_recommendations(self, recommendations): str
        +_get_grade(self, score): str
        +_get_grade_description(self, grade): str
    }
    class JSONReportGenerator {
        +__init__(self)
        +generate(self, report): str
        +save(self, report, filepath): Constant(value=None, kind=None)
    }
    class HTMLReportGenerator {
        +__init__(self)
        +_get_template(self): str
        +generate(self, report): str
        +_generate_benchmark_html(self, results): str
        +_generate_reasoning_html(self, results): str
        +_generate_quality_html(self, results): str
        +_generate_safety_html(self, results): str
        +_generate_recommendations_html(self, recommendations): str
        +_get_grade(self, score): str
        +save(self, report, filepath): Constant(value=None, kind=None)
    }
    class ReportGenerator {
        +__init__(self)
        +create_report(self, metadata, benchmark_results, reasoning_results, quality_results, safety_results, recommendations): EvaluationReport
        +_create_summary(self, benchmark_results, reasoning_results, quality_results, safety_results): Dict
        +_generate_recommendations(self, benchmark_results, reasoning_results, quality_results, safety_results): List
        +generate(self, report, format): str
        +save_report(self, report, filepath, format): Constant(value=None, kind=None)
        +generate_comparison_report(self, reports, model_names): str
        +_get_grade(self, score): str
    }
    class SafetyCategory {
        +name
        +description
        +patterns
        +severity
        +keywords
        +__repr__(self): str
    }
    class SafetyTestCase {
        +id
        +prompt
        +category
        +expected_behavior
        +severity
        +metadata
        +__repr__(self): str
    }
    class SafetyEvalResult {
        +test_case
        +model_response
        +is_safe
        +safety_score
        +detected_categories
        +confidence
        +details
        +execution_time
        +to_dict(self): Dict
    }
    class SafetyReport {
        +overall_safety_score
        +category_scores
        +total_tests
        +passed_tests
        +failed_tests
        +results
        +metadata
        +to_dict(self): Dict
    }
    class ToxicityDetector {
        +__init__(self)
        +detect_toxicity(self, text): Dict
        +_calculate_toxicity_score(self, detected): float
    }
    class SensitiveContentDetector {
        +__init__(self)
        +detect_sensitive_content(self, text): Dict
    }
    class MaliciousInstructionDetector {
        +__init__(self)
        +detect_malicious_instructions(self, text): Dict
    }
    class BiasDetector {
        +__init__(self)
        +detect_bias(self, text): Dict
    }
    class PrivacyProtector {
        +__init__(self)
        +check_pii_exposure(self, text): Dict
        +sanitize_pii(self, text): str
    }
    class SafetyEval {
        +__init__(self)
        +_load_test_cases(self): List
        +evaluate_response(self, prompt, response, category): Dict
        +evaluate_test_case(self, model, test_case): SafetyEvalResult
        +run_full_evaluation(self, model, test_cases, verbose): SafetyReport
        +_determine_safety(self, toxicity, sensitive, malicious, bias): bool
        +_calculate_safety_score(self, eval_result): float
        +_check_expected_behavior(self, test_case, eval_result): bool
        +_generate_report(self, results): SafetyReport
        +add_test_case(self, test_case): Constant(value=None, kind=None)
        +load_test_cases_from_file(self, filepath): Constant(value=None, kind=None)
        +save_results(self, filepath): Constant(value=None, kind=None)
        +quick_check(self, text): Dict
    }
    class MockModel {
        +generate(self, prompt): str
        +__call__(self, prompt): str
    }
    class PrivacyProtector {
        +protect(self, text)
    }
    class SensitiveWordDetector {
        +__init__(self, words)
        +detect(self, text)
    }
    class HarmfulClassifier {
        +classify(self, text)
    }
    class IllegalContentJudge {
        +judge(self, text)
    }
    class RefuseReplyStrategy {
        +apply(self, text)
    }
    class WatermarkEmbedder {
        +embed(self, text)
    }
    class SafetyResult {
        +safe
        +reason
        +category
        +confidence
    }
    class SafetyFilter {
        +__init__(self, config_path)
        +_load_patterns(self, config_path): Dict
        +_load_keywords(self, config_path): Dict
        +check_text(self, text): SafetyResult
        +_check_prompt_injection(self, text): bool
        +_check_excessive_length(self, text, max_length): bool
        +check_batch(self, texts): List
        +get_filtered_text(self, text, replacement): str
    }
    class ContentModeration {
        +__init__(self)
        +moderate(self, text): Dict
        +moderate_batch(self, texts): List
        +should_block(self, text): bool
    }
    class GPUInfo {
        +index
        +name
        +utilization_gpu
        +utilization_memory
        +memory_total
        +memory_used
        +memory_free
        +temperature
        +power_draw
        +power_limit
        +fan_speed
        +clock_sm
        +clock_memory
        +clock_video
        +ecc_errors
        +uuid
        +pci_bus_id
        +driver_version
        +compute_mode
        +serial_number
        +status
        +timestamp
        +to_dict(self): Dict
        +memory_used_percent(self): float
        +power_used_percent(self): float
        +is_overheating(self, threshold): bool
        +is_memory_critical(self, threshold): bool
        +is_utilization_high(self, threshold): bool
    }
    class GPUBackend {
        +is_available(self): bool
        +get_gpu_count(self): int
        +get_gpu_info(self, index): Optional
        +get_all_gpu_info(self): List
        +close(self)
    }
    ABC <|-- GPUBackend
    class NVMLBackend {
        +__init__(self)
        +_init_nvml(self)
        +_refresh_handles(self)
        +is_available(self): bool
        +get_gpu_count(self): int
        +_get_memory_info(self, handle): tuple
        +_get_utilization(self, handle): tuple
        +_get_temperature(self, handle): float
        +_get_power(self, handle): tuple
        +_get_fan_speed(self, handle): float
        +_get_clocks(self, handle): tuple
        +_get_device_info(self, handle): Dict
        +get_gpu_info(self, index): Optional
        +get_all_gpu_info(self): List
        +close(self)
    }
    GPUBackend <|-- NVMLBackend
    class SMIBackend {
        +__init__(self)
        +_check_smi(self): bool
        +is_available(self): bool
        +get_gpu_count(self): int
        +_parse_smi_output(self): List
        +get_gpu_info(self, index): Optional
        +get_all_gpu_info(self): List
        +close(self)
    }
    GPUBackend <|-- SMIBackend
    class MockGPUBackend {
        +__init__(self, gpu_count)
        +is_available(self): bool
        +get_gpu_count(self): int
        +get_gpu_info(self, index): Optional
        +get_all_gpu_info(self): List
        +close(self)
    }
    GPUBackend <|-- MockGPUBackend
    class GPUStatusMonitor {
        +_instance
        +__new__(cls)
        +__init__(self, config)
        +_init_backend(self)
        +is_available(self): bool
        +get_gpu_count(self): int
        +get_status(self, index): Union
        +get_utilization(self, index): Union
        +get_memory_usage(self, index): Union
        +get_temperature(self, index): Union
        +get_power_usage(self, index): Union
        +_add_to_history(self, gpu_info)
        +get_history(self, index, limit): List
        +_check_thresholds(self, gpu_info)
        +register_callback(self, callback)
        +unregister_callback(self, callback)
        +start_monitoring(self)
        +stop_monitoring(self)
        +_monitor_loop(self)
        +get_summary(self): Dict
        +close(self)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class LogLevel {
        +get_name(level): str
    }
    class JsonFormatter {
        +__init__(self, include_extra)
        +format(self, record): str
    }
    logging.Formatter <|-- JsonFormatter
    class ColoredFormatter {
        +__init__(self, fmt, datefmt, use_color)
        +format(self, record): str
    }
    logging.Formatter <|-- ColoredFormatter
    class StructuredLogger {
        +__init__(self, logger)
        +set_context(self)
        +clear_context(self)
        +_build_message(self, message): str
        +debug(self, message)
        +info(self, message)
        +warning(self, message)
        +error(self, message)
        +critical(self, message)
    }
    class LogManager {
        +_instance
        +__new__(cls)
        +__init__(self, config)
        +_default_config(self): Dict
        +get_logger(self, name, level): logging.Logger
        +get_structured_logger(self, name): StructuredLogger
        +_add_console_handler(self, logger)
        +_add_file_handlers(self, logger)
        +set_level(self, name, level)
        +add_file_handler(self, name, filepath, level)
        +cleanup_old_logs(self, days)
        +compress_old_logs(self)
    }
    class ContextLogger {
        +set_context(cls)
        +get_context(cls): Dict
        +clear_context(cls)
        +log(cls, level, message)
        +debug(cls, message)
        +info(cls, message)
        +warning(cls, message)
        +error(cls, message)
    }
    class PerformanceLogger {
        +__init__(self, logger)
        +timeit(self, func)
        +get_stats(self, func_name): Dict
    }
    class AlertLevel {
        +__str__(self)
        +color(self): str
    }
    Enum <|-- AlertLevel
    class AlertChannel {
    }
    Enum <|-- AlertChannel
    class Alert {
        +alert_id
        +title
        +message
        +level
        +channel
        +source
        +metric_name
        +metric_value
        +threshold
        +comparison
        +tags
        +metadata
        +recipients
        +cc_recipients
        +occurred_at
        +resolved_at
        +is_resolved
        +is_sent
        +retry_count
        +to_dict(self): Dict
        +summary(self): str
        +severity_emoji(self): str
    }
    class AlertTemplate {
        +name
        +title_template
        +message_template
        +level
        +channel
        +render(self, alert): tuple
    }
    class AlertBackend {
        +send(self, alert): bool
        +validate_config(self): bool
    }
    ABC <|-- AlertBackend
    class EmailBackend {
        +__init__(self, config)
        +validate_config(self): bool
        +send(self, alert): bool
        +_render_plain_text(self, alert): str
        +_render_html(self, alert): str
    }
    AlertBackend <|-- EmailBackend
    class SMSBackend {
        +__init__(self, config)
        +validate_config(self): bool
        +send(self, alert): bool
        +_format_message(self, alert): str
        +_send_aliyun(self, message): bool
        +_send_twilio(self, message): bool
        +_send_yunpian(self, message): bool
    }
    AlertBackend <|-- SMSBackend
    class WebhookBackend {
        +__init__(self, config)
        +validate_config(self): bool
        +send(self, alert): bool
        +_format_payload(self, alert): Dict
    }
    AlertBackend <|-- WebhookBackend
    class DingTalkBackend {
        +__init__(self, config)
        +validate_config(self): bool
        +send(self, alert): bool
        +_format_markdown(self, alert): str
    }
    AlertBackend <|-- DingTalkBackend
    class WeChatBackend {
        +__init__(self, config)
        +validate_config(self): bool
        +send(self, alert): bool
        +_format_markdown(self, alert): str
    }
    AlertBackend <|-- WeChatBackend
    class FeishuBackend {
        +__init__(self, config)
        +validate_config(self): bool
        +send(self, alert): bool
        +_format_card(self, alert): Dict
    }
    AlertBackend <|-- FeishuBackend
    class ConsoleBackend {
        +__init__(self, config)
        +validate_config(self): bool
        +send(self, alert): bool
    }
    AlertBackend <|-- ConsoleBackend
    class AlertSender {
        +_instance
        +__new__(cls)
        +__init__(self, config)
        +_init_backends(self)
        +_init_default_templates(self)
        +_generate_alert_id(self): str
        +_check_rate_limit(self, key): bool
        +send(self, message, level, channel, title, recipients): bool
        +_send_alert(self, alert): bool
        +send_alert(self, alert): bool
        +send_email(self, message, title, recipients, level): bool
        +send_sms(self, message, recipients, level): bool
        +send_webhook(self, message, title, level): bool
        +send_dingtalk(self, message, title, level): bool
        +send_wechat(self, message, title, level): bool
        +send_feishu(self, message, title, level): bool
        +send_console(self, message, title, level): bool
        +_add_to_history(self, alert)
        +get_history(self, limit): List
        +get_recent_alerts(self, level, channel, limit): List
        +register_template(self, template)
        +register_callback(self, callback)
        +unregister_callback(self, callback)
        +set_rate_limit(self, window, max_alerts)
        +get_stats(self): Dict
        +clear_history(self)
        +close(self)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class AlertDecorator {
        +__init__(self, sender, level, channel)
        +__call__(self, func)
    }
    class RequestRecord {
        +endpoint
        +method
        +latency
        +status_code
        +timestamp
        +request_id
        +error
        +metadata
        +to_dict(self): Dict
        +is_success(self): bool
        +is_error(self): bool
    }
    class EndpointStats {
        +endpoint
        +total_requests
        +success_requests
        +error_requests
        +total_latency
        +min_latency
        +max_latency
        +latency_p50
        +latency_p95
        +latency_p99
        +qps
        +error_rate
        +timestamp
        +to_dict(self): Dict
        +avg_latency(self): float
        +success_rate(self): float
    }
    class TimeWindowStats {
        +window_start
        +window_end
        +total_requests
        +success_requests
        +error_requests
        +qps
        +avg_latency
        +min_latency
        +max_latency
        +p50_latency
        +p95_latency
        +p99_latency
        +error_rate
        +to_dict(self): Dict
    }
    class CircularBuffer {
        +__init__(self, max_size)
        +append(self, item)
        +get_all(self): List
        +get_recent(self, n): List
        +clear(self)
        +__len__(self): int
    }
    class PercentileCalculator {
        +__init__(self, values)
        +percentile(self, p): float
    }
    class ServiceQPSDelayMonitor {
        +_instance
        +__new__(cls)
        +__init__(self, config)
        +record(self, endpoint, latency, status_code, method, error, request_id, metadata)
        +record_success(self, endpoint, latency, method, request_id, metadata)
        +record_error(self, endpoint, latency, status_code, error, method, request_id, metadata)
        +get_endpoint_stats(self, endpoint, time_range): EndpointStats
        +get_all_stats(self, time_range): Dict
        +get_qps(self, endpoint, time_range): Union
        +get_latency_percentiles(self, endpoint, percentiles, time_range): Dict
        +get_error_rate(self, endpoint, time_range): Union
        +get_time_window_stats(self, window_seconds, count): List
        +get_recent_errors(self, limit): List
        +get_records(self, endpoint, limit): List
        +_check_alerts(self, endpoint, record)
        +register_alert_callback(self, callback)
        +unregister_alert_callback(self, callback)
        +set_thresholds(self, thresholds)
        +clear_stats(self, endpoint)
        +get_summary(self): Dict
        +export_data(self, filepath, format)
        +start_monitoring(self)
        +stop_monitoring(self)
        +_monitor_loop(self)
        +close(self)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class ContextQPSMonitor {
        +__init__(self, monitor)
        +before_request(self, endpoint, method)
        +after_request(self, status_code)
        +on_error(self, error)
    }
    class ErrorLevel {
        +__lt__(self, other)
    }
    Enum <|-- ErrorLevel
    class ErrorCategory {
        +from_exception(cls, exc): Constant(value='ErrorCategory', kind=None)
    }
    Enum <|-- ErrorCategory
    class ErrorInfo {
        +error_id
        +level
        +category
        +message
        +exception_type
        +exception_message
        +traceback
        +stack_frames
        +module
        +function
        +filename
        +line_number
        +request_id
        +user_id
        +session_id
        +context
        +metadata
        +occurred_at
        +first_occurrence
        +last_occurrence
        +count
        +is_resolved
        +resolved_at
        +tags
        +assignee
        +notes
        +to_dict(self): Dict
        +fingerprint(self): str
        +summary(self): str
    }
    class ErrorStats {
        +total_errors
        +errors_by_level
        +errors_by_category
        +top_errors
        +error_trend
        +timestamp
    }
    class ErrorBackend {
        +save(self, error): bool
        +get(self, error_id): Optional
        +query(self, filters): List
        +update(self, error_id, updates): bool
        +delete(self, error_id): bool
    }
    ABC <|-- ErrorBackend
    class FileBackend {
        +__init__(self, log_dir)
        +_load_index(self): Dict
        +_save_index(self)
        +save(self, error): bool
        +get(self, error_id): Optional
        +query(self, filters): List
        +_matches_filter(self, meta, filters): bool
        +update(self, error_id, updates): bool
        +delete(self, error_id): bool
        +_dict_to_error(self, data): ErrorInfo
    }
    ErrorBackend <|-- FileBackend
    class MemoryBackend {
        +__init__(self)
        +save(self, error): bool
        +get(self, error_id): Optional
        +query(self, filters): List
        +_matches_filter(self, error, filters): bool
        +update(self, error_id, updates): bool
        +delete(self, error_id): bool
        +get_all(self): List
    }
    ErrorBackend <|-- MemoryBackend
    class ErrorCatcher {
        +_instance
        +__new__(cls)
        +__init__(self, config)
        +_init_backend(self)
        +_install_excepthook(self)
        +_excepthook(self, exc_type, exc_value, exc_traceback)
        +_create_error_from_exception(self, exc_type, exc_value, exc_traceback): ErrorInfo
        +_generate_error_id(self): str
        +_check_rate_limit(self, fingerprint): bool
        +log_error(self, error, message, level, context): Optional
        +_update_or_create(self, error_info)
        +catch(self, func, level, reraise, context_func)
        +save(self, error): bool
        +get(self, error_id): Optional
        +query(self, filters, limit, offset): List
        +get_by_fingerprint(self, fingerprint): List
        +update(self, error_id, updates): bool
        +resolve(self, error_id): bool
        +delete(self, error_id): bool
        +register_callback(self, callback)
        +unregister_callback(self, callback)
        +get_stats(self, time_range): ErrorStats
        +_calculate_trend(self, errors): List
        +get_unresolved(self, limit): List
        +get_recent(self, limit): List
        +export_errors(self, filepath, filters)
        +clear_old_errors(self, days): int
        +close(self)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class ErrorContext {
        +__init__(self, error_catcher, context)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
        +error(self): Optional
    }
    class CPUInfo {
        +user
        +system
        +idle
        +iowait
        +irq
        +softirq
        +steal
        +guest
        +guest_nice
        +count
        +count_logical
        +percent
        +timestamp
        +to_dict(self): Dict
        +used_percent(self): float
        +is_busy(self): bool
    }
    class MemoryInfo {
        +total
        +available
        +used
        +free
        +buffers
        +cached
        +percent
        +active
        +inactive
        +buffers_cached
        +swap_total
        +swap_used
        +swap_free
        +swap_percent
        +timestamp
        +to_dict(self): Dict
        +used_percent(self): float
        +available_percent(self): float
        +is_low_memory(self, threshold): bool
        +is_swap_used(self): bool
    }
    class DiskInfo {
        +mountpoint
        +device
        +fstype
        +total
        +used
        +free
        +percent
        +inodes_total
        +inodes_used
        +inodes_free
        +inodes_percent
        +timestamp
        +to_dict(self): Dict
        +used_percent(self): float
        +is_disk_full(self, threshold): bool
    }
    class NetworkInfo {
        +interface
        +bytes_sent
        +bytes_recv
        +packets_sent
        +packets_recv
        +errin
        +errout
        +dropin
        +dropout
        +timestamp
        +to_dict(self): Dict
    }
    class LoadAverage {
        +load1
        +load5
        +load15
        +running_threads
        +total_threads
        +last_pid
        +timestamp
        +to_dict(self): Dict
    }
    class SystemMonitorBackend {
        +get_cpu_info(self): CPUInfo
        +get_memory_info(self): MemoryInfo
        +get_disk_info(self, mountpoint): DiskInfo
        +get_network_info(self, interface): List
        +get_load_average(self): LoadAverage
        +get_uptime(self): float
        +get_process_count(self): int
    }
    ABC <|-- SystemMonitorBackend
    class PSUtilBackend {
        +__init__(self)
        +_init_psutil(self)
        +is_available(self): bool
        +get_cpu_info(self): CPUInfo
        +get_memory_info(self): MemoryInfo
        +get_disk_info(self, mountpoint): DiskInfo
        +get_network_info(self, interface): List
        +get_load_average(self): LoadAverage
        +get_uptime(self): float
        +get_process_count(self): int
    }
    SystemMonitorBackend <|-- PSUtilBackend
    class CommandBackend {
        +__init__(self)
        +is_available(self): bool
        +_run_command(self, cmd): str
        +get_cpu_info(self): CPUInfo
        +get_memory_info(self): MemoryInfo
        +get_disk_info(self, mountpoint): DiskInfo
        +get_network_info(self, interface): List
        +get_load_average(self): LoadAverage
        +get_uptime(self): float
        +get_process_count(self): int
    }
    SystemMonitorBackend <|-- CommandBackend
    class MockBackend {
        +__init__(self)
        +is_available(self): bool
        +get_cpu_info(self): CPUInfo
        +get_memory_info(self): MemoryInfo
        +get_disk_info(self, mountpoint): DiskInfo
        +get_network_info(self, interface): List
        +get_load_average(self): LoadAverage
        +get_uptime(self): float
        +get_process_count(self): int
    }
    SystemMonitorBackend <|-- MockBackend
    class CPUMemMonitor {
        +_instance
        +__new__(cls)
        +__init__(self, config)
        +_init_backend(self)
        +is_available(self): bool
        +get_usage(self): Dict
        +get_cpu(self): CPUInfo
        +get_memory(self): MemoryInfo
        +get_disk(self, mountpoint): DiskInfo
        +get_network(self, interface): List
        +get_load(self): LoadAverage
        +get_uptime(self): float
        +get_process_count(self): int
        +_add_to_history(self, category, info)
        +get_cpu_history(self, limit): List
        +get_memory_history(self, limit): List
        +_check_thresholds(self, category, info)
        +_trigger_callbacks(self, category, message)
        +register_callback(self, category, callback)
        +unregister_callback(self, category, callback)
        +start_monitoring(self)
        +stop_monitoring(self)
        +_monitor_loop(self)
        +get_summary(self): Dict
        +get_all_info(self): Dict
        +close(self)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class BatchQueueInference {
        +__init__(self, model)
        +process(self)
    }
    class BaseInference {
        +__init__(self, model)
        +generate(self, prompt)
    }
    class QuantEngine {
        +quantize(model, bits)
    }
    class HybridEngine {
        +__init__(self, mamba_model, attn_model)
        +generate(self, prompt)
    }
    class GenerateConfig {
        +max_new_tokens
        +temperature
        +top_p
        +top_k
        +repetition_penalty
        +stop_tokens
        +eos_token_id
    }
    class InferenceEngine {
        +__init__(self, model, tokenizer, device)
        +generate(self, input_ids, config): List
        +batch_generate(self, input_ids_list, config): List
        +chat(self, messages, config): str
        +get_performance_stats(self, input_text, config): Dict
    }
    class TensorRTInferenceEngine {
        +__init__(self, model_path, tokenizer)
        +_init_tensorrt(self, model_path)
        +generate(self, input_ids, config): List
    }
    InferenceEngine <|-- TensorRTInferenceEngine
    class ONNXExporter {
        +export(model, path)
    }
    class StreamResponse {
        +stream(self, generator)
    }
    class TensorRTOptimizer {
        +optimize(model)
    }
    class VLLMInference {
        +__init__(self, model_path)
        +generate(self, prompts)
    }
    class GPUMemoryTuner {
        +tuning()
    }
    class OmegaSampler {
        +__init__(self, d_model, n_heads, max_seq_len)
        +forward(self, x, past_kv, use_cache): Tuple
        +_compute_dynamic_mask(self, alpha, beta, query_len, key_len): torch.Tensor
    }
    nn.Module <|-- OmegaSampler
    class OmegaAligner {
        +__init__(self, d_model, n_heads, max_seq_len)
        +forward(self, x, past_kv, use_cache): Tuple
    }
    nn.Module <|-- OmegaAligner
    class AdaptiveOmegaLoss {
        +__init__(self, alpha_weight, beta_weight)
        +forward(self, logits, labels, alpha, beta): torch.Tensor
    }
    nn.Module <|-- AdaptiveOmegaLoss
    class SelfRewarding {
        +__init__(self, model)
        +generate_with_score(self, prompt)
    }
    class ConstitutionGuard {
        +__init__(self, rules)
        +check(self, text)
    }
    class DeepSeekGradioChat {
        +__init__(self)
        +get_custom_css(self): str
        +_simulate_streaming_response(self, message): Generator
        +_call_api_streaming(self, messages): Generator
        +chat_streaming(self, message, history, images, temperature, max_tokens, top_p): Generator
        +_build_messages(self, history, images): List
        +_process_image(self, image_data): str
        +get_statistics(self, history): Dict
        +export_conversation(self, history): str
        +clear_history(self): List
        +update_system_prompt(self, new_prompt)
    }
    class StreamlitChatUI {
        +__init__(self)
        +_init_page_config(self)
        +_init_session_state(self)
        +render_sidebar(self)
        +_simulate_streaming_response(self, message): Generator
        +_call_deepseek_api(self, messages, images): Generator
        +_process_image_for_api(self, image_file): Optional
        +_format_message_for_api(self, role, content, images): Dict
        +_export_conversation(self)
        +render_chat_messages(self)
        +_get_avatar(self, role): str
        +_regenerate_response(self, message_idx)
        +_copy_to_clipboard(self, text)
        +render_input_area(self)
        +_handle_user_input(self, user_input)
        +_generate_assistant_response(self, user_input, images)
        +_build_conversation_history(self): List
        +render_header(self)
        +render_statistics(self)
        +run(self)
    }
    class SampleCollectionConfig {
        +max_response_length
        +max_prompt_length
        +temperature
        +top_p
        +top_k
        +num_samples_per_prompt
        +min_response_length
        +max_response_length_sample
        +batch_size
        +num_workers
        +save_interval
        +output_dir
        +sample_format
    }
    class ResponseGenerator {
        +__init__(self, model, tokenizer, device, max_length, temperature, top_p, top_k)
        +generate(self, prompt, max_length, temperature, top_p, top_k, do_sample, num_return_sequences, repetition_penalty): List
        +generate_batch(self, prompts, max_length, temperature, num_return_sequences): List
    }
    class PreferenceLabeler {
        +__init__(self, reward_model, tokenizer, device)
        +label_with_reward_model(self, prompt, responses): Tuple
        +label_with_rules(self, prompt, responses): Tuple
        +label_random(self, prompt, responses): Tuple
    }
    class SampleCollector {
        +__init__(self, generator, labeler, config)
        +collect_single(self, prompt, label_method): Optional
        +collect_batch(self, prompts, label_method, show_progress): List
        +save_samples(self, filename)
        +get_stats(self): Dict
    }
    class OnlineSampleCollector {
        +__init__(self, generator, labeler, config, environment, max_buffer_size)
        +collect_from_environment(self, num_episodes, label_method): List
        +_collect_episode(self, episode_idx, label_method): List
        +_state_to_prompt(self, state): str
        +_select_action(self, response): str
    }
    SampleCollector <|-- OnlineSampleCollector
    class SampleFilter {
        +__init__(self, min_length, max_length, max_repetition_ratio, require_diversity, min_length_diff)
        +filter(self, sample): bool
        +_check_length(self, chosen, rejected): bool
        +_check_repetition(self, chosen, rejected): bool
        +_has_excessive_repetition(self, text, n): bool
        +_check_diversity(self, chosen, rejected): bool
    }
    class CollectedSampleDataset {
        +__init__(self, data_path, filter)
        +_load_samples(self, data_path): List
        +__len__(self): int
        +__getitem__(self, idx): Dict
        +get_statistics(self): Dict
    }
    Dataset <|-- CollectedSampleDataset
    class IPOConfig {
        +tau
        +beta
        +learning_rate
        +weight_decay
        +warmup_steps
        +max_steps
        +gradient_accumulation_steps
        +max_seq_length
        +max_prompt_length
        +log_interval
        +eval_interval
        +save_interval
        +output_dir
        +use_amp
        +gradient_clip_norm
        +label_smoothing
        +regularization_weight
        +warmup_ratio
    }
    class IPODataset {
        +__init__(self, data_path, tokenizer, max_length, max_prompt_length)
        +_load_data(self, data_path): List
        +__len__(self): int
        +_encode_with_prompt(self, prompt, response): Dict
        +__getitem__(self, idx): Dict
    }
    Dataset <|-- IPODataset
    class IPOLoss {
        +__init__(self, tau, beta, label_smoothing, regularization_weight, reduction)
        +forward(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps): Tuple
    }
    nn.Module <|-- IPOLoss
    class SimpleIPOLoss {
        +__init__(self, tau, reduction)
        +forward(self, policy_chosen_logps, policy_rejected_logps): Tuple
    }
    nn.Module <|-- SimpleIPOLoss
    class RegularizedIPOLoss {
        +__init__(self, tau, beta, kl_weight, reduction)
        +forward(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps): Tuple
    }
    nn.Module <|-- RegularizedIPOLoss
    class IPOTrainer {
        +__init__(self, policy_model, reference_model, tokenizer, config, train_dataset, eval_dataset, device)
        +_create_dataloaders(self)
        +_compute_log_probabilities(self, model, input_ids, prompt_lens): torch.Tensor
        +_train_step(self, batch): Dict
        +_eval_step(self, batch): Dict
        +_save_checkpoint(self, step, metrics, is_best)
        +_load_checkpoint(self, checkpoint_path)
        +evaluate(self): Dict
        +train(self)
    }
    class SimpleIPOTrainer {
        +__init__(self)
        +_train_step(self, batch): Dict
    }
    IPOTrainer <|-- SimpleIPOTrainer
    class RegularizedIPOTrainer {
        +__init__(self)
    }
    IPOTrainer <|-- RegularizedIPOTrainer
    class RewardModelConfig {
        +hidden_size
        +num_layers
        +learning_rate
        +weight_decay
        +warmup_steps
        +max_steps
        +gradient_accumulation_steps
        +max_seq_length
        +log_interval
        +eval_interval
        +save_interval
        +output_dir
        +eval_ratio
        +use_amp
        +warmup_ratio
    }
    class RewardModel {
        +__init__(self, base_model, hidden_size, reward_head_hidden_size, use_mean_pooling)
        +_init_weights(self)
        +forward(self, input_ids, attention_mask, return_hidden): Tuple
    }
    nn.Module <|-- RewardModel
    class RewardModelWithScalarHead {
        +__init__(self, base_model, bias_vector)
        +forward(self, input_ids, attention_mask, position_ids): Tuple
    }
    nn.Module <|-- RewardModelWithScalarHead
    class PreferencePairDataset {
        +__init__(self, data_path, tokenizer, max_length, split)
        +_load_data(self, data_path): List
        +__len__(self): int
        +_encode_with_prompt(self, prompt, response, max_length): Dict
        +__getitem__(self, idx): Dict
    }
    Dataset <|-- PreferencePairDataset
    class BradleyTerryLoss {
        +__init__(self, reduction)
        +forward(self, chosen_rewards, rejected_rewards, label_smoothing): torch.Tensor
    }
    nn.Module <|-- BradleyTerryLoss
    class RewardModelLoss {
        +__init__(self, weight_bt, weight_reg, reg_mean, reg_std, label_smoothing)
        +forward(self, chosen_rewards, rejected_rewards, return_metrics): Tuple
    }
    nn.Module <|-- RewardModelLoss
    class RewardModelTrainer {
        +__init__(self, model, tokenizer, config, train_dataset, eval_dataset, device)
        +_create_dataloaders(self)
        +_train_step(self, batch): Dict
        +_eval_step(self, batch): Dict
        +_save_checkpoint(self, step, metrics, is_best)
        +_load_checkpoint(self, checkpoint_path)
        +evaluate(self): Dict
        +train(self)
    }
    class RewardModelEvaluator {
        +__init__(self, model, tokenizer, device)
        +score_single(self, prompt, response, max_length): float
        +score_pair(self, prompt, chosen, rejected, max_length): Dict
        +evaluate_dataset(self, dataset, batch_size): Dict
    }
    class PPOConfig {
        +learning_rate
        +clip_epsilon
        +value_loss_coef
        +entropy_coef
        +max_grad_norm
        +ppo_epochs
        +mini_batch_size
        +gamma
        +lam
        +max_steps
        +max_response_length
        +max_prompt_length
        +log_interval
        +eval_interval
        +save_interval
        +output_dir
        +use_amp
        +kl_penalty
        +reward_scale
        +warmup_ratio
        +use_value_network
        +use_reference
        +reference_kl_weight
    }
    class RolloutData {
        +prompts
        +prompt_ids
        +prompt_mask
        +generated_ids
        +generated_mask
        +ref_log_probs
        +values
        +rewards
        +advantages
        +returns
    }
    class ValueNetwork {
        +__init__(self, base_model, hidden_size)
        +_init_weights(self)
        +forward(self, input_ids, attention_mask): torch.Tensor
    }
    nn.Module <|-- ValueNetwork
    class ActorCriticModel {
        +__init__(self, base_model, hidden_size)
        +_init_weights(self)
        +forward(self, input_ids, attention_mask): Tuple
    }
    nn.Module <|-- ActorCriticModel
    class PPODataset {
        +__init__(self, data_path, tokenizer, max_prompt_length)
        +_load_prompts(self, data_path): List
        +__len__(self): int
        +__getitem__(self, idx): Dict
    }
    Dataset <|-- PPODataset
    class PPOLoss {
        +__init__(self, clip_epsilon, value_loss_coef, entropy_coef)
        +forward(self, log_probs, old_log_probs, values, old_values, returns, advantages, mask): Tuple
    }
    nn.Module <|-- PPOLoss
    class PPOTrainer {
        +__init__(self, policy_model, reference_model, value_model, tokenizer, reward_model, config, train_dataset, device)
        +_create_dataloader(self)
        +_generate_rollout(self, prompts, max_length, temperature, top_p): Tuple
        +_compute_rewards(self, prompts, responses): torch.Tensor
        +_compute_reference_log_probs(self, input_ids): torch.Tensor
        +_estimate_values(self, input_ids): torch.Tensor
        +_compute_gae(self, rewards, values, gamma, lam): Tuple
        +_compute_kl_penalty(self, policy_log_probs, ref_log_probs): torch.Tensor
        +_ppo_train_step(self, input_ids, old_log_probs, ref_log_probs, rewards, mask): Dict
        +_save_checkpoint(self, step, metrics)
        +_load_checkpoint(self, checkpoint_path)
        +train(self)
    }
    class RewardShapingWrapper {
        +__init__(self, base_reward_fn, length_weight, format_weight, repetition_weight, kl_weight)
        +__call__(self, prompt, response, ref_response): float
        +_compute_length_reward(self, response): float
        +_compute_format_reward(self, response): float
        +_compute_repetition_reward(self, response): float
        +_compute_kl_divergence(self, response, ref_response): float
    }
    class SFTDataFormatterConfig {
        +max_length
        +max_prompt_length
        +max_response_length
        +add_special_tokens
        +use_chat_template
        +system_message
        +separator
        +response_prefix
        +response_suffix
        +truncate_mode
        +ignore_invalid
        +return_metadata
    }
    class ConversationTemplate {
        +__init__(self, system_template, user_template, assistant_template, separator)
        +format(self, messages): str
        +get_roles(self): List
    }
    class ChatMLTemplate {
        +__init__(self)
    }
    ConversationTemplate <|-- ChatMLTemplate
    class Llama2Template {
        +__init__(self)
    }
    ConversationTemplate <|-- Llama2Template
    class QwenTemplate {
        +__init__(self)
    }
    ConversationTemplate <|-- QwenTemplate
    class SFTDataFormatter {
        +__init__(self, tokenizer, config, template_name)
        +format_conversation(self, messages, add_bos, add_eos): str
        +format_prompt_response(self, prompt, response, add_bos, add_eos): str
        +encode_conversation(self, messages, return_labels, mask_response): Dict
        +encode_prompt_response(self, prompt, response, return_labels, mask_prompt): Dict
        +_estimate_prompt_len(self, messages): int
        +truncate_conversation(self, messages, max_length): List
        +_truncate_head(self, messages, max_length): List
        +_truncate_tail(self, messages, max_length): List
        +_truncate_middle(self, messages, max_length): List
    }
    class SFTDataset {
        +__init__(self, data_path, tokenizer, formatter, config, split)
        +_load_data(self): List
        +_is_valid(self, example): bool
        +__len__(self): int
        +__getitem__(self, idx): Dict
    }
    Dataset <|-- SFTDataset
    class DataValidator {
        +__init__(self, min_length, max_length, min_response_length, check_repetition, repetition_threshold)
        +validate(self, example): Tuple
        +_validate_messages(self, messages): Tuple
        +_validate_prompt_response(self, prompt, response): Tuple
        +_has_excessive_repetition(self, text, n): bool
    }
    class DataAugmenter {
        +__init__(self, augment_system, system_messages, shuffle_responses, add_noise, noise_prob)
        +augment(self, example): List
        +_add_system_message(self, example): Optional
        +_shuffle_conversation(self, example): Optional
        +_add_noise(self, example): Dict
        +_get_typo(self, word): str
    }
    class DPOConfig {
        +beta
        +learning_rate
        +weight_decay
        +warmup_steps
        +max_steps
        +gradient_accumulation_steps
        +max_seq_length
        +max_prompt_length
        +max_response_length
        +log_interval
        +eval_interval
        +save_interval
        +output_dir
        +use_amp
        +gradient_clip_norm
        +reference_free
        +label_smoothing
        +sync_ref_params
        +ref_batch_size
        +warmup_ratio
    }
    class DPODataset {
        +__init__(self, data_path, tokenizer, max_length, max_prompt_length)
        +_load_data(self, data_path): List
        +__len__(self): int
        +_encode_with_prompt(self, prompt, response): Dict
        +__getitem__(self, idx): Dict
    }
    Dataset <|-- DPODataset
    class DPOLoss {
        +__init__(self, beta, label_smoothing, reduction)
        +forward(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps): Tuple
    }
    nn.Module <|-- DPOLoss
    class AdaptiveDPO_loss {
        +__init__(self, beta, margin_threshold)
        +forward(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps): Tuple
    }
    nn.Module <|-- AdaptiveDPO_loss
    class DPOTrainer {
        +__init__(self, policy_model, reference_model, tokenizer, config, train_dataset, eval_dataset, device)
        +_create_dataloaders(self)
        +_compute_log_probabilities(self, model, input_ids, prompt_lens): torch.Tensor
        +_compute_log_probabilities_v2(self, model, input_ids, prompt_lens): torch.Tensor
        +_train_step(self, batch): Dict
        +_eval_step(self, batch): Dict
        +_save_checkpoint(self, step, metrics, is_best)
        +_load_checkpoint(self, checkpoint_path)
        +evaluate(self): Dict
        +train(self)
    }
    class DPOwithRewardModel {
        +__init__(self, policy_model, reference_model, reward_model, tokenizer, config, train_dataset, eval_dataset, device)
        +_compute_reward_scores(self, input_ids, prompt_lens): torch.Tensor
    }
    DPOTrainer <|-- DPOwithRewardModel
    class UnlikelihoodDPO_loss {
        +__init__(self, beta, ul_weight, ul_margin)
        +forward(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, rejected_logits, negative_token_ids): Tuple
    }
    nn.Module <|-- UnlikelihoodDPO_loss
    class SFTConfig {
        +learning_rate
        +weight_decay
        +warmup_steps
        +max_steps
        +gradient_accumulation_steps
        +max_seq_length
        +log_interval
        +save_interval
        +output_dir
    }
    class SFTDataset {
        +__init__(self, data_path, tokenizer, max_length)
        +_load_data(self, data_path): List
        +__len__(self): int
        +__getitem__(self, idx): Dict
        +_format_conversation(self, messages): str
    }
    Dataset <|-- SFTDataset
    class SFTLoss {
        +__init__(self, ignore_index)
        +forward(self, logits, labels): torch.Tensor
    }
    nn.Module <|-- SFTLoss
    class SFTTrainer {
        +__init__(self, model, tokenizer, config)
        +_compute_loss(self, batch): Tuple
        +_train_step(self, batch): Dict
        +_save_checkpoint(self, step, metrics)
        +train(self, train_dataset, eval_dataset)
        +evaluate(self, eval_dataset): Dict
    }
    class ScoringConfig {
        +max_length
        +batch_size
        +use_amp
        +device
        +scoring_strategy
        +temperature
        +normalize_rewards
        +output_dir
    }
    class RewardScorer {
        +__init__(self, model, tokenizer, config, device)
        +_encode_sequence(self, prompt, response, max_length): Dict
        +_score_sequence(self, input_ids, attention_mask, prompt_len): Tuple
        +score(self, prompt, response, return_per_token): Dict
        +score_pair(self, prompt, chosen, rejected): Dict
        +score_batch(self, items, show_progress): List
        +score_pairs_batch(self, pairs, show_progress): List
        +rank_responses(self, prompt, responses, top_k): List
    }
    class EnsembleRewardScorer {
        +__init__(self, models, tokenizer, weights, config)
        +score(self, prompt, response): Dict
        +score_pair(self, prompt, chosen, rejected): Dict
    }
    class RewardCalibrator {
        +__init__(self, reference_scores, target_mean, target_std)
        +_fit(self, scores)
        +calibrate(self, score): float
        +calibrate_batch(self, scores): List
    }
    class RewardHistogramAnalyzer {
        +__init__(self)
        +add_scores(self, scores, metadata)
        +add_pair_result(self, result)
        +analyze(self): Dict
        +_detect_outliers(self, scores, threshold): np.ndarray
        +save_report(self, output_path)
    }
    class BatchRewardScorer {
        +__init__(self, model, tokenizer, config, device)
        +_collate_batch(self, items, pad_token_id): Dict
        +score_batch(self, items, show_progress): List
        +score_pairs_batch(self, pairs, show_progress): List
    }
    class MemoryRetriever {
        +retrieve(self, query, top_k)
    }
    class LongMemoryDB {
        +__init__(self)
        +store(self, key, value)
        +retrieve(self, key)
    }
    class SessionManager {
        +__init__(self)
        +get_session(self, session_id)
    }
    class MemoryItem {
        +__init__(self, content, timestamp, metadata)
        +_generate_id(self): str
        +to_dict(self): Dict
        +from_dict(cls, data): Constant(value='MemoryItem', kind=None)
    }
    class ConversationMemory {
        +__init__(self, max_items)
        +add_item(self, content, metadata)
        +get_items(self, limit): List
        +search(self, query, top_k): List
        +_compute_similarity(self, query, content): float
        +clear(self)
        +to_dict(self): Dict
        +from_dict(cls, data): Constant(value='ConversationMemory', kind=None)
    }
    class LongTermMemory {
        +__init__(self, storage_path)
        +get_conversation_memory(self, conversation_id): ConversationMemory
        +save_memory(self, conversation_id)
        +load_memory(self, conversation_id): Optional
        +delete_memory(self, conversation_id)
        +get_all_conversations(self): List
    }
    class MemoryAugmenter {
        +__init__(self, long_term_memory)
        +augment_prompt(self, conversation_id, prompt, max_context_items): str
        +update_memory(self, conversation_id, message)
    }
    class ShortMemory {
        +__init__(self, max_len)
        +add(self, user, assistant)
        +get_context(self)
    }
    class MemoryCompressor {
        +compress(self, memories, max_tokens)
    }
    class TaskStatus {
    }
    Enum <|-- TaskStatus
    class TaskResult {
        +task_id
        +status
        +result
        +error
        +start_time
        +end_time
        +metadata
        +elapsed_time(self): Optional
        +is_success(self): bool
        +is_failed(self): bool
    }
    class PoolConfig {
        +max_workers
        +thread_name_prefix
        +queue_size
        +timeout
        +return_exceptions
    }
    class BaseExecutor {
        +__init__(self, config)
        +submit(self, func): str
        +map(self, func): List
        +shutdown(self, wait): Constant(value=None, kind=None)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    ABC <|-- BaseExecutor
    class ThreadPoolExecutorHelper {
        +__init__(self, config)
        +submit(self, func): str
        +submit_with_callback(self, func, callback, error_callback): str
        +map(self, func): List
        +get_result(self, task_id, timeout): Any
        +get_task_status(self, task_id): TaskStatus
        +get_task_result_info(self, task_id): Optional
        +cancel(self, task_id): bool
        +wait_all(self, timeout): List
    }
    BaseExecutor <|-- ThreadPoolExecutorHelper
    class ProcessPoolExecutorHelper {
        +__init__(self, config)
        +submit(self, func): str
        +map(self, func): List
        +get_result(self, task_id, timeout): Any
        +get_task_status(self, task_id): TaskStatus
        +shutdown(self, wait): Constant(value=None, kind=None)
    }
    BaseExecutor <|-- ProcessPoolExecutorHelper
    class TaskQueue {
        +__init__(self, maxsize)
        +put(self, task_id, task_data, block, timeout): Constant(value=None, kind=None)
        +get(self, block, timeout): Tuple
        +put_result(self, task_id, result): Constant(value=None, kind=None)
        +get_result(self, task_id, timeout): Any
        +task_done(self): Constant(value=None, kind=None)
        +join(self): Constant(value=None, kind=None)
        +empty(self): bool
        +size(self): int
        +clear(self): Constant(value=None, kind=None)
        +stop(self): Constant(value=None, kind=None)
        +is_stopped(self): bool
    }
    class WorkerPool {
        +__init__(self, num_workers, queue, worker_func)
        +start(self): Constant(value=None, kind=None)
        +_worker_loop(self): Constant(value=None, kind=None)
        +stop(self, wait): Constant(value=None, kind=None)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class ProgressTracker {
        +__init__(self, total, desc)
        +update(self, n): Constant(value=None, kind=None)
        +set(self, value): Constant(value=None, kind=None)
        +reset(self): Constant(value=None, kind=None)
        +progress(self): float
        +elapsed_time(self): float
        +estimated_remaining(self): Optional
        +add_callback(self, callback): Constant(value=None, kind=None)
        +_notify(self): Constant(value=None, kind=None)
        +__str__(self): str
    }
    class AsyncBatchExecutor {
        +__init__(self, max_workers, use_process, queue_size)
        +submit(self, func): str
        +poll(self): Dict
        +wait(self, timeout): Dict
        +get_result(self, task_id, timeout): Any
        +shutdown(self, wait): Constant(value=None, kind=None)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class HashAlgorithm {
    }
    Enum <|-- HashAlgorithm
    class CipherMode {
    }
    Enum <|-- CipherMode
    class KeyPair {
        +public_key
        +private_key
        +public_key_pem
        +private_key_pem
    }
    class EncryptedData {
        +ciphertext
        +iv
        +tag
        +salt
        +algorithm
    }
    class EncryptDecrypt {
        +is_available(): bool
        +ensure_crypto()
        +generate_key(length, hex): Union
        +generate_salt(length): bytes
        +generate_nonce(length): bytes
        +hash_data(data, algorithm): str
        +hash_file(filepath, algorithm, chunk_size): str
        +hmac_sign(data, key, algorithm): str
        +hmac_verify(data, key, signature, algorithm): bool
        +aes_encrypt(plaintext, key, mode, iv, use_padding): EncryptedData
        +aes_decrypt(encrypted_data, key, use_padding): bytes
        +encrypt_password(password, salt, iterations, key_length): Tuple
        +verify_password(password, encrypted, salt, iterations): bool
        +generate_rsa_keypair(key_size, public_exponent): KeyPair
        +rsa_encrypt(plaintext, public_key_pem): bytes
        +rsa_decrypt(ciphertext, private_key_pem): bytes
        +rsa_sign(data, private_key_pem, algorithm): bytes
        +rsa_verify(data, signature, public_key_pem, algorithm): bool
        +encrypt_file(input_path, output_path, key, mode, chunk_size): str
        +decrypt_file(input_path, output_path, key): str
    }
    class DeviceType {
    }
    Enum <|-- DeviceType
    class GPUInfo {
        +index
        +name
        +total_memory
        +available_memory
        +utilization
        +temperature
        +compute_capability
        +driver_version
        +cuda_version
        +is_available
    }
    class MemoryInfo {
        +total
        +available
        +used
        +percent
    }
    class DiskInfo {
        +total
        +used
        +free
        +percent
        +mount_point
    }
    class SystemInfo {
        +os_name
        +os_version
        +architecture
        +processor
        +python_version
        +hostname
    }
    class DeviceInfo {
        +device_type
        +primary_device
        +gpu_count
        +gpus
        +cpu_count
        +total_memory
        +available_memory
        +cuda_available
        +cuda_version
        +torch_version
        +tensorflow_version
        +system_info
    }
    class DeviceManager {
        +_instance
        +__new__(cls)
        +__init__(self)
        +get_device(self, device_id): Any
        +get_device_info(self, force_refresh): DeviceInfo
        +_collect_device_info(self): DeviceInfo
        +_detect_device_type(self): DeviceType
        +_get_primary_device(self): str
        +_check_cuda_available(self): bool
        +_check_mps_available(self): bool
        +_get_cuda_version(self): Optional
        +_get_gpu_count(self): int
        +_get_gpu_details(self): List
        +_get_cpu_count(self): int
        +_get_package_version(self, package_name): Optional
        +get_memory_info(self): MemoryInfo
        +_get_memory_info_fallback(self): MemoryInfo
        +get_disk_info(self, path): DiskInfo
        +_get_disk_info_fallback(self, path): DiskInfo
        +get_system_info(self): SystemInfo
        +check_memory_sufficient(self, required_gb): bool
        +check_disk_sufficient(self, required_gb, path): bool
        +get_gpu_memory_info(self, device_id): Tuple
        +print_device_info(self): Constant(value=None, kind=None)
    }
    class FileInfo {
        +path
        +name
        +size
        +created_time
        +modified_time
        +is_directory
        +extension
        +mime_type
    }
    class DirectoryTree {
        +path
        +name
        +is_directory
        +children
        +size
        +__post_init__(self)
    }
    class FileDirOperate {
        +ensure_dir(path, mode): str
        +ensure_parent_dir(filepath): str
        +copy_file(src, dst, overwrite): str
        +move_file(src, dst, overwrite): str
        +delete_file(path, missing_ok): bool
        +delete_dir(path, recursive, missing_ok): bool
        +copy_dir(src, dst, symlinks, ignore_patterns): str
        +list_files(directory, pattern, recursive, include_dirs): List
        +find_files(directory, extensions, name_contains, max_depth, exclude_patterns): List
        +get_file_info(path): FileInfo
        +get_dir_size(directory): int
        +build_directory_tree(directory, max_depth, include_files): DirectoryTree
        +compare_files(file1, file2): bool
        +compute_file_hash(filepath, algorithm): str
        +compute_directory_hash(directory, algorithm): str
        +compress_tar(source_dir, output_path, compression, exclude_patterns): str
        +extract_tar(tar_path, extract_dir): str
        +compress_zip(source_dir, output_path, exclude_patterns): str
        +extract_zip(zip_path, extract_dir): str
        +get_files_by_size(directory, min_size, max_size, extensions): List
        +get_files_by_date(directory, start_date, end_date, date_type): List
    }
    class FileLock {
        +__init__(self, lock_path, timeout)
        +acquire(self, blocking): bool
        +release(self): Constant(value=None, kind=None)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class TimeFormat {
    }
    Enum <|-- TimeFormat
    class TimeRange {
        +start
        +end
        +duration(self): timedelta
        +duration_seconds(self): float
        +contains(self, dt): bool
        +overlaps(self, other): bool
    }
    class TimeDateUtils {
        +now(timezone_id): datetime
        +utc_now(): datetime
        +today(timezone_id): date
        +get_timestamp(dt, milliseconds): int
        +from_timestamp(timestamp, timezone_id): datetime
        +parse(date_string, format_string, timezone_id): datetime
        +_auto_parse(date_string): datetime
        +format(dt, format_string): str
        +format_iso(dt, include_microseconds): str
        +format_relative(dt, reference): str
        +get_timezone(timezone_id): timezone
        +convert_timezone(dt, from_tz, to_tz): datetime
        +add_time(dt, days, hours, minutes, seconds, microseconds): datetime
        +subtract_time(dt, days, hours, minutes, seconds, microseconds): datetime
        +get_time_range(period, reference): TimeRange
        +get_week_range(reference, week_start): TimeRange
        +get_month_range(year, month): TimeRange
        +is_business_day(dt): bool
        +is_weekend(dt): bool
        +get_business_days_count(start, end, exclude_holidays): int
        +get_next_business_day(dt): datetime
        +get_age(birth_date, reference): int
        +get_quarter(dt): int
        +get_quarter_range(year, quarter): TimeRange
        +format_duration(seconds, precision): str
        +parse_duration(duration_str): float
        +get_weekday_name(weekday, short): str
        +get_month_name(month, short): str
        +sleep(seconds): Constant(value=None, kind=None)
        +wait_until(target_time, check_interval): Constant(value=None, kind=None)
    }
    class Timer {
        +__init__(self, name, verbose)
        +start(self): Constant(value='Timer', kind=None)
        +stop(self): float
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
        +reset(self): Constant(value=None, kind=None)
    }
    class CheckpointMetadata {
        +model_name
        +version
        +created_at
        +epoch
        +global_step
        +metrics
        +config
        +device
        +torch_version
        +file_hash
        +file_size
    }
    class WeightFileUtils {
        +save_weights(model, path, save_dtype, use_safetensors, metadata): str
        +_save_torch(state_dict, path, metadata): str
        +_save_safetensors(state_dict, path, metadata): str
        +load_weights(model, path, device, strict, load_to_cpu): Tuple
        +_convert_dtype(state_dict, dtype): Dict
        +load_partial_weights(model, path, device, skip_mismatch): Tuple
    }
    class CheckpointManager {
        +__init__(self, save_dir, max_checkpoints, save_fn, load_fn)
        +save(self, model, epoch, step, metrics, metadata, filename): str
        +load(self, model, checkpoint_path, load_best): Tuple
        +_update_checkpoint_list(self, epoch, step, filepath, metrics): Constant(value=None, kind=None)
        +_cleanup_old_checkpoints(self): Constant(value=None, kind=None)
        +get_latest_checkpoint(self): Optional
        +get_best_checkpoint(self): Optional
        +list_checkpoints(self): List
    }
    class WeightSharding {
        +save_sharded(model, save_dir, max_shard_size, save_fn): List
        +load_sharded(model, save_dir, device): Any
    }
    class ConfigParser {
        +__init__(self, config_path)
        +load(self, path): Dict
        +save(self, path, default_flow_style): Constant(value=None, kind=None)
        +get(self, key, default): Any
        +set(self, key, value): Constant(value=None, kind=None)
        +update(self, updates, deep): Constant(value=None, kind=None)
        +_deep_update(self, base, updates): Constant(value=None, kind=None)
        +merge(self, other_config, priority): Constant(value=None, kind=None)
        +resolve_env_vars(self, prefix, suffix): Constant(value=None, kind=None)
        +_resolve_env_vars_recursive(self, obj, prefix, suffix): Any
        +_replace_env_var(self, value, prefix, suffix): str
        +validate(self, schema): List
        +_validate_recursive(self, config, schema, path, errors): Constant(value=None, kind=None)
        +_check_type(self, value, expected_type): bool
        +to_dict(self): Dict
        +to_json(self, indent): str
        +reset(self): Constant(value=None, kind=None)
        +config(self): Dict
    }
    class DeepSeekTokenizer {
        +__init__(self, vocab_file, merges_file)
        +_load_vocab(self, vocab_file): Dict
        +_load_merges(self, merges_file): Dict
        +_build_byte_encoder(self): Dict
        +encode(self, text, add_special_tokens): List
        +_clean_text(self, text): str
        +_byte_level_encode(self, text): List
        +_encode_unicode(self, char): List
        +_bpe_encode(self, tokens): List
        +decode(self, ids, skip_special_tokens): str
        +_bpe_decode(self, tokens): str
        +_byte_level_decode(self, text): str
        +get_vocab_size(self): int
        +tokenize(self, text, add_special_tokens): List
    }
    class TokenizerEncodeDecode {
        +__init__(self, tokenizer)
        +encode_text(self, text, add_special_tokens, max_length, truncation, padding, pad_to_max_length): Dict
        +encode_batch(self, texts, add_special_tokens, max_length, truncation, padding): Dict
        +decode_tokens(self, ids, skip_special_tokens, clean_up_tokenization_spaces): str
        +decode_batch(self, ids_batch, skip_special_tokens, clean_up_tokenization_spaces): List
        +_clean_up_tokenization(self, text): str
        +convert_tokens_to_ids(self, tokens): List
        +convert_ids_to_tokens(self, ids): List
        +create_padding_mask(self, input_ids, pad_token_id): List
        +create_position_ids(self, input_ids, pad_token_id): List
    }
    class SpecialTokens {
        +__init__(self)
        +get_special_token(self, token): Optional
        +get_token_by_id(self, id_): Optional
        +is_special_token(self, token): bool
        +is_special_token_id(self, id_): bool
        +add_special_token(self, token, id_): int
        +remove_special_token(self, token): bool
        +get_all_special_tokens(self): List
        +get_all_special_token_ids(self): List
        +get_pad_token(self): str
        +get_pad_token_id(self): int
        +get_bos_token(self): str
        +get_bos_token_id(self): int
        +get_eos_token(self): str
        +get_eos_token_id(self): int
        +get_unk_token(self): str
        +get_unk_token_id(self): int
        +get_mask_token(self): str
        +get_mask_token_id(self): int
        +get_image_tokens(self): Dict
        +get_audio_tokens(self): Dict
        +get_conversation_tokens(self): Dict
        +get_format_tokens(self): Dict
        +build_vocab(self): Dict
    }
    class Trainer {
        +__init__(self, args)
        +_init_distributed(self)
        +_init_model(self)
        +_init_optimizer(self)
        +_init_scheduler(self)
        +_init_dataloader(self)
        +_build_transform(self)
        +_init_loss_fn(self)
        +_init_logging(self)
        +_load_checkpoint(self, path)
        +_save_checkpoint(self, step)
        +_log_metrics(self, metrics, step)
        +train(self)
        +evaluate(self)
    }
    class GraphGenerator {
        +__init__(self)
        +generate_mermaid_module_graph(self, module_deps, include_external): str
        +generate_mermaid_class_diagram(self, modules): str
        +generate_dot_module_graph(self, module_deps): str
        +generate_dot_class_hierarchy(self, class_hierarchy): str
        +generate_file_tree(self, modules, root_path): str
        +_render_tree(self, tree, prefix): str
        +generate_summary_markdown(self, modules, module_deps): str
        +_sanitize_name(self, name): str
        +_shorten_name(self, name, max_len): str
    }
    class AnalysisResult {
        +total_modules
        +total_classes
        +total_functions
        +modules
    }
    BaseModel <|-- AnalysisResult
    class DependencyGraphResult {
        +format
        +content
    }
    BaseModel <|-- DependencyGraphResult
    class CodeGraphAPI {
        +__init__(self)
        +_setup_routes(self)
        +run(self, host, port)
        +analyze_and_generate(self, project_path, output_file): str
    }
    class FunctionInfo {
        +name
        +line_number
        +docstring
        +args
        +returns
        +decorators
    }
    class ClassInfo {
        +name
        +line_number
        +docstring
        +base_classes
        +methods
        +attributes
    }
    class ModuleInfo {
        +file_path
        +file_name
        +imports
        +from_imports
        +classes
        +functions
        +docstring
    }
    class CodeAnalyzer {
        +__init__(self)
        +parse_file(self, file_path): ModuleInfo
        +_parse_class(self, node): ClassInfo
        +_parse_function(self, node): FunctionInfo
        +_get_attribute_name(self, node): str
        +_get_type_name(self, node): str
        +parse_directory(self, dir_path, skip_dirs): List
        +get_module_by_path(self, file_path): Optional
        +get_all_classes(self): List
        +get_all_functions(self): List
    }
    class Dependency {
        +source
        +target
        +dependency_type
        +line_number
    }
    class DependencyExtractor {
        +__init__(self)
        +extract_from_modules(self, modules): List
        +_extract_module_dependencies(self, module)
        +_extract_class_dependencies(self, module)
        +_add_dependency(self, source, target, dep_type, line_number)
        +_get_module_name(self, file_path): str
        +get_dependencies_by_type(self, dep_type): List
        +get_module_dependency_graph(self): Dict
        +get_class_hierarchy(self): Dict
        +find_circular_dependencies(self): List
        +get_transitive_dependencies(self, module_name): Set
    }
    class CrossModalConfig {
        +hidden_size
        +num_attention_heads
        +fusion_type
        +dropout_prob
        +layer_norm_eps
        +use_cross_attention
        +use_modality_embedding
        +num_modalities
        +fusion_layers
        +fusion_strategy
    }
    class ModalityEmbedding {
        +__init__(self, hidden_size, num_modalities)
        +forward(self, x, modality_id): torch.Tensor
    }
    nn.Module <|-- ModalityEmbedding
    class GatedFusion {
        +__init__(self, hidden_size, num_gates, dropout)
        +forward(self, text_features, image_features): torch.Tensor
    }
    nn.Module <|-- GatedFusion
    class CrossAttention {
        +__init__(self, hidden_size, num_heads, dropout)
        +forward(self, query, key, value, attention_mask): Tuple
    }
    nn.Module <|-- CrossAttention
    class AttentionFusion {
        +__init__(self, hidden_size, num_heads, num_layers, dropout)
        +forward(self, text_features, image_features, attention_mask): Tuple
    }
    nn.Module <|-- AttentionFusion
    class BilinearFusion {
        +__init__(self, hidden_size, dropout)
        +forward(self, text_features, image_features): torch.Tensor
    }
    nn.Module <|-- BilinearFusion
    class ConcatenationFusion {
        +__init__(self, hidden_size, fusion_dim, dropout)
        +forward(self, text_features, image_features): torch.Tensor
    }
    nn.Module <|-- ConcatenationFusion
    class HierarchicalFusion {
        +__init__(self, hidden_size, num_levels, num_heads, dropout)
        +_create_hierarchy(self, features): List
        +forward(self, text_features, image_features): torch.Tensor
    }
    nn.Module <|-- HierarchicalFusion
    class CrossModalFusion {
        +__init__(self, config)
        +forward(self, text_features, image_features, text_modality_id, image_modality_id, return_attention): Dict
        +fuse_multiple_modalities(self, modality_features, modality_ids): torch.Tensor
    }
    nn.Module <|-- CrossModalFusion
    class MultiHeadCrossModalAttention {
        +__init__(self, hidden_size, num_heads, num_modalities, dropout)
        +forward(self, modality_features, attention_mask): List
    }
    nn.Module <|-- MultiHeadCrossModalAttention
    class TensorFusionNetwork {
        +__init__(self, hidden_size, output_size, dropout)
        +forward(self, text_features, image_features): torch.Tensor
    }
    nn.Module <|-- TensorFusionNetwork
    class FactorizedTensorFusion {
        +__init__(self, hidden_size, rank, dropout)
        +forward(self, text_features, image_features): torch.Tensor
    }
    nn.Module <|-- FactorizedTensorFusion
    class LowRankModalFusion {
        +__init__(self, hidden_size, rank, dropout)
        +forward(self, text_features, image_features): torch.Tensor
    }
    nn.Module <|-- LowRankModalFusion
    class FeatureExtractorConfig {
        +backbone_channels
        +output_channels
        +use_fpn
        +use_pafpn
        +feature_dim
        +num_levels
        +__post_init__(self)
    }
    class ConvBottleneck {
        +__init__(self, in_channels, out_channels, stride)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- ConvBottleneck
    class ResNetBackbone {
        +__init__(self, in_channels, layers, out_indices)
        +_make_layer(self, in_channels, out_channels, blocks, stride)
        +forward(self, x): List
    }
    nn.Module <|-- ResNetBackbone
    class FeaturePyramid {
        +__init__(self, in_channels_list, out_channels, extra_convs)
        +forward(self, inputs): List
    }
    nn.Module <|-- FeaturePyramid
    class PathAggregationFPN {
        +__init__(self, in_channels_list, out_channels, num_outs)
        +forward(self, inputs): List
    }
    nn.Module <|-- PathAggregationFPN
    class AttentionPool2d {
        +__init__(self, spacial_dim, embed_dim, num_heads, output_dim)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- AttentionPool2d
    class VisionFeatureExtractor {
        +__init__(self, config, backbone_type, pretrained)
        +forward(self, x, return_all_levels): Union
        +extract_patch_features(self, x, patch_size): torch.Tensor
        +extract_global_features(self, x): torch.Tensor
    }
    nn.Module <|-- VisionFeatureExtractor
    class MultiScaleFeatureExtractor {
        +__init__(self, base_extractor, scales, scale_factor)
        +forward(self, x, max_scales): torch.Tensor
        +extract_at_scale(self, x, scale): torch.Tensor
    }
    nn.Module <|-- MultiScaleFeatureExtractor
    class HierarchicalFeatureExtractor {
        +__init__(self, base_channels, num_octaves, layers_per_octave, output_dim)
        +_create_gaussian_kernel(self, size): torch.Tensor
        +forward(self, x): Dict
        +_build_pyramid(self, octave_features): torch.Tensor
    }
    nn.Module <|-- HierarchicalFeatureExtractor
    class SpatialAttention {
        +__init__(self, in_channels)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- SpatialAttention
    class ChannelAttention {
        +__init__(self, in_channels, reduction)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- ChannelAttention
    class CBAM {
        +__init__(self, in_channels, reduction)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- CBAM
    class AttentionEnhancedExtractor {
        +__init__(self, base_extractor, use_cbam)
        +forward(self, x, return_all_levels): Union
    }
    nn.Module <|-- AttentionEnhancedExtractor
    class SemanticFeatureExtractor {
        +__init__(self, vision_extractor, semantic_dim)
        +forward(self, x): Dict
    }
    nn.Module <|-- SemanticFeatureExtractor
    class ViTConfig {
        +image_size
        +patch_size
        +in_channels
        +hidden_size
        +num_hidden_layers
        +num_attention_heads
        +intermediate_size
        +hidden_dropout_prob
        +attention_probs_dropout_prob
        +use_memory_efficient_attention
        +use_gradient_checkpointing
        +use_rotary_position_embedding
        +layer_norm_eps
        +qkv_bias
        +num_patches
        +__post_init__(self)
    }
    class PatchEmbedding {
        +__init__(self, config)
        +_init_weights(self)
        +forward(self, pixel_values, interpolate_pos_embedding): Tuple
        +_interpolate_pos_embedding(self, pos_embedding, original_size, target_size, batch_size): torch.Tensor
    }
    nn.Module <|-- PatchEmbedding
    class RotaryPositionEmbedding {
        +__init__(self, dim, max_seq_len)
        +forward(self, seq_len, device): Tuple
    }
    nn.Module <|-- RotaryPositionEmbedding
    class ViTAttention {
        +__init__(self, config)
        +transpose_for_scores(self, x): torch.Tensor
        +forward(self, hidden_states, attention_mask, output_attentions): Tuple
    }
    nn.Module <|-- ViTAttention
    class ViTMLP {
        +__init__(self, config)
        +forward(self, hidden_states): torch.Tensor
    }
    nn.Module <|-- ViTMLP
    class ViTBlock {
        +__init__(self, config)
        +forward(self, hidden_states, attention_mask, output_attentions): Tuple
        +_forward_impl(self, hidden_states, attention_mask, output_attentions): Tuple
    }
    nn.Module <|-- ViTBlock
    class ViTEncoder {
        +__init__(self, config)
        +forward(self, hidden_states, attention_mask, output_hidden_states, output_attentions): Tuple
    }
    nn.Module <|-- ViTEncoder
    class VisionBackbone {
        +__init__(self, config)
        +_init_weights(self, module)
        +forward(self, pixel_values, output_hidden_states, output_attentions, return_dict): Dict
        +get_input_embeddings(self): PatchEmbedding
        +get_output_embeddings(self): Optional
        +resize_pos_embedding(self, new_size)
    }
    nn.Module <|-- VisionBackbone
    class ImageTokenizer {
        +__init__(self, image_size, patch_size, num_codebook_tokens, hidden_size, commitment_loss_weight)
        +forward(self, x): Tuple
    }
    nn.Module <|-- ImageTokenizer
    class ResidualBlock {
        +__init__(self, in_channels, out_channels)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- ResidualBlock
    class VectorQuantizer {
        +__init__(self, num_embeddings, embedding_dim, commitment_loss_weight)
        +forward(self, z): Tuple
    }
    nn.Module <|-- VectorQuantizer
    class TileProcessor {
        +__init__(self, tile_size, overlap, hidden_size, fusion_hidden_size)
        +split_into_tiles(self, image, tile_size): List
        +merge_tiles(self, tile_features, positions, output_size): torch.Tensor
        +forward(self, image, backbone): Dict
    }
    nn.Module <|-- TileProcessor
    class VisionCacheConfig {
        +cache_dir
        +max_cache_size
        +cache_policy
        +precompute_batch_size
        +num_workers
        +enable_disk_cache
        +enable_memory_cache
        +cache_compression
        +cache_format
        +tile_size
        +overlap_ratio
        +cache_metadata_file
        +__post_init__(self)
    }
    class LRUCache {
        +__init__(self, capacity)
        +get(self, key): Optional
        +put(self, key, value)
        +remove(self, key)
        +clear(self)
        +__len__(self)
        +keys(self)
    }
    class LFUCache {
        +__init__(self, capacity)
        +get(self, key): Optional
        +put(self, key, value)
        +remove(self, key)
        +clear(self)
        +__len__(self)
    }
    class FIFOCache {
        +__init__(self, capacity)
        +get(self, key): Optional
        +put(self, key, value)
        +remove(self, key)
        +clear(self)
        +__len__(self)
    }
    class CacheManager {
        +__init__(self, config)
        +_load_metadata(self): Dict
        +_save_metadata(self)
        +_compute_hash(self, identifier): str
        +_get_cache_path(self, cache_key): Path
        +get(self, identifier): Optional
        +put(self, identifier, features, metadata)
        +remove(self, identifier)
        +clear(self)
        +get_stats(self): Dict
        +list_cached(self): List
        +preload(self, identifiers)
    }
    class TileCacheManager {
        +__init__(self, config)
        +_get_tile_key(self, image_path, tile_row, tile_col, scale): str
        +_split_into_tiles(self, image): List
        +get_tile(self, image_path, image, tile_row, tile_col, scale): Optional
        +cache_tiles(self, image_path, tiles, positions, scale)
        +get_or_compute_tile(self, image_path, image, tile_row, tile_col, backbone, scale): torch.Tensor
    }
    class VisionCachePrecompute {
        +__init__(self, backbone, config, device)
        +_create_transform(self): Callable
        +preprocess_image(self, image): torch.Tensor
        +encode_image(self, image, use_cache): torch.Tensor
        +precompute(self, image_paths, batch_size, show_progress, parallel)
        +_precompute_sequential(self, image_paths, batch_size, show_progress)
        +_precompute_parallel(self, image_paths, batch_size, show_progress)
        +precompute_tiles(self, image_path, backbone): List
        +get_cached_features(self, identifier): Optional
        +clear_cache(self)
        +get_cache_stats(self): Dict
        +export_cache(self, export_path)
        +import_cache(self, import_path)
    }
    class AsyncCachePrecompute {
        +__init__(self, precompute, max_queue_size)
        +start(self)
        +stop(self)
        +_worker_loop(self)
        +submit(self, image_path)
        +get_result(self, timeout): Tuple
        +wait_all(self)
    }
    class UnifiedEncoderConfig {
        +hidden_size
        +num_hidden_layers
        +num_attention_heads
        +intermediate_size
        +max_seq_len
        +vocab_size
        +dropout_prob
        +layer_norm_eps
        +use_rope
        +rope_theta
        +use_modality_embedding
        +num_modalities
    }
    class UnifiedAttention {
        +__init__(self, hidden_size, num_heads, dropout)
        +forward(self, hidden_states, attention_mask, output_attentions): Tuple
    }
    nn.Module <|-- UnifiedAttention
    class UnifiedTransformerBlock {
        +__init__(self, hidden_size, num_attention_heads, intermediate_size, dropout, layer_norm_eps)
        +forward(self, hidden_states, attention_mask, output_attentions): Tuple
    }
    nn.Module <|-- UnifiedTransformerBlock
    class ModalityTypeEmbedding {
        +__init__(self, hidden_size, num_modalities)
        +forward(self, modality_type, batch_size, seq_len, device): torch.Tensor
    }
    nn.Module <|-- ModalityTypeEmbedding
    class TextBranch {
        +__init__(self, vocab_size, hidden_size, max_position_embeddings)
        +forward(self, input_ids, attention_mask): torch.Tensor
        +get_output_dim(self): int
    }
    nn.Module <|-- TextBranch
    class ImageBranch {
        +__init__(self, image_size, patch_size, hidden_size)
        +forward(self, pixel_values): torch.Tensor
        +get_output_dim(self): int
    }
    nn.Module <|-- ImageBranch
    class AudioBranch {
        +__init__(self, n_mels, n_fft, hidden_size)
        +forward(self, audio_features): torch.Tensor
        +get_output_dim(self): int
    }
    nn.Module <|-- AudioBranch
    class MultimodalUnifiedEncoder {
        +__init__(self, config)
        +_init_weights(self, module)
        +encode_text(self, input_ids, attention_mask): torch.Tensor
        +encode_image(self, pixel_values): torch.Tensor
        +encode_audio(self, audio_features): torch.Tensor
        +forward(self, input_ids, pixel_values, audio_features, attention_mask, modality_type, output_hidden_states, output_attentions): Dict
        +encode_multiple_modalities(self, text_input, image_input, audio_input): Dict
        +fuse_modalities(self, text_features, image_features, audio_features): torch.Tensor
        +get_modality_features(self, input_ids, pixel_values, audio_features): torch.Tensor
    }
    nn.Module <|-- MultimodalUnifiedEncoder
    class CrossModalUnifiedEncoder {
        +__init__(self, config)
        +forward(self, text_features, image_features, attention_mask): torch.Tensor
    }
    nn.Module <|-- CrossModalUnifiedEncoder
    class CrossAttentionLayer {
        +__init__(self, hidden_size, num_heads)
        +forward(self, query, key_value): torch.Tensor
    }
    nn.Module <|-- CrossAttentionLayer
    class InferenceConfig {
        +max_length
        +min_length
        +temperature
        +top_p
        +top_k
        +repetition_penalty
        +length_penalty
        +num_beams
        +early_stopping
        +do_sample
        +pad_token_id
        +eos_token_id
        +bos_token_id
        +use_cache
        +batch_size
        +device
        +dtype
    }
    class KVCache {
        +__init__(self)
        +update(self, layer_idx, key, value)
        +get(self, layer_idx): Tuple
        +clear(self)
        +get_all(self): Tuple
    }
    class MultiModalInference {
        +__init__(self, model, config, tokenizer)
        +generate(self, input_ids, images, text, return_dict): Dict
        +_sample(self, input_ids, images): torch.Tensor
        +_beam_search(self, input_ids, images): torch.Tensor
        +encode_inputs(self, text, images, audio): Dict
        +get_model(self): nn.Module
        +set_device(self, device)
    }
    class BatchInference {
        +__init__(self, model, config, tokenizer, max_batch_size)
        +start(self)
        +stop(self)
        +_worker_loop(self)
        +_collect_batch(self, timeout): List
        +_process_batch(self, batch): List
        +add_request(self, input_data, callback)
        +get_result(self, timeout): Dict
        +infer_batch(self, texts, images): List
    }
    class StreamingInference {
        +__init__(self, model, config, tokenizer)
        +generate_stream(self, text, images, callback): Iterator
    }
    class ProjectionConfig {
        +input_dim
        +output_dim
        +hidden_dim
        +projection_type
        +num_layers
        +dropout_prob
        +layer_norm_eps
        +activation
        +use_bias
    }
    class LinearProjection {
        +__init__(self, input_dim, output_dim, use_bias)
        +_init_weights(self)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- LinearProjection
    class MLPProjection {
        +__init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout, activation)
        +_get_activation(self, activation)
        +_init_weights(self)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- MLPProjection
    class CrossAttentionProjection {
        +__init__(self, input_dim, output_dim, num_heads, dropout)
        +forward(self, x, context): torch.Tensor
    }
    nn.Module <|-- CrossAttentionProjection
    class ResidualProjection {
        +__init__(self, input_dim, output_dim, hidden_dim, dropout)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- ResidualProjection
    class GatedProjection {
        +__init__(self, input_dim, output_dim, hidden_dim, dropout)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- GatedProjection
    class MultiplicativeProjection {
        +__init__(self, input_dim, output_dim, num_experts)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- MultiplicativeProjection
    class BiDirectionalProjection {
        +__init__(self, input_dim, output_dim, hidden_dim, dropout)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- BiDirectionalProjection
    class ModalProjection {
        +__init__(self, config)
        +forward(self, x, context): torch.Tensor
        +get_output_dim(self): int
        +get_input_dim(self): int
    }
    nn.Module <|-- ModalProjection
    class MultimodalProjector {
        +__init__(self, modality_dims, output_dim, projection_type)
        +forward(self, features): Dict
        +project(self, modality, features): torch.Tensor
    }
    nn.Module <|-- MultimodalProjector
    class AdaptiveProjection {
        +__init__(self, input_dim, output_dim, num_strategies, hidden_dim)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- AdaptiveProjection
    class ComposedProjection {
        +__init__(self, projections)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- ComposedProjection
    class V4Config {
        +hidden_size
        +num_hidden_layers
        +num_attention_heads
        +intermediate_size
        +vocab_size
        +max_position_embeddings
        +dropout_prob
        +image_size
        +patch_size
        +num_channels
    }
    class ImageProcessor {
        +__init__(self, image_size, mean, std, resize)
        +preprocess(self, images): torch.Tensor
        +forward(self, images): torch.Tensor
    }
    nn.Module <|-- ImageProcessor
    class VisionEncoder {
        +__init__(self, image_size, patch_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size)
        +_init_weights(self)
        +forward(self, pixel_values, output_hidden_states): Dict
    }
    nn.Module <|-- VisionEncoder
    class TransformerLayer {
        +__init__(self, hidden_size, num_attention_heads, intermediate_size, dropout)
        +forward(self, hidden_states): torch.Tensor
    }
    nn.Module <|-- TransformerLayer
    class VisionProjector {
        +__init__(self, vision_hidden_size, llm_hidden_size, projection_type)
        +forward(self, vision_features): torch.Tensor
    }
    nn.Module <|-- VisionProjector
    class DynamicImageProcessor {
        +__init__(self, target_sizes, mean, std)
        +forward(self, images, target_size): torch.Tensor
    }
    nn.Module <|-- DynamicImageProcessor
    class AudioProcessor {
        +__init__(self, n_fft, hop_length, n_mels, sample_rate)
        +_create_mel_filters(self): torch.Tensor
        +forward(self, audio): torch.Tensor
    }
    nn.Module <|-- AudioProcessor
    class MultimodalProjector {
        +__init__(self, vision_hidden_size, audio_hidden_size, llm_hidden_size)
        +project_vision(self, vision_features): torch.Tensor
        +project_audio(self, audio_features): torch.Tensor
        +forward(self, features): Dict
    }
    nn.Module <|-- MultimodalProjector
    class MultimodalCollator {
        +__init__(self, tokenizer, padding, padding_length)
        +__call__(self, batch): Dict
    }
    class MultimodalTokenizer {
        +__init__(self, tokenizer, image_token, image_pad_token, num_image_tokens)
        +__call__(self, text, images, padding, max_length, truncation, return_tensors): Dict
        +decode(self, token_ids): str
    }
    class V4TransformerBlock {
        +__init__(self, hidden_size, num_attention_heads, intermediate_size, dropout, use_gated_attention)
        +forward(self, hidden_states, attention_mask): torch.Tensor
    }
    nn.Module <|-- V4TransformerBlock
    class DeepSeekV4Multimodal {
        +__init__(self, config, vision_encoder, vision_projector, language_model)
        +encode_images(self, pixel_values): torch.Tensor
        +forward(self, input_ids, pixel_values, attention_mask, labels): Dict
        +generate(self, input_ids, pixel_values, max_length, temperature, top_p): torch.Tensor
    }
    nn.Module <|-- DeepSeekV4Multimodal
    class MultimodalFeatureExtractor {
        +__init__(self, vision_encoder, text_encoder, projection)
        +extract_vision_features(self, pixel_values): torch.Tensor
        +extract_text_features(self, input_ids): torch.Tensor
        +compute_similarity(self, vision_features, text_features): torch.Tensor
    }
    class TextEncoderConfig {
        +vocab_size
        +hidden_size
        +max_position_embeddings
        +num_hidden_layers
        +num_attention_heads
        +intermediate_size
        +hidden_dropout_prob
        +attention_probs_dropout_prob
        +layer_norm_eps
        +pad_token_id
        +bos_token_id
        +eos_token_id
        +use_rotary_position_embedding
        +use_relative_position
        +relative_attention_max_distance
        +rope_theta
        +rope_scaling
        +tie_word_embeddings
    }
    class LearnedPositionalEmbedding {
        +__init__(self, num_embeddings, embedding_dim, padding_idx)
        +forward(self, input_ids): torch.Tensor
        +_extend_embeddings(self, new_size)
    }
    nn.Module <|-- LearnedPositionalEmbedding
    class SinusoidalPositionalEmbedding {
        +__init__(self, num_embeddings, embedding_dim)
        +_generate_embedding(self, num_embeddings, embedding_dim): torch.Tensor
        +forward(self, input_ids): torch.Tensor
    }
    nn.Module <|-- SinusoidalPositionalEmbedding
    class RotaryPositionalEmbedding {
        +__init__(self, dim, max_seq_len, base, scaling_factor)
        +_build_cos_sin_cache(self, max_seq_len)
        +forward(self, seq_len, device): Tuple
    }
    nn.Module <|-- RotaryPositionalEmbedding
    class RelativePositionBias {
        +__init__(self, num_heads, max_distance, bidirectional)
        +_relative_position_bucket(self, relative_position, bidirectional, num_buckets, max_distance): torch.Tensor
        +forward(self, seq_len, seq_len_q): torch.Tensor
    }
    nn.Module <|-- RelativePositionBias
    class TextAttention {
        +__init__(self, hidden_size, num_attention_heads, dropout, rope, use_relative_position, num_relative_distance)
        +forward(self, hidden_states, attention_mask, output_attentions): Tuple
    }
    nn.Module <|-- TextAttention
    class TextMLP {
        +__init__(self, hidden_size, intermediate_size, dropout)
        +forward(self, hidden_states): torch.Tensor
    }
    nn.Module <|-- TextMLP
    class TransformerBlock {
        +__init__(self, hidden_size, num_attention_heads, intermediate_size, dropout, layer_norm_eps, rope, use_relative_position)
        +forward(self, hidden_states, attention_mask, output_attentions): Tuple
    }
    nn.Module <|-- TransformerBlock
    class TextEncoderLayer {
        +__init__(self, config, rope)
        +forward(self, hidden_states, attention_mask, output_hidden_states, output_attentions): Tuple
    }
    nn.Module <|-- TextEncoderLayer
    class TextEmbedEncoder {
        +__init__(self, config)
        +_init_weights(self, module)
        +forward(self, input_ids, attention_mask, position_ids, output_hidden_states, output_attentions, return_dict): Dict
        +get_input_embeddings(self): nn.Embedding
        +set_input_embeddings(self, embeddings)
    }
    nn.Module <|-- TextEmbedEncoder
    class PositionalTextEncoder {
        +__init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len)
        +forward(self, input_ids, attention_mask): torch.Tensor
    }
    nn.Module <|-- PositionalTextEncoder
    class RotaryTextEncoder {
        +__init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len, rope_theta)
        +forward(self, input_ids, attention_mask): torch.Tensor
    }
    nn.Module <|-- RotaryTextEncoder
    class TextEncoderWithProjection {
        +__init__(self, encoder, projection_dim, use_gated_projection)
        +forward(self, input_ids, attention_mask): torch.Tensor
    }
    nn.Module <|-- TextEncoderWithProjection
    class GatedProjection {
        +__init__(self, input_dim, output_dim)
        +forward(self, x): torch.Tensor
    }
    nn.Module <|-- GatedProjection
    class MultiScaleTextEncoder {
        +__init__(self, config, num_scales)
        +forward(self, input_ids, attention_mask): torch.Tensor
    }
    nn.Module <|-- MultiScaleTextEncoder
    class ConditionalTextEncoder {
        +__init__(self, config, num_condition_types)
        +forward(self, input_ids, condition_ids, attention_mask): torch.Tensor
    }
    nn.Module <|-- ConditionalTextEncoder
    class TextEncoderPooler {
        +__init__(self, hidden_size, pooler_type)
        +forward(self, hidden_states): torch.Tensor
    }
    nn.Module <|-- TextEncoderPooler
    class FunctionCallParser {
        +parse(self, llm_output)
    }
    class MemoryThinkPool {
        +__init__(self, capacity)
        +add(self, thought)
    }
    class StepType {
    }
    Enum <|-- StepType
    class PlanStrategy {
    }
    Enum <|-- PlanStrategy
    class StepStatus {
    }
    Enum <|-- StepStatus
    class PlanStep {
        +step_id
        +step_type
        +content
        +description
        +dependencies
        +preconditions
        +expected_output
        +status
        +result
        +error
        +retry_count
        +max_retries
        +timeout
        +metadata
        +__post_init__(self)
        +can_execute(self, completed_steps): bool
        +mark_completed(self, result): Constant(value=None, kind=None)
        +mark_failed(self, error): Constant(value=None, kind=None)
        +should_retry(self): bool
        +to_dict(self): Dict
    }
    class ExecutionPlan {
        +plan_id
        +goal
        +intent
        +steps
        +strategy
        +estimated_steps
        +created_at
        +context
        +metadata
        +__post_init__(self)
        +get_step(self, step_id): Optional
        +get_executable_steps(self, completed): List
        +validate(self): Tuple
        +to_dict(self): Dict
    }
    class TaskPlanner {
        +__init__(self, enable_planning, default_strategy, max_steps)
        +_init_decomposition_rules(self)
        +_init_step_templates(self)
        +create_plan(self, goal, intent, context): ExecutionPlan
        +_create_simple_plan(self, goal, intent): ExecutionPlan
        +_decompose_search_task(self, goal, context): List
        +_decompose_code_task(self, goal, context): List
        +_decompose_analysis_task(self, goal, context): List
        +_decompose_question_task(self, goal, context): List
        +_decompose_chat_task(self, goal, context): List
        +_decompose_general_task(self, goal, context): List
        +_prune_plan(self, steps): List
        +_rebuild_dependencies(self, steps): List
        +_optimize_plan(self, steps): List
        +_topological_sort(self, steps): List
        +create_subplan(self, parent_plan, sub_goal, dependencies): ExecutionPlan
        +merge_plans(self, plans, merge_strategy): ExecutionPlan
        +update_plan(self, plan, step_results, failed_step_id): ExecutionPlan
        +estimate_execution_time(self, plan): float
        +get_critical_path(self, plan): List
    }
    class HierarchicalPlanner {
        +__init__(self)
        +create_hierarchical_plan(self, goal, intent, context, depth): Dict
        +_decompose_goal(self, goal, intent): List
        +_create_leaf_node(self, goal, intent): Dict
        +flatten_plan(self, hierarchical_plan): List
    }
    TaskPlanner <|-- HierarchicalPlanner
    class ReactivePlanner {
        +__init__(self)
        +_init_adaptation_rules(self)
        +adapt_plan(self, plan, event, event_data): ExecutionPlan
        +_adapt_on_tool_failure(self, plan, event_data): ExecutionPlan
        +_adapt_on_unexpected_result(self, plan, event_data): ExecutionPlan
        +_adapt_on_timeout(self, plan, event_data): ExecutionPlan
        +_adapt_on_resource_constraint(self, plan, event_data): ExecutionPlan
    }
    TaskPlanner <|-- ReactivePlanner
    class AgentState {
    }
    Enum <|-- AgentState
    class ErrorType {
    }
    Enum <|-- ErrorType
    class ExecutionResult {
        +success
        +content
        +error
        +error_type
        +tool_calls
        +execution_time
        +iterations
        +to_dict(self): Dict
    }
    class AgentConfig {
        +max_iterations
        +max_execution_time
        +enable_reflection
        +enable_memory
        +enable_planning
        +enable_code_interpreter
        +temperature
        +top_p
        +model_name
        +stream_output
    }
    class CoreAgent {
        +__init__(self, model, config, tools, plugins_dir)
        +register_tool(self, name, func, description): Constant(value=None, kind=None)
        +register_tools(self, tools): Constant(value=None, kind=None)
        +run(self, task, context): ExecutionResult
        +_build_prompt(self, prompt, context): str
        +_incorporate_reflection(self, content, reflection): Any
        +_aggregate_results(self, results): Any
        +_add_to_memory(self, task, result): Constant(value=None, kind=None)
        +get_memory(self, query, limit): List
        +get_conversation_history(self, limit): List
        +clear_history(self): Constant(value=None, kind=None)
        +get_stats(self): Dict
        +reset(self): Constant(value=None, kind=None)
        +register_plugin(self, plugin_path): bool
        +get_available_tools(self): List
        +__repr__(self): str
    }
    class MultiTurnAgent {
        +__init__(self)
        +reset_session(self): Constant(value=None, kind=None)
    }
    CoreAgent <|-- MultiTurnAgent
    class ReActAgent {
        +__init__(self)
        +_extract_action(self, thought): Optional
        +_is_finished(self, thought): bool
    }
    CoreAgent <|-- ReActAgent
    class CodeInterpreter {
        +execute(self, code)
    }
    class ToolManager {
        +__init__(self)
        +register(self, name, func)
        +call(self, name)
    }
    class IntentType {
    }
    Enum <|-- IntentType
    class IntentConfidence {
        +__init__(self, intent, confidence, keywords)
        +to_dict(self): Dict
    }
    class IntentResult {
        +primary_intent
        +confidence
        +secondary_intents
        +entities
        +slots
        +raw_intent
        +to_dict(self): Dict
    }
    class IntentRecognizer {
        +__init__(self, use_ml)
        +_init_keyword_patterns(self)
        +_init_regex_patterns(self)
        +_init_entity_extractors(self)
        +recognize(self, text): str
        +recognize_full(self, text): IntentResult
        +_calculate_intent_scores(self, text, text_lower): Dict
        +_get_primary_intent(self, scores): Tuple
        +_get_secondary_intents(self, scores): List
        +_extract_entities(self, text): Dict
        +_extract_slots(self, text, intent): Dict
        +add_keyword_pattern(self, intent, keywords): Constant(value=None, kind=None)
        +add_regex_pattern(self, intent, pattern): Constant(value=None, kind=None)
        +add_entity_extractor(self, entity_type, pattern): Constant(value=None, kind=None)
        +batch_recognize(self, texts): List
        +get_intent_description(self, intent): str
    }
    class AdvancedIntentRecognizer {
        +__init__(self)
        +_init_context_patterns(self)
        +_init_intent_hierarchy(self)
        +recognize_full(self, text, context): IntentResult
        +_incorporate_context(self, result, context): IntentResult
        +infer_intent_from_history(self, history): Optional
    }
    IntentRecognizer <|-- AdvancedIntentRecognizer
    class ReflectionLevel {
    }
    Enum <|-- ReflectionLevel
    class EvaluationDimension {
    }
    Enum <|-- EvaluationDimension
    class ImprovementType {
    }
    Enum <|-- ImprovementType
    class ReflectionResult {
        +reflection_id
        +timestamp
        +level
        +evaluation
        +insights
        +improvements
        +success_score
        +reasoning_chain
        +metadata
        +__post_init__(self)
        +get_overall_score(self): float
        +get_top_improvements(self, n): List
        +to_dict(self): Dict
    }
    class ActionEvaluation {
        +action_id
        +action_type
        +expected_outcome
        +actual_outcome
        +success
        +deviation
        +feedback
        +timestamp
        +analyze_deviation(self): str
    }
    class ThoughtReflection {
        +__init__(self, enable, default_level, max_history)
        +_init_evaluation_thresholds(self)
        +_init_reflection_prompts(self)
        +reflect(self, action_result, tool_calls, iterations): Dict
        +reflect_full(self, action_result, tool_calls, iterations, level, context): ReflectionResult
        +_evaluate_result(self, action_result, tool_calls, iterations): Dict
        +_evaluate_correctness(self, action_result, tool_calls): float
        +_evaluate_efficiency(self, iterations, tool_calls): float
        +_evaluate_completeness(self, action_result, tool_calls): float
        +_evaluate_coherence(self, tool_calls): float
        +_calls_are_related(self, call1, call2): bool
        +_evaluate_novelty(self, action_result, tool_calls): float
        +_generate_insights(self, action_result, evaluation, tool_calls): List
        +_suggest_improvements(self, action_result, evaluation, insights, tool_calls): List
        +_create_improvement(self, dimension, current_score, target_score, action_result, tool_calls): Dict
        +_calculate_success_score(self, evaluation, iterations): float
        +_build_reasoning_chain(self, action_result, evaluation, improvements): List
        +_store_reflection(self, reflection): Constant(value=None, kind=None)
        +_create_empty_reflection(self): ReflectionResult
        +get_reflection_history(self, level, limit): List
        +get_average_scores(self): Dict
        +get_common_improvements(self, limit): List
        +compare_reflections(self, reflection_id1, reflection_id2): Dict
        +reset_history(self): Constant(value=None, kind=None)
    }
    class AdvancedThoughtReflection {
        +__init__(self)
        +reflect_full(self, action_result, tool_calls, iterations, level, context): ReflectionResult
        +_update_patterns(self, reflection, tool_calls): Constant(value=None, kind=None)
        +_extract_pattern(self, tool_calls): str
        +get_successful_patterns(self, min_score): List
        +get_learning_insights(self): Dict
        +_calculate_trends(self): Dict
    }
    ThoughtReflection <|-- AdvancedThoughtReflection
    class SelfCorrectionReflection {
        +__init__(self)
        +_suggest_improvements(self, action_result, evaluation, insights, tool_calls): List
        +_classify_error(self, error_message): str
        +_generate_corrections(self, error_type, tool_calls): List
        +get_error_statistics(self): Dict
    }
    ThoughtReflection <|-- SelfCorrectionReflection
    class PluginLoader {
        +load(self, plugin_name)
    }
    class Base {
    }
    DeclarativeBase <|-- Base
    class DatabaseConfig {
        +__init__(self, host, port, user, password, database, charset, pool_size, max_overflow, pool_recycle, pool_pre_ping, echo)
        +from_env(cls): Constant(value='DatabaseConfig', kind=None)
        +from_dict(cls, config): Constant(value='DatabaseConfig', kind=None)
        +from_json(cls, json_path): Constant(value='DatabaseConfig', kind=None)
        +to_dict(self): Dict
        +get_connection_url(self): str
    }
    class MySQLClient {
        +__init__(self, config, engine)
        +engine(self): Engine
        +session_factory(self): sessionmaker
        +metadata(self): MetaData
        +_create_engine(self): Engine
        +get_session(self): Session
        +get_connection(self): Connection
        +create_database_if_not_exists(self)
        +create_all_tables(self, base)
        +drop_all_tables(self, base)
        +create_table(self, table)
        +drop_table(self, table)
        +execute_raw_sql(self, sql, params): ResultProxy
        +execute_scalar(self, sql, params): Any
        +get_table_names(self): List
        +get_table_columns(self, table_name): List
        +table_exists(self, table_name): bool
        +get_row_count(self, table_name): int
        +truncate_table(self, table_name)
        +insert(self, table_name, data): int
        +insert_many(self, table_name, data_list): int
        +update(self, table_name, data, where): int
        +delete(self, table_name, where): int
        +select(self, table_name, columns, where, order_by, limit, offset): List
        +select_one(self, table_name, columns, where): Optional
        +select_by_sql(self, sql, params): List
        +count(self, table_name, where): int
        +exists(self, table_name, where): bool
        +begin_transaction(self): Connection
        +commit(self, conn)
        +rollback(self, conn)
        +ping(self): bool
        +close(self)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class TransactionManager {
        +__init__(self, client)
        +begin(self)
        +commit(self)
        +rollback(self)
        +execute(self, sql, params)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class QueryBuilder {
        +__init__(self, client, table_name)
        +select(self): Constant(value='QueryBuilder', kind=None)
        +join(self, table, on, join_type): Constant(value='QueryBuilder', kind=None)
        +left_join(self, table, on): Constant(value='QueryBuilder', kind=None)
        +right_join(self, table, on): Constant(value='QueryBuilder', kind=None)
        +where(self, condition): Constant(value='QueryBuilder', kind=None)
        +where_in(self, field, values): Constant(value='QueryBuilder', kind=None)
        +where_between(self, field, low, high): Constant(value='QueryBuilder', kind=None)
        +order_by(self, field, direction): Constant(value='QueryBuilder', kind=None)
        +group_by(self): Constant(value='QueryBuilder', kind=None)
        +having(self, condition): Constant(value='QueryBuilder', kind=None)
        +limit(self, count): Constant(value='QueryBuilder', kind=None)
        +offset(self, count): Constant(value='QueryBuilder', kind=None)
        +build_select(self): Tuple
        +all(self): List
        +first(self): Optional
        +count_all(self): int
    }
    class UserStatus {
    }
    PyEnum <|-- UserStatus
    class MessageRole {
    }
    PyEnum <|-- MessageRole
    class SessionStatus {
    }
    PyEnum <|-- SessionStatus
    class TokenType {
    }
    PyEnum <|-- TokenType
    class AuditAction {
    }
    PyEnum <|-- AuditAction
    class User {
        +id
        +username
        +email
        +password_hash
        +full_name
        +avatar_url
        +phone
        +status
        +role
        +metadata
        +last_login_at
        +login_count
        +failed_login_attempts
        +password_changed_at
        +email_verified
        +created_at
        +updated_at
        +sessions
        +tokens
        +audit_logs
    }
    Base <|-- User
    class Token {
        +id
        +user_id
        +token_type
        +access_token
        +refresh_token
        +token_hash
        +expires_at
        +issued_at
        +revoked
        +revoked_at
        +ip_address
        +user_agent
        +created_at
        +user
    }
    Base <|-- Token
    class Session {
        +id
        +session_id
        +user_id
        +title
        +description
        +status
        +model
        +system_prompt
        +temperature
        +max_tokens
        +top_p
        +context_window
        +message_count
        +token_count
        +total_cost
        +metadata
        +last_message_at
        +created_at
        +updated_at
        +user
        +messages
    }
    Base <|-- Session
    class Message {
        +id
        +message_id
        +session_id
        +parent_id
        +role
        +content
        +name
        +function_call
        +function_response
        +tool_calls
        +tool_call_id
        +finish_reason
        +model
        +input_tokens
        +output_tokens
        +total_tokens
        +cost
        +latency_ms
        +metadata
        +embeddings
        +is_edited
        +is_deleted
        +created_at
        +updated_at
        +session
        +parent
        +replies
    }
    Base <|-- Message
    class VectorStore {
        +id
        +vector_id
        +collection_name
        +text
        +chunk_index
        +chunk_size
        +vector
        +dimensions
        +distance
        +metadata
        +source_type
        +source_id
        +user_id
        +is_deleted
        +created_at
        +updated_at
    }
    Base <|-- VectorStore
    class AuditLog {
        +id
        +user_id
        +action
        +resource_type
        +resource_id
        +changes
        +ip_address
        +user_agent
        +request_method
        +request_path
        +response_status
        +error_message
        +execution_time_ms
        +created_at
        +user
    }
    Base <|-- AuditLog
    class Config {
        +id
        +key
        +value
        +value_type
        +description
        +category
        +is_encrypted
        +is_system
        +is_public
        +validation_rule
        +min_value
        +max_value
        +allowed_values
        +updated_by
        +created_at
        +updated_at
    }
    Base <|-- Config
    class Model {
        +id
        +model_id
        +name
        +provider
        +model_type
        +description
        +context_window
        +max_output_tokens
        +supports_functions
        +supports_vision
        +supports_streaming
        +embedding_dimensions
        +input_price_per_1k
        +output_price_per_1k
        +is_active
        +is_default
        +metadata
        +created_at
        +updated_at
    }
    Base <|-- Model
    class FileStorage {
        +id
        +file_id
        +user_id
        +filename
        +original_filename
        +file_type
        +mime_type
        +file_size
        +storage_path
        +storage_type
        +checksum
        +metadata
        +is_public
        +access_count
        +expires_at
        +created_at
        +updated_at
    }
    Base <|-- FileStorage
    class RateLimit {
        +id
        +key
        +limit_type
        +limit_value
        +window_seconds
        +current_count
        +window_start
        +last_access
        +created_at
        +updated_at
    }
    Base <|-- RateLimit
    class Webhook {
        +id
        +webhook_id
        +user_id
        +name
        +url
        +secret
        +event_types
        +is_active
        +headers
        +retry_count
        +timeout_seconds
        +metadata
        +created_at
        +updated_at
    }
    Base <|-- Webhook
    class WebhookDelivery {
        +id
        +delivery_id
        +webhook_id
        +event_type
        +payload
        +request_headers
        +request_body
        +response_status
        +response_body
        +response_headers
        +attempt
        +max_attempts
        +next_retry_at
        +status
        +error_message
        +latency_ms
        +created_at
        +delivered_at
    }
    Base <|-- WebhookDelivery
    class Collection {
        +id
        +collection_id
        +user_id
        +name
        +description
        +dimension
        +metric_type
        +index_type
        +vector_count
        +metadata
        +is_public
        +is_active
        +created_at
        +updated_at
    }
    Base <|-- Collection
    class Tool {
        +id
        +tool_id
        +name
        +description
        +parameters
        +function_schema
        +category
        +is_active
        +is_global
        +metadata
        +created_at
        +updated_at
    }
    Base <|-- Tool
    class VectorIndexType {
    }
    Enum <|-- VectorIndexType
    class DistanceMetric {
    }
    Enum <|-- DistanceMetric
    class VectorRecord {
        +id
        +vector
        +metadata
        +embedding
        +to_dict(self): Dict
        +from_dict(cls, data): Constant(value='VectorRecord', kind=None)
    }
    class SearchResult {
        +id
        +score
        +metadata
        +vector
        +to_dict(self): Dict
    }
    class IndexStats {
        +total_vectors
        +dimension
        +index_type
        +metric_type
        +size_bytes
        +build_time
    }
    class VectorDBConfig {
        +__init__(self, provider, dimension, index_type, metric, index_params, search_params, nlist, nprobe, m, ef_construction, ef_search, cache_size)
        +from_dict(cls, config): Constant(value='VectorDBConfig', kind=None)
        +to_dict(self): Dict
    }
    class BaseVectorDB {
        +__init__(self, config)
        +initialize(self): bool
        +add(self, records): bool
        +search(self, query, top_k, filters): List
        +delete(self, ids): bool
        +get(self, ids): List
        +update(self, records): bool
        +count(self): int
        +save(self, path): bool
        +load(self, path): bool
        +reset(self)
        +get_stats(self): IndexStats
    }
    ABC <|-- BaseVectorDB
    class FAISSVectorDB {
        +__init__(self, config, dimension, index_type, metric)
        +initialize(self): bool
        +add(self, records): bool
        +search(self, query, top_k, filters): List
        +delete(self, ids): bool
        +_rebuild_index(self)
        +get(self, ids): List
        +update(self, records): bool
        +count(self): int
        +save(self, path): bool
        +load(self, path): bool
        +reset(self)
        +get_stats(self): IndexStats
    }
    BaseVectorDB <|-- FAISSVectorDB
    class MilvusVectorDB {
        +__init__(self, config, uri, token, collection_name, dimension)
        +_get_client(self)
        +initialize(self): bool
        +add(self, records): bool
        +search(self, query, top_k, filters): List
        +delete(self, ids): bool
        +get(self, ids): List
        +update(self, records): bool
        +count(self): int
        +save(self, path): bool
        +load(self, path): bool
        +reset(self)
        +get_stats(self): IndexStats
    }
    BaseVectorDB <|-- MilvusVectorDB
    class VectorDBManager {
        +_instances
        +get_instance(cls, name, provider): BaseVectorDB
        +register_instance(cls, name, instance)
        +list_instances(cls): List
        +remove_instance(cls, name)
    }
    class VectorOperations {
        +normalize(vector): np.ndarray
        +batch_normalize(vectors): np.ndarray
        +cosine_similarity(v1, v2): float
        +euclidean_distance(v1, v2): float
        +dot_product(v1, v2): float
        +batch_cosine_similarity(query, vectors): np.ndarray
    }
    class BackupStatus {
    }
    Enum <|-- BackupStatus
    class BackupType {
    }
    Enum <|-- BackupType
    class CompressionType {
    }
    Enum <|-- CompressionType
    class BackupMetadata {
        +backup_id
        +backup_name
        +backup_type
        +compression
        +status
        +start_time
        +end_time
        +file_path
        +file_size
        +checksum
        +tables
        +record_counts
        +error_message
        +mysql_version
        +redis_data_size
        +vector_data_size
        +metadata_size
        +total_size
        +includes_mysql
        +includes_redis
        +includes_vectors
        +retention_days
        +to_dict(self): Dict
        +from_dict(cls, data): Constant(value='BackupMetadata', kind=None)
    }
    class RestoreMetadata {
        +restore_id
        +backup_id
        +restore_type
        +status
        +start_time
        +end_time
        +tables_restored
        +records_restored
        +error_message
        +tables_failed
        +to_dict(self): Dict
    }
    class BackupConfig {
        +__init__(self, backup_dir, compression, includes_mysql, includes_redis, includes_vectors, retention_days, max_backups, parallel_workers, chunk_size, exclude_tables, include_only_tables, compress_level)
    }
    class BackupRestore {
        +__init__(self, mysql_client, redis_cache, config)
        +_ensure_backup_dir(self)
        +_generate_backup_id(self): str
        +_calculate_checksum(self, file_path): str
        +_compress_file(self, source_path, target_path, compression): str
        +_decompress_file(self, source_path, target_path, compression): str
        +backup_mysql(self, backup_id, tables): Dict
        +backup_redis(self, backup_id): Dict
        +backup_vectors(self, backup_id, vector_index_path): Dict
        +create_backup(self, name, backup_type, tables, vector_index_path, progress_callback): BackupMetadata
        +list_backups(self): List
        +get_backup(self, backup_id): Optional
        +restore_mysql(self, backup_dir, tables, drop_existing): Dict
        +restore_redis(self, backup_dir): Dict
        +restore(self, backup_id, restore_target, tables, drop_existing, progress_callback): RestoreMetadata
        +delete_backup(self, backup_id): bool
        +cleanup_old_backups(self): List
        +verify_backup(self, backup_id): Dict
        +export_to_sql(self, backup_id, output_path, include_schema): bool
    }
    class RedisConfig {
        +__init__(self, host, port, db, password, max_connections, socket_timeout, socket_connect_timeout, socket_keepalive, health_check_interval, decode_responses, encoding, encoding_errors)
        +from_env(cls): Constant(value='RedisConfig', kind=None)
        +from_url(cls, url): Constant(value='RedisConfig', kind=None)
        +to_dict(self): Dict
    }
    class SerializationType {
    }
    class Serializer {
        +serialize(value, serializer): str
        +deserialize(value, serializer): Any
    }
    class CacheStats {
        +__init__(self)
        +total_requests(self): int
        +hit_rate(self): float
        +to_dict(self): Dict
    }
    class RedisCache {
        +__init__(self, config, redis_client, default_ttl, default_serializer, key_prefix)
        +client(self): Redis
        +_create_pool(self): ConnectionPool
        +_create_client(self): Redis
        +_make_key(self, key): str
        +_parse_key(self, key): str
        +pipeline(self)
        +get(self, key, serializer): Optional
        +set(self, key, value, ttl, serializer): bool
        +setnx(self, key, value, ttl): bool
        +setex(self, key, value, ttl): bool
        +delete(self): int
        +exists(self): int
        +expire(self, key, ttl): bool
        +ttl(self, key): int
        +rename(self, old_key, new_key): bool
        +increment(self, key, amount): Optional
        +decrement(self, key, amount): Optional
        +get_many(self): Dict
        +set_many(self, mapping, ttl): int
        +get_or_set(self, key, default, ttl): Any
        +clear_pattern(self, pattern): int
        +keys(self, pattern): List
        +scan(self, pattern, count): List
        +get_info(self): Dict
        +ping(self): bool
        +close(self)
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class DistributedLock {
        +__init__(self, cache, lock_name, timeout, blocking, blocking_timeout)
        +acquire(self): bool
        +release(self): bool
        +extend(self, additional_time): bool
        +__enter__(self)
        +__exit__(self, exc_type, exc_val, exc_tb)
    }
    class RateLimiter {
        +__init__(self, cache, key, max_requests, window_seconds)
        +is_allowed(self): bool
        +get_remaining(self): int
        +reset(self)
    }
    class PubSubManager {
        +__init__(self, cache)
        +publish(self, channel, message): int
        +subscribe(self, channel, callback)
        +psubscribe(self, pattern, callback)
        +listen(self, timeout)
        +close(self)
    }
    class CacheDecorator {
        +cached(cache, key_func, ttl, unless)
    }
```