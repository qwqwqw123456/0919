#!/usr/bin/env python3
"""
DeepSeek V4 Complete - 模块导入测试
验证所有核心模块是否可以正常导入
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_module(module_name: str, imports: list) -> bool:
    """测试单个模块的导入"""
    print(f"\n{'='*50}")
    print(f"测试模块: {module_name}")
    print('='*50)
    
    try:
        for import_name in imports:
            module = __import__(f"{module_name}.{import_name}", fromlist=[import_name])
            print(f"  ✅ 成功导入: {module_name}.{import_name}")
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  其他错误: {e}")
        return False

def main():
    print("=" * 60)
    print("DeepSeek V4 Complete - 模块导入测试")
    print("=" * 60)
    
    modules = {
        'core': ['RMSNorm', 'SwiGLU', 'RoPE', 'MultiHeadLatentAttention', 
                 'DeepSeekMoE', 'MultiTokenPredictor', 'KVCacheManager'],
        'tokenizer': ['DeepSeekTokenizer', 'TokenizerEncodeDecode', 'SpecialTokens'],
        'data_engine': ['RawDataCleaner', 'TextFilter', 'DataFormatConverter',
                        'SampleBuilder', 'DataAugmenter', 'StreamingDataset'],
        'human_alignment': ['SFTTrainer', 'DPOTrainer', 'PPOTrainer'],
        'omega_alignment': ['OmegaSampler', 'OmegaAligner'],
        'high_perf_infer': ['InferenceEngine', 'GenerateConfig'],
        'conversation_memory': ['ConversationMemory', 'LongTermMemory'],
        'safe_guard_system': ['SafetyFilter', 'ContentModeration'],
        'search_enhance': ['BaseSearchEngine', 'SearchJudge'],
        'model_evaluation': ['BasicMetricCalc', 'CommonBenchTest'],
        'operation_monitor': ['get_logger', 'get_gpu_monitor'],
        'common_utils': ['ConfigParser', 'DeviceManager', 'encrypt_text'],
    }
    
    results = []
    for module_name, imports in modules.items():
        result = test_module(module_name, imports)
        results.append((module_name, result))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for module_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {module_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个模块通过测试")
    
    if passed == total:
        print("\n🎉 所有核心模块测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个模块测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())