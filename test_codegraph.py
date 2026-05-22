#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('/workspace/deepseek_v4_complete'))

from deepseek_v4_complete.codegraph.code_analyzer import CodeAnalyzer
from deepseek_v4_complete.codegraph.dependency_extractor import DependencyExtractor
from deepseek_v4_complete.codegraph.graph_generator import GraphGenerator

def main():
    print("=" * 60)
    print("代码图可视化功能测试")
    print("=" * 60)
    
    # 1. 测试代码分析器
    print("\n[1/4] 测试代码分析器...")
    analyzer = CodeAnalyzer()
    
    # 解析项目
    project_path = "/workspace/deepseek_v4_complete"
    print(f"分析项目目录: {project_path}")
    modules = analyzer.parse_directory(project_path, skip_dirs={'__pycache__', '.git'})
    print(f"✅ 成功解析 {len(modules)} 个模块")
    
    # 显示概览
    total_classes = sum(len(m.classes) for m in modules)
    total_functions = sum(len(m.functions) + sum(len(c.methods) for c in m.classes) 
                           for m in modules)
    print(f"   - 类数量: {total_classes}")
    print(f"   - 函数/方法数量: {total_functions}")
    
    # 2. 测试依赖关系提取
    print("\n[2/4] 测试依赖关系提取器...")
    extractor = DependencyExtractor()
    dependencies = extractor.extract_from_modules(modules)
    module_deps = extractor.get_module_dependency_graph()
    class_hierarchy = extractor.get_class_hierarchy()
    cycles = extractor.find_circular_dependencies()
    print(f"✅ 提取了 {len(dependencies)} 个依赖关系")
    print(f"   - 模块依赖图: {len(module_deps)} 个节点")
    print(f"   - 类继承关系: {len(class_hierarchy)} 条")
    print(f"   - 循环依赖: {len(cycles)} 个")
    
    # 3. 测试图表生成器
    print("\n[3/4] 测试图表生成器...")
    generator = GraphGenerator()
    
    # 生成文件树
    print("   生成文件目录树...")
    file_tree = generator.generate_file_tree(modules, project_path)
    print("✅ 文件目录树生成成功:")
    print(file_tree[:500], "..." if len(file_tree) > 500 else "")
    
    # 生成摘要Markdown
    print("\n   生成项目摘要...")
    summary = generator.generate_summary_markdown(modules, module_deps)
    summary_path = "/workspace/project_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"✅ 项目摘要已保存到: {summary_path}")
    
    # 生成Mermaid类图
    print("\n   生成Mermaid类图...")
    mermaid_class = generator.generate_mermaid_class_diagram(modules[:5])  # 只取前5个模块作为示例
    mermaid_path = "/workspace/class_diagram.mmd"
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_class)
    print(f"✅ Mermaid类图已保存到: {mermaid_path}")
    
    # 生成DOT模块图
    print("\n   生成DOT模块图...")
    dot_module = generator.generate_dot_module_graph(module_deps)
    dot_path = "/workspace/module_graph.dot"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_module)
    print(f"✅ DOT模块图已保存到: {dot_path}")
    
    # 4. 测试对用户训练脚本的分析
    print("\n[4/4] 分析用户训练脚本...")
    train_script_path = "/workspace/train_script.py"
    if os.path.exists(train_script_path):
        train_module = analyzer.parse_file(train_script_path)
        print(f"✅ 成功解析训练脚本")
        print(f"   - 类: {[c.name for c in train_module.classes]}")
        print(f"   - 函数: {[f.name for f in train_module.functions]}")
        print(f"   - 导入: {train_module.imports}")
    else:
        print(f"⚠️ 训练脚本不存在: {train_script_path}")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print(f"  1. {summary_path} - 项目结构摘要")
    print(f"  2. {mermaid_path} - Mermaid类图")
    print(f"  3. {dot_path} - DOT模块图")

if __name__ == "__main__":
    main()
