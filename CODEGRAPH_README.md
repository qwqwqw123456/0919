# 代码图可视化功能使用说明

## 概述

我们已经为 DeepSeek V4 完整项目添加了强大的代码图可视化功能，可以帮助您：

- 分析项目的代码结构
- 提取模块依赖关系
- 生成可视化图表（Mermaid、DOT 格式）
- 查看项目的类继承关系
- 检测循环依赖问题

## 文件结构

代码图可视化模块位于 `/workspace/deepseek_v4_complete/codegraph/`，包含以下文件：

```
codegraph/
├── __init__.py            # 模块初始化文件
├── code_analyzer.py       # 代码分析器 - 解析 Python 代码
├── dependency_extractor.py # 依赖关系提取器
├── graph_generator.py     # 图表生成器
└── visualization_api.py   # 可视化 API（基于 FastAPI）
```

## 核心组件

### 1. CodeAnalyzer - 代码分析器

负责解析 Python 代码文件，提取：
- 类定义和属性
- 函数/方法定义
- 导入语句
- 文档字符串

使用示例：
```python
from deepseek_v4_complete.codegraph.code_analyzer import CodeAnalyzer

analyzer = CodeAnalyzer()
modules = analyzer.parse_directory("/path/to/your/project")
```

### 2. DependencyExtractor - 依赖关系提取器

从已解析的模块中提取依赖关系：
- 模块级依赖（import 语句）
- 类继承关系
- 循环依赖检测

使用示例：
```python
from deepseek_v4_complete.codegraph.dependency_extractor import DependencyExtractor

extractor = DependencyExtractor()
dependencies = extractor.extract_from_modules(modules)
module_deps = extractor.get_module_dependency_graph()
circular_deps = extractor.find_circular_dependencies()
```

### 3. GraphGenerator - 图表生成器

生成多种格式的可视化图表：
- **Mermaid** 格式图表（适用于 Markdown）
- **DOT** 格式图表（可使用 Graphviz 渲染）
- 文件目录树
- 项目摘要报告

使用示例：
```python
from deepseek_v4_complete.codegraph.graph_generator import GraphGenerator

generator = GraphGenerator()
mermaid_class = generator.generate_mermaid_class_diagram(modules)
dot_module = generator.generate_dot_module_graph(module_deps)
summary = generator.generate_summary_markdown(modules, module_deps)
```

### 4. CodeGraphAPI - FastAPI 服务

提供 REST API 接口，可以作为服务运行：

```python
from deepseek_v4_complete.codegraph.visualization_api import CodeGraphAPI

api = CodeGraphAPI()
api.analyze_and_generate("/path/to/project", "output.md")
# 或者启动服务器
# api.run(host="0.0.0.0", port=8000)
```

## 快速开始

### 1. 基础使用

我们已经为您创建了测试脚本，您可以直接运行：

```bash
python3 /workspace/test_codegraph.py
```

这将会：
- 分析整个 DeepSeek V4 项目
- 提取依赖关系
- 生成可视化文件

### 2. 生成可视化文件

运行测试后，您会得到以下文件：

- `project_summary.md` - 完整的项目结构分析报告
- `class_diagram.mmd` - 类继承关系的 Mermaid 图表
- `module_graph.dot` - 模块依赖的 DOT 图表

### 3. 可视化 Mermaid 图表

要查看 Mermaid 图表，您可以：
1. 使用支持 Mermaid 的 Markdown 编辑器（如 VS Code、Typora）
2. 在线查看：https://mermaid.live/
3. 使用 GitHub/GitLab 查看（它们原生支持 Mermaid）

### 4. 可视化 DOT 图表

要查看 DOT 图表，您需要安装 Graphviz：

```bash
# 安装 Graphviz（Ubuntu/Debian）
sudo apt-get install graphviz

# 或使用 pip 安装
pip install graphviz

# 渲染图表
dot -Tpng module_graph.dot -o module_graph.png
```

## API 端点（可选）

如果您需要使用 API 服务，需要先安装依赖：

```bash
pip install fastapi uvicorn
```

然后 API 提供以下端点：
- `GET /analyze?path=/your/project` - 分析项目
- `GET /dependencies/module?path=/your/project` - 获取模块依赖图
- `GET /dependencies/class?path=/your/project` - 获取类继承图
- `GET /file-tree?path=/your/project` - 获取文件目录树
- `GET /circular-deps?path=/your/project` - 检测循环依赖
- `GET /summary?path=/your/project` - 生成项目摘要

## 功能特性

✅ **代码解析** - 自动分析 Python 项目结构
✅ **依赖提取** - 识别模块和类的依赖关系
✅ **图表生成** - 支持 Mermaid 和 DOT 格式
✅ **循环依赖检测** - 查找可能的问题依赖
✅ **摘要报告** - 自动生成 Markdown 格式的项目分析
✅ **API 服务** - 提供 RESTful 接口
✅ **轻量设计** - 无复杂依赖（API 部分除外）

## 示例分析

我们已经对您提供的训练脚本进行了分析，发现：
- 5个类：RMSNorm, VisionProjector, ModelConfig, SimpleMultiModalModel, TrainDataset
- 主要依赖：torch, transformers, peft, PIL

## 下一步

1. 运行 `test_codegraph.py` 查看项目的完整分析
2. 使用生成的可视化图表理解项目结构
3. 根据需要集成到您的开发工作流中

祝您使用愉快！
