from typing import Dict, List, Optional, Set
from .code_analyzer import ModuleInfo, ClassInfo, FunctionInfo


class GraphGenerator:
    def __init__(self):
        self.graph_styles = {
            'module': {'fillcolor': '#E3F2FD', 'color': '#1976D2', 'style': 'filled'},
            'class': {'fillcolor': '#E8F5E9', 'color': '#388E3C', 'style': 'filled'},
            'function': {'fillcolor': '#FFF3E0', 'color': '#F57C00', 'style': 'filled'},
            'external': {'fillcolor': '#FCE4EC', 'color': '#C2185B', 'style': 'filled,dashed'},
        }
    
    def generate_mermaid_module_graph(self, module_deps: Dict[str, List[str]], 
                                      include_external: bool = False) -> str:
        """生成模块依赖的Mermaid图表"""
        lines = ["```mermaid", "graph TD"]
        
        internal_modules = set(module_deps.keys())
        
        for source, targets in module_deps.items():
            lines.append(f"    subgraph \"{self._shorten_name(source)}\"")
            
            for target in targets:
                if include_external or target in internal_modules:
                    target_label = self._shorten_name(target)
                    lines.append(f"    {self._sanitize_name(source)} --> {self._sanitize_name(target)}")
            
            lines.append("    end")
        
        lines.append("```")
        return '\n'.join(lines)
    
    def generate_mermaid_class_diagram(self, modules: List[ModuleInfo]) -> str:
        """生成类继承关系的Mermaid图表"""
        lines = ["```mermaid", "classDiagram"]
        
        for module in modules:
            for cls in module.classes:
                cls_name = cls.name
                
                lines.append(f"    class {cls_name} {{")
                
                for attr in cls.attributes:
                    lines.append(f"        +{attr}")
                
                for method in cls.methods:
                    args_str = ", ".join(method.args)
                    return_str = f": {method.returns}" if method.returns else ""
                    lines.append(f"        +{method.name}({args_str}){return_str}")
                
                lines.append("    }")
                
                for base in cls.base_classes:
                    lines.append(f"    {base} <|-- {cls_name}")
        
        lines.append("```")
        return '\n'.join(lines)
    
    def generate_dot_module_graph(self, module_deps: Dict[str, List[str]]) -> str:
        """生成模块依赖的DOT图表"""
        lines = ["digraph module_deps {", "    rankdir=LR;"]
        
        for style_name, style in self.graph_styles.items():
            lines.append(f"    node [shape=box, style=\"{style['style']}\", "
                        f"fillcolor=\"{style['fillcolor']}\", color=\"{style['color']}\"];")
        
        internal_modules = set(module_deps.keys())
        
        for source, targets in module_deps.items():
            for target in targets:
                source_node = self._sanitize_name(source)
                target_node = self._sanitize_name(target)
                source_label = self._shorten_name(source)
                target_label = self._shorten_name(target)
                
                lines.append(f"    {source_node} [label=\"{source_label}\"];")
                lines.append(f"    {target_node} [label=\"{target_label}\"");
                if target not in internal_modules:
                    lines.append(f", style=\"filled,dashed\", fillcolor=\"#FCE4EC\"");
                lines.append("];")
                lines.append(f"    {source_node} -> {target_node};")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def generate_dot_class_hierarchy(self, class_hierarchy: Dict[str, List[str]]) -> str:
        """生成类继承层次的DOT图表"""
        lines = ["digraph class_hierarchy {", "    rankdir=BT;"]
        
        for cls_name, bases in class_hierarchy.items():
            cls_node = self._sanitize_name(cls_name)
            cls_label = cls_name.split('.')[-1]
            lines.append(f"    {cls_node} [label=\"{cls_label}\", shape=box, "
                        f"style=\"filled\", fillcolor=\"#E8F5E9\"];")
            
            for base in bases:
                base_node = self._sanitize_name(base)
                base_label = base.split('.')[-1]
                lines.append(f"    {base_node} [label=\"{base_label}\", shape=box, "
                            f"style=\"filled\", fillcolor=\"#C8E6C9\"];")
                lines.append(f"    {cls_node} -> {base_node} [arrowhead=empty];")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def generate_file_tree(self, modules: List[ModuleInfo], root_path: str = "") -> str:
        """生成文件目录树"""
        tree = {}
        
        for module in modules:
            rel_path = module.file_path
            if root_path and rel_path.startswith(root_path):
                rel_path = rel_path[len(root_path):].lstrip('/')
            
            parts = rel_path.split('/')
            current = tree
            
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        return self._render_tree(tree)
    
    def _render_tree(self, tree: Dict[str, dict], prefix: str = "") -> str:
        """递归渲染目录树"""
        lines = []
        items = sorted(tree.items())
        
        for i, (name, children) in enumerate(items):
            is_last = i == len(items) - 1
            
            if children:
                lines.append(f"{prefix}{'└── ' if is_last else '├── '}{name}/")
                new_prefix = prefix + ('    ' if is_last else '│   ')
                lines.append(self._render_tree(children, new_prefix))
            else:
                lines.append(f"{prefix}{'└── ' if is_last else '├── '}{name}")
        
        return '\n'.join(lines)
    
    def generate_summary_markdown(self, modules: List[ModuleInfo], 
                                  module_deps: Dict[str, List[str]]) -> str:
        """生成项目摘要Markdown"""
        lines = ["# 项目代码结构分析", ""]
        
        lines.append("## 概览")
        total_classes = sum(len(m.classes) for m in modules)
        total_functions = sum(len(m.functions) + sum(len(c.methods) for c in m.classes) 
                            for m in modules)
        lines.append(f"- 模块数量: {len(modules)}")
        lines.append(f"- 类数量: {total_classes}")
        lines.append(f"- 函数/方法数量: {total_functions}")
        lines.append("")
        
        lines.append("## 文件结构")
        lines.append("```")
        lines.append(self.generate_file_tree(modules))
        lines.append("```")
        lines.append("")
        
        lines.append("## 模块依赖图")
        lines.append(self.generate_mermaid_module_graph(module_deps))
        lines.append("")
        
        lines.append("## 类继承图")
        lines.append(self.generate_mermaid_class_diagram(modules))
        
        return '\n'.join(lines)
    
    def _sanitize_name(self, name: str) -> str:
        """清理名称以用于图表节点"""
        return name.replace('.', '_').replace('/', '_').replace('-', '_')
    
    def _shorten_name(self, name: str, max_len: int = 20) -> str:
        """缩短名称以适应图表显示"""
        parts = name.split('.')
        if len(parts) > 2:
            return '.'.join(parts[-2:])
        if len(name) > max_len:
            return name[:max_len-3] + '...'
        return name
