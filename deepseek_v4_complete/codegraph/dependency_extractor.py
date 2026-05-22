import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from .code_analyzer import ModuleInfo, ClassInfo


@dataclass
class Dependency:
    source: str
    target: str
    dependency_type: str
    line_number: int


class DependencyExtractor:
    def __init__(self):
        self.dependencies: List[Dependency] = []
        self.module_deps: Dict[str, Set[str]] = defaultdict(set)
        self.class_deps: Dict[str, Set[str]] = defaultdict(set)
    
    def extract_from_modules(self, modules: List[ModuleInfo]) -> List[Dependency]:
        """从模块列表中提取依赖关系"""
        self.dependencies = []
        
        for module in modules:
            self._extract_module_dependencies(module)
            self._extract_class_dependencies(module)
        
        return self.dependencies
    
    def _extract_module_dependencies(self, module: ModuleInfo):
        """提取模块级别的依赖"""
        module_name = self._get_module_name(module.file_path)
        
        for imp in module.imports:
            self._add_dependency(module_name, imp, "module_import", 0)
        
        for from_module, items in module.from_imports.items():
            self._add_dependency(module_name, from_module, "from_import", 0)
    
    def _extract_class_dependencies(self, module: ModuleInfo):
        """提取类级别的依赖"""
        module_name = self._get_module_name(module.file_path)
        
        for cls in module.classes:
            class_full_name = f"{module_name}.{cls.name}"
            
            for base in cls.base_classes:
                self._add_dependency(class_full_name, base, "inheritance", cls.line_number)
                self.class_deps[class_full_name].add(base)
    
    def _add_dependency(self, source: str, target: str, dep_type: str, line_number: int):
        """添加依赖关系"""
        if source != target:
            self.dependencies.append(Dependency(
                source=source,
                target=target,
                dependency_type=dep_type,
                line_number=line_number
            ))
            self.module_deps[source].add(target)
    
    def _get_module_name(self, file_path: str) -> str:
        """从文件路径获取模块名称"""
        parts = []
        path = file_path
        
        while path and path != '/':
            path, tail = os.path.split(path)
            if tail == '__init__.py':
                continue
            if tail.endswith('.py'):
                tail = tail[:-3]
            parts.append(tail)
        
        return '.'.join(reversed(parts))
    
    def get_dependencies_by_type(self, dep_type: str) -> List[Dependency]:
        """按类型筛选依赖"""
        return [d for d in self.dependencies if d.dependency_type == dep_type]
    
    def get_module_dependency_graph(self) -> Dict[str, List[str]]:
        """获取模块依赖图"""
        graph = {}
        for source, targets in self.module_deps.items():
            graph[source] = list(targets)
        return graph
    
    def get_class_hierarchy(self) -> Dict[str, List[str]]:
        """获取类继承层次"""
        hierarchy = {}
        for source, targets in self.class_deps.items():
            hierarchy[source] = list(targets)
        return hierarchy
    
    def find_circular_dependencies(self) -> List[Tuple[str, str]]:
        """查找循环依赖"""
        cycles = []
        for source, targets in self.module_deps.items():
            for target in targets:
                if target in self.module_deps and source in self.module_deps[target]:
                    if (target, source) not in cycles:
                        cycles.append((source, target))
        return cycles
    
    def get_transitive_dependencies(self, module_name: str) -> Set[str]:
        """获取模块的传递依赖"""
        visited = set()
        queue = [module_name]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current in self.module_deps:
                for dep in self.module_deps[current]:
                    if dep not in visited:
                        queue.append(dep)
        
        visited.discard(module_name)
        return visited
