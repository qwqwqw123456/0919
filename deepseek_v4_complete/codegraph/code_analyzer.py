import ast
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any


@dataclass
class FunctionInfo:
    name: str
    line_number: int
    docstring: str = ""
    args: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    decorators: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    line_number: int
    docstring: str = ""
    base_classes: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)


@dataclass
class ModuleInfo:
    file_path: str
    file_name: str
    imports: List[str] = field(default_factory=list)
    from_imports: Dict[str, List[str]] = field(default_factory=dict)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    docstring: str = ""


class CodeAnalyzer:
    def __init__(self):
        self.parsed_modules: Dict[str, ModuleInfo] = {}
    
    def parse_file(self, file_path: str) -> ModuleInfo:
        """解析单个Python文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return ModuleInfo(file_path=file_path, file_name=os.path.basename(file_path))
        
        module_info = ModuleInfo(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            docstring=ast.get_docstring(tree) or ""
        )
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_info.imports.append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module:
                    if module not in module_info.from_imports:
                        module_info.from_imports[module] = []
                    for alias in node.names:
                        module_info.from_imports[module].append(alias.name)
            
            elif isinstance(node, ast.ClassDef):
                class_info = self._parse_class(node)
                module_info.classes.append(class_info)
            
            elif isinstance(node, ast.FunctionDef):
                func_info = self._parse_function(node)
                module_info.functions.append(func_info)
        
        self.parsed_modules[file_path] = module_info
        return module_info
    
    def _parse_class(self, node: ast.ClassDef) -> ClassInfo:
        """解析类定义"""
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(self._get_attribute_name(base))
        
        class_info = ClassInfo(
            name=node.name,
            line_number=node.lineno,
            docstring=ast.get_docstring(node) or "",
            base_classes=base_classes
        )
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._parse_function(item)
                class_info.methods.append(method_info)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    class_info.attributes.append(item.target.id)
        
        return class_info
    
    def _parse_function(self, node: ast.FunctionDef) -> FunctionInfo:
        """解析函数/方法定义"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        
        returns = None
        if node.returns:
            returns = self._get_type_name(node.returns)
        
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(self._get_attribute_name(dec))
        
        return FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            docstring=ast.get_docstring(node) or "",
            args=args,
            returns=returns,
            decorators=decorators
        )
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """获取属性节点的完整名称"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))
    
    def _get_type_name(self, node: ast.AST) -> str:
        """获取类型注解的名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Subscript):
            return self._get_type_name(node.value)
        return str(node)
    
    def parse_directory(self, dir_path: str, skip_dirs: Optional[Set[str]] = None) -> List[ModuleInfo]:
        """解析目录下所有Python文件"""
        if skip_dirs is None:
            skip_dirs = {'__pycache__', '.git', 'node_modules'}
        
        results = []
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    module_info = self.parse_file(file_path)
                    results.append(module_info)
        
        return results
    
    def get_module_by_path(self, file_path: str) -> Optional[ModuleInfo]:
        """根据路径获取已解析的模块"""
        return self.parsed_modules.get(file_path)
    
    def get_all_classes(self) -> List[ClassInfo]:
        """获取所有解析到的类"""
        all_classes = []
        for module in self.parsed_modules.values():
            all_classes.extend(module.classes)
        return all_classes
    
    def get_all_functions(self) -> List[FunctionInfo]:
        """获取所有解析到的函数"""
        all_functions = []
        for module in self.parsed_modules.values():
            all_functions.extend(module.functions)
            for cls in module.classes:
                all_functions.extend(cls.methods)
        return all_functions
