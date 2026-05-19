from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
import os
import json

from .code_analyzer import CodeAnalyzer, ModuleInfo
from .dependency_extractor import DependencyExtractor
from .graph_generator import GraphGenerator


class AnalysisResult(BaseModel):
    total_modules: int
    total_classes: int
    total_functions: int
    modules: List[Dict[str, Any]]


class DependencyGraphResult(BaseModel):
    format: str
    content: str


class CodeGraphAPI:
    def __init__(self):
        self.app = FastAPI(title="CodeGraph API", description="代码图可视化服务")
        self.analyzer = CodeAnalyzer()
        self.extractor = DependencyExtractor()
        self.generator = GraphGenerator()
        self._setup_routes()
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.get("/analyze", response_model=AnalysisResult, 
                     summary="分析项目代码结构")
        def analyze_project(path: str = Query(..., description="项目路径")):
            try:
                if not os.path.exists(path):
                    raise HTTPException(status_code=404, detail="路径不存在")
                
                modules = self.analyzer.parse_directory(path)
                total_classes = sum(len(m.classes) for m in modules)
                total_functions = sum(len(m.functions) + sum(len(c.methods) for c in m.classes) 
                                    for m in modules)
                
                module_list = []
                for m in modules:
                    module_list.append({
                        "file_path": m.file_path,
                        "file_name": m.file_name,
                        "classes": [c.name for c in m.classes],
                        "functions": [f.name for f in m.functions],
                        "imports": m.imports
                    })
                
                return AnalysisResult(
                    total_modules=len(modules),
                    total_classes=total_classes,
                    total_functions=total_functions,
                    modules=module_list
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dependencies/module", response_model=DependencyGraphResult,
                     summary="获取模块依赖图")
        def get_module_dependencies(path: str = Query(..., description="项目路径"),
                                    format: str = Query("mermaid", description="输出格式: mermaid/dot")):
            try:
                if not os.path.exists(path):
                    raise HTTPException(status_code=404, detail="路径不存在")
                
                modules = self.analyzer.parse_directory(path)
                self.extractor.extract_from_modules(modules)
                module_deps = self.extractor.get_module_dependency_graph()
                
                if format == "mermaid":
                    content = self.generator.generate_mermaid_module_graph(module_deps)
                elif format == "dot":
                    content = self.generator.generate_dot_module_graph(module_deps)
                else:
                    raise HTTPException(status_code=400, detail="不支持的格式")
                
                return DependencyGraphResult(format=format, content=content)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dependencies/class", response_model=DependencyGraphResult,
                     summary="获取类继承图")
        def get_class_hierarchy(path: str = Query(..., description="项目路径"),
                                format: str = Query("mermaid", description="输出格式: mermaid/dot")):
            try:
                if not os.path.exists(path):
                    raise HTTPException(status_code=404, detail="路径不存在")
                
                modules = self.analyzer.parse_directory(path)
                
                if format == "mermaid":
                    content = self.generator.generate_mermaid_class_diagram(modules)
                elif format == "dot":
                    self.extractor.extract_from_modules(modules)
                    class_hierarchy = self.extractor.get_class_hierarchy()
                    content = self.generator.generate_dot_class_hierarchy(class_hierarchy)
                else:
                    raise HTTPException(status_code=400, detail="不支持的格式")
                
                return DependencyGraphResult(format=format, content=content)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/file-tree", summary="获取文件目录树")
        def get_file_tree(path: str = Query(..., description="项目路径")):
            try:
                if not os.path.exists(path):
                    raise HTTPException(status_code=404, detail="路径不存在")
                
                modules = self.analyzer.parse_directory(path)
                tree = self.generator.generate_file_tree(modules, path)
                
                return {"tree": tree}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/circular-deps", summary="检测循环依赖")
        def detect_circular_dependencies(path: str = Query(..., description="项目路径")):
            try:
                if not os.path.exists(path):
                    raise HTTPException(status_code=404, detail="路径不存在")
                
                modules = self.analyzer.parse_directory(path)
                self.extractor.extract_from_modules(modules)
                cycles = self.extractor.find_circular_dependencies()
                
                return {"circular_dependencies": [{"source": c[0], "target": c[1]} for c in cycles]}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/summary", summary="生成项目摘要")
        def generate_summary(path: str = Query(..., description="项目路径")):
            try:
                if not os.path.exists(path):
                    raise HTTPException(status_code=404, detail="路径不存在")
                
                modules = self.analyzer.parse_directory(path)
                self.extractor.extract_from_modules(modules)
                module_deps = self.extractor.get_module_dependency_graph()
                
                summary = self.generator.generate_summary_markdown(modules, module_deps)
                
                return {"summary": summary}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """启动API服务"""
        uvicorn.run(self.app, host=host, port=port)
    
    def analyze_and_generate(self, project_path: str, output_file: Optional[str] = None) -> str:
        """分析项目并生成可视化内容"""
        modules = self.analyzer.parse_directory(project_path)
        self.extractor.extract_from_modules(modules)
        module_deps = self.extractor.get_module_dependency_graph()
        
        summary = self.generator.generate_summary_markdown(modules, module_deps)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
        
        return summary


if __name__ == "__main__":
    api = CodeGraphAPI()
    api.run()
