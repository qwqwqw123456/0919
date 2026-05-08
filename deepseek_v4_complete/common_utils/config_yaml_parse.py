"""
YAML 配置文件解析工具模块

提供完整的 YAML 配置文件的加载、保存、合并、验证等功能
支持嵌套配置、环境变量替换、配置验证等高级特性
"""

import os
import yaml
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from copy import deepcopy


class ConfigParser:
    """
    YAML 配置文件解析器类

    提供完整的配置解析功能，支持：
    - 基础加载和保存
    - 配置合并和覆盖
    - 环境变量替换
    - 配置验证
    - 配置模板
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置解析器

        Args:
            config_path: 配置文件路径，可选
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._original_config: Dict[str, Any] = {}
        if config_path and os.path.exists(config_path):
            self.load(config_path)

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        从文件加载 YAML 配置

        Args:
            path: 配置文件路径，如果为 None 则使用初始化时的路径

        Returns:
            加载的配置字典

        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML 解析错误
        """
        target_path = path or self.config_path
        if not target_path:
            raise ValueError("未指定配置文件路径")

        target_path = os.path.expanduser(target_path)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"配置文件不存在: {target_path}")

        with open(target_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}
        self._original_config = deepcopy(self._config)
        self.config_path = target_path
        return self._config

    def save(self, path: Optional[str] = None, default_flow_style: bool = False) -> None:
        """
        将配置保存到 YAML 文件

        Args:
            path: 保存路径，如果为 None 则使用加载时的路径
            default_flow_style: 是否使用简化的 YAML 格式（单行格式）
        """
        target_path = path or self.config_path
        if not target_path:
            raise ValueError("未指定保存路径")

        target_path = os.path.expanduser(target_path)
        os.makedirs(os.path.dirname(target_path) or '.', exist_ok=True)

        with open(target_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self._config,
                f,
                default_flow_style=default_flow_style,
                allow_unicode=True,
                sort_keys=False
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键

        Args:
            key: 配置键，支持嵌套（如 'model.learning_rate'）
            default: 默认值

        Returns:
            配置值，如果键不存在则返回默认值
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """
        设置配置值，支持点号分隔的嵌套键

        Args:
            key: 配置键，支持嵌套（如 'model.learning_rate'）
            value: 要设置的值
        """
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any], deep: bool = True) -> None:
        """
        批量更新配置

        Args:
            updates: 要更新的配置字典
            deep: 是否进行深度合并（True）或浅合并（False）
        """
        if deep:
            self._deep_update(self._config, updates)
        else:
            self._config.update(updates)

    def _deep_update(self, base: Dict, updates: Dict) -> None:
        """深度合并更新配置"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def merge(self, other_config: Union[Dict[str, Any], 'ConfigParser'], 
              priority: str = 'other') -> None:
        """
        合并另一个配置

        Args:
            other_config: 要合并的配置字典或 ConfigParser 对象
            priority: 优先级，'other' 表示 other_config 优先，'self' 表示 self 优先
        """
        if isinstance(other_config, ConfigParser):
            other_dict = other_config._config
        else:
            other_dict = other_config

        if priority == 'other':
            self._deep_update(self._config, other_dict)
        else:
            temp = deepcopy(other_dict)
            self._deep_update(temp, self._config)
            self._config = temp

    def resolve_env_vars(self, prefix: str = '${', suffix: str = '}') -> None:
        """
        替换配置中的环境变量

        格式：${VAR_NAME} 或 ${VAR_NAME:default_value}

        Args:
            prefix: 环境变量前缀
            suffix: 环境变量后缀
        """
        self._config = self._resolve_env_vars_recursive(self._config, prefix, suffix)

    def _resolve_env_vars_recursive(self, obj: Any, prefix: str, suffix: str) -> Any:
        """递归解析环境变量"""
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars_recursive(v, prefix, suffix) 
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars_recursive(item, prefix, suffix) 
                    for item in obj]
        elif isinstance(obj, str):
            return self._replace_env_var(obj, prefix, suffix)
        return obj

    def _replace_env_var(self, value: str, prefix: str, suffix: str) -> str:
        """替换字符串中的环境变量"""
        if prefix not in value:
            return value

        result = value
        start_idx = result.find(prefix)
        while start_idx != -1:
            end_idx = result.find(suffix, start_idx)
            if end_idx == -1:
                break

            expr = result[start_idx + len(prefix):end_idx]
            if ':' in expr:
                var_name, default_value = expr.split(':', 1)
            else:
                var_name, default_value = expr, ''

            env_value = os.environ.get(var_name.strip(), default_value.strip())
            result = result[:start_idx] + env_value + result[end_idx + len(suffix):]
            start_idx = result.find(prefix, start_idx + len(env_value))

        return result

    def validate(self, schema: Dict[str, Any]) -> List[str]:
        """
        验证配置是否符合 schema

        Args:
            schema: 验证模式字典

        Returns:
            错误列表，如果为空则表示验证通过
        """
        errors = []
        self._validate_recursive(self._config, schema, '', errors)
        return errors

    def _validate_recursive(self, config: Any, schema: Any, path: str, errors: List[str]) -> None:
        """递归验证配置"""
        if not isinstance(schema, dict):
            return

        required = schema.get('required', [])
        properties = schema.get('properties', {})

        for key in required:
            if self.get(f"{path}.{key}" if path else key) is None:
                errors.append(f"缺少必需的配置项: {path}.{key}" if path else f"缺少必需的配置项: {key}")

        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            if key in properties:
                expected_type = properties[key].get('type')
                if expected_type and not self._check_type(value, expected_type):
                    errors.append(f"类型错误 [{current_path}]: 期望 {expected_type}, 实际 {type(value).__name__}")

                if isinstance(value, dict) and 'properties' in properties[key]:
                    self._validate_recursive(value, properties[key], current_path, errors)

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """检查值类型"""
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)

    def to_dict(self) -> Dict[str, Any]:
        """返回配置的深拷贝字典"""
        return deepcopy(self._config)

    def to_json(self, indent: int = 2) -> str:
        """返回 JSON 格式的配置字符串"""
        return json.dumps(self._config, indent=indent, ensure_ascii=False)

    def reset(self) -> None:
        """重置配置为原始加载状态"""
        self._config = deepcopy(self._original_config)

    @property
    def config(self) -> Dict[str, Any]:
        """获取当前配置字典"""
        return self._config


def load_config(path: str, resolve_env: bool = True) -> Dict[str, Any]:
    """
    便捷函数：从文件加载 YAML 配置

    Args:
        path: 配置文件路径
        resolve_env: 是否自动解析环境变量

    Returns:
        配置字典

    Example:
        >>> config = load_config('config.yaml')
        >>> learning_rate = config['model']['learning_rate']
    """
    parser = ConfigParser(path)
    if resolve_env:
        parser.resolve_env_vars()
    return parser.config


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    便捷函数：保存配置到 YAML 文件

    Args:
        config: 配置字典
        path: 保存路径
    """
    parser = ConfigParser()
    parser._config = config
    parser.save(path)


def merge_configs(*configs: Dict[str, Any], priority: str = 'last') -> Dict[str, Any]:
    """
    合并多个配置字典

    Args:
        *configs: 要合并的配置字典
        priority: 优先级，'last' 表示后面的优先，'first' 表示前面的优先

    Returns:
        合并后的配置字典
    """
    if priority == 'last':
        configs = reversed(configs)

    result = {}
    for config in configs:
        result = _deep_merge(result, config)
    return result


def _deep_merge(base: Dict, update: Dict) -> Dict:
    """深度合并两个字典"""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_config_template(schema: Dict[str, Any], output_path: str) -> None:
    """
    根据 schema 创建配置模板文件

    Args:
        schema: 配置模式字典
        output_path: 输出路径
    """
    template = _create_template_from_schema(schema)
    save_config(template, output_path)


def _create_template_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """根据 schema 创建模板"""
    template = {}
    properties = schema.get('properties', {})

    for key, prop in properties.items():
        if 'properties' in prop:
            template[key] = _create_template_from_schema(prop)
        elif 'default' in prop:
            template[key] = prop['default']
        elif prop.get('type') == 'string':
            template[key] = ''
        elif prop.get('type') == 'number' or prop.get('type') == 'integer':
            template[key] = 0
        elif prop.get('type') == 'boolean':
            template[key] = False
        elif prop.get('type') == 'array':
            template[key] = []
        elif prop.get('type') == 'object':
            template[key] = {}

    return template
