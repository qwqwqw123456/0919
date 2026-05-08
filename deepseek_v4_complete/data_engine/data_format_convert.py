import json
import csv
import yaml
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

class DataFormatConverter:
    def __init__(self):
        pass
    
    def json_to_dict(self, json_str: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    def dict_to_json(self, data: Dict[str, Any], indent: int = 2) -> str:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    
    def csv_to_list(self, csv_str: str, delimiter: str = ',') -> List[Dict[str, Any]]:
        lines = csv_str.strip().split('\n')
        if not lines:
            return []
        
        reader = csv.DictReader(lines, delimiter=delimiter)
        return list(reader)
    
    def list_to_csv(self, data: List[Dict[str, Any]], delimiter: str = ',') -> str:
        if not data:
            return ''
        
        fieldnames = data[0].keys()
        output = []
        output.append(delimiter.join(fieldnames))
        
        for row in data:
            values = []
            for field in fieldnames:
                value = row.get(field, '')
                if isinstance(value, str) and delimiter in value:
                    value = f'"{value}"'
                values.append(str(value))
            output.append(delimiter.join(values))
        
        return '\n'.join(output)
    
    def yaml_to_dict(self, yaml_str: str) -> Optional[Dict[str, Any]]:
        try:
            return yaml.safe_load(yaml_str)
        except yaml.YAMLError:
            return None
    
    def dict_to_yaml(self, data: Dict[str, Any]) -> str:
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)
    
    def xml_to_dict(self, xml_str: str) -> Optional[Dict[str, Any]]:
        try:
            root = ET.fromstring(xml_str)
            return self._xml_element_to_dict(root)
        except ET.ParseError:
            return None
    
    def _xml_element_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        result = {}
        
        if element.attrib:
            result.update(element.attrib)
        
        if element.text and element.text.strip():
            result['_text'] = element.text.strip()
        
        children = {}
        for child in element:
            child_dict = self._xml_element_to_dict(child)
            if child.tag in children:
                if not isinstance(children[child.tag], list):
                    children[child.tag] = [children[child.tag]]
                children[child.tag].append(child_dict)
            else:
                children[child.tag] = child_dict
        
        if children:
            result.update(children)
        
        return {element.tag: result}
    
    def dict_to_xml(self, data: Dict[str, Any], root_tag: str = 'root') -> str:
        root = ET.Element(root_tag)
        self._dict_to_xml_element(data, root)
        return ET.tostring(root, encoding='unicode')
    
    def _dict_to_xml_element(self, data: Dict[str, Any], parent: ET.Element):
        for key, value in data.items():
            if key == '_text':
                parent.text = value
            elif isinstance(value, dict):
                child = ET.SubElement(parent, key)
                self._dict_to_xml_element(value, child)
            elif isinstance(value, list):
                for item in value:
                    child = ET.SubElement(parent, key)
                    if isinstance(item, dict):
                        self._dict_to_xml_element(item, child)
                    else:
                        child.text = str(item)
            else:
                child = ET.SubElement(parent, key)
                child.text = str(value)
    
    def read_file(self, file_path: Union[str, Path]) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    def write_file(self, file_path: Union[str, Path], content: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def convert_file(self, input_path: Union[str, Path], output_path: Union[str, Path], 
                     input_format: str = None, output_format: str = None):
        input_format = input_format or Path(input_path).suffix[1:].lower()
        output_format = output_format or Path(output_path).suffix[1:].lower()
        
        content = self.read_file(input_path)
        if content is None:
            raise ValueError(f"Cannot read file: {input_path}")
        
        data = None
        
        if input_format == 'json':
            data = self.json_to_dict(content)
        elif input_format == 'csv':
            data = self.csv_to_list(content)
        elif input_format == 'yaml' or input_format == 'yml':
            data = self.yaml_to_dict(content)
        elif input_format == 'xml':
            data = self.xml_to_dict(content)
        
        if data is None:
            raise ValueError(f"Unsupported input format: {input_format}")
        
        output_content = None
        
        if output_format == 'json':
            output_content = self.dict_to_json(data)
        elif output_format == 'csv':
            if isinstance(data, list):
                output_content = self.list_to_csv(data)
            else:
                raise ValueError("Cannot convert non-list data to CSV")
        elif output_format == 'yaml' or output_format == 'yml':
            output_content = self.dict_to_yaml(data)
        elif output_format == 'xml':
            output_content = self.dict_to_xml(data)
        
        if output_content is None:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        self.write_file(output_path, output_content)
    
    def format_to_huggingface(self, data: List[Dict[str, Any]], 
                              text_key: str = 'text',
                              label_key: str = None) -> List[Dict[str, Any]]:
        result = []
        for item in data:
            formatted = {'text': item.get(text_key, '')}
            if label_key and label_key in item:
                formatted['label'] = item[label_key]
            result.append(formatted)
        return result
    
    def huggingface_to_format(self, data: List[Dict[str, Any]],
                             text_key: str = 'text',
                             label_key: str = 'label') -> List[Dict[str, Any]]:
        result = []
        for item in data:
            formatted = {text_key: item.get('text', '')}
            if 'label' in item:
                formatted[label_key] = item['label']
            result.append(formatted)
        return result