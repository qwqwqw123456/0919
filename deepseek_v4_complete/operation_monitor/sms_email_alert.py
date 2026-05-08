"""
短信/邮件告警 - SMS and Email Alert
支持多种告警通道：邮件、短信、Webhook、企业微信、钉钉等
提供告警模板、告警分级、告警抑制、告警聚合等功能
"""

import os
import sys
import time
import json
import smtplib
import threading
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
import asyncio


class AlertLevel(Enum):
    """告警级别"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    
    def __str__(self):
        return self.name
    
    @property
    def color(self) -> str:
        """获取级别对应的颜色"""
        colors = {
            AlertLevel.DEBUG: "#808080",
            AlertLevel.INFO: "#00FF00",
            AlertLevel.WARNING: "#FFFF00",
            AlertLevel.ERROR: "#FF0000",
            AlertLevel.CRITICAL: "#FF00FF"
        }
        return colors.get(self, "#FFFFFF")


class AlertChannel(Enum):
    """告警通道"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    WECHAT = "wechat"
    DINGTALK = "dingtalk"
    FEISHU = "feishu"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    CONSOLE = "console"


@dataclass
class Alert:
    """告警信息"""
    alert_id: str = ""
    title: str = ""
    message: str = ""
    level: AlertLevel = AlertLevel.INFO
    channel: AlertChannel = AlertChannel.EMAIL
    
    source: str = ""
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    comparison: str = ">"
    
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    recipients: List[str] = field(default_factory=list)
    cc_recipients: List[str] = field(default_factory=list)
    
    occurred_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    is_resolved: bool = False
    is_sent: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'title': self.title,
            'message': self.message,
            'level': self.level.value,
            'level_name': self.level.name,
            'channel': self.channel.value,
            'source': self.source,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'comparison': self.comparison,
            'tags': self.tags,
            'metadata': self.metadata,
            'recipients': self.recipients,
            'cc_recipients': self.cc_recipients,
            'occurred_at': self.occurred_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'is_resolved': self.is_resolved,
            'is_sent': self.is_sent,
            'retry_count': self.retry_count
        }
    
    @property
    def summary(self) -> str:
        """告警摘要"""
        if self.metric_name:
            return f"{self.title} [{self.metric_name}={self.metric_value:.2f} {self.comparison} {self.threshold}]"
        return self.title
    
    @property
    def severity_emoji(self) -> str:
        """级别对应的表情符号"""
        emojis = {
            AlertLevel.DEBUG: "",
            AlertLevel.INFO: "",
            AlertLevel.WARNING: "",
            AlertLevel.ERROR: "",
            AlertLevel.CRITICAL: ""
        }
        return emojis.get(self.level, "")


@dataclass
class AlertTemplate:
    """告警模板"""
    name: str
    title_template: str
    message_template: str
    level: AlertLevel = AlertLevel.INFO
    channel: AlertChannel = AlertChannel.EMAIL
    
    def render(self, alert: Alert) -> tuple:
        """渲染模板"""
        context = {
            'alert_id': alert.alert_id,
            'title': alert.title,
            'message': alert.message,
            'level': alert.level.name,
            'level_emoji': alert.severity_emoji,
            'source': alert.source,
            'metric_name': alert.metric_name,
            'metric_value': alert.metric_value,
            'threshold': alert.threshold,
            'comparison': alert.comparison,
            'occurred_at': alert.occurred_at.strftime('%Y-%m-%d %H:%M:%S'),
            'tags': ', '.join([f"{k}={v}" for k, v in alert.tags.items()])
        }
        context.update(alert.metadata)
        
        title = self.title_template.format(**context)
        message = self.message_template.format(**context)
        
        return title, message


class AlertBackend(ABC):
    """告警发送后端抽象基类"""
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """发送告警"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置"""
        pass


class EmailBackend(AlertBackend):
    """邮件告警后端"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._smtp_host = self._config.get('smtp_host', 'localhost')
        self._smtp_port = self._config.get('smtp_port', 587)
        self._smtp_user = self._config.get('smtp_user', '')
        self._smtp_password = self._config.get('smtp_password', '')
        self._use_tls = self._config.get('use_tls', True)
        self._from_addr = self._config.get('from_addr', self._smtp_user)
        self._logger = logging.getLogger(__name__)
    
    def validate_config(self) -> bool:
        """验证邮件配置"""
        return bool(self._smtp_host and self._from_addr)
    
    def send(self, alert: Alert) -> bool:
        """发送邮件告警"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.level.name}] {alert.title}"
            msg['From'] = self._from_addr
            msg['To'] = ', '.join(alert.recipients) if alert.recipients else self._from_addr
            
            if alert.cc_recipients:
                msg['Cc'] = ', '.join(alert.cc_recipients)
            
            plain_text = self._render_plain_text(alert)
            html_content = self._render_html(alert)
            
            msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            recipients = alert.recipients.copy()
            recipients.extend(alert.cc_recipients)
            
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                if self._use_tls:
                    server.starttls()
                if self._smtp_user and self._smtp_password:
                    server.login(self._smtp_user, self._smtp_password)
                server.sendmail(self._from_addr, recipients, msg.as_string())
            
            self._logger.info(f"邮件告警已发送: {alert.title}")
            return True
        except Exception as e:
            self._logger.error(f"邮件发送失败: {e}")
            return False
    
    def _render_plain_text(self, alert: Alert) -> str:
        """渲染纯文本告警"""
        lines = [
            f"告警级别: {alert.level.name}",
            f"告警时间: {alert.occurred_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"告警标题: {alert.title}",
            "",
            "告警详情:",
            alert.message,
            ""
        ]
        
        if alert.metric_name:
            lines.extend([
                f"监控指标: {alert.metric_name}",
                f"当前值: {alert.metric_value}",
                f"阈值: {alert.comparison} {alert.threshold}",
                ""
            ])
        
        if alert.tags:
            lines.append("标签:")
            for k, v in alert.tags.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
        
        lines.append(f"告警ID: {alert.alert_id}")
        
        return '\n'.join(lines)
    
    def _render_html(self, alert: Alert) -> str:
        """渲染HTML告警"""
        level_colors = {
            AlertLevel.DEBUG: "#808080",
            AlertLevel.INFO: "#00FF00",
            AlertLevel.WARNING: "#FFA500",
            AlertLevel.ERROR: "#FF4500",
            AlertLevel.CRITICAL: "#FF0000"
        }
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {level_colors.get(alert.level, '#333')}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
                .metric {{ background-color: #e8f4e8; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .footer {{ margin-top: 20px; color: #666; font-size: 12px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .label {{ font-weight: bold; width: 120px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{alert.level.severity_emoji} {alert.title}</h2>
                <p>告警级别: <strong>{alert.level.name}</strong></p>
                <p>告警时间: {alert.occurred_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="content">
                <h3>告警详情</h3>
                <p>{alert.message}</p>
        """
        
        if alert.metric_name:
            html += f"""
                <div class="metric">
                    <h4>监控指标</h4>
                    <table>
                        <tr><td class="label">指标名称</td><td>{alert.metric_name}</td></tr>
                        <tr><td class="label">当前值</td><td><strong>{alert.metric_value}</strong></td></tr>
                        <tr><td class="label">阈值</td><td>{alert.comparison} {alert.threshold}</td></tr>
                    </table>
                </div>
            """
        
        if alert.tags:
            html += "<h4>标签</h4><ul>"
            for k, v in alert.tags.items():
                html += f"<li><strong>{k}</strong>: {v}</li>"
            html += "</ul>"
        
        html += f"""
            </div>
            <div class="footer">
                <p>告警ID: {alert.alert_id}</p>
                <p>来源: {alert.source or 'Unknown'}</p>
            </div>
        </body>
        </html>
        """
        
        return html


class SMSBackend(AlertBackend):
    """短信告警后端"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._provider = self._config.get('provider', 'aliyun')
        self._api_key = self._config.get('api_key', '')
        self._api_secret = self._config.get('api_secret', '')
        self._sign_name = self._config.get('sign_name', '')
        self._template_code = self._config.get('template_code', '')
        self._logger = logging.getLogger(__name__)
    
    def validate_config(self) -> bool:
        """验证短信配置"""
        return bool(self._api_key and self._sign_name and self._template_code)
    
    def send(self, alert: Alert) -> bool:
        """发送短信告警"""
        try:
            message = self._format_message(alert)
            
            if self._provider == 'aliyun':
                return self._send_aliyun(message)
            elif self._provider == 'twilio':
                return self._send_twilio(message)
            elif self._provider == 'yunpian':
                return self._send_yunpian(message)
            else:
                self._logger.warning(f"未知的短信提供商: {self._provider}")
                return False
        except Exception as e:
            self._logger.error(f"短信发送失败: {e}")
            return False
    
    def _format_message(self, alert: Alert) -> str:
        """格式化短信内容"""
        message = f"[{alert.level.name}] {alert.title}"
        if alert.metric_name:
            message += f" {alert.metric_name}={alert.metric_value}"
        return message[:70]
    
    def _send_aliyun(self, message: str) -> bool:
        """通过阿里云发送短信"""
        try:
            import aliyunsms
            client = aliyunsms.Client(self._api_key, self._api_secret)
            for phone in alert.recipients:
                result = client.send_sms(phone, self._sign_name, self._template_code, {'message': message})
                if not result.get('success'):
                    self._logger.error(f"短信发送失败: {result}")
                    return False
            return True
        except ImportError:
            self._logger.warning("阿里云SDK未安装")
            return False
    
    def _send_twilio(self, message: str) -> bool:
        """通过Twilio发送短信"""
        try:
            from twilio.rest import Client
            client = Client(self._api_key, self._api_secret)
            for phone in alert.recipients:
                client.messages.create(body=message, from_=self._config.get('from_phone'), to=phone)
            return True
        except ImportError:
            self._logger.warning("Twilio SDK未安装")
            return False
    
    def _send_yunpian(self, message: str) -> bool:
        """通过云片发送短信"""
        try:
            import requests
            url = "https://sms.yunpian.com/v2/sms/single_send.json"
            for phone in alert.recipients:
                data = {
                    'apikey': self._api_key,
                    'mobile': phone,
                    'text': message
                }
                response = requests.post(url, data=data)
                result = response.json()
                if result.get('code') != 0:
                    self._logger.error(f"短信发送失败: {result}")
                    return False
            return True
        except ImportError:
            self._logger.warning("requests库未安装")
            return False


class WebhookBackend(AlertBackend):
    """Webhook告警后端"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._webhook_url = self._config.get('webhook_url', '')
        self._webhook_secret = self._config.get('webhook_secret', '')
        self._timeout = self._config.get('timeout', 10)
        self._logger = logging.getLogger(__name__)
    
    def validate_config(self) -> bool:
        """验证Webhook配置"""
        return bool(self._webhook_url)
    
    def send(self, alert: Alert) -> bool:
        """发送Webhook告警"""
        try:
            import requests
            
            payload = self._format_payload(alert)
            headers = {
                'Content-Type': 'application/json'
            }
            
            if self._webhook_secret:
                import hmac
                import hashlib
                signature = hmac.new(
                    self._webhook_secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                headers['X-Signature'] = signature
            
            response = requests.post(
                self._webhook_url,
                json=payload,
                headers=headers,
                timeout=self._timeout
            )
            
            if response.status_code < 400:
                self._logger.info(f"Webhook告警已发送: {alert.title}")
                return True
            else:
                self._logger.error(f"Webhook返回错误: {response.status_code}")
                return False
        except Exception as e:
            self._logger.error(f"Webhook发送失败: {e}")
            return False
    
    def _format_payload(self, alert: Alert) -> Dict[str, Any]:
        """格式化Webhook载荷"""
        return {
            'alert_id': alert.alert_id,
            'title': alert.title,
            'message': alert.message,
            'level': alert.level.name,
            'level_value': alert.level.value,
            'source': alert.source,
            'metric_name': alert.metric_name,
            'metric_value': alert.metric_value,
            'threshold': alert.threshold,
            'tags': alert.tags,
            'occurred_at': alert.occurred_at.isoformat(),
            'metadata': alert.metadata
        }


class DingTalkBackend(AlertBackend):
    """钉钉告警后端"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._webhook_url = self._config.get('webhook_url', '')
        self._secret = self._config.get('secret', '')
        self._logger = logging.getLogger(__name__)
    
    def validate_config(self) -> bool:
        """验证钉钉配置"""
        return bool(self._webhook_url)
    
    def send(self, alert: Alert) -> bool:
        """发送钉钉告警"""
        try:
            import requests
            import base64
            import hmac
            import hashlib
            import time
            import urllib.parse
            
            timestamp = str(round(time.time() * 1000))
            secret_enc = self._secret.encode('utf-8')
            string_to_sign = f'{timestamp}\n{self._secret}'
            string_to_sign_enc = string_to_sign.encode('utf-8')
            sign = base64.b64encode(hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()).decode('utf-8')
            sign = urllib.parse.quote_plus(sign)
            
            url = f"{self._webhook_url}&timestamp={timestamp}&sign={sign}"
            
            level_colors = {
                AlertLevel.WARNING: "warning",
                AlertLevel.ERROR: "red",
                AlertLevel.CRITICAL: "red"
            }
            
            payload = {
                'msgtype': 'markdown',
                'markdown': {
                    'title': f"[{alert.level.name}] {alert.title}",
                    'text': self._format_markdown(alert)
                }
            }
            
            response = requests.post(url, json=payload)
            result = response.json()
            
            if result.get('errcode') == 0:
                self._logger.info(f"钉钉告警已发送: {alert.title}")
                return True
            else:
                self._logger.error(f"钉钉发送失败: {result}")
                return False
        except Exception as e:
            self._logger.error(f"钉钉告警发送失败: {e}")
            return False
    
    def _format_markdown(self, alert: Alert) -> str:
        """格式化Markdown内容"""
        md = f"### {alert.severity_emoji} {alert.title}\n\n"
        md += f"**告警级别**: {alert.level.name}\n\n"
        md += f"**告警时间**: {alert.occurred_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += f"**告警详情**: {alert.message}\n\n"
        
        if alert.metric_name:
            md += f"**监控指标**: {alert.metric_name}\n\n"
            md += f"- 当前值: `{alert.metric_value}`\n"
            md += f"- 阈值: {alert.comparison} `{alert.threshold}`\n\n"
        
        if alert.tags:
            md += "**标签**:\n\n"
            for k, v in alert.tags.items():
                md += f"- {k}: {v}\n"
        
        md += f"\n**告警ID**: `{alert.alert_id}`"
        
        return md


class WeChatBackend(AlertBackend):
    """企业微信告警后端"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._webhook_url = self._config.get('webhook_url', '')
        self._logger = logging.getLogger(__name__)
    
    def validate_config(self) -> bool:
        """验证企业微信配置"""
        return bool(self._webhook_url)
    
    def send(self, alert: Alert) -> bool:
        """发送企业微信告警"""
        try:
            import requests
            
            payload = {
                'msgtype': 'markdown',
                'markdown': {
                    'content': self._format_markdown(alert)
                }
            }
            
            response = requests.post(self._webhook_url, json=payload)
            result = response.json()
            
            if result.get('errcode') == 0:
                self._logger.info(f"企业微信告警已发送: {alert.title}")
                return True
            else:
                self._logger.error(f"企业微信发送失败: {result}")
                return False
        except Exception as e:
            self._logger.error(f"企业微信告警发送失败: {e}")
            return False
    
    def _format_markdown(self, alert: Alert) -> str:
        """格式化Markdown内容"""
        md = f"<strong>{alert.severity_emoji} {alert.title}</strong>\n\n"
        md += f"> 告警级别: <font color=\"red\">{alert.level.name}</font>\n\n"
        md += f"> 告警时间: {alert.occurred_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += f"> {alert.message}\n\n"
        
        if alert.metric_name:
            md += f"**监控指标**: {alert.metric_name}\n"
            md += f"- 当前值: {alert.metric_value}\n"
            md += f"- 阈值: {alert.comparison} {alert.threshold}\n"
        
        return md


class FeishuBackend(AlertBackend):
    """飞书告警后端"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._webhook_url = self._config.get('webhook_url', '')
        self._logger = logging.getLogger(__name__)
    
    def validate_config(self) -> bool:
        """验证飞书配置"""
        return bool(self._webhook_url)
    
    def send(self, alert: Alert) -> bool:
        """发送飞书告警"""
        try:
            import requests
            
            payload = {
                'msg_type': 'interactive',
                'card': self._format_card(alert)
            }
            
            response = requests.post(self._webhook_url, json=payload)
            result = response.json()
            
            if result.get('code') == 0:
                self._logger.info(f"飞书告警已发送: {alert.title}")
                return True
            else:
                self._logger.error(f"飞书发送失败: {result}")
                return False
        except Exception as e:
            self._logger.error(f"飞书告警发送失败: {e}")
            return False
    
    def _format_card(self, alert: Alert) -> Dict[str, Any]:
        """格式化卡片内容"""
        header_color = {
            AlertLevel.WARNING: "yellow",
            AlertLevel.ERROR: "red",
            AlertLevel.CRITICAL: "red"
        }.get(alert.level, "blue")
        
        elements = [
            {
                'tag': 'div',
                'text': {
                    'tag': 'lark_md',
                    'content': f"**告警级别**: {alert.level.name}\n\n**告警时间**: {alert.occurred_at.strftime('%Y-%m-%d %H:%M:%S')}"
                }
            },
            {
                'tag': 'hr'
            },
            {
                'tag': 'div',
                'text': {
                    'tag': 'lark_md',
                    'content': alert.message
                }
            }
        ]
        
        if alert.metric_name:
            elements.append({
                'tag': 'div',
                'text': {
                    'tag': 'lark_md',
                    'content': f"**监控指标**: {alert.metric_name}\n\n- 当前值: `{alert.metric_value}`\n- 阈值: {alert.comparison} `{alert.threshold}`"
                }
            })
        
        return {
            'header': {
                'title': {
                    'tag': 'plain_text',
                    'content': f"{alert.severity_emoji} {alert.title}"
                },
                'template': header_color
            },
            'elements': elements
        }


class ConsoleBackend(AlertBackend):
    """控制台告警后端（用于测试）"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._use_color = self._config.get('use_color', True)
        self._logger = logging.getLogger(__name__)
    
    def validate_config(self) -> bool:
        return True
    
    def send(self, alert: Alert) -> bool:
        """输出到控制台"""
        if self._use_color:
            color_codes = {
                AlertLevel.DEBUG: '\033[90m',
                AlertLevel.INFO: '\033[92m',
                AlertLevel.WARNING: '\033[93m',
                AlertLevel.ERROR: '\033[91m',
                AlertLevel.CRITICAL: '\033[95m',
                'RESET': '\033[0m'
            }
            color = color_codes.get(alert.level, color_codes['RESET'])
            reset = color_codes['RESET']
            
            print(f"{color}[{alert.level.name}]{reset} [{alert.occurred_at.strftime('%H:%M:%S')}] {alert.title}")
            print(f"  {alert.message}")
            if alert.metric_name:
                print(f"  {alert.metric_name}={alert.metric_value} {alert.comparison} {alert.threshold}")
        else:
            print(f"[{alert.level.name}] [{alert.occurred_at.strftime('%H:%M:%S')}] {alert.title}")
            print(f"  {alert.message}")
        
        return True


class AlertSender:
    """
    告警发送器
    
    功能特性：
    - 多通道支持（邮件、短信、Webhook、钉钉、企业微信、飞书等）
    - 告警模板渲染
    - 告警分级和过滤
    - 告警抑制和聚合
    - 重试机制
    - 告警历史记录
    """
    
    _instance: Optional['AlertSender'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._config = config or {}
        
        self._backends: Dict[AlertChannel, AlertBackend] = {}
        self._templates: Dict[str, AlertTemplate] = {}
        self._alert_history: List[Alert] = []
        self._max_history = self._config.get('max_history', 1000)
        
        self._rate_limiter: Dict[str, List[datetime]] = defaultdict(list)
        self._rate_limit_window = self._config.get('rate_limit_window', 300)
        self._rate_limit_max = self._config.get('rate_limit_max', 10)
        
        self._callbacks: List[Callable] = []
        self._logger = logging.getLogger(__name__)
        
        self._init_backends()
        self._init_default_templates()
    
    def _init_backends(self):
        """初始化告警后端"""
        channel_configs = self._config.get('channels', {})
        
        if 'email' in channel_configs:
            self._backends[AlertChannel.EMAIL] = EmailBackend(channel_configs['email'])
        
        if 'sms' in channel_configs:
            self._backends[AlertChannel.SMS] = SMSBackend(channel_configs['sms'])
        
        if 'webhook' in channel_configs:
            self._backends[AlertChannel.WEBHOOK] = WebhookBackend(channel_configs['webhook'])
        
        if 'dingtalk' in channel_configs:
            self._backends[AlertChannel.DINGTALK] = DingTalkBackend(channel_configs['dingtalk'])
        
        if 'wechat' in channel_configs:
            self._backends[AlertChannel.WECHAT] = WeChatBackend(channel_configs['wechat'])
        
        if 'feishu' in channel_configs:
            self._backends[AlertChannel.FEISHU] = FeishuBackend(channel_configs['feishu'])
        
        if not self._backends or self._config.get('enable_console', False):
            self._backends[AlertChannel.CONSOLE] = ConsoleBackend()
    
    def _init_default_templates(self):
        """初始化默认模板"""
        self._templates['default'] = AlertTemplate(
            name='default',
            title_template='[{level_emoji}{level}] {title}',
            message_template='{message}\n\n时间: {occurred_at}\n指标: {metric_name}={metric_value} {comparison} {threshold}'
        )
        
        self._templates['critical'] = AlertTemplate(
            name='critical',
            title_template='🚨 {title}',
            message_template='【严重告警】\n\n{level_emoji}{message}\n\n发生时间: {occurred_at}\n\n请立即处理！',
            level=AlertLevel.CRITICAL
        )
    
    def _generate_alert_id(self) -> str:
        """生成告警ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"ALT_{timestamp}"
    
    def _check_rate_limit(self, key: str) -> bool:
        """检查速率限制"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self._rate_limit_window)
        
        self._rate_limiter[key] = [
            t for t in self._rate_limiter[key] if t > cutoff
        ]
        
        if len(self._rate_limiter[key]) >= self._rate_limit_max:
            return False
        
        self._rate_limiter[key].append(now)
        return True
    
    def send(self, message: str,
             level: AlertLevel = AlertLevel.INFO,
             channel: AlertChannel = AlertChannel.EMAIL,
             title: str = "",
             recipients: Optional[List[str]] = None,
             **kwargs) -> bool:
        """
        发送告警
        
        Args:
            message: 告警消息
            level: 告警级别
            channel: 告警通道
            title: 告警标题
            recipients: 接收人列表
            **kwargs: 其他参数
        
        Returns:
            是否发送成功
        """
        alert_id = self._generate_alert_id()
        
        alert = Alert(
            alert_id=alert_id,
            title=title or message[:50],
            message=message,
            level=level,
            channel=channel,
            recipients=recipients or [],
            **kwargs
        )
        
        return self._send_alert(alert)
    
    def _send_alert(self, alert: Alert) -> bool:
        """发送告警"""
        rate_limit_key = f"{alert.channel.value}:{alert.level.name}"
        if not self._check_rate_limit(rate_limit_key):
            self._logger.warning(f"告警被速率限制: {alert.title}")
            return False
        
        backend = self._backends.get(alert.channel)
        if not backend:
            self._logger.error(f"未配置的告警通道: {alert.channel}")
            return False
        
        max_retries = self._config.get('max_retries', 3)
        for attempt in range(max_retries):
            if backend.send(alert):
                alert.is_sent = True
                self._add_to_history(alert)
                for callback in self._callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        self._logger.error(f"告警回调执行失败: {e}")
                return True
            
            alert.retry_count = attempt + 1
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        
        self._logger.error(f"告警发送失败 (已重试{max_retries}次): {alert.title}")
        return False
    
    def send_alert(self, alert: Alert) -> bool:
        """发送告警对象"""
        return self._send_alert(alert)
    
    def send_email(self, message: str, title: str = "",
                   recipients: Optional[List[str]] = None,
                   level: AlertLevel = AlertLevel.INFO) -> bool:
        """发送邮件告警"""
        return self.send(message, level, AlertChannel.EMAIL, title, recipients)
    
    def send_sms(self, message: str,
                 recipients: Optional[List[str]] = None,
                 level: AlertLevel = AlertLevel.WARNING) -> bool:
        """发送短信告警"""
        return self.send(message, level, AlertChannel.SMS, recipients=recipients)
    
    def send_webhook(self, message: str, title: str = "",
                     level: AlertLevel = AlertLevel.INFO) -> bool:
        """发送Webhook告警"""
        return self.send(message, level, AlertChannel.WEBHOOK, title)
    
    def send_dingtalk(self, message: str, title: str = "",
                      level: AlertLevel = AlertLevel.INFO) -> bool:
        """发送钉钉告警"""
        return self.send(message, level, AlertChannel.DINGTALK, title)
    
    def send_wechat(self, message: str, title: str = "",
                   level: AlertLevel = AlertLevel.INFO) -> bool:
        """发送企业微信告警"""
        return self.send(message, level, AlertChannel.WECHAT, title)
    
    def send_feishu(self, message: str, title: str = "",
                   level: AlertLevel = AlertLevel.INFO) -> bool:
        """发送飞书告警"""
        return self.send(message, level, AlertChannel.FEISHU, title)
    
    def send_console(self, message: str, title: str = "",
                    level: AlertLevel = AlertLevel.INFO) -> bool:
        """发送控制台告警"""
        return self.send(message, level, AlertChannel.CONSOLE, title)
    
    def _add_to_history(self, alert: Alert):
        """添加到历史记录"""
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]
    
    def get_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        return self._alert_history[-limit:]
    
    def get_recent_alerts(self, level: Optional[AlertLevel] = None,
                         channel: Optional[AlertChannel] = None,
                         limit: int = 100) -> List[Alert]:
        """获取最近的告警"""
        alerts = self._alert_history
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        if channel:
            alerts = [a for a in alerts if a.channel == channel]
        
        return alerts[-limit:]
    
    def register_template(self, template: AlertTemplate):
        """注册告警模板"""
        self._templates[template.name] = template
    
    def register_callback(self, callback: Callable):
        """注册告警回调"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable):
        """取消注册回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def set_rate_limit(self, window: int, max_alerts: int):
        """设置速率限制"""
        self._rate_limit_window = window
        self._rate_limit_max = max_alerts
    
    def get_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        total = len(self._alert_history)
        by_level = defaultdict(int)
        by_channel = defaultdict(int)
        
        for alert in self._alert_history:
            by_level[alert.level.name] += 1
            by_channel[alert.channel.value] += 1
        
        return {
            'total': total,
            'sent': sum(1 for a in self._alert_history if a.is_sent),
            'failed': sum(1 for a in self._alert_history if not a.is_sent and a.retry_count > 0),
            'by_level': dict(by_level),
            'by_channel': dict(by_channel)
        }
    
    def clear_history(self):
        """清空告警历史"""
        self._alert_history.clear()
    
    def close(self):
        """关闭发送器"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_sender(config: Optional[Dict[str, Any]] = None) -> AlertSender:
    """
    获取告警发送器单例
    
    Args:
        config: 配置字典
    
    Returns:
        AlertSender实例
    """
    return AlertSender(config)


class AlertDecorator:
    """告警装饰器"""
    
    def __init__(self, sender: Optional[AlertSender] = None,
                 level: AlertLevel = AlertLevel.ERROR,
                 channel: AlertChannel = AlertChannel.CONSOLE):
        self._sender = sender or AlertSender()
        self._level = level
        self._channel = channel
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self._sender.send(
                    message=f"函数 {func.__name__} 执行失败: {str(e)}",
                    title=f"函数执行错误: {func.__name__}",
                    level=self._level,
                    channel=self._channel,
                    metadata={
                        'function': func.__name__,
                        'args': str(args)[:100],
                        'kwargs': str(kwargs)[:100]
                    }
                )
                raise
        return wrapper


def alert_on_error(level: AlertLevel = AlertLevel.ERROR,
                  channel: AlertChannel = AlertChannel.CONSOLE):
    """
    错误告警装饰器
    
    Args:
        level: 告警级别
        channel: 告警通道
    
    Usage:
        @alert_on_error()
        def my_function():
            ...
    """
    return AlertDecorator(level=level, channel=channel)


__all__ = [
    'AlertLevel',
    'AlertChannel',
    'Alert',
    'AlertTemplate',
    'AlertBackend',
    'EmailBackend',
    'SMSBackend',
    'WebhookBackend',
    'DingTalkBackend',
    'WeChatBackend',
    'FeishuBackend',
    'ConsoleBackend',
    'AlertSender',
    'AlertDecorator',
    'alert_on_error',
    'get_sender'
]
