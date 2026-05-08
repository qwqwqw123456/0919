"""
DeepSeek V4 前端 Web UI 模块
提供 Streamlit 和 Gradio 两种聊天界面
"""

from .streamlit_chat_ui import StreamlitChatUI, main as run_streamlit
from .gradio_chat_app import DeepSeekGradioChat, create_gradio_interface, main as run_gradio

__all__ = [
    'StreamlitChatUI',
    'run_streamlit',
    'DeepSeekGradioChat',
    'create_gradio_interface',
    'run_gradio',
]

__version__ = '1.0.0'
