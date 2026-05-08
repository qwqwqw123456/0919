"""
DeepSeek V4 Gradio 聊天应用
支持流式输出、自定义 CSS、多轮对话等高级功能
"""

import gradio as gr
import json
import time
import base64
from datetime import datetime
from typing import List, Dict, Optional, Iterator, Generator
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekGradioChat:
    """Gradio 聊天应用主类"""

    def __init__(self):
        """初始化聊天应用配置"""
        self.api_url = "http://localhost:8000/api/chat/stream"
        self.model_params = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "stream": True
        }
        self.conversation_history: List[Dict] = []
        self.system_prompt = "你是一个有帮助的AI助手，请用中文回答用户的问题。"

    def get_custom_css(self) -> str:
        """
        获取自定义 CSS 样式

        返回:
            自定义 CSS 样式字符串
        """
        return """
        /* 全局样式 */
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }

        /* 标题样式 */
        .main-title {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        /* 聊天消息样式 */
        .user-message {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 20px 20px 5px 20px !important;
            padding: 12px 18px !important;
            max-width: 80% !important;
            margin: 8px 0 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        }

        .bot-message {
            background-color: #E3F2FD !important;
            color: #333 !important;
            border-radius: 20px 20px 20px 5px !important;
            padding: 12px 18px !important;
            max-width: 80% !important;
            margin: 8px 0 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        }

        /* 输入框样式 */
        textarea {
            border-radius: 25px !important;
            border: 2px solid #667eea !important;
            padding: 15px !important;
            font-size: 16px !important;
        }

        textarea:focus {
            border-color: #764ba2 !important;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.5) !important;
        }

        /* 按钮样式 */
        .submit-btn, .clear-btn {
            border-radius: 25px !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
        }

        .submit-btn:hover, .clear-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        /* 参数滑块样式 */
        .slider-container {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }

        /* 图片上传区域 */
        .image-preview {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
        }

        /* 对话历史 */
        .chat-history {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
        }

        /* 统计信息 */
        .stats-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 10px 0;
        }

        /* 加载动画 */
        .loading {
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Markdown 内容样式 */
        .bot-message p {
            margin: 8px 0;
            line-height: 1.6;
        }

        .bot-message code {
            background-color: #f5f5f5;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }

        .bot-message pre {
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }

        /* 滚动条样式 */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        """

    def _simulate_streaming_response(self, message: str) -> Generator[str, None, None]:
        """
        模拟流式响应（实际项目中应替换为真实的 API 调用）

        参数:
            message: 用户输入的消息

        返回:
            生成器，逐步返回响应内容
        """
        template_responses = [
            f"好的，让我来回答您的问题：{message}\n\n",
            f"关于您提到的{message}，我的理解是：\n\n",
            f"感谢您的提问！针对\"{message}\"这个问题，\n\n",
        ]

        response = template_responses[hash(message) % len(template_responses)]
        response += f"这是一个详细的回答。首先，{message}涉及到很多方面。\n\n"
        response += "1. **第一点**：这个问题可以从多个角度来分析。\n"
        response += "2. **第二点**：需要考虑不同的应用场景。\n"
        response += "3. **第三点**：实际应用中需要结合具体情况。\n\n"
        response += "```python\n"
        response += "# 示例代码\n"
        response += "def example():\n"
        response += "    return 'Hello, DeepSeek!'\n"
        response += "```\n\n"
        response += "综上所述，我建议您可以这样理解和处理这个问题。"

        for char in response:
            yield char
            time.sleep(0.02)

    def _call_api_streaming(self, messages: List[Dict]) -> Generator[str, None, None]:
        """
        调用 DeepSeek API 获取流式响应

        参数:
            messages: 对话历史消息列表

        返回:
            生成器，逐步返回响应内容
        """
        try:
            import requests

            payload = {
                "messages": messages,
                **self.model_params
            }

            response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=60
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'content' in data:
                                yield data['content']
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"API请求失败: {response.status_code}"

        except ImportError:
            logger.warning("requests 库未安装，使用模拟响应")
            for chunk in self._simulate_streaming_response(messages[-1]["content"]):
                yield chunk

        except Exception as e:
            logger.error(f"API 调用错误: {e}")
            for chunk in self._simulate_streaming_response(messages[-1]["content"]):
                yield chunk

    def chat_streaming(
        self,
        message: str,
        history: List[List[str]],
        images: Optional[List] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9
    ) -> Generator[str, None, None]:
        """
        处理聊天请求并返回流式响应

        参数:
            message: 用户输入的消息
            history: 对话历史 [[用户消息, 助手回复], ...]
            images: 上传的图片列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            top_p: top_p 参数

        返回:
            生成器，返回流式响应
        """
        self.model_params["temperature"] = temperature
        self.model_params["max_tokens"] = max_tokens
        self.model_params["top_p"] = top_p

        history.append([message, ""])
        full_response = ""

        messages = self._build_messages(history, images)

        try:
            for chunk in self._call_api_streaming(messages):
                full_response += chunk
                history[-1][1] = full_response
                yield history, ""

        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            logger.error(error_msg)
            history[-1][1] = error_msg
            yield history, ""

    def _build_messages(self, history: List[List[str]], images: Optional[List] = None) -> List[Dict]:
        """
        构建 API 消息格式

        参数:
            history: 对话历史
            images: 图片列表

        返回:
            消息列表
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        for user_msg, bot_msg in history[:-1]:
            messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

        if history:
            last_user_msg = history[-1][0]
            if images:
                messages.append({
                    "role": "user",
                    "content": last_user_msg,
                    "images": [self._process_image(img) for img in images]
                })
            else:
                messages.append({"role": "user", "content": last_user_msg})

        return messages

    def _process_image(self, image_data) -> str:
        """
        处理图片为 base64 格式

        参数:
            image_data: 图片数据

        返回:
            base64 编码的图片字符串
        """
        try:
            if hasattr(image_data, 'read'):
                image_data.seek(0)
                img_bytes = image_data.read()
            else:
                img_bytes = image_data

            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/png;base64,{img_b64}"
        except Exception as e:
            logger.error(f"图片处理失败: {e}")
            return ""

    def get_statistics(self, history: List[List[str]]) -> Dict[str, any]:
        """
        获取对话统计信息

        参数:
            history: 对话历史

        返回:
            统计信息字典
        """
        total_messages = len(history)
        total_words = sum(len(msg[0].split()) + len(msg[1].split()) for msg in history)
        total_chars = sum(len(msg[0]) + len(msg[1]) for msg in history)
        estimated_tokens = total_words // 4

        return {
            "total_conversations": total_messages,
            "total_words": total_words,
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens
        }

    def export_conversation(self, history: List[List[str]]) -> str:
        """
        导出会话历史为 JSON 格式

        参数:
            history: 对话历史

        返回:
            JSON 格式的对话内容
        """
        export_data = {
            "export_time": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "model_params": self.model_params,
            "conversations": history
        }
        return json.dumps(export_data, ensure_ascii=False, indent=2)

    def clear_history(self) -> List[List[str]]:
        """
        清空对话历史

        返回:
            空的历史列表
        """
        self.conversation_history = []
        return []

    def update_system_prompt(self, new_prompt: str):
        """
        更新系统提示词

        参数:
            new_prompt: 新的系统提示词
        """
        self.system_prompt = new_prompt


def create_gradio_interface() -> gr.Blocks:
    """
    创建 Gradio 界面

    返回:
        Gradio Blocks 应用实例
    """
    chat_app = DeepSeekGradioChat()

    with gr.Blocks(
        title="DeepSeek V4 Chat",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
        ),
        css=chat_app.get_custom_css()
    ) as demo:

        gr.Markdown("""
        <div class="main-title">
            <h1>🤖 DeepSeek V4 智能对话助手</h1>
            <p>支持多模态交互的智能 AI 助手</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group():
                    chatbot = gr.Chatbot(
                        label="💬 对话历史",
                        bubble_full_width=False,
                        height=500,
                        show_copy_button=True,
                        avatar_images=("👤", "🤖")
                    )

                    with gr.Row():
                        with gr.Column(scale=8):
                            msg_input = gr.Textbox(
                                label="✏️ 输入您的消息",
                                placeholder="请输入您的问题...",
                                lines=3,
                                max_lines=10
                            )

                        with gr.Column(scale=2):
                            submit_btn = gr.Button(
                                "🚀 发送",
                                variant="primary",
                                size="lg"
                            )

                    with gr.Row():
                        with gr.Column(scale=8):
                            image_input = gr.File(
                                label="📷 上传图片（可选）",
                                file_count="multiple",
                                file_types=["image"]
                            )
                        with gr.Column(scale=2):
                            clear_btn = gr.Button(
                                "🗑️ 清空对话",
                                variant="secondary"
                            )

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### ⚙️ 参数设置")

                    with gr.Group():
                        temperature = gr.Slider(
                            label="🌡️ Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            info="控制输出的随机性"
                        )

                        max_tokens = gr.Slider(
                            label="📝 最大 Token 数",
                            minimum=256,
                            maximum=8192,
                            value=2048,
                            step=256,
                            info="生成的最大 token 数量"
                        )

                        top_p = gr.Slider(
                            label="🎯 Top P",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            info="核采样参数"
                        )

                    gr.Markdown("### 📤 导出功能")
                    with gr.Group():
                        export_btn = gr.Button(
                            "💾 导出会话",
                            size="sm"
                        )
                        export_output = gr.File(
                            label="导出的文件",
                            visible=False
                        )

                    gr.Markdown("### 📊 统计信息")
                    stats_display = gr.JSON(
                        label="当前会话统计",
                        value=chat_app.get_statistics([])
                    )

                    gr.Markdown("### 🔧 系统设置")
                    with gr.Group():
                        system_prompt = gr.Textbox(
                            label="系统提示词",
                            value=chat_app.system_prompt,
                            lines=3,
                            max_lines=5
                        )
                        update_prompt_btn = gr.Button(
                            "✅ 更新提示词",
                            size="sm"
                        )

                    gr.Markdown("""
                    ### ℹ️ 使用说明

                    1. **发送消息**: 在文本框输入内容，点击发送按钮
                    2. **上传图片**: 支持上传多张图片，会附加到消息中
                    3. **调整参数**: 侧边栏可调整模型生成参数
                    4. **导出对话**: 点击导出按钮下载 JSON 格式对话记录
                    5. **清空对话**: 点击清空按钮重置对话历史

                    ### ⌨️ 快捷键

                    - `Enter`: 发送消息
                    - `Shift + Enter`: 换行
                    """)

        def respond(
            message: str,
            history: List[List[str]],
            images: Optional[List],
            temp: float,
            tokens: int,
            top: float
        ):
            """
            处理用户输入并生成响应

            参数:
                message: 用户消息
                history: 对话历史
                images: 图片列表
                temp: 温度参数
                tokens: 最大 token 数
                top: top_p 参数

            返回:
                更新后的历史和空消息
            """
            if not message.strip():
                return history, ""

            response_generator = chat_app.chat_streaming(
                message, history, images, temp, tokens, top
            )

            for updated_history, _ in response_generator:
                yield updated_history, ""

        def update_stats(history: List[List[str]]):
            """更新统计信息"""
            return chat_app.get_statistics(history)

        def update_system_prompt(prompt: str):
            """更新系统提示词"""
            chat_app.update_system_prompt(prompt)
            return gr.Textbox(value=prompt)

        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, image_input, temperature, max_tokens, top_p],
            outputs=[chatbot, msg_input]
        )

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, image_input, temperature, max_tokens, top_p],
            outputs=[chatbot, msg_input]
        )

        clear_btn.click(
            fn=chat_app.clear_history,
            inputs=[],
            outputs=[chatbot]
        )

        export_btn.click(
            fn=chat_app.export_conversation,
            inputs=[chatbot],
            outputs=[export_output]
        )

        chatbot.change(
            fn=update_stats,
            inputs=[chatbot],
            outputs=[stats_display]
        )

        update_prompt_btn.click(
            fn=update_system_prompt,
            inputs=[system_prompt],
            outputs=[system_prompt]
        )

    return demo


def main():
    """应用入口函数"""
    demo = create_gradio_interface()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=False
    )


if __name__ == "__main__":
    main()
