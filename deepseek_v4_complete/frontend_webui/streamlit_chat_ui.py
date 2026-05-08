"""
DeepSeek V4 Streamlit 聊天界面
支持多轮对话、图片上传、流式输出等高级功能
"""

import streamlit as st
import time
import base64
import json
from datetime import datetime
from typing import List, Dict, Optional, Generator
import io
from PIL import Image


class StreamlitChatUI:
    """Streamlit 聊天界面主类"""

    def __init__(self):
        """初始化聊天界面配置"""
        self._init_page_config()
        self._init_session_state()

    def _init_page_config(self):
        """配置页面基本信息"""
        st.set_page_config(
            page_title="DeepSeek V4 Chat",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def _init_session_state(self):
        """初始化会话状态变量"""
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []

        if "uploaded_images" not in st.session_state:
            st.session_state.uploaded_images = []

        if "streaming_enabled" not in st.session_state:
            st.session_state.streaming_enabled = True

        if "model_params" not in st.session_state:
            st.session_state.model_params = {
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9
            }

        if "theme" not in st.session_state:
            st.session_state.theme = "light"

    def render_sidebar(self):
        """渲染侧边栏配置面板"""
        with st.sidebar:
            st.title("⚙️ 设置")

            st.subheader("🤖 模型参数")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.model_params["temperature"],
                step=0.1,
                help="控制输出的随机性，值越高输出越随机"
            )
            max_tokens = st.slider(
                "最大 Token 数",
                min_value=256,
                max_value=8192,
                value=st.session_state.model_params["max_tokens"],
                step=256,
                help="生成的最大token数量"
            )
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.model_params["top_p"],
                step=0.05,
                help="核采样参数"
            )

            st.session_state.model_params["temperature"] = temperature
            st.session_state.model_params["max_tokens"] = max_tokens
            st.session_state.model_params["top_p"] = top_p

            st.subheader("🎨 界面设置")
            st.session_state.streaming_enabled = st.checkbox(
                "启用流式输出",
                value=st.session_state.streaming_enabled,
                help="是否启用打字机效果的流式输出"
            )

            st.subheader("📤 上传图片")
            uploaded_file = st.file_uploader(
                "上传图片（支持多张）",
                type=["png", "jpg", "jpeg", "gif", "webp"],
                accept_multiple_files=True,
                help="上传的图片会附加到下一条消息中"
            )

            if uploaded_file:
                for file in uploaded_file:
                    if file not in st.session_state.uploaded_images:
                        st.session_state.uploaded_images.append(file)
                        st.success(f"已上传: {file.name}")

            if st.session_state.uploaded_images:
                st.write("已上传的图片：")
                cols = st.columns(3)
                for idx, img_file in enumerate(st.session_state.uploaded_images):
                    with cols[idx % 3]:
                        image = Image.open(img_file)
                        st.image(image, width=100, caption=img_file.name)
                        if st.button("🗑️", key=f"del_{idx}"):
                            st.session_state.uploaded_images.remove(img_file)
                            st.rerun()

            st.divider()

            st.subheader("📋 会话管理")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ 清空对话", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.conversation_history = []
                    st.rerun()

            with col2:
                if st.button("💾 导出对话", use_container_width=True):
                    self._export_conversation()

            st.divider()

            st.subheader("ℹ️ 使用说明")
            st.markdown("""
            1. **发送消息**: 在下方输入框输入内容并发送
            2. **上传图片**: 在侧边栏上传图片，将附加到消息中
            3. **多轮对话**: 系统自动维护对话上下文
            4. **流式输出**: 实时显示AI回复，无需等待完成
            5. **调整参数**: 侧边栏可调整模型生成参数
            """)

            with st.expander("🔧 高级设置"):
                st.number_input(
                    "系统提示词",
                    value="你是一个有帮助的AI助手",
                    key="system_prompt"
                )

    def _simulate_streaming_response(self, message: str) -> Generator[str, None, None]:
        """
        模拟流式响应（实际项目中应替换为真实的API调用）
        生成器函数，逐字 yield 响应内容
        """
        template_responses = [
            f"好的，让我来回答您的问题：{message}\n\n",
            f"关于您提到的{message}，我的理解是：\n\n",
            f"感谢您的提问！针对\"{message}\"这个问题，\n\n",
        ]

        response = template_responses[hash(message) % len(template_responses)]
        response += f"这是一个详细的回答。首先，{message}涉及到很多方面。\n\n"
        response += "1. 第一点：这个问题可以从多个角度来分析。\n"
        response += "2. 第二点：需要考虑不同的应用场景。\n"
        response += "3. 第三点：实际应用中需要结合具体情况。\n\n"
        response += "综上所述，我建议您可以这样理解和处理这个问题。"

        for char in response:
            yield char
            time.sleep(0.02)

    def _call_deepseek_api(self, messages: List[Dict], images: List = None) -> Generator[str, None, None]:
        """
        调用 DeepSeek API 获取流式响应

        参数:
            messages: 对话历史消息列表
            images: 可选的图片列表

        返回:
            生成器，逐步返回响应内容
        """
        try:
            api_url = "http://localhost:8000/api/chat/stream"
            import requests

            payload = {
                "messages": messages,
                "images": images,
                **st.session_state.model_params
            }

            with st.spinner("正在等待响应..."):
                response = requests.post(
                    api_url,
                    json=payload,
                    stream=True,
                    timeout=60
                )

                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            if 'content' in data:
                                yield data['content']
                else:
                    yield f"API请求失败: {response.status_code}"

        except requests.exceptions.ConnectionError:
            for chunk in self._simulate_streaming_response(messages[-1]["content"]):
                yield chunk
        except Exception as e:
            yield f"发生错误: {str(e)}"

    def _process_image_for_api(self, image_file) -> Optional[str]:
        """
        处理图片，转换为 base64 编码

        参数:
            image_file: 图片文件对象

        返回:
            base64 编码的图片字符串
        """
        try:
            image = Image.open(image_file)
            buffered = io.BytesIO()
            image.save(buffered, format=image.format or "PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/{image.format.lower()};base64,{img_str}"
        except Exception as e:
            st.error(f"图片处理失败: {e}")
            return None

    def _format_message_for_api(self, role: str, content: str, images: List = None) -> Dict:
        """
        格式化消息为 API 所需格式

        参数:
            role: 消息角色 (user/assistant/system)
            content: 消息内容
            images: 可选的图片列表

        返回:
            格式化后的消息字典
        """
        message = {"role": role, "content": content}
        if images:
            message["images"] = images
        return message

    def _export_conversation(self):
        """导出对话历史为 JSON 文件"""
        if not st.session_state.messages:
            st.warning("没有可导出的对话内容")
            return

        conversation_data = {
            "export_time": datetime.now().isoformat(),
            "model_params": st.session_state.model_params,
            "messages": st.session_state.messages
        }

        json_str = json.dumps(conversation_data, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()

        st.download_button(
            label="下载对话记录",
            data=json_str,
            file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    def render_chat_messages(self):
        """渲染聊天消息列表"""
        for idx, message in enumerate(st.session_state.messages):
            role = message["role"]
            content = message["content"]
            images = message.get("images", [])

            with st.chat_message(role, avatar=self._get_avatar(role)):
                if images:
                    cols = st.columns([1, 3])
                    with cols[0]:
                        for img in images:
                            st.image(img, width=150)
                    with cols[1]:
                        st.markdown(content)
                else:
                    st.markdown(content)

                if idx > 0:
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("🔄", key=f"regen_{idx}", help="重新生成"):
                            self._regenerate_response(idx)
                    with col2:
                        if st.button("📋", key=f"copy_{idx}", help="复制"):
                            self._copy_to_clipboard(content)

    def _get_avatar(self, role: str) -> str:
        """根据角色返回对应的头像标识"""
        avatars = {
            "user": "👤",
            "assistant": "🤖",
            "system": "⚙️"
        }
        return avatars.get(role, "💬")

    def _regenerate_response(self, message_idx: int):
        """重新生成指定位置的回复"""
        if message_idx > 0 and st.session_state.messages[message_idx - 1]["role"] == "user":
            st.session_state.messages = st.session_state.messages[:message_idx - 1]
            st.rerun()

    def _copy_to_clipboard(self, text: str):
        """将文本复制到剪贴板"""
        st.session_state.clipboard_text = text
        st.success("已复制到剪贴板！")

    def render_input_area(self):
        """渲染输入区域"""
        prompt = st.chat_input(
            "输入您的问题...",
            key="chat_input",
            accept_file=True
        )

        if prompt:
            self._handle_user_input(prompt)

    def _handle_user_input(self, user_input: str):
        """
        处理用户输入

        参数:
            user_input: 用户输入的文本
        """
        images = []
        for img_file in st.session_state.uploaded_images:
            img = Image.open(img_file)
            images.append(img)

        user_message = {
            "role": "user",
            "content": user_input,
            "images": images,
            "timestamp": datetime.now().isoformat()
        }

        st.session_state.messages.append(user_message)
        st.session_state.conversation_history.append(user_message)
        st.session_state.uploaded_images = []

        with st.chat_message("user", avatar="👤"):
            if images:
                cols = st.columns([1, 3])
                with cols[0]:
                    for img in images:
                        st.image(img, width=150)
                with cols[1]:
                    st.markdown(user_input)
            else:
                st.markdown(user_input)

        self._generate_assistant_response(user_input, images)

    def _generate_assistant_response(self, user_input: str, images: List):
        """
        生成助手回复

        参数:
            user_input: 用户输入
            images: 用户上传的图片列表
        """
        with st.chat_message("assistant", avatar="🤖"):
            if st.session_state.streaming_enabled:
                full_response = st.write_stream(
                    self._call_deepseek_api(
                        self._build_conversation_history(),
                        [self._process_image_for_api(img) for img in images] if images else None
                    )
                )
            else:
                with st.spinner("AI 正在思考..."):
                    response_generator = self._call_deepseek_api(
                        self._build_conversation_history(),
                        [self._process_image_for_api(img) for img in images] if images else None
                    )
                    full_response = ""
                    placeholder = st.empty()
                    for chunk in response_generator:
                        full_response += chunk
                        placeholder.markdown(full_response)

            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            st.session_state.conversation_history.append(assistant_message)

    def _build_conversation_history(self) -> List[Dict]:
        """构建对话历史用于 API 调用"""
        history = []
        for msg in st.session_state.messages:
            history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return history

    def render_header(self):
        """渲染页面头部"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🤖 DeepSeek V4 智能对话助手")
            st.markdown("*支持多模态交互的智能 AI 助手*")

    def render_statistics(self):
        """渲染统计信息"""
        if st.session_state.messages:
            user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
            assistant_msgs = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总消息数", len(st.session_state.messages))
            with col2:
                st.metric("用户消息", user_msgs)
            with col3:
                st.metric("AI 回复", assistant_msgs)
            with col4:
                total_tokens = sum(
                    len(m["content"].split()) for m in st.session_state.messages
                )
                st.metric("估计 Token", total_tokens)

    def run(self):
        """运行主应用"""
        self.render_sidebar()
        self.render_header()
        self.render_statistics()

        st.divider()

        self.render_chat_messages()
        self.render_input_area()


def main():
    """应用入口函数"""
    app = StreamlitChatUI()
    app.run()


if __name__ == "__main__":
    main()
