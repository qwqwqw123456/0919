import gradio as gr
def chat(message, history):
    return "reply"
gr.ChatInterface(chat).launch()
