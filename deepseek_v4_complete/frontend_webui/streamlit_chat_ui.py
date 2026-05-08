import streamlit as st
st.title("DeepSeek V4 Chat")
user_input = st.text_input("Your message:")
if user_input:
    st.write(f"Assistant: {user_input[::-1]}")
