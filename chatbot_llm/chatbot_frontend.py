#1 import streamlit and chatbot file
import streamlit as st
import chatbot_backend as demo

#2 set the title for the bot
st.title("Hi, This is Chatbot Sarala! :sunglasses:")

#3 langchain memory to the session cache - Session State
if 'memory' not in st.session_state:
    st.session_state.memory = demo.demo_memory()

#4 add the UI chat history to the session cache - Session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])


input_text = st.chat_input("Chat with Amijeet's bedrock udemy course bot here")
if input_text:

    with st.chat_message("user"):
        st.markdown(input_text)

    st.session_state.chat_history.append({"role":"user", "text":input_text})

    chat_response = demo.demo_conversation(input_text = input_text, memory=st.session_state.memory)

    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append({"role":"assistant", "text":chat_response})
