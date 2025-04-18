import streamlit as st
import requests

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something here")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    rslt = requests.post("http://localhost:8888/answer", json = {"question": prompt})
    with st.chat_message("ai"):
        st.markdown(rslt.json()["resp"])
    st.session_state.messages.append({"role": "ai", "content": rslt.json()["resp"]})