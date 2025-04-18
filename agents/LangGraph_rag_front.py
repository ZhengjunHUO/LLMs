import streamlit as st
import requests

question = st.text_input("Question", "What security issue could we face while using the LLM ?")

if st.button("Submit"):
    rslt = requests.post("http://localhost:8888/answer", json = {"question": question})
    st.write("", rslt.json()["resp"])