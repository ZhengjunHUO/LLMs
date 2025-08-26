import streamlit as st
import requests

comment = st.text_input("Comment", value="You look smart")

if st.button("Submit"):
    rslt = requests.post(f"http://localhost:8080/classify", json = {"comment": comment})
    st.write(rslt.json()[0]["label"])