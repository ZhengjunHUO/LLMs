import streamlit as st
import requests

x = st.number_input("Val1", value=1)
y = st.number_input("Val2", value=2)

if st.button("Add"):
    rslt = requests.get(f"http://localhost:8888/add?x={x}&y={y}")
    st.write("Got:", rslt.json()["result"])