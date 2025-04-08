#!/bin/sh

uvicorn fastapi_demo:app --host localhost --port 8888
#curl "http://localhost:8888/add?x=2&y=3"

streamlit run streamlit_demo.py
