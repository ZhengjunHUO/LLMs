import streamlit as st
import extra_streamlit_components as stx
import requests

SELF_URL = "http://127.0.0.1:8501"
LOGIN_URL = f"http://127.0.0.1:3000/auth/login?redirect_uri={SELF_URL}"
VERIFY_URL = "http://127.0.0.1:3000/auth/whoami"

@st.fragment
def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

st.title("OIDC test")
if st.button("Login"):
    st.markdown(f"""
        <meta http-equiv="refresh" content="0;url={LOGIN_URL}" />
        """, unsafe_allow_html=True)

if st.button("Req"):
    rslt = requests.get(f"{VERIFY_URL}")
    st.write("Resp: ", rslt)

if st.button("connect.sid"):
    value = cookie_manager.get(cookie="connect.sid")
    st.write(value)

with st.form(key="Cookie"):
    hide_streamlit_style = """
    <style>
    [data-testid="stForm"] {border: none; padding: 0;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    submitted = st.form_submit_button("Get cookies")
    if submitted:
        st.subheader("All Cookies:")
        cookies = cookie_manager.get_all()
        st.write(cookies)