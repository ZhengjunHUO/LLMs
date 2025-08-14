import streamlit as st
import streamlit.components.v1 as components

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

SELF_URL = "http://127.0.0.1:8501"
LOGIN_URL = f"http://127.0.0.1:3000/auth/login?redirect_uri={SELF_URL}"
VERIFY_URL = "http://127.0.0.1:3000/auth/whoami"

#components.html(f"""
#<script>
#async function checkAuth() {{
#    try {{
#        let resp = await fetch("{VERIFY_URL}", {{
#            method: "GET",
#            credentials: "include"
#        }});
#        if (resp.status === 200) {{
#            window.parent.postMessage({{ isStreamlitMessage: true, loggedIn: true }}, "*");
#        }} else {{
#            window.parent.postMessage({{ isStreamlitMessage: true, loggedIn: false }}, "*");
#        }}
#    }} catch (err) {{
#        console.error(err);
#    }}
#}}
#checkAuth();
#</script>
#""", height=0)
#
##msg = st.experimental_get_query_params().get("loggedIn")
#msg = st.query_params["loggedIn"]
#print(f"msg: {msg}")
#if msg:
#    st.session_state.logged_in = msg[0] == "true"

#def check_login_success():
#    return st.query_params.get("auth") == ["success"]
#
#if check_login_success():
#    st.session_state.logged_in = True

def check_login_success_via_session() -> bool:
    """
    Returns:
      True  -> user has a valid session
      False -> no valid session / first run (JS not returned yet)
    """
    result = components.html(
        f"""
        <script>
        async function run() {{
          try {{
            const resp = await fetch("{VERIFY_URL}", {{
              method: "GET",
              credentials: "include"
            }});
            // pass boolean back to Streamlit
            const ok = resp.status === 200;
            Streamlit.setComponentValue(ok);
          }} catch (e) {{
            Streamlit.setComponentValue(false);
          }}
        }}
        run();
        </script>
        """,
        height=0,
    )
    print(f"result: {result}")
    return result

probe = check_login_success_via_session()
print(f"probe: {probe} === {bool(probe)}")
st.session_state.logged_in = bool(probe)

if not st.session_state.logged_in:
    st.title("Welcome")
    st.write("Please log in to continue.")
    if st.button("Login"):
        st.markdown(f"""
            <meta http-equiv="refresh" content="0;url={LOGIN_URL}" />
        """, unsafe_allow_html=True)
else:
    st.title("Logged in")
    st.write("✅ You are now logged in!")
