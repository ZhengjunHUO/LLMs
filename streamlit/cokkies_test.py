# See discussion: https://github.com/streamlit/streamlit/pull/5457

import re
import streamlit as st
try:
    # Streamlit >= 1.12.0
    from streamlit.web.server.server import Server
    from streamlit.runtime.runtime import Runtime, SessionInfo
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    # from streamlit.runtime.scriptrunner import get_script_run_ctx
except:
    raise Exception('You must use Streamlit >= v1.12.0')

# Mega hack walking the GC heap.
# Look only for singletons that you know exist for the entire streamlit server lifetime, e.g.: Server and Runtime!!
def st_instance_of_type(type_obj: object) -> object:
    import gc
    st_obj = None
    for obj in gc.get_objects():
        if type(obj) is type_obj:
            st_obj = obj
            break
    return st_obj

def st_server_props():
    st_server = st_instance_of_type(Server)

    st_server_runtime = st_server._runtime
    st_gc_runtime = st_instance_of_type(Runtime)
    assert(st_server_runtime == st_gc_runtime)

    main_script_path = st_server.main_script_path
    browser_is_connected = st_server.browser_is_connected

    return {'st_server_runtime': st_server_runtime, 'st_gc_runtime': st_gc_runtime, 'main_script_path': main_script_path, 'browser_is_connected': browser_is_connected}
    
def st_session_info() -> SessionInfo:
    st_runtime = st_instance_of_type(Runtime)
    # get session id from the current script runner thread
    session_id = add_script_run_ctx().streamlit_script_run_ctx.session_id
    # use the session id to retrieve the session info
    session_info = st_runtime._get_session_info(session_id)
    return session_info

def st_client_headers() -> dict:
    session_info = st_session_info()
    client_headers = session_info.client.request.headers._dict
    return dict(client_headers)

def st_client_cookies() -> dict:
    client_headers = st_client_headers()
    cookies_str = client_headers["Cookie"]
    results = re.findall(r"([\w]+)=([^;]+)", cookies_str)
    cookies = dict(results)
    return cookies

st.subheader('Server Props')
st.write(st_server_props())

st.subheader('Client Headers')
st.write(st_client_headers())

st.subheader('Client Cookies')
st.write(st_client_cookies())

