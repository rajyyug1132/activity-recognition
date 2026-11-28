"""Entry point. Run from the repo root: streamlit run app/streamlit_app.py

Top navigation over four pages; each page file draws its own content.
"""
import streamlit as st

st.set_page_config(page_title="Activity Recognition", page_icon=":material/directions_run:", layout="wide")
st.navigation(
    [
        st.Page("overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page("home.py", title="Phone recording", icon=":material/upload_file:", url_path="Phone"),
        st.Page("pages/1_Live.py", title="Live", icon=":material/sensors:", url_path="Live"),
        st.Page("pages/2_Video.py", title="Video", icon=":material/videocam:", url_path="Video"),
    ],
    position="top",
).run()
