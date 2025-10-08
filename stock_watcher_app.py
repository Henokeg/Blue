import streamlit as st

st.set_page_config(page_title="Hello Streamlit", layout="centered")
st.markdown("# ✅ Streamlit is running")
st.write("If you can read this, the deploy is good.")
st.write("Streamlit version:", st.__version__)
