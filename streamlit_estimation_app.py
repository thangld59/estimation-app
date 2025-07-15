
import streamlit as st
import pandas as pd
from datetime import datetime
import os

# App title and logo
st.set_page_config(page_title="BuildWise", page_icon="🏗️")
st.image("assets/logo.png", width=120)
st.title("BuildWise: Smart Estimation & Quotation Comparison")

# Login simulation
st.sidebar.title("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login = st.sidebar.button("Login")

if login:
    st.session_state["user"] = username

if "user" not in st.session_state:
    st.warning("Please log in to continue.")
    st.stop()

user = st.session_state["user"]
user_folder = f"user_data/{user}"
os.makedirs(user_folder, exist_ok=True)

# Upload price list
st.subheader("📁 Upload Price List (Database) Files")
uploaded_dbs = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_dbs:
    for f in uploaded_dbs:
        with open(os.path.join(user_folder, f.name), "wb") as out_file:
            out_file.write(f.read())
    st.success("Uploaded successfully!")

# List current database files
st.subheader("📂 Your Uploaded Price List Files")
db_files = os.listdir(user_folder)
if db_files:
    for file in db_files:
        st.markdown(f"- {file}")
        if st.button(f"🗑️ Delete {file}", key=f"del_{file}"):
            os.remove(os.path.join(user_folder, file))
            st.experimental_rerun()
else:
    st.info("No price list files uploaded yet.")

# Upload estimation request
st.subheader("📄 Upload Estimation Request File")
estimation_file = st.file_uploader("Upload estimation request", type=["xlsx"], key="estimation")

if estimation_file and db_files:
    st.success("Estimation file uploaded. Matching will be added here.")

    # Placeholder for full matching logic
    # Would include fuzzy match, result preview, unmatched, export, etc.
else:
    st.info("Please upload at least one price list and one estimation file.")
