
import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

# Clean and normalize text
def clean(text):
    text = str(text).lower()
    text = re.sub(r"0[,.]?6kv|1[,.]?0kv", "", text)
    text = text.replace("mm2", "").replace("mm²", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("/", " ").replace(",", "")
    text = text.replace("-", " ")
    text = text.replace("cáp", "").replace("cable", "").replace("dây", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Extract size like 4x25, 2x10, etc. Normalize from formats like "4C x 25mm2"
def extract_size(text):
    text = str(text).lower()
    text = text.replace("mm2", "").replace("mm²", "")
    text = re.sub(r"(\d)c", r"\1", text)  # convert 4C -> 4
    match = re.search(r'\b\d{1,2}\s*[x×]\s*\d{1,3}\b', text)
    return match.group(0).replace(" ", "") if match else ""

st.set_page_config(page_title="BuildWise", page_icon="📐", layout="wide")
st.image("assets/logo.png", width=120)
st.title("📐 BuildWise - Smart Estimation Tool")

username = st.sidebar.text_input("Username")
if not username:
    st.warning("Please enter your username to continue.")
    st.stop()

user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)

st.subheader("📁 Upload Price List Files")
uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(user_folder, file.name), "wb") as f:
            f.write(file.read())
    st.success("✅ Price list uploaded successfully.")

st.subheader("📂 Manage Price Lists")
price_list_files = os.listdir(user_folder)
selected_file = st.radio("Choose one file to match or use all", ["All files"] + price_list_files)

st.subheader("📄 Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est")

if estimation_file and price_list_files:
    est = pd.read_excel(estimation_file).dropna(how='all')
    est_cols = est.columns.tolist()
    if len(est_cols) < 3:
        st.error("Estimation file must have at least 3 columns.")
        st.stop()

    est["combined"] = (
        est.get(est_cols[0], "").fillna('') + " " +
        est.get(est_cols[1], "").fillna('') + " " +
        est.get(est_cols[2], "").fillna('')
    ).apply(clean)

    est["size"] = (
        est.get(est_cols[0], "").fillna('') + " " +
        est.get(est_cols[1], "").fillna('') + " " +
        est.get(est_cols[2], "").fillna('')
    ).apply(extract_size)

    db_frames = []
    if selected_file == "All files":
        for f in price_list_files:
            df = pd.read_excel(os.path.join(user_folder, f)).dropna(how='all')
            df["source"] = f
            db_frames.append(df)
        db = pd.concat(db_frames, ignore_index=True)
    else:
        db = pd.read_excel(os.path.join(user_folder, selected_file)).dropna(how='all')

    db_cols = db.columns.tolist()
    if len(db_cols) < 3:
        st.error("Database file must have at least 3 columns.")
        st.stop()

    db["combined"] = (
        db.get(db_cols[0], "").fillna('') + " " +
        db.get(db_cols[1], "").fillna('') + " " +
        db.get(db_cols[2], "").fillna('')
    ).apply(clean)

    db["size"] = (
        db.get(db_cols[0], "").fillna('') + " " +
        db.get(db_cols[1], "").fillna('') + " " +
        db.get(db_cols[2], "").fillna('')
    ).apply(extract_size)

    matches = []
    for _, est_row in est.iterrows():
        query = est_row["combined"]
        query_size = est_row["size"]

        if not query_size:
            continue

        db_filtered = db[db["size"] == query_size]

        if db_filtered.empty:
            matches.append({
                "Estimation": query,
                "Size": query_size,
                "Match Status": "❌ No size match",
                "Matched DB": "",
                "Score": 0,
                "Original Match": ""
            })
        else:
            db_filtered = db_filtered.copy()
            db_filtered["score"] = db_filtered["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
            best = db_filtered.loc[db_filtered["score"].idxmax()]
            matches.append({
                "Estimation": query,
                "Size": query_size,
                "Match Status": "✅ Size matched",
                "Matched DB": best["combined"],
                "Score": best["score"],
                "Original Match": best.get(db_cols[0], "")
            })

    result_df = pd.DataFrame(matches)
    st.subheader("🔍 Matching Result")
    st.dataframe(result_df)

    buffer = BytesIO()
    result_df.to_excel(buffer, index=False)
    st.download_button("📥 Download Matching Result", data=buffer.getvalue(), file_name="Matching_Result_SizeAware.xlsx")
