
import streamlit as st
import pandas as pd
import os
from io import BytesIO
from rapidfuzz import fuzz
import re

# --- SETUP ---
st.set_page_config(page_title="BuildWise", page_icon="🏗️", layout="wide")
st.image("assets/logo.png", width=120)
st.title("BuildWise: Smart Estimation & Quotation Comparison")

# --- LOGIN SIMULATION ---
username = st.sidebar.text_input("Username")
if not username:
    st.warning("Please enter your username to continue.")
    st.stop()

user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)

# --- PRICE LIST UPLOAD ---
st.subheader("📁 Upload Price List Files")
uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(user_folder, file.name), "wb") as f:
            f.write(file.read())
    st.success("✅ Price list uploaded successfully.")

# --- SHOW AND DELETE FILES ---
st.subheader("📂 Manage Price Lists")
price_list_files = os.listdir(user_folder)
selected_file = st.radio("Choose one file to match or use all", ["All files"] + price_list_files)
file_to_delete = st.selectbox("Delete a file", [""] + price_list_files)
if file_to_delete and st.button("Delete"):
    os.remove(os.path.join(user_folder, file_to_delete))
    st.success(f"{file_to_delete} deleted.")
    st.experimental_rerun()

# --- ESTIMATION FILE ---
st.subheader("📄 Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est")

# --- TEXT CLEANING ---
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

# --- MATCHING ---
if estimation_file and price_list_files:
    est = pd.read_excel(estimation_file)
    est_cols = est.columns.tolist()
    if len(est_cols) < 3:
        st.error("Estimation file must have at least 3 columns.")
        st.stop()
    est["combined"] = (est[est_cols[0]].astype(str) + " " + est[est_cols[1]].astype(str) + " " + est[est_cols[2]].astype(str)).apply(clean)

    db_frames = []
    if selected_file == "All files":
        for f in price_list_files:
            df = pd.read_excel(os.path.join(user_folder, f))
            df["source"] = f
            db_frames.append(df)
        db = pd.concat(db_frames, ignore_index=True)
    else:
        db = pd.read_excel(os.path.join(user_folder, selected_file))

    db_cols = db.columns.tolist()
    if len(db_cols) < 3:
        st.error("Database file must have at least 3 columns.")
        st.stop()

    db["combined"] = (db[db_cols[0]].astype(str) + " " + db[db_cols[1]].astype(str) + " " + db[db_cols[2]].astype(str)).apply(clean)

    if "Material cost" not in db.columns or "Labour cost" not in db.columns:
        st.error("Database file must include 'Material cost' and 'Labour cost' columns.")
        st.stop()

    results = []
    unmatched = []

    for i, row in est.iterrows():
        query = row["combined"]
        db["score"] = db["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
        best = db.loc[db["score"].idxmax()]
        if best["score"] >= 75:
            result = {
                "Model": row[est_cols[0]],
                "Description (Requested)": row[est_cols[1]],
                "Description (Proposed)": best[db_cols[1]],
                "Specification": row[est_cols[2]],
                "Quantity": row.get("Khối lượng", row.get("Quantity", 0)),
                "Unit": row.get("Đơn vị", row.get("Unit", "")),
                "Material Cost": best.get("Material cost", 0),
                "Labour Cost": best.get("Labour cost", 0),
            }
            result["Amount Material"] = result["Quantity"] * result["Material Cost"]
            result["Amount Labour"] = result["Quantity"] * result["Labour Cost"]
            result["Total"] = result["Amount Material"] + result["Amount Labour"]
            results.append(result)
        else:
            unmatched.append(row)

    result_df = pd.DataFrame(results)
    total = result_df["Total"].sum() if not result_df.empty else 0
    if not result_df.empty:
        total_row = pd.Series([""] * 10 + [total], index=result_df.columns)
        result_df = pd.concat([result_df, pd.DataFrame([total_row])], ignore_index=True)

        st.success("✅ Matching completed")
        st.dataframe(result_df)
        buffer = BytesIO()
        result_df.to_excel(buffer, index=False)
        st.download_button("📥 Download Estimation Result", data=buffer.getvalue(), file_name="BuildWise_Result.xlsx")

    if unmatched:
        unmatched_df = pd.DataFrame(unmatched)
        st.warning("⚠️ Some items could not be matched.")
        st.dataframe(unmatched_df)
        u_buffer = BytesIO()
        unmatched_df.to_excel(u_buffer, index=False)
        st.download_button("📥 Download Unmatched Items", data=u_buffer.getvalue(), file_name="BuildWise_Unmatched.xlsx")
