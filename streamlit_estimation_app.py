
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from io import BytesIO
from rapidfuzz import fuzz
import re

# --- SETUP ---
st.set_page_config(page_title="BuildWise", page_icon="🏗️", layout="wide")
st.image("assets/logo.png", width=120)
st.title("BuildWise: Smart Estimation & Quotation Comparison")

# --- USER LOGIN SIMULATION ---
st.sidebar.title("🔐 Login")
username = st.sidebar.text_input("Username", key="user")
if not username:
    st.warning("Please enter your username in the sidebar to continue.")
    st.stop()

user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)

# --- PRICE LIST UPLOAD ---
st.subheader("📁 Upload Price List (Database) Files")
uploaded_files = st.file_uploader("Upload one or more price list files (.xlsx)", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(user_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.success("Price list files uploaded successfully.")

# --- SHOW PRICE LIST FILES ---
st.subheader("📂 Your Price List Files")
price_list_files = sorted(os.listdir(user_folder))
selected_file = st.radio("Select a price list to use for matching or choose all:", ["All files"] + price_list_files)

# --- DELETE PRICE LIST FILES ---
st.markdown("#### 🗑️ Delete a file")
file_to_delete = st.selectbox("Select file to delete", [""] + price_list_files)
if file_to_delete and st.button("Delete file"):
    os.remove(os.path.join(user_folder, file_to_delete))
    st.success(f"{file_to_delete} deleted.")
    st.experimental_rerun()

# --- ESTIMATION REQUEST FILE UPLOAD ---
st.subheader("📄 Upload Estimation Request")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est")

# --- Cleaning function ---
def clean(text):
    text = str(text).lower()
    text = text.replace("cáp điện", "").replace("cable", "")
    text = text.replace("mm2", "").replace("(", "").replace(")", "")
    text = text.replace("4c x", "4x")
    text = text.replace("/", " ").replace(",", "")
    return text.strip()

# --- Matching logic ---
if estimation_file and price_list_files:
    est_df = pd.read_excel(estimation_file)
    est_df = est_df.rename(columns={"Desciption": "Description"})

    if selected_file == "All files":
        db_frames = []
        for f in price_list_files:
            df = pd.read_excel(os.path.join(user_folder, f))
            df["source_file"] = f
            db_frames.append(df)
        db = pd.concat(db_frames, ignore_index=True)
    else:
        db = pd.read_excel(os.path.join(user_folder, selected_file))
        db["source_file"] = selected_file

    db = db.rename(columns={"Desciption": "Description"})

    result = []
    unmatched_rows = []

    for _, row in est_df.iterrows():
        match_found = False
        match_row = {}
        for col in ["Model", "Description", "Specification"]:
            value = clean(row.get(col, ""))
            if not value:
                continue
            matches = db.copy()
            matches["score"] = db[col].astype(str).apply(lambda x: fuzz.token_set_ratio(clean(x), value))
            best_match = matches.loc[matches["score"].idxmax()]
            if best_match["score"] > 75:
                match_row = best_match
                match_found = True
                break
        if match_found:
            result.append({
                "Model": row.get("Model", ""),
                "Description (Requested)": row.get("Description", ""),
                "Description (Proposed)": match_row.get("Description", ""),
                "Specification": row.get("Specification", ""),
                "Quantity": row.get("Quantity", 0),
                "Unit": row.get("Unit", ""),
                "Material Cost": match_row.get("Material cost", 0),
                "Labour Cost": match_row.get("Labour cost", 0)
            })
        else:
            unmatched_rows.append(row)

    result_df = pd.DataFrame(result)
    result_df["Amount Material"] = result_df["Quantity"] * result_df["Material Cost"]
    result_df["Amount Labour"] = result_df["Quantity"] * result_df["Labour Cost"]
    result_df["Total"] = result_df["Amount Material"] + result_df["Amount Labour"]

    result_df["Total"] = result_df["Total"].round(2)
    result_df["Amount Material"] = result_df["Amount Material"].round(2)
    result_df["Amount Labour"] = result_df["Amount Labour"].round(2)

    result_df = result_df[[
        "Model", "Description (Requested)", "Description (Proposed)", "Specification",
        "Quantity", "Unit", "Material Cost", "Labour Cost",
        "Amount Material", "Amount Labour", "Total"
    ]]

    grand_total = result_df["Total"].sum()
    total_row = pd.Series([""] * 10 + [grand_total], index=result_df.columns)
    result_df = pd.concat([result_df, pd.DataFrame([total_row])], ignore_index=True)

    st.success("✅ Matching completed!")
    st.dataframe(result_df)

    output = BytesIO()
    result_df.to_excel(output, index=False)
    st.download_button("📥 Download Cleaned Estimation File", data=output.getvalue(),
                       file_name="BuildWise_Estimation_Result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if unmatched_rows:
        st.subheader("❗ Unmatched Items")
        unmatched_df = pd.DataFrame(unmatched_rows)
        st.dataframe(unmatched_df)
        unmatched_output = BytesIO()
        unmatched_df.to_excel(unmatched_output, index=False)
        st.download_button("📥 Download Unmatched Items", data=unmatched_output.getvalue(),
                           file_name="BuildWise_Unmatched_Items.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload an estimation request and at least one price list file.")
