import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

# === Utility functions ===
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

def extract_size(text):
    text = str(text).lower()
    text = text.replace("mm2", "").replace("mm²", "")
    text = re.sub(r"(\d)c", r"\1", text)  # convert 4C -> 4
    match = re.search(r'\b\d{1,2}\s*[x×]\s*\d{1,3}\b', text)
    return match.group(0).replace(" ", "") if match else ""

# === UI Setup ===
st.set_page_config(page_title="BuildWise", page_icon="📀", layout="wide")
st.image("assets/logo.png", width=120)
st.title("📀 BuildWise - Smart Estimation Tool")

username = st.sidebar.text_input("Username")
if not username:
    st.warning("Please enter your username to continue.")
    st.stop()

user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)

# === Upload Price List ===
st.subheader("📁 Upload Price List Files")
uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(user_folder, file.name), "wb") as f:
            f.write(file.read())
    st.success("✅ Price list uploaded successfully.")

# === Manage Price List ===
st.subheader("📂 Manage Price Lists")
price_list_files = os.listdir(user_folder)
selected_file = st.radio("Choose one file to match or use all", ["All files"] + price_list_files)

# === Upload Estimation File ===
st.subheader("📄 Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est")
if estimation_file and price_list_files:
    est = pd.read_excel(estimation_file).dropna(how='all')
    est_cols = est.columns.tolist()
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    est["combined"] = (est[est_cols[0]].astype(str).fillna('') + " " + est[est_cols[1]].astype(str).fillna('') + " " + est[est_cols[2]].astype(str).fillna('')).apply(clean)
    est["size"] = (est[est_cols[0]].astype(str).fillna('') + " " + est[est_cols[1]].astype(str).fillna('') + " " + est[est_cols[2]].astype(str).fillna('')).apply(extract_size)

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
    if len(db_cols) < 7:
        st.error("Price list file must have at least 7 columns.")
        st.stop()

    db["combined"] = (db[db_cols[0]].astype(str).fillna('') + " " + db[db_cols[1]].astype(str).fillna('') + " " + db[db_cols[2]].astype(str).fillna('')).apply(clean)
    db["size"] = (db[db_cols[0]].astype(str).fillna('') + " " + db[db_cols[1]].astype(str).fillna('') + " " + db[db_cols[2]].astype(str).fillna('')).apply(extract_size)

    output_data = []
    for i, row in est.iterrows():
        query = row["combined"]
        query_size = row["size"]
        qty = row[est_cols[3]]
        unit = row[est_cols[4]]
        spec = row[est_cols[2]]
        model = row[est_cols[0]]
        desc_req = row[est_cols[1]]

        match_row = {"Description": "", "Material Cost": 0, "Labour Cost": 0}

        if query_size:
            db_filtered = db[db["size"] == query_size]
            if not db_filtered.empty:
                db_filtered = db_filtered.copy()
                db_filtered["score"] = db_filtered["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
                best = db_filtered.loc[db_filtered["score"].idxmax()]
                match_row = {
                    "Description": best[db_cols[1]],
                    "Material Cost": pd.to_numeric(best[db_cols[5]], errors="coerce"),
                    "Labour Cost": pd.to_numeric(best[db_cols[6]], errors="coerce")
                }

        qty_val = pd.to_numeric(qty, errors="coerce") if not pd.isna(qty) else 0
        m_cost = match_row["Material Cost"] if not pd.isna(match_row["Material Cost"]) else 0
        l_cost = match_row["Labour Cost"] if not pd.isna(match_row["Labour Cost"]) else 0
        amt_mat = qty_val * m_cost
        amt_lab = qty_val * l_cost
        total = amt_mat + amt_lab

        output_data.append([
            model, desc_req, match_row["Description"], spec,
            qty_val, unit, m_cost, l_cost, amt_mat, amt_lab, total
        ])

    result_df = pd.DataFrame(output_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Quantity", "Unit",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
    grand_row = pd.DataFrame([[""] * 10 + [grand_total]], columns=result_df.columns)
    result_final = pd.concat([result_df, grand_row], ignore_index=True)

    st.subheader("🔍 Matched Estimation")
    display_df = result_final.copy()
    for col in ["Quantity", "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0).map(lambda x: f"{int(x):,}")

    st.dataframe(display_df)

    # Unmatched rows
    unmatched_df = result_df[result_df["Description (proposed)"] == ""]
    st.subheader("❌ Unmatched Rows")
    if not unmatched_df.empty:
        st.dataframe(unmatched_df)
    else:
        st.info("✅ All rows matched successfully!")

    # Excel export
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Matched Results")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")

    st.download_button("📥 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")
