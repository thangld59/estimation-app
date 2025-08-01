import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

# ------------------------------
# Utility Functions
# ------------------------------
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
    text = re.sub(r"(\d)c", r"\1", text)
    match = re.search(r'\b\d{1,2}\s*[x×]\s*\d{1,3}\b', text)
    return match.group(0).replace(" ", "") if match else ""

def extract_material_structure(text):
    text = str(text).lower()
    return re.findall(r'(cu|al|xlpe|pvc|pe|lszh|hdpe)', text)

def weighted_score(query_tokens, target_tokens):
    weights = {
        'cu': 1.0, 'al': 1.0,
        'xlpe': 0.8, 'pvc': 0.6,
        'lszh': 0.5, 'pe': 0.5, 'hdpe': 0.5
    }
    score = 0
    max_score = 0
    for token in set(query_tokens + target_tokens):
        max_score += weights.get(token, 0)
        if token in query_tokens and token in target_tokens:
            score += weights.get(token, 0)
    return round((score / max_score) * 100, 2) if max_score > 0 else 0

# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(page_title="BuildWise", page_icon="📀", layout="wide")
st.image("assets/logo.png", width=120)
st.title(":triangular_ruler: BuildWise - Smart Estimation Tool")

username = st.sidebar.text_input("Username")
if not username:
    st.warning("Please enter your username to continue.")
    st.stop()

user_folder = f"user_data/{username}"
form_folder = "shared_forms"
os.makedirs(user_folder, exist_ok=True)
os.makedirs(form_folder, exist_ok=True)

# ------------------------------
# Shared Forms
# ------------------------------
st.subheader(":scroll: Price List and Estimation Request Form (Mẫu Bảng Giá và Mẫu Yêu Cầu Vào Giá)")
form_files = os.listdir(form_folder)
if username == "Admin123":
    form_uploads = st.file_uploader("Upload form files", type=["xlsx", "xls"], accept_multiple_files=True, key="form_upload")
    if form_uploads:
        for f in form_uploads:
            with open(os.path.join(form_folder, f.name), "wb") as out_file:
                out_file.write(f.read())
        st.success("Form file(s) uploaded successfully.")

    form_to_delete = st.selectbox("Select a form file to delete", [""] + form_files, key="form_delete")
    if form_to_delete and st.button("Delete Selected Form File"):
        try:
            os.remove(os.path.join(form_folder, form_to_delete))
            st.success(f"Deleted form file: {form_to_delete}")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error deleting form file: {e}")
else:
    for file in form_files:
        with open(os.path.join(form_folder, file), "rb") as f:
            st.download_button(f"📄 Download {file}", f.read(), file_name=file)

# ------------------------------
# Upload Price List Files
# ------------------------------
st.subheader(":file_folder: Upload Price List Files")
uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(user_folder, file.name), "wb") as f:
            f.write(file.read())
    st.success(":white_check_mark: Price list uploaded successfully.")

# ------------------------------
# Manage Price Lists
# ------------------------------
st.subheader(":open_file_folder: Manage Price Lists")
price_list_files = os.listdir(user_folder)
selected_file = st.radio("Choose one file to match or use all", ["All files"] + price_list_files)

file_to_delete = st.selectbox("Select a file to delete", [""] + price_list_files)
if file_to_delete:
    if st.button("Delete Selected File"):
        try:
            os.remove(os.path.join(user_folder, file_to_delete))
            st.success(f"Deleted file: {file_to_delete}")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error deleting file: {e}")

# ------------------------------
# Upload Estimation File
# ------------------------------
st.subheader(":page_facing_up: Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est")
if estimation_file and price_list_files:
    est = pd.read_excel(estimation_file).dropna(how='all')
    est_cols = est.columns.tolist()
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    est["combined"] = (est[est_cols[0]].fillna('') + " " + est[est_cols[1]].fillna('') + " " + est[est_cols[2]].fillna('')).apply(clean)
    est["size"] = (est[est_cols[0]].fillna('') + " " + est[est_cols[1]].fillna('') + " " + est[est_cols[2]].fillna('')).apply(extract_size)
    est["materials"] = (est[est_cols[0]].fillna('') + " " + est[est_cols[1]].fillna('') + " " + est[est_cols[2]].fillna('')).apply(extract_material_structure)

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
    if len(db_cols) < 6:
        st.error("Price list file must have at least 6 columns.")
        st.stop()

    db["combined"] = (db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna('')).apply(clean)
    db["size"] = (db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna('')).apply(extract_size)
    db["materials"] = (db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna('')).apply(extract_material_structure)

    output_data = []
    for i, row in est.iterrows():
        query = row["combined"]
        query_size = row["size"]
        query_materials = row["materials"]
        unit = row[est_cols[3]]
        qty = row[est_cols[4]]

        best = None
        if query_size:
            db_filtered = db[db["size"] == query_size]
            if not db_filtered.empty:
                db_filtered = db_filtered.copy()
                db_filtered["score"] = db_filtered["materials"].apply(lambda m: weighted_score(query_materials, m))
                best = db_filtered.loc[db_filtered["score"].idxmax()]

        if best is not None:
            desc_proposed = best[db_cols[1]]
            m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce")
        else:
            desc_proposed = ""
            m_cost = l_cost = 0

        qty_val = pd.to_numeric(qty, errors="coerce")
        if pd.isna(qty_val): qty_val = 0
        amt_mat = qty_val * m_cost
        amt_lab = qty_val * l_cost
        total = amt_mat + amt_lab

        output_data.append([
            row[est_cols[0]], row[est_cols[1]], desc_proposed, row[est_cols[2]], unit, qty,
            m_cost, l_cost, amt_mat, amt_lab, total
        ])

    result_df = pd.DataFrame(output_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Unit", "Quantity",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
    result_df.loc[len(result_df.index)] = [""] * 10 + [grand_total]

    st.subheader(":mag: Matched Estimation")
    display_df = result_df.copy()
    display_df["Quantity"] = pd.to_numeric(display_df["Quantity"], errors="coerce").fillna(0).astype(int).map("{:,}".format)
    for col in ["Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0).astype(int).map("{:,}".format)
    st.dataframe(display_df)

    st.subheader(":x: Unmatched Rows")
    unmatched_df = result_df[result_df["Description (proposed)"] == ""]
    if not unmatched_df.empty:
        st.dataframe(unmatched_df)
    else:
        st.info(":white_check_mark: All rows matched successfully!")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Matched Results")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")

    st.download_button("📥 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")
