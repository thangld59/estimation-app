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
    text = re.sub(r"(\d)c", r"\1", text)  # convert 4C -> 4
    match = re.search(r'\b\d{1,2}\s*[x×]\s*\d{1,3}(\.\d+)?\b', text)
    return match.group(0).replace(" ", "") if match else ""

def extract_conduit_size(text):
    text = str(text).lower()
    match = re.search(r'(ø|phi)?\s*(\d{1,3})(mm)?', text)
    return match.group(2) if match else ""

def extract_dimensions(text):
    text = str(text).lower().replace(" ", "")
    match = re.search(r'(\d{2,4})[x×](\d{2,4})', text)
    return match.group(0) if match else ""

def category_of(text):
    text = text.lower()
    if any(word in text for word in ["cáp", "cable", "dây điện"]):
        return "cable"
    elif any(word in text for word in ["ống", "conduit"]):
        return "conduit"
    elif any(word in text for word in ["máng cáp", "cable tray", "cable duct"]):
        return "cable_tray"
    elif any(word in text for word in ["thang cáp", "cable rack", "cable ladder"]):
        return "cable_rack"
    else:
        return "other"

def extract_attribute(row, cat):
    desc = f"{row[0]} {row[1]} {row[2]}".lower()
    if cat == "cable":
        return extract_size(desc)
    elif cat == "conduit":
        return extract_conduit_size(desc)
    elif cat in ["cable_tray", "cable_rack"]:
        return extract_dimensions(desc)
    return ""

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
os.makedirs(user_folder, exist_ok=True)

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

    output_data = []
    for _, row in est.iterrows():
        category = category_of(" ".join([str(row[c]) for c in est_cols[:3]]))
        attr = extract_attribute(row, category)

        qty = row[est_cols[4]]
        unit = row[est_cols[3]]
        best = None

        db["combined"] = (db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna('')).apply(clean)

        if attr:
            db_matches = db[db["combined"].str.contains(attr, na=False)]
            if not db_matches.empty:
                db_matches = db_matches.copy()
                db_matches["score"] = db_matches["combined"].apply(lambda x: fuzz.token_set_ratio(" ".join([str(row[c]) for c in est_cols[:3]]), x))
                db_matches = db_matches[db_matches["score"] >= 70]  # new rule
                if not db_matches.empty:
                    best = db_matches.loc[db_matches["score"].idxmax()]

        m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce") if best is not None else 0
        l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce") if best is not None else 0
        desc_proposed = best[db_cols[1]] if best is not None else ""
        qty_val = pd.to_numeric(qty, errors="coerce")
        if pd.isna(qty_val): qty_val = 0
        amt_mat = qty_val * m_cost
        amt_lab = qty_val * l_cost
        total = amt_mat + amt_lab

        output_data.append([
            row[est_cols[0]],  # Model
            row[est_cols[1]],  # Description (requested)
            desc_proposed,     # Description (proposed)
            row[est_cols[2]],  # Specification
            unit,              # Unit
            qty,               # Quantity
            m_cost,            # Material Cost
            l_cost,            # Labour Cost
            amt_mat,           # Amount Material
            amt_lab,           # Amount Labour
            total              # Total
        ])

    result_df = pd.DataFrame(output_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Unit", "Quantity",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
    grand_row = pd.DataFrame([[""] * 10 + [grand_total]], columns=result_df.columns)
    result_final = pd.concat([result_df, grand_row], ignore_index=True)

    st.subheader(":mag: Matched Estimation")
    display_df = result_final.copy()
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
        result_final.to_excel(writer, index=False, sheet_name="Matched Results")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")

    st.download_button("\U0001F4E5 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")
