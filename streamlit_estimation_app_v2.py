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
    match = re.search(r'\b\d{1,2}\s*[x×]\s*\d{1,3}(\.\d+)?\b', text)
    return match.group(0).replace(" ", "") if match else ""

def extract_voltage(text):
    text = str(text).lower()
    match = re.search(r"\b0[.,]?6[ /-]?1[.,]?0?k?[vV]?\b", text)
    return "0.6/1kV" if match else ""

def extract_material(text):
    text = str(text).lower()
    if "nhôm" in text or "aluminium" in text or "al" in text:
        return "al"
    elif "cu" in text:
        return "cu"
    return ""

def extract_insulation(text):
    text = str(text).lower()
    if "xlpe" in text: return "xlpe"
    if "pvc" in text: return "pvc"
    if "lszh" in text: return "lszh"
    if "pe" in text: return "pe"
    return ""

def extract_shielding(text):
    text = str(text).lower()
    if any(x in text for x in ["swa", "sta", "armored", "a", "tape", "screen", "shield"]):
        return "shielded"
    return "unshielded"

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

    est["combined"] = (est[est_cols[0]].fillna('') + " " + est[est_cols[1]].fillna('') + " " + est[est_cols[2]].fillna('')).apply(clean)
    est["size"] = (est[est_cols[0]] + " " + est[est_cols[1]] + " " + est[est_cols[2]]).apply(extract_size)
    est["voltage"] = (est[est_cols[0]] + " " + est[est_cols[1]] + " " + est[est_cols[2]]).apply(extract_voltage)
    est["material"] = (est[est_cols[0]] + " " + est[est_cols[1]] + " " + est[est_cols[2]]).apply(extract_material)
    est["insulation"] = (est[est_cols[0]] + " " + est[est_cols[1]] + " " + est[est_cols[2]]).apply(extract_insulation)

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
    db["size"] = (db[db_cols[0]] + " " + db[db_cols[1]] + " " + db[db_cols[2]]).apply(extract_size)
    db["voltage"] = (db[db_cols[0]] + " " + db[db_cols[1]] + " " + db[db_cols[2]]).apply(extract_voltage)
    db["material"] = (db[db_cols[0]] + " " + db[db_cols[1]] + " " + db[db_cols[2]]).apply(extract_material)
    db["insulation"] = (db[db_cols[0]] + " " + db[db_cols[1]] + " " + db[db_cols[2]]).apply(extract_insulation)

    output_data = []
    for i, row in est.iterrows():
        query = row["combined"]
        query_size = row["size"]
        query_voltage = row["voltage"]
        query_material = row["material"]
        query_insul = row["insulation"]
        unit = row[est_cols[3]]
        qty = row[est_cols[4]]

        best = None
        if query_size and query_voltage:
            db_filtered = db[(db["size"] == query_size) & (db["voltage"] == query_voltage)]
            if not db_filtered.empty:
                db_filtered = db_filtered.copy()
                db_filtered["score"] = db_filtered["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
                db_filtered["boost"] = 0
                db_filtered.loc[db_filtered["material"] == query_material, "boost"] += 10
                db_filtered.loc[db_filtered["insulation"] == query_insul, "boost"] += 5
                db_filtered["total_score"] = db_filtered["score"] + db_filtered["boost"]
                top = db_filtered.sort_values(by="total_score", ascending=False).iloc[0]
                if top["total_score"] >= 70:
                    best = top

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
            row[est_cols[0]], row[est_cols[1]], desc_proposed, row[est_cols[2]],
            unit, qty, m_cost, l_cost, amt_mat, amt_lab, total
        ])

    result_df = pd.DataFrame(output_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Unit", "Quantity",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
    result_final = pd.concat([result_df, pd.DataFrame([[''] * 10 + [grand_total]], columns=result_df.columns)], ignore_index=True)

    st.subheader(":mag: Matched Estimation")
    st.dataframe(result_final)

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

    st.download_button("📥 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")
