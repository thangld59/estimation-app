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
    text = re.sub(r"[()\[\],]", " ", text)
    text = text.replace("mm2", "").replace("mm²", "")
    text = text.replace("/", " ").replace(",", "")
    text = text.replace("-", " ")
    text = text.replace("cáp", "").replace("cable", "").replace("dây", "").replace("wire", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_size(text):
    text = str(text).lower()
    text = re.sub(r"(mm2|mm²)", "", text)
    size_match = re.search(r"\b\d{1,2}\s*[x×]\s*\d{1,3}(\.\d+)?\b", text)
    if size_match:
        return size_match.group(0).replace(" ", "")
    alt_match = re.search(r"\b(d|ø|phi)?\s*\d{1,3}(mm)?\b", text)
    if alt_match:
        return alt_match.group(0).replace(" ", "")
    return ""

def extract_voltage(text):
    match = re.search(r"\b0[.,]?6[ /-]?1[.,]?0?k?[vV]?\b", str(text))
    return "0.6/1kV" if match else ""

def extract_material(text):
    text = text.lower()
    if "nhôm" in text or "al" in text or "aluminium" in text:
        return "al"
    if "cu" in text or "đồng" in text:
        return "cu"
    return ""

def extract_insulation(text):
    for ins in ["xlpe", "pvc", "pe", "lszh"]:
        if ins in text.lower():
            return ins
    return ""

def extract_type_conduit(text):
    for t in ["pvc", "hdpe", "imc", "emt", "rsc", "flexible", "corrugated", "ống mềm", "ống cứng"]:
        if t.lower() in text.lower():
            return t.lower()
    return ""

def extract_category(text, type_):
    text = text.lower()
    if type_ == "cable":
        return any(k in text for k in ["cáp", "cable", "dây điện", "wire"])
    if type_ == "conduit":
        return any(k in text for k in ["ống", "conduit", "ống luồn", "ống dây", "ống mềm", "flexible"])
    return False

def match_row(row, db, db_cols, item_type="cable"):
    query_text = clean(row["combined"])
    query_size = extract_size(row["combined"])
    query_voltage = extract_voltage(row["combined"]) if item_type == "cable" else ""
    query_material = extract_material(row["combined"])
    query_insulation = extract_insulation(row["combined"]) if item_type == "cable" else ""
    query_type = extract_type_conduit(row["combined"]) if item_type == "conduit" else ""

    filtered = db.copy()

    # Category keywords mandatory
    filtered = filtered[filtered["category_match"] == True]

    # Filter by size (mandatory)
    if query_size:
        filtered = filtered[filtered["size"] == query_size]
    else:
        return None  # Size required

    if filtered.empty:
        return None

    # Optional boosting
    def score_func(x):
        score = fuzz.token_set_ratio(query_text, x["cleaned"])
        if item_type == "cable":
            if query_voltage and extract_voltage(x["combined"]) == query_voltage:
                score += 10
            if query_material and extract_material(x["combined"]) == query_material:
                score += 15
            if query_insulation and extract_insulation(x["combined"]) == query_insulation:
                score += 10
        elif item_type == "conduit":
            if query_type and extract_type_conduit(x["combined"]) == query_type:
                score += 10
            if query_material and extract_material(x["combined"]) == query_material:
                score += 15
        return score

    filtered["score"] = filtered.apply(score_func, axis=1)
    best = filtered.sort_values("score", ascending=False).iloc[0]
    return best if best["score"] >= 70 else None

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
    est["combined"] = (est[est_cols[0]].fillna('') + " " + est[est_cols[1]].fillna('') + " " + est[est_cols[2]].fillna(''))

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
    db["combined"] = (db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna(''))
    db["cleaned"] = db["combined"].apply(clean)
    db["size"] = db["combined"].apply(extract_size)
    db["category_match"] = db["combined"].apply(lambda x: extract_category(x, "cable") or extract_category(x, "conduit"))

    output_data = []
    for _, row in est.iterrows():
        unit = row[est_cols[3]]
        qty = row[est_cols[4]]
        row["combined"] = str(row[est_cols[0]]) + " " + str(row[est_cols[1]]) + " " + str(row[est_cols[2]])
        best_match = match_row(row, db, db_cols, "cable")
        if not best_match:
            best_match = match_row(row, db, db_cols, "conduit")

        if best_match is not None:
            desc_proposed = best_match[db_cols[1]]
            m_cost = pd.to_numeric(best_match[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best_match[db_cols[5]], errors="coerce")
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
    grand_row = pd.DataFrame([[''] * 10 + [grand_total]], columns=result_df.columns)
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

    st.download_button("📥 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")
