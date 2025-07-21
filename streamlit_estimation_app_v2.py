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
    text = re.sub(r"[()/,-]", " ", text)
    text = re.sub(r"mm2|mm²", "", text)
    text = re.sub(r"0[.,]?6[ /-]?1[.,]?0?k?[vV]?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_size_cable(text):
    text = str(text).lower().replace(" ", "")
    match = re.search(r"(\d{1,2}[cx×]\d{1,3}(\.\d+)?)", text)
    return match.group(1) if match else ""

def extract_voltage(text):
    text = str(text).lower()
    if re.search(r"0[.,]?6[ /-]?1[.,]?0?k?[vV]?", text):
        return "0.6/1kV"
    return ""

def extract_material(text):
    text = text.lower()
    if "nhôm" in text or "al" in text or "aluminium" in text:
        return "Al"
    elif "cu" in text:
        return "Cu"
    return ""

def extract_insulation(text):
    for ins in ["xlpe", "pvc", "pe", "lszh"]:
        if ins in text.lower():
            return ins.upper()
    return ""

def extract_shield(text):
    for s in ["swa", "sta", "armored", "tape", "shield"]:
        if s in text.lower():
            return s.upper()
    return ""

def extract_size_conduit(text):
    text = str(text).lower()
    match = re.search(r"\b(d|ø|phi)?\s*(\d{1,3})(mm)?\b", text)
    return match.group(2) if match else ""

def extract_type_conduit(text):
    for t in ["pvc", "hdpe", "imc", "emt", "rsc", "flexible", "corrugated", "ống mềm", "ống cứng"]:
        if t.lower() in text.lower():
            return t.upper()
    return ""

def extract_material_conduit(text):
    for m in ["thép", "inox", "nhựa", "aluminum", "galvanized"]:
        if m.lower() in text.lower():
            return m.lower()
    return ""

def is_cable(text):
    return any(x in text.lower() for x in ["cáp", "cable", "dây điện", "wire"])

def is_conduit(text):
    return any(x in text.lower() for x in ["ống", "conduit", "ống luồn", "ống dây", "ống mềm", "flexible"])

def cable_score(row1, row2):
    score = 0
    if row1["size"] and row1["size"] == row2["size"]:
        score += 40
    if row1["material"] == row2["material"]:
        score += 20
    if row1["insulation"] == row2["insulation"]:
        score += 10
    if row1["shield"] == row2["shield"]:
        score += 5
    if row1["voltage"] == row2["voltage"]:
        score += 10
    token_score = fuzz.token_set_ratio(row1["combined"], row2["combined"])
    return score + token_score * 0.15

def conduit_score(row1, row2):
    score = 0
    if row1["size"] and row1["size"] == row2["size"]:
        score += 40
    if row1["type"] == row2["type"]:
        score += 25
    if row1["material"] == row2["material"]:
        score += 20
    token_score = fuzz.token_set_ratio(row1["combined"], row2["combined"])
    return score + token_score * 0.15

# ------------------------------
# Streamlit UI
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

st.subheader(":file_folder: Upload Price List Files")
uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(user_folder, file.name), "wb") as f:
            f.write(file.read())
    st.success(":white_check_mark: Price list uploaded successfully.")

st.subheader(":open_file_folder: Manage Price Lists")
price_list_files = os.listdir(user_folder)
selected_file = st.radio("Choose one file to match or use all", ["All files"] + price_list_files)

st.subheader(":page_facing_up: Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est")

if estimation_file and price_list_files:
    est = pd.read_excel(estimation_file).dropna(how='all')
    est_cols = est.columns.tolist()
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    est["combined"] = (est[est_cols[0]].fillna("") + " " + est[est_cols[1]].fillna("") + " " + est[est_cols[2]].fillna("")).apply(clean)
    est["category"] = est["combined"].apply(lambda x: "cable" if is_cable(x) else ("conduit" if is_conduit(x) else "other"))
    est["size"] = est["combined"].apply(lambda x: extract_size_cable(x) if is_cable(x) else extract_size_conduit(x))
    est["voltage"] = est["combined"].apply(extract_voltage)
    est["material"] = est["combined"].apply(extract_material)
    est["insulation"] = est["combined"].apply(extract_insulation)
    est["shield"] = est["combined"].apply(extract_shield)
    est["type"] = est["combined"].apply(extract_type_conduit)
    est["mat2"] = est["combined"].apply(extract_material_conduit)

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
    db["combined"] = (db[db_cols[0]].fillna("") + " " + db[db_cols[1]].fillna("") + " " + db[db_cols[2]].fillna("")).apply(clean)
    db["category"] = db["combined"].apply(lambda x: "cable" if is_cable(x) else ("conduit" if is_conduit(x) else "other"))
    db["size"] = db["combined"].apply(lambda x: extract_size_cable(x) if is_cable(x) else extract_size_conduit(x))
    db["voltage"] = db["combined"].apply(extract_voltage)
    db["material"] = db["combined"].apply(extract_material)
    db["insulation"] = db["combined"].apply(extract_insulation)
    db["shield"] = db["combined"].apply(extract_shield)
    db["type"] = db["combined"].apply(extract_type_conduit)
    db["mat2"] = db["combined"].apply(extract_material_conduit)

    output_data = []
    for _, row in est.iterrows():
        cat = row["category"]
        filtered_db = db[db["category"] == cat]
        if row["size"]:  # apply size filtering
            filtered_db = filtered_db[filtered_db["size"] == row["size"]]

        if not filtered_db.empty:
            scores = filtered_db.apply(lambda x: cable_score(row, x) if cat == "cable" else conduit_score(row, x), axis=1)
            best_match = filtered_db.loc[scores.idxmax()]
            best_score = scores.max()
        else:
            best_match = None
            best_score = 0

        if best_match is not None and best_score >= 70:
            desc_proposed = best_match[db_cols[1]]
            m_cost = pd.to_numeric(best_match[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best_match[db_cols[5]], errors="coerce")
        else:
            desc_proposed = ""
            m_cost = l_cost = 0

        qty_val = pd.to_numeric(row[est_cols[4]], errors="coerce")
        qty_val = 0 if pd.isna(qty_val) else qty_val
        amt_mat = qty_val * m_cost
        amt_lab = qty_val * l_cost
        total = amt_mat + amt_lab

        output_data.append([
            row[est_cols[0]], row[est_cols[1]], desc_proposed, row[est_cols[2]],
            row[est_cols[3]], row[est_cols[4]], m_cost, l_cost, amt_mat, amt_lab, total
        ])

    result_df = pd.DataFrame(output_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification",
        "Unit", "Quantity", "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
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
