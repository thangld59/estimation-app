
import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

# ---------------- Utility Functions ----------------
def clean(text):
    text = str(text).lower()
    text = re.sub(r"0[,.]?6kv|1[,.]?0kv", "", text)
    text = text.replace("mm2", "").replace("mm²", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("/", " ").replace(",", "")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_conduit_size(text):
    text = str(text).lower()
    match = re.search(r"(d|ø|phi)?\s*(\d{1,3})(mm)?", text)
    return match.group(2) if match else ""

def extract_dimensions(text):
    text = str(text).lower()
    dims = {}
    width = re.search(r"(w[=\s]*|)(\d{2,4})(?=\D|$)", text)
    height = re.search(r"(h[=\s]*|)(\d{2,4})(?=\D|$)", text)
    thickness = re.search(r"(t[=\s]*|dày[=\s]*)(\d+(\.\d+)?)", text)
    if width:
        dims["w"] = int(width.group(2))
    if height:
        dims["h"] = int(height.group(2))
    if thickness:
        dims["t"] = float(thickness.group(2))
    return dims

def get_category(description):
    desc = str(description).lower()
    if "cable" in desc or "dây" in desc:
        return "cable"
    elif "conduit" in desc or "ống luồn" in desc or "pipe" in desc:
        return "conduit"
    elif "tray" in desc or "máng cáp" in desc or "duct" in desc:
        return "cable_tray"
    elif "rack" in desc or "thang cáp" in desc:
        return "cable_rack"
    return "other"

def match_conduit(req_size, db_df):
    db_df["match"] = db_df["size"].apply(lambda x: req_size == x)
    return db_df[db_df["match"]]

def match_dimensions(req_dims, db_df):
    def is_match(row):
        db_dims = row["dimensions"]
        for key in ["w", "h", "t"]:
            if key in req_dims and key in db_dims:
                if abs(req_dims[key] - db_dims[key]) > 1e-1:
                    return False
            elif key in req_dims or key in db_dims:
                return False
        return True
    return db_df[db_df.apply(is_match, axis=1)]

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="BuildWise", page_icon="🧮", layout="wide")
st.title("📐 BuildWise Estimation Tool (v2)")

username = st.sidebar.text_input("Username")
if not username:
    st.warning("Please enter your username to continue.")
    st.stop()

user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)

# Upload Price List Files
st.subheader("📁 Upload Price List Files")
uploaded_files = st.file_uploader("Upload Excel price list(s)", type="xlsx", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(user_folder, file.name), "wb") as f:
            f.write(file.read())
    st.success("✅ Price list uploaded.")

# Select Price List
st.subheader("📄 Select Price List to Match")
price_list_files = os.listdir(user_folder)
selected_file = st.radio("Use a specific file or all files:", ["All files"] + price_list_files)

# Upload Estimation File
st.subheader("📥 Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request Excel", type="xlsx", key="est")
if estimation_file and price_list_files:
    est = pd.read_excel(estimation_file).dropna(how="all")
    est_cols = est.columns.tolist()
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    est["category"] = est[est_cols[1]].apply(get_category)
    est["size"] = est[est_cols[1]].apply(extract_conduit_size)
    est["dimensions"] = est[est_cols[1]].apply(extract_dimensions)
    est["combined"] = (est[est_cols[0]].fillna("") + " " + est[est_cols[1]].fillna("") + " " + est[est_cols[2]].fillna("")).apply(clean)

    # Load database
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
    db["category"] = db[db_cols[1]].apply(get_category)
    db["size"] = db[db_cols[1]].apply(extract_conduit_size)
    db["dimensions"] = db[db_cols[1]].apply(extract_dimensions)
    db["combined"] = (db[db_cols[0]].fillna("") + " " + db[db_cols[1]].fillna("") + " " + db[db_cols[2]].fillna("")).apply(clean)

    # Matching Logic
    result_data = []
    for i, row in est.iterrows():
        cat = row["category"]
        query = row["combined"]
        qty = row[est_cols[4]]
        unit = row[est_cols[3]]
        best_match = None

        subset = db[db["category"] == cat].copy()

        if cat == "conduit":
            subset = match_conduit(row["size"], subset)
        elif cat in ["cable_tray", "cable_rack"]:
            subset = match_dimensions(row["dimensions"], subset)

        if not subset.empty:
            subset["score"] = subset["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
            subset = subset[subset["score"] > 70]  # Threshold
            if not subset.empty:
                best_match = subset.loc[subset["score"].idxmax()]

        if best_match is not None:
            m_cost = pd.to_numeric(best_match[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best_match[db_cols[5]], errors="coerce")
            desc_proposed = best_match[db_cols[1]]
        else:
            m_cost = l_cost = 0
            desc_proposed = ""

        qty_val = pd.to_numeric(qty, errors="coerce") or 0
        amt_mat = qty_val * m_cost
        amt_lab = qty_val * l_cost
        total = amt_mat + amt_lab

        result_data.append([
            row[est_cols[0]], row[est_cols[1]], desc_proposed, row[est_cols[2]],
            unit, qty_val, m_cost, l_cost, amt_mat, amt_lab, total
        ])

    result_df = pd.DataFrame(result_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Unit", "Quantity",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
    result_df.loc[len(result_df.index)] = [""] * 10 + [grand_total]

    st.subheader("✅ Matched Results")
    display_df = result_df.copy()
    for col in ["Quantity", "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0).astype(int).map("{:,}".format)
    st.dataframe(display_df)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Matched Results")
    st.download_button("📥 Download Estimation Results", buffer.getvalue(), file_name="Estimation_Result_v2.xlsx")
