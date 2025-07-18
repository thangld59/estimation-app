
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

def extract_dimensions(text):
    text = text.lower()
    dimensions = {"w": None, "h": None, "t": None}
    matches = re.findall(r"(w|h|t)[=x:]?\s*(\d+(?:\.\d+)?)", text)
    for k, v in matches:
        dimensions[k] = float(v)
    if not any(dimensions.values()):
        wh_match = re.search(r"(\d{2,4})[x×](\d{2,4})", text)
        if wh_match:
            dimensions["w"], dimensions["h"] = map(float, wh_match.groups())
        t_match = re.search(r"(?:t|dày)[=:\s]*(\d+(?:\.\d+)?)", text)
        if t_match:
            dimensions["t"] = float(t_match.group(1))
    return dimensions

def dimension_similarity(dim1, dim2):
    if not dim1 or not dim2:
        return 0
    matches = 0
    total = 0
    for k in ["w", "h", "t"]:
        if dim1[k] is not None and dim2[k] is not None:
            total += 1
            if abs(dim1[k] - dim2[k]) <= 1:
                matches += 1
    return matches / total if total > 0 else 0

def get_category(description):
    desc = description.lower()
    if any(x in desc for x in ["cable", "dây", "cáp"]) and any(s in desc for s in ["mm2", "mm²"]):
        return "cable"
    elif "conduit" in desc or "ống" in desc:
        return "conduit"
    elif "tray" in desc or "máng cáp" in desc:
        return "tray"
    elif "rack" in desc or "thang cáp" in desc:
        return "rack"
    return "general"

# ------------------------------
# App Configuration
# ------------------------------
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
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    est["category"] = est[est_cols[1]].apply(get_category)
    est["dimensions"] = est[est_cols[1]].apply(extract_dimensions)
    est["combined"] = (est[est_cols[0]].fillna('') + " " + est[est_cols[1]].fillna('') + " " + est[est_cols[2]].fillna('')).apply(clean)

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
    db["dimensions"] = db[db_cols[1]].apply(extract_dimensions)
    db["combined"] = (db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna('')).apply(clean)

    output_data = []
    for i, row in est.iterrows():
        query = row["combined"]
        cat = row["category"]
        qty = row[est_cols[4]]
        unit = row[est_cols[3]]
        dimensions = row["dimensions"]

        matches = db[db["category"] == cat].copy()
        if cat in ["rack", "tray"]:
            matches["dim_score"] = matches["dimensions"].apply(lambda d: dimension_similarity(d, dimensions))
            matches = matches[matches["dim_score"] >= 0.7]
        if matches.empty:
            best = None
        else:
            matches["score"] = matches["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
            best = matches.loc[matches["score"].idxmax()] if matches["score"].max() >= 75 else None

        if best is not None:
            m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce")
            desc_proposed = best[db_cols[1]]
        else:
            m_cost = l_cost = 0
            desc_proposed = ""

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
    grand_row = pd.DataFrame([[""] * 10 + [grand_total]], columns=result_df.columns)
    result_final = pd.concat([result_df, grand_row], ignore_index=True)

    st.subheader("🔍 Matched Estimation")
    display_df = result_final.copy()
    for col in ["Quantity", "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0).astype(int).map("{:,}".format)
    st.dataframe(display_df)

    unmatched_df = result_df[result_df["Description (proposed)"] == ""]
    st.subheader("❌ Unmatched Rows")
    if not unmatched_df.empty:
        st.dataframe(unmatched_df)
    else:
        st.info("✅ All rows matched successfully!")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_final.to_excel(writer, index=False, sheet_name="Matched Results")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")

    st.download_button("📥 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")
