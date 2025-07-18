import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

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
    match = re.search(r"\b\d{1,2}\s*[cC×x]\s*\d{1,3}(\.\d+)?\b", str(text))
    return match.group(0).replace(" ", "") if match else ""

def extract_voltage(text):
    match = re.search(r"\b0[.,]?6[ /-]?1[.,]?0?k?[vV]?\b", str(text))
    return "0.6/1kV" if match else ""

def extract_material(text):
    text = str(text).lower()
    if "nhôm" in text or "al" in text or "aluminium" in text:
        return "al"
    if "cu" in text:
        return "cu"
    return ""

def extract_insulation(text):
    text = str(text).upper()
    for ins in ["XLPE", "PVC", "PE", "LSZH"]:
        if ins in text:
            return ins
    return ""

def is_cable(text):
    text = str(text).lower()
    return any(keyword in text for keyword in ["cáp", "cable", "dây"])

def get_best_match(row, db):
    if not is_cable(row["combined"]):
        return None

    row_size = extract_size(row["combined"])
    row_voltage = extract_voltage(row["combined"])
    row_material = extract_material(row["combined"])
    row_insulation = extract_insulation(row["combined"])

    candidates = db[db["combined"].apply(is_cable)].copy()
    candidates["score"] = candidates["combined"].apply(lambda x: fuzz.token_set_ratio(row["combined"], x))

    if row_size:
        candidates = candidates[candidates["combined"].str.contains(row_size)]
    if row_voltage:
        candidates = candidates[candidates["combined"].str.contains("0.6/1kV")]
    if row_material:
        candidates = candidates[candidates["combined"].str.contains(row_material)]
    if row_insulation:
        candidates = candidates[candidates["combined"].str.contains(row_insulation.lower())]

    if candidates.empty:
        return None

    top = candidates.loc[candidates["score"].idxmax()]
    if top["score"] < 70:
        return None

    return top

st.set_page_config(page_title="BuildWise", layout="wide")
st.title("🔌 BuildWise Cable Matching Only")

est_file = st.file_uploader("Upload Estimation File", type=["xlsx"], key="est")
db_file = st.file_uploader("Upload Cable Price List", type=["xlsx"], key="db")

if est_file and db_file:
    est = pd.read_excel(est_file).dropna(how="all")
    db = pd.read_excel(db_file).dropna(how="all")

    est_cols = est.columns.tolist()
    db_cols = db.columns.tolist()

    est["combined"] = (est[est_cols[0]].astype(str) + " " + est[est_cols[1]].astype(str) + " " + est[est_cols[2]].astype(str)).apply(clean)
    db["combined"] = (db[db_cols[0]].astype(str) + " " + db[db_cols[1]].astype(str) + " " + db[db_cols[2]].astype(str)).apply(clean)

    results = []
    for _, row in est.iterrows():
        match = get_best_match(row, db)
        qty = pd.to_numeric(row[est_cols[4]], errors="coerce")
        qty = qty if pd.notna(qty) else 0

        if match is not None:
            mat_cost = pd.to_numeric(match[db_cols[4]], errors="coerce")
            lab_cost = pd.to_numeric(match[db_cols[5]], errors="coerce")
            desc_proposed = match[db_cols[1]]
        else:
            mat_cost = lab_cost = 0
            desc_proposed = ""

        amt_mat = qty * mat_cost
        amt_lab = qty * lab_cost
        total = amt_mat + amt_lab

        results.append([
            row[est_cols[0]], row[est_cols[1]], desc_proposed,
            row[est_cols[2]], row[est_cols[3]], qty,
            mat_cost, lab_cost, amt_mat, amt_lab, total
        ])

    result_df = pd.DataFrame(results, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification",
        "Unit", "Quantity", "Material Cost", "Labour Cost", "Amount Material",
        "Amount Labour", "Total"
    ])

    st.subheader("Matched Results")
    st.dataframe(result_df)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False)
    st.download_button("Download Matched Results", buffer.getvalue(), file_name="Cable_Matching_Result.xlsx")
