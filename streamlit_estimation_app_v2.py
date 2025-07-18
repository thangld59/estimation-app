# Streamlit Estimation App - Cable Matching Fixed
import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

def clean(text):
    text = str(text).lower()
    text = re.sub(r"[()/-]", " ", text)
    text = re.sub(r"mm2|mm²", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_cable_attributes(text):
    text = text.lower()

    size_match = re.search(r"\b\d{1,2}\s*[cCx×x]\s*\d{1,3}(\.\d+)?", text)
    voltage_match = re.search(r"\b0[.,]?6[ /-]?1[.,]?0?k?[vV]?", text)
    material_match = re.search(r"\b(cu|nhôm|al|aluminium)\b", text)
    insulation_match = re.search(r"\b(xlpe|pvc|pe|lszh)\b", text)
    shielding_match = re.search(r"(screen|tape|copper shield|armored|swa|sta)", text)

    return {
        "size": size_match.group(0).replace(" ", "").lower() if size_match else "",
        "voltage": "0.6/1kV" if voltage_match else "",
        "material": material_match.group(0).capitalize() if material_match else "",
        "insulation": insulation_match.group(0).upper() if insulation_match else "",
        "shielding": shielding_match.group(0) if shielding_match else ""
    }

def is_cable(text):
    return any(keyword in text.lower() for keyword in ["cáp", "cable", "dây"])

st.set_page_config(page_title="BuildWise", layout="wide")
st.title("BuildWise - Estimation Tool")

username = st.sidebar.text_input("Enter your username")
if not username:
    st.stop()

user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)

estimation_file = st.file_uploader("Upload Estimation File", type="xlsx", key="est")
price_list_file = st.file_uploader("Upload Price List File", type="xlsx", key="price")

if estimation_file and price_list_file:
    est = pd.read_excel(estimation_file).dropna(how="all")
    price = pd.read_excel(price_list_file).dropna(how="all")

    if len(est.columns) < 5 or len(price.columns) < 7:
        st.error("Estimation must have 5 cols and Price list at least 7 cols.")
        st.stop()

    est.columns = [f"col{i}" for i in range(len(est.columns))]
    price.columns = [f"col{i}" for i in range(len(price.columns))]

    est["desc"] = est["col0"].fillna("") + " " + est["col1"].fillna("") + " " + est["col2"].fillna("")
    price["desc"] = price["col0"].fillna("") + " " + price["col1"].fillna("") + " " + price["col2"].fillna("")

    est["cleaned"] = est["desc"].apply(clean)
    price["cleaned"] = price["desc"].apply(clean)

    est["attr"] = est["desc"].apply(extract_cable_attributes)
    price["attr"] = price["desc"].apply(extract_cable_attributes)

    results = []
    for i, row in est.iterrows():
        best_score = 0
        best_row = None
        for j, prow in price.iterrows():
            if not is_cable(prow["desc"]):
                continue
            score = 0
            e_attr = row["attr"]
            p_attr = prow["attr"]

            if e_attr["size"] and e_attr["size"] == p_attr["size"]: score += 30
            if e_attr["voltage"] and e_attr["voltage"] == p_attr["voltage"]: score += 25
            if e_attr["material"] and e_attr["material"] == p_attr["material"]: score += 20
            if e_attr["insulation"] and e_attr["insulation"] == p_attr["insulation"]: score += 10
            if e_attr["shielding"] and e_attr["shielding"] in p_attr["shielding"]: score += 5

            fuzzy_score = fuzz.token_set_ratio(row["cleaned"], prow["cleaned"]) // 10
            score += fuzzy_score

            if score > best_score:
                best_score = score
                best_row = prow

        if best_score >= 70 and best_row is not None:
            m_cost = best_row["col5"]
            l_cost = best_row["col6"]
            desc_proposed = best_row["col1"]
        else:
            m_cost = l_cost = 0
            desc_proposed = ""

        qty = row["col4"]
        try:
            qty_val = float(qty)
        except:
            qty_val = 0
        amt_mat = qty_val * m_cost
        amt_lab = qty_val * l_cost
        total = amt_mat + amt_lab

        results.append([
            row["col0"], row["col1"], desc_proposed, row["col2"],
            row["col3"], qty,
            m_cost, l_cost, amt_mat, amt_lab, total
        ])

    out_df = pd.DataFrame(results, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification",
        "Unit", "Quantity", "Material Cost", "Labour Cost",
        "Amount Material", "Amount Labour", "Total"
    ])
    st.dataframe(out_df)
