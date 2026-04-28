# ===============================
# BUILDWISE - FINAL (WITH PASTE)
# ===============================

import streamlit as st
import pandas as pd
import os
import re
import json
import io
from io import BytesIO
from datetime import datetime
from rapidfuzz import fuzz
from openpyxl import load_workbook

# ------------------------------
# PASTE EXCEL PARSE + MAP
# ------------------------------
def parse_paste_to_df(paste_text):
    try:
        df = pd.read_csv(io.StringIO(paste_text), sep="\t")
        return df
    except:
        return None


def map_columns(df):
    import re

    def is_number(val):
        try:
            float(str(val).replace(",", ""))
            return True
        except:
            return False

    def is_cable(text):
        text = str(text).lower()
        return bool(re.search(r"\d+x\d+|\d+mm2|cu|xlpe|pvc", text))

    col_scores = {}

    for col in df.columns:
        values = df[col].astype(str)

        score = {
            "Mô tả": 0,
            "Số lượng": 0,
            "Model": 0,
            "Hãng": 0,
            "Đơn vị": 0,
        }

        for v in values.head(10):
            v_low = str(v).lower()

            if is_number(v):
                score["Số lượng"] += 2

            if is_cable(v):
                score["Mô tả"] += 2

            if len(str(v).split()) <= 3:
                score["Model"] += 1

            if any(b in v_low for b in ["cadisun", "cadivi", "ls", "lapp"]):
                score["Hãng"] += 2

            if v_low in ["m", "mtr", "pcs"]:
                score["Đơn vị"] += 2

        col_scores[col] = score

    assigned = {}

    for target in ["Mô tả", "Số lượng", "Model", "Hãng", "Đơn vị"]:
        best_col = None
        best_score = 0

        for col, scores in col_scores.items():
            if scores[target] > best_score and col not in assigned.values():
                best_score = scores[target]
                best_col = col

        if best_col:
            assigned[target] = best_col

    result = pd.DataFrame()

    for target in ["Model", "Mô tả", "Hãng", "Đơn vị", "Số lượng"]:
        if target in assigned:
            result[target] = df[assigned[target]]
        else:
            result[target] = ""

    return result


# ===============================
# STREAMLIT APP (RÚT GỌN UI CHỈ MATCH)
# ===============================
st.set_page_config(page_title="BuildWise", layout="wide")

user_folder = "user_data"
os.makedirs(user_folder, exist_ok=True)

# ------------------------------
# Upload price list
# ------------------------------
st.subheader("1. Upload price list")

uploads = st.file_uploader("Upload price list", type=["xlsx"], accept_multiple_files=True)

if uploads:
    for f in uploads:
        with open(os.path.join(user_folder, f.name), "wb") as out:
            out.write(f.read())
    st.success("Uploaded!")

price_list_files = [f for f in os.listdir(user_folder) if f.endswith(".xlsx")]

# ------------------------------
# Matching
# ------------------------------
st.subheader("2. Matching")

estimation_file = st.file_uploader("Upload estimation file", type=["xlsx"])

st.markdown("### 📥 Paste from Excel")

paste_text = st.text_area("Paste data", height=200)

if st.button("Chuẩn hóa dữ liệu"):
    df_raw = parse_paste_to_df(paste_text)

    if df_raw is None:
        st.error("Cannot read data")
    else:
        df_std = map_columns(df_raw)
        df_std.insert(0, "TT", range(1, len(df_std)+1))
        st.session_state["est_table"] = df_std

if "est_table" in st.session_state:
    st.dataframe(st.session_state["est_table"], use_container_width=True)

# ------------------------------
# MATCH
# ------------------------------
if st.button("Match now"):

    if estimation_file is None and "est_table" not in st.session_state:
        st.error("Need input")
        st.stop()

    if not price_list_files:
        st.error("Need price list")
        st.stop()

    # READ EST
    if estimation_file:
        est = pd.read_excel(estimation_file)
    else:
        est = st.session_state["est_table"]

    # READ DB
    frames = []
    for f in price_list_files:
        df = pd.read_excel(os.path.join(user_folder, f))
        frames.append(df)

    db = pd.concat(frames)

    # SIMPLE MATCH (fallback demo)
    results = []

    for _, row in est.iterrows():
        q = str(row.get("Mô tả", "")).lower()

        best = None
        best_score = -1

        for _, r in db.iterrows():
            score = fuzz.token_set_ratio(q, str(r[1]).lower())

            if score > best_score:
                best_score = score
                best = r

        if best is not None:
            results.append([
                best[0],
                row.get("Mô tả", ""),
                best[1],
                row.get("Đơn vị", ""),
                row.get("Số lượng", ""),
            ])
        else:
            results.append(["", row.get("Mô tả", ""), "", "", ""])

    df_res = pd.DataFrame(results, columns=[
        "Model", "Requested", "Matched", "Unit", "Qty"
    ])

    st.dataframe(df_res, use_container_width=True)
