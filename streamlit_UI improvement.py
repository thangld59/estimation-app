# =============================
# BUILDWISE - CLEAN VERSION
# =============================

import streamlit as st
import pandas as pd
import os
import io
import re
from rapidfuzz import fuzz

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="BuildWise", layout="wide")

user_folder = "user_data"
os.makedirs(user_folder, exist_ok=True)

# -------------------------
# HELPERS
# -------------------------
def list_price_list_files(folder):
    return [f for f in os.listdir(folder) if f.endswith(".xlsx")]

# -------------------------
# CLEAN TEXT
# -------------------------
def clean(text):
    text = str(text).lower()
    text = re.sub(r"(sqmm|sqm|mm2)", "mm2", text)
    text = re.sub(r"(\d)\s*mm2", r"\1mm2", text)
    text = text.replace("mm2", "")
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------
# PARSE PASTE (SMART)
# -------------------------
def parse_paste_to_df(paste_text):
    try:
        lines = paste_text.strip().split("\n")
        data = []

        for line in lines:
            parts = re.split(r"\t| {2,}", line.strip())

            if len(parts) == 1:
                parts = line.split()

            data.append(parts)

        df = pd.DataFrame(data)

        # detect header
        header_keywords = ["mô tả", "qty", "sl", "unit"]

        first_row = df.iloc[0].astype(str).str.lower().tolist()

        if any(any(k in c for k in header_keywords) for c in first_row):
            df.columns = df.iloc[0]
            df = df[1:]
            df.reset_index(drop=True, inplace=True)
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        return df

    except:
        return None

# -------------------------
# MAP COLUMNS (SMART)
# -------------------------
def map_columns(df):

    def is_number(x):
        try:
            float(str(x))
            return True
        except:
            return False

    result = pd.DataFrame()

    # detect quantity = column nhiều số nhất
    scores = {}

    for col in df.columns:
        values = df[col].astype(str)
        num_count = sum(is_number(v) for v in values[:10])
        scores[col] = num_count

    qty_col = max(scores, key=scores.get)

    # detect description = column có cable pattern
    desc_col = None
    for col in df.columns:
        if col == qty_col:
            continue
        if any(re.search(r"\d+x\d+|cu|pvc|xlpe", str(v).lower()) for v in df[col][:10]):
            desc_col = col
            break

    result["Mô tả"] = df[desc_col] if desc_col else ""
    result["Số lượng"] = df[qty_col] if qty_col else ""

    result["Model"] = ""
    result["Hãng"] = ""
    result["Đơn vị"] = ""

    return result

# -------------------------
# MATCH FUNCTION (SIMPLE)
# -------------------------
def match_simple(est, db):

    results = []

    for _, row in est.iterrows():
        q = clean(row["Mô tả"])

        best = None
        best_score = -1

        for _, r in db.iterrows():
            score = fuzz.token_set_ratio(q, clean(r[1]))

            if score > best_score:
                best_score = score
                best = r

        if best is not None:
            results.append([
                best[0],
                row["Mô tả"],
                best[1],
                row["Đơn vị"],
                row["Số lượng"],
            ])
        else:
            results.append(["", row["Mô tả"], "", "", ""])

    return pd.DataFrame(results, columns=[
        "Model",
        "Description (requested)",
        "Description (proposed)",
        "Unit",
        "Quantity"
    ])

# =========================
# UI
# =========================

# 1. UPLOAD PRICE LIST
st.subheader("1. Upload price list files")

uploads = st.file_uploader("Upload", type=["xlsx"], accept_multiple_files=True)

if uploads:
    for f in uploads:
        with open(os.path.join(user_folder, f.name), "wb") as out:
            out.write(f.read())
    st.success("Uploaded!")

# -------------------------
# 2. MANAGE
# -------------------------
st.subheader("2. Manage price lists")

price_list_files = list_price_list_files(user_folder)

if price_list_files:
    st.write(price_list_files)
else:
    st.info("No price list")

selected_file = st.selectbox("Select file", ["All files"] + price_list_files)

# -------------------------
# 3. MATCHING
# -------------------------
st.subheader("3. Matching estimation request file")

estimation_file = st.file_uploader("Upload estimation", type=["xlsx"])

st.markdown("### 📥 Paste from Excel")

paste_text = st.text_area("Paste here", height=200)

if st.button("Chuẩn hóa dữ liệu"):
    df_raw = parse_paste_to_df(paste_text)

    if df_raw is None:
        st.error("Cannot read paste")
    else:
        df_std = map_columns(df_raw)
        df_std.insert(0, "TT", range(1, len(df_std)+1))
        st.session_state["est"] = df_std

if "est" in st.session_state:
    st.dataframe(st.session_state["est"], use_container_width=True)

# -------------------------
# MATCH BUTTON
# -------------------------
if st.button("Match now"):

    if estimation_file is None and "est" not in st.session_state:
        st.error("Need input")
        st.stop()

    if not price_list_files:
        st.error("Need price list")
        st.stop()

    # READ EST
    if estimation_file:
        est = pd.read_excel(estimation_file)
        est.columns = ["Model","Mô tả","Hãng","Đơn vị","Số lượng"]
    else:
        est = st.session_state["est"]

    # READ DB
    frames = []
    for f in price_list_files:
        df = pd.read_excel(os.path.join(user_folder, f))
        frames.append(df)

    db = pd.concat(frames)

    result = match_simple(est, db)

    st.dataframe(result, use_container_width=True)

# -------------------------
# 4. QUOTATION
# -------------------------
st.subheader("4. Quotation generation")
