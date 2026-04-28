# ===============================
# BUILDWISE - FULL MATCHING + PASTE
# ===============================

import streamlit as st
import pandas as pd
import os
import re
import io
from rapidfuzz import fuzz

# ===============================
# CLEAN + NORMALIZE
# ===============================
def clean(text):
    text = str(text).lower()
    text = re.sub(r"(sqmm|sqm|mm2|mm²)", "mm2", text)
    text = re.sub(r"(\d)\s*mm2", r"\1mm2", text)
    text = re.sub(r"(\d)\s*[cC]\s*[x×]\s*(\d+)", r"\1x\2", text)
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ===============================
# PARSE CABLE
# ===============================
MAIN_RE = re.compile(r"(\d+)x(\d+)")

def parse_cable_spec(text):
    text = clean(text)
    m = MAIN_RE.search(text)

    if m:
        return {
            "main_key": f"{m.group(1)}x{m.group(2)}",
            "aux_key": ""
        }

    return {"main_key": "", "aux_key": ""}

# ===============================
# MATERIAL
# ===============================
MATERIAL_RE = re.compile(r"(cu|xlpe|pvc|dsta|data|al)", re.I)

def extract_material_structure_tokens(text):
    return MATERIAL_RE.findall(str(text).lower())

def material_structure_score(q, r):
    if not q or not r:
        return 0
    return len(set(q)&set(r)) / len(set(q)|set(r)) * 100

# ===============================
# VOLTAGE
# ===============================
VOLTAGE_RE = re.compile(r"(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*k?v", re.I)

def extract_voltage(text):
    m = VOLTAGE_RE.search(str(text))
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)))

# ===============================
# MATCH SCORE
# ===============================
def combined_match_score(q, qm, qa, qmat, qv, r, rm, ra, rmat, rv):
    size_score = fuzz.token_set_ratio(qm, rm)
    core_score = fuzz.token_set_ratio(qa, ra)
    mat_score = material_structure_score(qmat, rmat)

    # HARD RULE VOLTAGE
    if qv and rv:
        if rv[1] < qv[1]:
            return 0

    return 0.45*size_score + 0.25*core_score + 0.30*mat_score

# ===============================
# PASTE PARSE
# ===============================
def parse_paste_to_df(paste_text):
    try:
        return pd.read_csv(io.StringIO(paste_text), sep="\t")
    except:
        return None

def map_columns(df):

    def is_number(x):
        try:
            float(str(x))
            return True
        except:
            return False

    result = pd.DataFrame()

    # detect quantity
    scores = {}
    for col in df.columns:
        scores[col] = sum(is_number(v) for v in df[col].head(10))

    qty_col = max(scores, key=scores.get)

    # detect description
    desc_col = None
    for col in df.columns:
        if col == qty_col:
            continue
        if any(re.search(r"\d+x\d+|cu|xlpe|pvc", str(v).lower()) for v in df[col][:10]):
            desc_col = col
            break

    result["Mô tả"] = df[desc_col] if desc_col else ""
    result["Số lượng"] = df[qty_col] if qty_col else ""
    result["Đơn vị"] = ""
    result["Model"] = ""

    return result

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(layout="wide")
st.title("BuildWise Estimation Tool")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------
# PRICE LIST
# -------------------------------
st.subheader("1. Upload price list")

uploads = st.file_uploader("Upload price list", type=["xlsx"], accept_multiple_files=True)

if uploads:
    for f in uploads:
        with open(os.path.join(DATA_DIR, f.name), "wb") as out:
            out.write(f.read())

price_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")]

# -------------------------------
# MATCHING
# -------------------------------
st.subheader("2. Matching")

est_file = st.file_uploader("Upload estimation file", type=["xlsx"])

st.markdown("### 📥 Paste from Excel")
paste_text = st.text_area("Paste here", height=200)

if st.button("Chuẩn hóa dữ liệu"):
    df_raw = parse_paste_to_df(paste_text)

    if df_raw is not None:
        df_std = map_columns(df_raw)
        df_std.insert(0, "TT", range(1, len(df_std)+1))
        st.session_state["est"] = df_std

if "est" in st.session_state:
    st.dataframe(st.session_state["est"], use_container_width=True)

# -------------------------------
# MATCH
# -------------------------------
if st.button("Match now"):

    if est_file is None and "est" not in st.session_state:
        st.error("No estimation input")
        st.stop()

    if not price_files:
        st.error("No price list")
        st.stop()

    # READ EST
    if est_file:
        est = pd.read_excel(est_file)
    else:
        est = st.session_state["est"]

    # PREPROCESS EST
    base_est = est["Mô tả"].fillna("")
    est["combined"] = base_est.apply(clean)

    parsed_est = base_est.apply(parse_cable_spec)
    est["main_key"] = parsed_est.apply(lambda d: d["main_key"])
    est["aux_key"] = parsed_est.apply(lambda d: d["aux_key"])
    est["materials"] = base_est.apply(extract_material_structure_tokens)
    est["voltage"] = base_est.apply(extract_voltage)

    # READ DB
    db = pd.concat([pd.read_excel(os.path.join(DATA_DIR, f)) for f in price_files])

    base_db = db.iloc[:,1].fillna("")
    db["combined"] = base_db.apply(clean)

    parsed_db = base_db.apply(parse_cable_spec)
    db["main_key"] = parsed_db.apply(lambda d: d["main_key"])
    db["aux_key"] = parsed_db.apply(lambda d: d["aux_key"])
    db["materials"] = base_db.apply(extract_material_structure_tokens)
    db["voltage"] = base_db.apply(extract_voltage)

    results = []

    for _, row in est.iterrows():
        q = row["combined"]

        best = None
        best_score = -1

        for _, r in db.iterrows():
            score = combined_match_score(
                q,
                row["main_key"],
                row["aux_key"],
                row["materials"],
                row["voltage"],
                r["combined"],
                r["main_key"],
                r["aux_key"],
                r["materials"],
                r["voltage"],
            )

            if score > best_score:
                best_score = score
                best = r

        results.append([
            row["Mô tả"],
            best.iloc[1] if best is not None else "",
            best_score
        ])

    df_res = pd.DataFrame(results, columns=["Requested", "Matched", "Score"])

    st.dataframe(df_res, use_container_width=True)
