# streamlit_estimation_app_cable_admin.py
import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

# =========================================================
# PAGE / HEADER
# =========================================================
st.set_page_config(page_title="BuildWise", page_icon="📐", layout="wide")
try:
    st.image("assets/logo.png", width=120)
except Exception:
    pass
st.title(":triangular_ruler: BuildWise - Smart Estimation Tool (Cable improved)")

# ---------------------------------------------------------
# Simple identity (not secure auth; just role switch)
# ---------------------------------------------------------
username = st.sidebar.text_input("Username")
if not username:
    st.warning("Please enter your username to continue.")
    st.stop()

# Threshold control (you can adjust if matches look too strict/loose)
cable_threshold = st.sidebar.slider("Cable match threshold", 0, 100, 70)

# Folders
USER_ROOT = "user_data"
FORM_ROOT = "shared_forms"
os.makedirs(USER_ROOT, exist_ok=True)
os.makedirs(FORM_ROOT, exist_ok=True)

user_folder = os.path.join(USER_ROOT, username)
os.makedirs(user_folder, exist_ok=True)

# =========================================================
# TEXT NORMALIZATION / FEATURE EXTRACTION
# =========================================================
MATERIAL_KEYWORDS = {
    "cu": "CU",
    "đồng": "CU",
    "dong": "CU",
    "al": "AL",
    "nhom": "AL",
    "nhôm": "AL",
    "aluminium": "AL",
}

INSULATION_KEYWORDS = {
    "xlpe": "XLPE",
    "pvc": "PVC",
    "pe": "PE",
    "lszh": "LSZH",
}

SHIELD_KEYWORDS = ["screen", "tape", "shield", "armored", "swa", "sta", "a", "armour", "armored"]
CABLE_WORDS = ["cáp", "cap", "cable", "dây điện", "day dien", "wire"]

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def clean_text(text: str) -> str:
    s = str(text).lower()
    s = s.replace("mm^2", "mm2").replace("mm²", "mm2")
    s = s.replace("/", " / ").replace("-", " ")
    s = s.replace(",", " ")
    s = re.sub(r"\b0[.,]?\s*6\s*kv\b", "0.6kv", s)
    s = re.sub(r"\b1[.,]?\s*0\s*kv\b", "1.0kv", s)
    s = normalize_spaces(s)
    return s

def seems_like_cable(s: str) -> bool:
    # If has any cable keywords OR a cores×size pattern OR XLPE/PVC words, treat as cable
    if any(w in s for w in CABLE_WORDS):
        return True
    if re.search(r"\b\d{1,2}\s*[x×]\s*\d{1,3}(\.\d+)?\b", s):
        return True
    if any(k in s for k in ["xlpe", "pvc", "lszh", "pe"]):
        return True
    return False

def detect_category(text: str) -> str:
    s = clean_text(text)
    return "cable" if seems_like_cable(s) else "other"

def extract_voltage(text: str) -> str:
    s = clean_text(text)
    m = re.search(r"0[.,]?\s*6\s*[/\- ]\s*1(?:[.,]?\s*0)?\s*k?v?", s)
    return "0.6/1kV" if m else ""

def extract_materials(text: str):
    s = clean_text(text)
    mats = set()
    for k, std in MATERIAL_KEYWORDS.items():
        if re.search(rf"\b{k}\b", s):
            mats.add(std)
    return sorted(mats)

def extract_insulations(text: str):
    s = clean_text(text)
    ins = set()
    for k, std in INSULATION_KEYWORDS.items():
        if k in s:
            ins.add(std)
    return sorted(ins)

def has_shield(text: str) -> bool:
    s = clean_text(text)
    return any(k in s for k in SHIELD_KEYWORDS)

def extract_core_size_main_and_extra(text: str):
    """
    Returns: (main_cores:int|None, main_size:float|None, extra_type:str|None ('E','N','1C'), extra_size:float|None)
    Supports:
      '3C x 70mm2 + 1C x 50mm2'
      '3x2.5 + E50'
      '3Cx70 + N50'
    """
    s = clean_text(text).replace("(", " ").replace(")", " ")
    s = normalize_spaces(s)

    main_cores = None
    main_size = None
    # MAIN cores × size
    m = re.search(r"\b(\d{1,2})\s*(?:c|core|cores|sợi|soi)?\s*[x×]\s*(\d{1,3}(?:[.,]\d{1,2})?)\b", s)
    if m:
        main_cores = int(m.group(1))
        main_size = float(m.group(2).replace(",", "."))

    # Fallback for size-only like "2.5mm2"
    if main_size is None:
        ms = re.search(r"\b(\d{1,3}(?:[.,]\d{1,2})?)\s*mm2\b", s)
        if ms:
            main_size = float(ms.group(1).replace(",", "."))

    extra_type = None
    extra_size = None
    # + 1C x 50
    mA = re.search(r"\+\s*1\s*c?\s*[x×]?\s*(\d{1,3}(?:[.,]\d{1,2})?)\b", s)
    if mA:
        extra_type = "1C"
        extra_size = float(mA.group(1).replace(",", "."))
    # + E50 or + N50
    mB = re.search(r"\+\s*([en])\s*(\d{1,3}(?:[.,]\d{1,2})?)\b", s)
    if mB:
        extra_type = mB.group(1).upper()
        extra_size = float(mB.group(2).replace(",", "."))

    return (main_cores, main_size, extra_type, extra_size)

def features_from_text(text: str):
    return {
        "category": detect_category(text),
        "voltage": extract_voltage(text),
        "materials": extract_materials(text),
        "insulations": extract_insulations(text),
        "shield": has_shield(text),
        "main_cores_size": extract_core_size_main_and_extra(text),
    }

# Weighted scoring
WEIGHTS = {
    "material_overlap": 35,
    "insulation_overlap": 25,
    "shield_match": 10,
    "voltage_match": 10,   # medium
    "cores_exact": 40,     # strong
    "size_exact": 40,      # strong
    "extra_line_match": 20 # E/N/1C exact
}

def list_overlap_score(q_list, t_list):
    if not q_list or not t_list:
        return 0
    q = set(q_list); t = set(t_list)
    if not q:
        return 0
    return int(round(100 * len(q & t) / max(1, len(q))))

def calc_weighted_score(qf, tf, size_tol_percent=None):
    score = 0

    # Lists
    score += WEIGHTS["material_overlap"] * (list_overlap_score(qf.get("materials", []), tf.get("materials", [])) / 100)
    score += WEIGHTS["insulation_overlap"] * (list_overlap_score(qf.get("insulations", []), tf.get("insulations", [])) / 100)

    # Booleans
    if qf.get("shield", False) == tf.get("shield", False):
        score += WEIGHTS["shield_match"]

    # Voltage
    if qf.get("voltage") and tf.get("voltage") and qf["voltage"] == tf["voltage"]:
        score += WEIGHTS["voltage_match"]

    # Cores & size
    q_cores, q_size, q_extra_type, q_extra_size = qf.get("main_cores_size", (None, None, None, None))
    t_cores, t_size, t_extra_type, t_extra_size = tf.get("main_cores_size", (None, None, None, None))

    if q_cores is not None and t_cores is not None and q_cores == t_cores:
        score += WEIGHTS["cores_exact"]

    if q_size is not None and t_size is not None:
        if size_tol_percent is None:
            if abs(q_size - t_size) < 1e-9:
                score += WEIGHTS["size_exact"]
        else:
            if t_size > 0 and abs(q_size - t_size) / t_size * 100 <= size_tol_percent:
                score += max(WEIGHTS["size_exact"] - 4, 0)

    if q_extra_type and t_extra_type:
        if q_extra_type == t_extra_type and q_extra_size is not None and t_extra_size is not None:
            if abs(q_extra_size - t_extra_size) < 1e-9:
                score += WEIGHTS["extra_line_match"]

    return int(round(score))

# =========================================================
# SHARED FORMS (Admin123 upload/delete; users download)
# =========================================================
st.subheader(":scroll: Price List & Estimation Request Forms (Mẫu Bảng Giá & Yêu Cầu Vào Giá)")
form_files = sorted(os.listdir(FORM_ROOT))

if username == "Admin123":
    form_uploads = st.file_uploader("Upload form files", type=["xlsx", "xls"], accept_multiple_files=True, key="form_upload")
    if form_uploads:
        for f in form_uploads:
            with open(os.path.join(FORM_ROOT, f.name), "wb") as out_file:
                out_file.write(f.read())
        st.success("Form file(s) uploaded successfully.")
        st.rerun()

    if form_files:
        form_to_delete = st.selectbox("Delete a shared form file", [""] + form_files, key="form_delete")
        if form_to_delete and st.button("Delete Selected Form File"):
            try:
                os.remove(os.path.join(FORM_ROOT, form_to_delete))
                st.success(f"Deleted form file: {form_to_delete}")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting form file: {e}")
else:
    if form_files:
        st.caption("Download shared forms:")
        for file in form_files:
            with open(os.path.join(FORM_ROOT, file), "rb") as f:
                st.download_button(f"📄 Download {file}", f.read(), file_name=file)
    else:
        st.info("No shared forms yet.")

# =========================================================
# USER PRICE LISTS (upload / manage / delete)
# =========================================================
st.subheader(":file_folder: Upload Price List Files")
uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(user_folder, file.name), "wb") as f:
            f.write(file.read())
    st.success(":white_check_mark: Price list uploaded successfully.")
    st.rerun()

st.subheader(":open_file_folder: Manage Price Lists")
price_list_files = sorted(os.listdir(user_folder))
selected_file = st.radio("Choose one file to match or use all", ["All files"] + price_list_files)

# Deletion
if price_list_files:
    file_to_delete = st.selectbox("Select a price list to delete", [""] + price_list_files, key="delete_pl")
    if file_to_delete and st.button("Delete Selected Price List"):
        try:
            os.remove(os.path.join(user_folder, file_to_delete))
            st.success(f"Deleted file: {file_to_delete}")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting file: {e}")

# =========================================================
# ESTIMATION FILE & MATCHING
# =========================================================
st.subheader(":page_facing_up: Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est")

if estimation_file and price_list_files:
    est = pd.read_excel(estimation_file).dropna(how="all")
    est_cols = est.columns.tolist()
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    # Prepare Estimation features
    est["combined_raw"] = est[est_cols[0]].astype(str).fillna('') + " " + est[est_cols[1]].astype(str).fillna('') + " " + est[est_cols[2]].astype(str).fillna('')
    est["combined"] = est["combined_raw"].apply(clean_text)
    est_feats = est["combined"].apply(features_from_text)
    est = pd.concat([est, est_feats.apply(pd.Series)], axis=1)

    # Read DB(s)
    if selected_file == "All files":
        frames = []
        for f in price_list_files:
            df = pd.read_excel(os.path.join(user_folder, f)).dropna(how="all")
            df["__source__"] = f
            frames.append(df)
        db = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        db = pd.read_excel(os.path.join(user_folder, selected_file)).dropna(how="all")
        db["__source__"] = selected_file

    if db.empty:
        st.error("Your selected price list(s) are empty.")
        st.stop()

    db_cols = db.columns.tolist()
    if len(db_cols) < 6:
        st.error("Price list file must have at least 6 columns (with description in column 2, material in col 5, labour in col 6).")
        st.stop()

    # DB features from first 3 columns
    db["combined_raw"] = db[db_cols[0]].astype(str).fillna('') + " " + db[db_cols[1]].astype(str).fillna('') + " " + db[db_cols[2]].astype(str).fillna('')
    db["combined"] = db["combined_raw"].apply(clean_text)
    db_feats = db["combined"].apply(features_from_text)
    db = pd.concat([db, db_feats.apply(pd.Series)], axis=1)

    # MATCHING
    output_rows = []

    for i, row in est.iterrows():
        # Only attempt cable match if row looks like cable. Otherwise, output unchanged (unmatched).
        if row["category"] != "cable":
            desc_proposed = ""
            m_cost = l_cost = 0
            unit = row[est_cols[3]]
            qty = row[est_cols[4]]
            qty_val = pd.to_numeric(qty, errors="coerce"); qty_val = 0 if pd.isna(qty_val) else qty_val
            amt_mat = qty_val * m_cost
            amt_lab = qty_val * l_cost
            total = amt_mat + amt_lab
            output_rows.append([
                row[est_cols[0]], row[est_cols[1]], desc_proposed, row[est_cols[2]],
                unit, qty, m_cost, l_cost, amt_mat, amt_lab, total
            ])
            continue

        # Build query features
        qf = {
            "materials": row.get("materials", []) or [],
            "insulations": row.get("insulations", []) or [],
            "shield": bool(row.get("shield", False)),
            "voltage": row.get("voltage", "") or "",
            "main_cores_size": row.get("main_cores_size", (None, None, None, None)),
        }

        # Start with all cable rows in DB
        cand = db[db["category"] == "cable"].copy()

        # Soft filters: if we have cores/size, try to keep items that at least parsed something comparable.
        q_cores, q_size, _, _ = qf["main_cores_size"]
        if q_cores is not None:
            cand = cand[cand["main_cores_size"].apply(lambda t: isinstance(t, tuple) and t[0] is not None)]
        if cand.empty:
            cand = db[db["category"] == "cable"].copy()

        if q_size is not None:
            cand = cand[cand["main_cores_size"].apply(lambda t: isinstance(t, tuple) and t[1] is not None)]
        if cand.empty:
            cand = db[db["category"] == "cable"].copy()

        # Compute weighted scores per row
        def row_score(r):
            tf = {
                "materials": r.get("materials", []) or [],
                "insulations": r.get("insulations", []) or [],
                "shield": bool(r.get("shield", False)),
                "voltage": r.get("voltage", "") or "",
                "main_cores_size": r.get("main_cores_size", (None, None, None, None)),
            }
            w = calc_weighted_score(qf, tf, size_tol_percent=None)
            fuzzy = fuzz.token_set_ratio(row["combined"], r["combined"])
            return w + 0.1 * fuzzy

        if not cand.empty:
            cand = cand.copy()
            cand["score_final"] = cand.apply(row_score, axis=1)
            cand = cand.sort_values("score_final", ascending=False)

        best = None
        if not cand.empty and cand.iloc[0]["score_final"] >= cable_threshold:
            best = cand.iloc[0]

        if best is not None:
            desc_proposed = best[db_cols[1]]
            m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce")
            m_cost = 0 if pd.isna(m_cost) else m_cost
            l_cost = 0 if pd.isna(l_cost) else l_cost
        else:
            desc_proposed = ""
            m_cost = l_cost = 0

        unit = row[est_cols[3]]
        qty = row[est_cols[4]]
        qty_val = pd.to_numeric(qty, errors="coerce"); qty_val = 0 if pd.isna(qty_val) else qty_val
        amt_mat = qty_val * m_cost
        amt_lab = qty_val * l_cost
        total = amt_mat + amt_lab

        output_rows.append([
            row[est_cols[0]],  # Model
            row[est_cols[1]],  # Description (requested)
            desc_proposed,     # Description (proposed)
            row[est_cols[2]],  # Specification
            unit,              # Unit
            qty,               # Quantity
            m_cost,            # Material Cost
            l_cost,            # Labour Cost
            amt_mat,           # Amount Material
            amt_lab,           # Amount Labour
            total              # Total
        ])

    # Result tables
    result_df = pd.DataFrame(output_rows, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification",
        "Unit", "Quantity", "Material Cost", "Labour Cost",
        "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum(skipna=True)
    grand_row = pd.DataFrame([[""] * 10 + [grand_total]], columns=result_df.columns)
    result_final = pd.concat([result_df, grand_row], ignore_index=True)

    # Display
    st.subheader(":mag: Matched Estimation")
    display_df = result_final.copy()
    display_df["Quantity"] = pd.to_numeric(display_df["Quantity"], errors="coerce").fillna(0).astype(int).map("{:,}".format)
    for col in ["Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0).astype(int).map("{:,}".format)
    st.dataframe(display_df, use_container_width=True)

    st.subheader(":x: Unmatched Rows")
    unmatched_df = result_df[result_df["Description (proposed)"] == ""]
    if not unmatched_df.empty:
        st.dataframe(unmatched_df, use_container_width=True)
    else:
        st.info(":white_check_mark: All rows matched successfully!")

    # Export
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_final.to_excel(writer, index=False, sheet_name="Matched Results")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")
    st.download_button("📥 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")
