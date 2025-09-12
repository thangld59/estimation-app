import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

# =========================
# Page / Header
# =========================
st.set_page_config(page_title="BuildWise", page_icon="📐", layout="wide")
try:
    st.image("assets/logo.png", width=120)
except Exception:
    pass
st.title(":triangular_ruler: BuildWise - Smart Estimation Tool (Cable)")

username = st.sidebar.text_input("Username")
if not username:
    st.warning("Please enter your username to continue.")
    st.stop()

# Controls
cable_threshold = st.sidebar.slider("Cable match threshold", 0, 100, 60)
show_debug = st.sidebar.checkbox("Show debug features", value=False)

# Folders
USER_ROOT = "user_data"
FORM_ROOT = "shared_forms"
os.makedirs(USER_ROOT, exist_ok=True)
os.makedirs(FORM_ROOT, exist_ok=True)
user_folder = os.path.join(USER_ROOT, username)
os.makedirs(user_folder, exist_ok=True)

# =========================
# Normalization & Extraction
# =========================
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

MATERIAL_KEYWORDS = {
    "cu": "CU", "đồng": "CU", "dong": "CU",
    "al": "AL", "nhom": "AL", "nhôm": "AL", "aluminium": "AL",
}
INSULATION_KEYWORDS = {"xlpe":"XLPE","pvc":"PVC","pe":"PE","lszh":"LSZH"}
SHIELD_KEYWORDS = ["screen","tape","shield","armored","armour","swa","sta","a"]

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
      '(3x2.5)mm2', '3 x 2.5mm2', etc.
    """
    s = clean_text(text).replace("(", " ").replace(")", " ")
    s = normalize_spaces(s)

    main_cores = None
    main_size = None

    m = re.search(r"\b(\d{1,2})\s*(?:c|core|cores|sợi|soi)?\s*[x×]\s*(\d{1,3}(?:[.,]\d{1,2})?)\b", s)
    if m:
        main_cores = int(m.group(1))
        main_size = float(m.group(2).replace(",", "."))

    if main_size is None:
        ms = re.search(r"\b(\d{1,3}(?:[.,]\d{1,2})?)\s*mm2\b", s)
        if ms:
            main_size = float(ms.group(1).replace(",", "."))

    extra_type = None
    extra_size = None
    mA = re.search(r"\+\s*1\s*c?\s*[x×]?\s*(\d{1,3}(?:[.,]\d{1,2})?)\b", s)
    if mA:
        extra_type = "1C"
        extra_size = float(mA.group(1).replace(",", "."))
    mB = re.search(r"\+\s*([en])\s*(\d{1,3}(?:[.,]\d{1,2})?)\b", s)
    if mB:
        extra_type = mB.group(1).upper()
        extra_size = float(mB.group(2).replace(",", "."))

    return (main_cores, main_size, extra_type, extra_size)

def features_from_text(text: str):
    return {
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
    score += WEIGHTS["material_overlap"] * (list_overlap_score(qf.get("materials", []), tf.get("materials", [])) / 100)
    score += WEIGHTS["insulation_overlap"] * (list_overlap_score(qf.get("insulations", []), tf.get("insulations", [])) / 100)
    if qf.get("shield", False) == tf.get("shield", False):
        score += WEIGHTS["shield_match"]
    if qf.get("voltage") and tf.get("voltage") and qf["voltage"] == tf["voltage"]:
        score += WEIGHTS["voltage_match"]

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

# =========================
# SHARED FORMS (Admin123)
# =========================
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

# =========================
# USER PRICE LISTS (upload/manage/delete)
# =========================
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

if price_list_files:
    file_to_delete = st.selectbox("Select a price list to delete", [""] + price_list_files, key="delete_pl")
    if file_to_delete and st.button("Delete Selected Price List"):
        try:
            os.remove(os.path.join(user_folder, file_to_delete))
            st.success(f"Deleted file: {file_to_delete}")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting file: {e}")

# =========================
# ESTIMATION & MATCHING
# =========================
st.subheader(":page_facing_up: Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est")

if estimation_file and price_list_files:
    # ---------- Read Estimation ----------
    est_raw = pd.read_excel(estimation_file).dropna(how="all")
    if est_raw.shape[1] < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    st.markdown("**Map Estimation columns**")
    est_cols_all = est_raw.columns.tolist()
    est_model_col = st.selectbox("Est: Model column", est_cols_all, index=0)
    est_desc_col  = st.selectbox("Est: Description column", est_cols_all, index=1)
    est_spec_col  = st.selectbox("Est: Specification column", est_cols_all, index=2)
    est_unit_col  = st.selectbox("Est: Unit column", est_cols_all, index=3)
    est_qty_col   = st.selectbox("Est: Quantity column", est_cols_all, index=4)

    est = pd.DataFrame({
        "Model": est_raw[est_model_col].astype(str),
        "Desc":  est_raw[est_desc_col].astype(str),
        "Spec":  est_raw[est_spec_col].astype(str),
        "Unit":  est_raw[est_unit_col],
        "Qty":   est_raw[est_qty_col],
    })
    est["combined_raw"] = (est["Model"] + " " + est["Desc"] + " " + est["Spec"]).apply(str)
    est["combined"] = est["combined_raw"].apply(clean_text)
    est_feats = est["combined"].apply(features_from_text)
    est = pd.concat([est, est_feats.apply(pd.Series)], axis=1)

    # ---------- Read DB ----------
    if selected_file == "All files":
        frames = []
        for f in price_list_files:
            df = pd.read_excel(os.path.join(user_folder, f)).dropna(how="all")
            df["__source__"] = f
            frames.append(df)
        db_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        db_raw = pd.read_excel(os.path.join(user_folder, selected_file)).dropna(how="all")
        db_raw["__source__"] = selected_file

    if db_raw.empty:
        st.error("Your selected price list(s) are empty.")
        st.stop()

    if db_raw.shape[1] < 6:
        st.error("Price list must have at least 6 columns.")
        st.stop()

    st.markdown("**Map Price List columns**")
    db_cols_all = db_raw.columns.tolist()
    db_model_col = st.selectbox("DB: Model column", db_cols_all, index=0)
    db_desc_col  = st.selectbox("DB: Description column", db_cols_all, index=1)
    db_spec_col  = st.selectbox("DB: Specification column", db_cols_all, index=2)
    db_mcost_col = st.selectbox("DB: Material Cost column", db_cols_all, index=4)
    db_lcost_col = st.selectbox("DB: Labour Cost column", db_cols_all, index=5)

    db = pd.DataFrame({
        "DB_Model": db_raw[db_model_col].astype(str),
        "DB_Desc":  db_raw[db_desc_col].astype(str),
        "DB_Spec":  db_raw[db_spec_col].astype(str),
        "DB_MCost": pd.to_numeric(db_raw[db_mcost_col], errors="coerce"),
        "DB_LCost": pd.to_numeric(db_raw[db_lcost_col], errors="coerce"),
        "__source__": db_raw["__source__"]
    })
    db["combined_raw"] = (db["DB_Model"] + " " + db["DB_Desc"] + " " + db["DB_Spec"]).apply(str)
    db["combined"] = db["combined_raw"].apply(clean_text)
    db_feats = db["combined"].apply(features_from_text)
    db = pd.concat([db, db_feats.apply(pd.Series)], axis=1)

    if show_debug:
        st.markdown("#### 🔎 Debug: parsed Estimation features (first 10)")
        st.dataframe(est.head(10))
        st.markdown("#### 🔎 Debug: parsed DB features (first 10)")
        st.dataframe(db.head(10))

    # ---------- Matching ----------
    output_rows = []

    for i, row in est.iterrows():
        qf = {
            "materials": row.get("materials", []) or [],
            "insulations": row.get("insulations", []) or [],
            "shield": bool(row.get("shield", False)),
            "voltage": row.get("voltage", "") or "",
            "main_cores_size": row.get("main_cores_size", (None, None, None, None)),
        }

        cand = db.copy()

        q_cores, q_size, _, _ = qf["main_cores_size"]
        if q_cores is not None:
            tmp = cand[cand["main_cores_size"].apply(lambda t: isinstance(t, tuple) and t[0] is not None)]
            if not tmp.empty:
                cand = tmp
        if q_size is not None:
            tmp = cand[cand["main_cores_size"].apply(lambda t: isinstance(t, tuple) and t[1] is not None)]
            if not tmp.empty:
                cand = tmp
        if cand.empty:
            cand = db.copy()

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

        best = None
        if not cand.empty:
            cand = cand.copy()
            cand["score_final"] = cand.apply(row_score, axis=1)
            cand = cand.sort_values("score_final", ascending=False)
            if not cand.empty and cand.iloc[0]["score_final"] >= cable_threshold:
                best = cand.iloc[0]

        if best is not None:
            desc_proposed = best["DB_Desc"]
            m_cost = 0 if pd.isna(best["DB_MCost"]) else float(best["DB_MCost"])
            l_cost = 0 if pd.isna(best["DB_LCost"]) else float(best["DB_LCost"])
        else:
            desc_proposed = ""
            m_cost = l_cost = 0

        unit = row["Unit"]
        qty = row["Qty"]
        qty_val = pd.to_numeric(qty, errors="coerce")
        qty_val = 0 if pd.isna(qty_val) else qty_val
        amt_mat = qty_val * m_cost
        amt_lab = qty_val * l_cost
        total = amt_mat + amt_lab

        output_rows.append([
            row["Model"],                 # Model
            row["Desc"],                  # Description (requested)
            desc_proposed,                # Description (proposed)
            row["Spec"],                  # Specification
            unit,                         # Unit
            qty,                          # Quantity
            m_cost,                       # Material Cost
            l_cost,                       # Labour Cost
            amt_mat,                      # Amount Material
            amt_lab,                      # Amount Labour
            total                         # Total
        ])

    # ---------- Results ----------
    result_df = pd.DataFrame(output_rows, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification",
        "Unit", "Quantity", "Material Cost", "Labour Cost",
        "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum(skipna=True)
    grand_row = pd.DataFrame([[""] * 10 + [grand_total]], columns=result_df.columns)
    result_final = pd.concat([result_df, grand_row], ignore_index=True)

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

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_final.to_excel(writer, index=False, sheet_name="Matched Results")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")
    st.download_button("📥 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")
