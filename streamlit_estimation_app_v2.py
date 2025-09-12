import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

# ------------------------------
# App Configuration / Header
# ------------------------------
st.set_page_config(page_title="BuildWise", page_icon="📀", layout="wide")
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
match_threshold = st.sidebar.slider("Cable match threshold", 0, 100, 60)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# Folders
user_folder = f"user_data/{username}"
form_folder = "shared_forms"
os.makedirs(user_folder, exist_ok=True)
os.makedirs(form_folder, exist_ok=True)

# ------------------------------
# Normalization & Extraction
# ------------------------------
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def clean(text: str) -> str:
    s = str(text).lower()
    s = s.replace("mm^2", "mm2").replace("mm²", "mm2")
    s = s.replace("/", " / ").replace("-", " ")
    s = s.replace(",", " ")
    s = re.sub(r"\b0[.,]?\s*6\s*kv\b", "0.6kv", s)
    s = re.sub(r"\b1[.,]?\s*0\s*kv\b", "1.0kv", s)
    return norm_space(s)

def extract_size(text: str) -> str:
    """
    Canonical size like '3x2.5'
    Accepts: '3C x 2.5', '3x2.5', '3×2.5', '3 * 2.5', '(3x2.5)mm2'
    """
    s = clean(text)
    s = s.replace("(", " ").replace(")", " ")
    # Make 3C -> 3
    s = re.sub(r"\b(\d{1,2})\s*c\b", r"\1", s)
    # cores x size (with x/×/*, decimal allowed)
    m = re.search(r"\b(\d{1,2})\s*[x×*]\s*(\d{1,3}(?:[.,]\d{1,2})?)\b", s)
    if m:
        cores = int(m.group(1))
        size = float(m.group(2).replace(",", "."))
        # normalize "2.50" -> "2.5" and integers keep as int-like string
        if abs(size - int(size)) < 1e-9:
            size_str = f"{int(size)}"
        else:
            size_str = f"{size}".rstrip("0").rstrip(".")
        return f"{cores}x{size_str}"
    # fallback: maybe only mm2 is present (e.g., "2.5mm2")
    m2 = re.search(r"\b(\d{1,3}(?:[.,]\d{1,2})?)\s*mm2\b", s)
    if m2:
        size = float(m2.group(1).replace(",", "."))
        if abs(size - int(size)) < 1e-9:
            size_str = f"{int(size)}"
        else:
            size_str = f"{size}".rstrip("0").rstrip(".")
        # cores unknown -> assume 1
        return f"1x{size_str}"
    return ""

# normalized material/insulation tokens
TOKEN_MAP = {
    "cu": "cu", "đồng": "cu", "dong": "cu",
    "al": "al", "nhôm": "al", "nhom": "al", "aluminium": "al",
    "xlpe": "xlpe", "pvc": "pvc", "pe": "pe", "lszh": "lszh", "hdpe": "hdpe",
    "swa": "swa", "sta": "sta", "armored": "armored", "armour": "armored", "shield": "shield", "screen": "shield", "tape":"tape",
}

def extract_material_structure(text: str):
    s = clean(text)
    found = []
    for raw, norm in TOKEN_MAP.items():
        if re.search(rf"\b{re.escape(raw)}\b", s):
            found.append(norm)
    # dedupe but keep order in a simple way
    seen = set()
    out = []
    for t in found:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def split_cores_size(size_str: str):
    """ '3x2.5' -> (3, 2.5) ; '' -> (None,None) """
    if not size_str:
        return (None, None)
    m = re.match(r"^(\d{1,2})x(\d+(?:\.\d+)?)$", size_str)
    if not m:
        return (None, None)
    cores = int(m.group(1))
    dia = float(m.group(2))
    return (cores, dia)

def weighted_material_score(q_tokens, t_tokens):
    weights = {
        'cu': 1.0, 'al': 1.0,
        'xlpe': 0.8, 'pvc': 0.7,
        'lszh': 0.6, 'pe': 0.6, 'hdpe': 0.6,
        'shield': 0.4, 'tape': 0.3, 'armored': 0.5, 'swa': 0.5, 'sta': 0.5
    }
    score = 0.0
    max_score = 0.0
    for tok in set((q_tokens or []) + (t_tokens or [])):
        w = weights.get(tok, 0.0)
        max_score += w
        if (tok in (q_tokens or [])) and (tok in (t_tokens or [])):
            score += w
    return int(round(100 * score / max_score)) if max_score > 0 else 0

def size_close(q_size, t_size, tol_pct=5.0):
    """percent tolerance on cross-sectional size (mm2)"""
    if q_size is None or t_size is None or t_size == 0:
        return False
    return abs(q_size - t_size) / t_size * 100.0 <= tol_pct

# ------------------------------
# Shared Forms (Admin123 can upload/delete; others can download)
# ------------------------------
st.subheader(":scroll: Price List and Estimation Request Form (Mẫu Bảng Giá và Mẫu Yêu Cầu Vào Giá)")
form_files = sorted(os.listdir(form_folder))

if username == "Admin123":
    form_uploads = st.file_uploader("Upload form files", type=["xlsx", "xls"], accept_multiple_files=True, key="form_upload")
    if form_uploads:
        for f in form_uploads:
            with open(os.path.join(form_folder, f.name), "wb") as out_file:
                out_file.write(f.read())
        st.success("Form file(s) uploaded successfully.")
        st.rerun()
    if form_files:
        form_to_delete = st.selectbox("Select a form file to delete", [""] + form_files, key="form_delete")
        if form_to_delete and st.button("Delete Selected Form File"):
            try:
                os.remove(os.path.join(form_folder, form_to_delete))
                st.success(f"Deleted form file: {form_to_delete}")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting form file: {e}")
else:
    if form_files:
        for file in form_files:
            with open(os.path.join(form_folder, file), "rb") as f:
                st.download_button(f"📄 Download {file}", f.read(), file_name=file)
    else:
        st.info("No shared forms yet.")

# ------------------------------
# Upload Price List Files
# ------------------------------
st.subheader(":file_folder: Upload Price List Files")
uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(user_folder, file.name), "wb") as f:
            f.write(file.read())
    st.success(":white_check_mark: Price list uploaded successfully.")
    st.rerun()

# ------------------------------
# Manage Price Lists
# ------------------------------
st.subheader(":open_file_folder: Manage Price Lists")
price_list_files = sorted(os.listdir(user_folder))
selected_file = st.radio("Choose one file to match or use all", ["All files"] + price_list_files)

file_to_delete = st.selectbox("Select a file to delete", [""] + price_list_files, key="del_price")
if file_to_delete:
    if st.button("Delete Selected File"):
        try:
            os.remove(os.path.join(user_folder, file_to_delete))
            st.success(f"Deleted file: {file_to_delete}")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting file: {e}")

# ------------------------------
# Upload Estimation File
# ------------------------------
st.subheader(":page_facing_up: Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est")

if estimation_file and price_list_files:
    # ===== Estimation =====
    est = pd.read_excel(estimation_file).dropna(how='all')
    est_cols = est.columns.tolist()
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    est["combined"] = (est[est_cols[0]].fillna('') + " " + est[est_cols[1]].fillna('') + " " + est[est_cols[2]].fillna('')).apply(clean)
    est["size"]      = (est[est_cols[0]].fillna('') + " " + est[est_cols[1]].fillna('') + " " + est[est_cols[2]].fillna('')).apply(extract_size)
    est["materials"] = (est[est_cols[0]].fillna('') + " " + est[est_cols[1]].fillna('') + " " + est[est_cols[2]].fillna('')).apply(extract_material_structure)
    est[["q_cores","q_size"]] = est["size"].apply(lambda s: pd.Series(split_cores_size(s)))

    # ===== Price List =====
    if selected_file == "All files":
        frames = []
        for f in price_list_files:
            df = pd.read_excel(os.path.join(user_folder, f)).dropna(how='all')
            df["_source_"] = f
            frames.append(df)
        db = pd.concat(frames, ignore_index=True)
    else:
        db = pd.read_excel(os.path.join(user_folder, selected_file)).dropna(how='all')
        db["_source_"] = selected_file

    db_cols = db.columns.tolist()
    if len(db_cols) < 6:
        st.error("Price list file must have at least 6 columns.")
        st.stop()

    # Clean / Parse DB
    db["combined"]  = (db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna('')).apply(clean)
    db["size"]      = (db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna('')).apply(extract_size)
    db["materials"] = (db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna('')).apply(extract_material_structure)
    db[["d_cores","d_size"]] = db["size"].apply(lambda s: pd.Series(split_cores_size(s)))

    if show_debug:
        st.markdown("#### 🔎 Debug: first 10 parsed Estimation rows")
        st.dataframe(est.head(10))
        st.markdown("#### 🔎 Debug: first 10 parsed DB rows")
        st.dataframe(db.head(10))

    # ------------------------------
    # Matching
    # ------------------------------
    output_data = []
    for _, row in est.iterrows():
        query = row["combined"]
        q_size_str = row["size"]
        q_cores = row["q_cores"]
        q_size = row["q_size"]
        q_mats = row["materials"]

        best = None

        # Stage 1: strict size string equality
        if q_size_str:
            cand1 = db[db["size"] == q_size_str].copy()
        else:
            cand1 = pd.DataFrame()

        if not cand1.empty:
            # score by materials + small fuzzy bonus
            def s1(r):
                mat_score = weighted_material_score(q_mats, r["materials"])
                fuzzy = fuzz.token_set_ratio(query, r["combined"])
                return mat_score + 0.1 * fuzzy
            cand1["score"] = cand1.apply(s1, axis=1)
            cand1 = cand1.sort_values("score", ascending=False)
            if cand1.iloc[0]["score"] >= match_threshold:
                best = cand1.iloc[0]

        # Stage 2: relax size (same cores + size within ±5%)
        if best is None and (q_cores is not None or q_size is not None):
            cand2 = db.copy()
            if q_cores is not None:
                cand2 = cand2[cand2["d_cores"].notna() & (cand2["d_cores"].astype(int) == int(q_cores))]
            if q_size is not None and not cand2.empty:
                cand2 = cand2[cand2["d_size"].notna() & cand2["d_size"].apply(lambda t: size_close(q_size, t, 5.0))]
            if not cand2.empty:
                def s2(r):
                    mat_score = weighted_material_score(q_mats, r["materials"])
                    fuzzy = fuzz.token_set_ratio(query, r["combined"])
                    return mat_score + 0.1 * fuzzy
                cand2["score"] = cand2.apply(s2, axis=1)
                cand2 = cand2.sort_values("score", ascending=False)
                if cand2.iloc[0]["score"] >= match_threshold:
                    best = cand2.iloc[0]

        # Stage 3: fuzzy fallback
        if best is None:
            cand3 = db.copy()
            cand3["score"] = cand3["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
            cand3 = cand3.sort_values("score", ascending=False)
            if not cand3.empty and cand3.iloc[0]["score"] >= match_threshold:
                best = cand3.iloc[0]

        # Costs & row assembly
        if best is not None:
            desc_proposed = best[db_cols[1]]
            m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce")
        else:
            desc_proposed = ""
            m_cost = l_cost = 0

        unit = row[est_cols[3]]
        qty  = row[est_cols[4]]
        qty_val = pd.to_numeric(qty, errors="coerce")
        qty_val = 0 if pd.isna(qty_val) else qty_val
        amt_mat = qty_val * (0 if pd.isna(m_cost) else m_cost)
        amt_lab = qty_val * (0 if pd.isna(l_cost) else l_cost)
        total = amt_mat + amt_lab

        output_data.append([
            row[est_cols[0]],  # Model
            row[est_cols[1]],  # Description (requested)
            desc_proposed,     # Description (proposed)
            row[est_cols[2]],  # Specification
            unit,              # Unit
            qty,               # Quantity
            m_cost if pd.notna(m_cost) else 0,
            l_cost if pd.notna(l_cost) else 0,
            amt_mat,           # Amount Material
            amt_lab,           # Amount Labour
            total              # Total
        ])

    # ------------------------------
    # Results / Export
    # ------------------------------
    result_df = pd.DataFrame(output_data, columns=[
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
