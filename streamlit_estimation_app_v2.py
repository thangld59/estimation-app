import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

# ------------------------------
# Utility Functions
# ------------------------------
def clean(text: str) -> str:
    text = str(text).lower()
    # strip voltages in free text (we don’t use them for size parsing here)
    text = re.sub(r"0[,.]?6kv|1[,.]?0kv", "", text)
    # normalize units & punctuation
    text = text.replace("mm2", "").replace("mm²", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("/", " ").replace(",", "")
    text = text.replace("-", " ")
    # Vietnamese/English generic cable words removed to reduce noise
    text = text.replace("cáp", "").replace("cable", "").replace("dây", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

MAIN_SIZE_RE = re.compile(r'\b(\d{1,2})\s*[cC]?\s*[x×]\s*(\d{1,3}(?:\.\d+)?)\b')
# Matches things like: + 1C x 50 / + 50 / + E50 / + PE50 / + N50 (spaces optional)
AUX_RE = re.compile(
    r'\+\s*(?:([1-9]\d*)\s*[cC]?\s*[x×]\s*)?((?:pe|PE|e|E|n|N)\s*)?(\d{1,3}(?:\.\d+)?)'
)

def parse_cable_spec(text: str):
    """
    Extract a structured spec:
      - main_cores, main_size (e.g., 3 and 70 for "3C x 70")
      - aux_type: 'E' (Earth), 'N' (Neutral), or '' if none/unknown
      - aux_cores (int or None), aux_size (float or None)
      - canonical keys:
          main_key: "3x70"
          aux_key:  "E50", "N50", "1x50", or "" if none
          full_key: "3x70+E50" (when aux exists)
    """
    text = str(text).lower().replace("mm2", "").replace("mm²", "")
    text = re.sub(r"\s+", " ", text)

    main_match = MAIN_SIZE_RE.search(text)
    main_cores, main_size = None, None
    if main_match:
        main_cores = int(main_match.group(1))
        main_size = float(main_match.group(2))

    aux_match = AUX_RE.search(text)
    aux_type = ""
    aux_cores = None
    aux_size = None
    if aux_match:
        # optional cores, optional type (E/PE/N), size
        cores_str = aux_match.group(1)
        type_str = aux_match.group(2)
        size_str = aux_match.group(3)

        if cores_str:
            try:
                aux_cores = int(cores_str)
            except:
                aux_cores = None

        if type_str:
            t = type_str.strip().upper()
            if t in ["E", "PE"]:
                aux_type = "E"
            elif t == "N":
                aux_type = "N"

        try:
            aux_size = float(size_str)
        except:
            aux_size = None

    # Canonical keys
    main_key = f"{int(main_cores)}x{int(main_size) if main_size and main_size.is_integer() else main_size}" if main_cores and main_size else ""
    if aux_type and aux_size:
        aux_key = f"{aux_type}{int(aux_size) if aux_size.is_integer() else aux_size}"
    elif aux_cores and aux_size:
        # no E/N given → just show as “1x50”
        aux_key = f"{aux_cores}x{int(aux_size) if aux_size.is_integer() else aux_size}"
    else:
        aux_key = ""

    full_key = f"{main_key}+{aux_key}" if main_key and aux_key else main_key
    return {
        "main_cores": main_cores,
        "main_size": main_size,
        "aux_type": aux_type,     # 'E', 'N', or ''
        "aux_cores": aux_cores,   # int or None
        "aux_size": aux_size,     # float or None
        "main_key": main_key,     # e.g., "3x70"
        "aux_key": aux_key,       # e.g., "E50", "N50", "1x50"
        "full_key": full_key      # e.g., "3x70+E50"
    }

def extract_material_structure_tokens(text: str):
    """
    Returns a list of normalized material/insulation tokens in order,
    e.g. "Cu/XLPE/PVC" -> ["cu","xlpe","pvc"].
    """
    text = str(text).lower()
    tokens = re.findall(r'(cu|al|aluminium|xlpe|pvc|pe|lszh|hdpe)', text)
    norm = []
    for t in tokens:
        if t in ["aluminium"]:
            norm.append("al")
        else:
            norm.append(t)
    return norm

def weighted_material_score(query_tokens, target_tokens) -> float:
    """
    Weighted overlap: gives more weight to conductor and insulation.
    """
    weights = {
        'cu': 1.0, 'al': 1.0,
        'xlpe': 0.8, 'pvc': 0.6,
        'lszh': 0.6, 'pe': 0.5, 'hdpe': 0.5
    }
    all_keys = set(query_tokens) | set(target_tokens)
    if not all_keys:
        return 0.0
    max_score = sum(weights.get(k, 0.3) for k in all_keys)  # small default for unknowns
    score = 0.0
    for k in all_keys:
        if (k in query_tokens) and (k in target_tokens):
            score += weights.get(k, 0.3)
    return (score / max_score) * 100.0

# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(page_title="BuildWise", page_icon="📀", layout="wide")
st.image("assets/logo.png", width=120)
st.title(":triangular_ruler: BuildWise - Smart Estimation Tool")

# Sidebar: username + threshold
username = st.sidebar.text_input("Username")
match_threshold = st.sidebar.slider("Cable match threshold", 0, 100, 45,
                                    help="Lower = more matches; higher = stricter")

if not username:
    st.warning("Please enter your username to continue.")
    st.stop()

user_folder = f"user_data/{username}"
form_folder = "shared_forms"
os.makedirs(user_folder, exist_ok=True)
os.makedirs(form_folder, exist_ok=True)

# ------------------------------
# Shared Forms Section (for all users)
# ------------------------------
st.subheader(":scroll: Price List and Estimation Request Form (Mẫu Bảng Giá và Mẫu Yêu Cầu Vào Giá)")

form_files = sorted(os.listdir(form_folder))
if username == "Admin123":
    form_uploads = st.file_uploader("Upload form files", type=["xlsx", "xls"],
                                    accept_multiple_files=True, key="form_upload")
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
        st.info("No shared forms uploaded yet.")

# ------------------------------
# Upload Price List Files
# ------------------------------
st.subheader(":file_folder: Upload Price List Files")
uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True, key="pl_upload")
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

# Allow deletion of uploaded price list files
if price_list_files:
    col_del1, col_del2 = st.columns([3,1])
    with col_del1:
        file_to_delete = st.selectbox("Select a file to delete", [""] + price_list_files, key="delete_pl")
    with col_del2:
        if st.button("Delete Selected File", use_container_width=True):
            if file_to_delete:
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
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est_file")

# Explicit action to run matching (prevents “need to refresh”)
run_matching = st.button("🔎 Match now")

if run_matching and estimation_file and price_list_files:
    # -------- Estimation --------
    est = pd.read_excel(estimation_file).dropna(how='all')
    est_cols = est.columns.tolist()
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    # Cleaned fields
    base_concat_est = (
        est[est_cols[0]].fillna('') + " " +
        est[est_cols[1]].fillna('') + " " +
        est[est_cols[2]].fillna('')
    )
    est["combined"] = base_concat_est.apply(clean)
    # Parse main + auxiliary (E/N/+extra)
    parsed_est = base_concat_est.apply(parse_cable_spec)
    est["main_key"] = parsed_est.apply(lambda d: d["main_key"])
    est["aux_key"]  = parsed_est.apply(lambda d: d["aux_key"])
    est["full_key"] = parsed_est.apply(lambda d: d["full_key"])
    est["materials"] = base_concat_est.apply(extract_material_structure_tokens)

    # -------- Price List (DB) --------
    db_frames = []
    if selected_file == "All files":
        for f in price_list_files:
            df = pd.read_excel(os.path.join(user_folder, f)).dropna(how='all')
            df["source"] = f
            db_frames.append(df)
        db = pd.concat(db_frames, ignore_index=True) if db_frames else pd.DataFrame()
    else:
        db = pd.read_excel(os.path.join(user_folder, selected_file)).dropna(how='all')
        db["source"] = selected_file

    if db.empty:
        st.error("No rows found in selected price list file(s).")
        st.stop()

    db_cols = db.columns.tolist()
    if len(db_cols) < 6:
        st.error("Price list file must have at least 6 columns.")
        st.stop()

    base_concat_db = (
        db[db_cols[0]].fillna('') + " " +
        db[db_cols[1]].fillna('') + " " +
        db[db_cols[2]].fillna('')
    )
    db["combined"]  = base_concat_db.apply(clean)
    parsed_db       = base_concat_db.apply(parse_cable_spec)
    db["main_key"]  = parsed_db.apply(lambda d: d["main_key"])
    db["aux_key"]   = parsed_db.apply(lambda d: d["aux_key"])
    db["full_key"]  = parsed_db.apply(lambda d: d["full_key"])
    db["materials"] = base_concat_db.apply(extract_material_structure_tokens)

    st.caption(f"Estimation rows: {len(est)}")
    st.caption(f"Price list rows: {len(db)} (files: {', '.join(sorted(set(db['source'])))} )")

    # -------- Matching --------
    output_data = []

    for _, row in est.iterrows():
        query = row["combined"]
        q_main = row["main_key"]
        q_aux  = row["aux_key"]
        q_mats = row["materials"]
        unit = row[est_cols[3]]
        qty  = row[est_cols[4]]

        best = None
        best_score = -1.0

        # Stage 0: strongest filter—exact main size match first
        c0 = db.copy()
        if q_main:
            c0 = c0[c0["main_key"] == q_main]

        # scoring function: prioritize aux match, then materials, then fuzzy
        def score_row(r):
            s = 0.0
            # aux match bonus
            if q_aux:
                if r["aux_key"] == q_aux:
                    s += 35.0
                else:
                    # partial credit if both have any aux (but different)
                    if r["aux_key"]:
                        s += 10.0
            # material similarity
            mat = weighted_material_score(q_mats, r["materials"])
            s += 0.6 * mat  # stronger weight than before
            # fuzzy similarity
            s += 0.4 * fuzz.token_set_ratio(query, r["combined"])
            return s

        # Stage 1: try within same main size
        if not c0.empty:
            c0 = c0.copy()
            c0["score"] = c0.apply(score_row, axis=1)
            top = c0.sort_values("score", ascending=False).head(1)
            if not top.empty and top.iloc[0]["score"] >= match_threshold:
                best = top.iloc[0]
                best_score = best["score"]

        # Stage 2: loosen main size (if Stage 1 failed)
        if best is None:
            c1 = db.copy()
            c1["score"] = c1.apply(score_row, axis=1)
            top2 = c1.sort_values("score", ascending=False).head(1)
            if not top2.empty and top2.iloc[0]["score"] >= match_threshold:
                best = top2.iloc[0]
                best_score = best["score"]

        # Stage 3: fuzzy-only fallback
        if best is None:
            c2 = db.copy()
            c2["score"] = c2["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
            top3 = c2.sort_values("score", ascending=False).head(1)
            if not top3.empty:
                best = top3.iloc[0]
                best_score = best["score"]

        if best is not None and best_score >= 0:
            desc_proposed = best[db_cols[1]]
            m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce")
            # If you want to mark too-low fuzzy as unmatched, uncomment:
            # if best_score < match_threshold:
            #     desc_proposed = ""
            #     m_cost = l_cost = 0
        else:
            desc_proposed = ""
            m_cost = l_cost = 0

        qty_val = pd.to_numeric(qty, errors="coerce")
        if pd.isna(qty_val):
            qty_val = 0
        amt_mat = qty_val * (m_cost if pd.notna(m_cost) else 0)
        amt_lab = qty_val * (l_cost if pd.notna(l_cost) else 0)
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

    result_df = pd.DataFrame(output_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Unit", "Quantity",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
    grand_row = pd.DataFrame([[''] * 10 + [grand_total]], columns=result_df.columns)
    result_final = pd.concat([result_df, grand_row], ignore_index=True)

    # ------------------------------
    # Display
    # ------------------------------
    st.subheader(":mag: Matched Estimation")
    display_df = result_final.copy()
    # format numbers
    if "Quantity" in display_df.columns:
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

    # ------------------------------
    # Export
    # ------------------------------
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_final.to_excel(writer, index=False, sheet_name="Matched Results")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")

    st.download_button("📥 Download Cleaned Estimation File",
                       buffer.getvalue(),
                       file_name="Estimation_Result_BuildWise.xlsx")
else:
    st.info("Upload your estimation file and price list(s), then click *Match now*.")
