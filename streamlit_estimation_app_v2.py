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
    # strip voltages in free text (we don't use them for cable size match here)
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

def extract_size(text: str) -> str:
    """
    Extracts patterns like '3x2.5', '4x16', '2x4', with optional spaces or ×.
    Also converts '4C' -> '4' before extraction, so '4C x 25' -> '4x25'.
    """
    text = str(text).lower()
    text = text.replace("mm2", "").replace("mm²", "")
    text = re.sub(r"(\d)\s*[cC]", r"\1", text)  # 4C -> 4
    match = re.search(r'\b\d{1,2}\s*[x×]\s*\d{1,3}(\.\d+)?\b', text)
    return match.group(0).replace(" ", "") if match else ""

def extract_material_structure_tokens(text: str):
    """
    Returns a list of normalized material/insulation tokens in order,
    e.g. "Cu/XLPE/PVC" -> ["cu","xlpe","pvc"].
    """
    text = str(text).lower()
    # common tokens
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

# Allow deletion of uploaded price list files
if price_list_files:
    file_to_delete = st.selectbox("Select a file to delete", [""] + price_list_files, key="delete_pl")
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
    # -------- Estimation --------
    est = pd.read_excel(estimation_file).dropna(how='all')
    est_cols = est.columns.tolist()
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    est["combined"] = (
        est[est_cols[0]].fillna('') + " " +
        est[est_cols[1]].fillna('') + " " +
        est[est_cols[2]].fillna('')
    ).apply(clean)

    est["size"] = (
        est[est_cols[0]].fillna('') + " " +
        est[est_cols[1]].fillna('') + " " +
        est[est_cols[2]].fillna('')
    ).apply(extract_size)

    est["materials"] = (
        est[est_cols[0]].fillna('') + " " +
        est[est_cols[1]].fillna('') + " " +
        est[est_cols[2]].fillna('')
    ).apply(extract_material_structure_tokens)

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

    # For matching: prepare cleaned fields
    db["combined"] = (
        db[db_cols[0]].fillna('') + " " +
        db[db_cols[1]].fillna('') + " " +
        db[db_cols[2]].fillna('')
    ).apply(clean)

    db["size"] = (
        db[db_cols[0]].fillna('') + " " +
        db[db_cols[1]].fillna('') + " " +
        db[db_cols[2]].fillna('')
    ).apply(extract_size)

    db["materials"] = (
        db[db_cols[0]].fillna('') + " " +
        db[db_cols[1]].fillna('') + " " +
        db[db_cols[2]].fillna('')
    ).apply(extract_material_structure_tokens)

    # Quick visibility: how many rows we have
    st.caption(f"Estimation rows: {len(est)}")
    st.caption(f"Price list rows: {len(db)} (files: {', '.join(sorted(set(db['source'])))} )")

    # -------- Matching --------
    output_data = []

    for _, row in est.iterrows():
        query = row["combined"]
        q_size = row["size"]
        q_mats = row["materials"]
        unit = row[est_cols[3]]
        qty = row[est_cols[4]]

        best = None
        best_score = -1.0

        # Stage 0: filter by size if we have one
        candidates = db.copy()
        if q_size:
            candidates = candidates[candidates["size"] == q_size]

        # Helper scorers: weighted materials + fuzzy
        def s1(r):
            mat_score = weighted_material_score(q_mats, r["materials"])
            fuzzy = fuzz.token_set_ratio(query, r["combined"])
            return mat_score + 0.4 * fuzzy  # stronger fuzzy weight

        # Stage 1: strict material+fuzzy on size-filtered set
        if not candidates.empty:
            candidates = candidates.copy()
            candidates["score"] = candidates.apply(s1, axis=1)
            top = candidates.sort_values("score", ascending=False).head(1)
            if not top.empty and top.iloc[0]["score"] >= match_threshold:
                best = top.iloc[0]
                best_score = best["score"]

        # Stage 2: relax size filter (if Stage 1 failed)
        if best is None:
            c2 = db.copy()
            c2["score"] = c2.apply(s1, axis=1)
            top2 = c2.sort_values("score", ascending=False).head(1)
            if not top2.empty and top2.iloc[0]["score"] >= match_threshold:
                best = top2.iloc[0]
                best_score = best["score"]

        # Stage 3: fuzzy fallback (always pick top-1 so we show something)
        if best is None:
            c3 = db.copy()
            c3["score"] = c3["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
            top3 = c3.sort_values("score", ascending=False).head(1)
            if not top3.empty:
                best = top3.iloc[0]
                best_score = best["score"]

        # Extract costs & description
        if best is not None and best_score >= 0:
            desc_proposed = best[db_cols[1]]
            m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce")
            # if Stage 3’s fuzzy-only best is very low and you want to mark as unmatched, uncomment:
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
