# buildwise_piping_estimation.py
import streamlit as st
import pandas as pd
import os
import re
import json
from io import BytesIO
from rapidfuzz import fuzz

# ------------------------------
# Files & user persistence
# ------------------------------
USERS_FILE = "users.json"
FORM_FOLDER = "piping_shared_forms"

DEFAULT_USERS = {
    "Admin123": {"password": "BuildWise2025", "role": "admin"},
    "User123": {"password": "User2025", "role": "user"}
}

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_USERS.copy()
    else:
        save_users(DEFAULT_USERS)
        return DEFAULT_USERS.copy()

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

USERS = load_users()

# ------------------------------
# Utility: parsing and token extraction (pipes + fittings)
# ------------------------------

# size regex for pipes/fittings: D25, Ø25, phi 32, 25mm, DN50
SIZE_RE = re.compile(r'\b(?:d|ø|phi|φ|dn)?\s*(\d{1,3})(?:\s*mm)?\b', flags=re.IGNORECASE)

# category keywords
PIPE_CATEGORY = ['ống', 'ong', 'pipe', 'ống nước', 'ống dẫn']
FITTING_KEYWORDS = [
    "cút", "cut", "elbow", "góc", "goc", "tee", "tê", "te", "t", "nối",
    "coupling", "thu", "reducer", "bịt", "bit", "cap", "măng sông", "mang song",
    "union", "rắc co", "rac co", "flange", "bích", "bich", "adapter", "reducer"
]

# Material tokens we support for fittings release
MATERIAL_TOKENS = {
    'pvc': 'PVC',
    'upvc': 'uPVC',
    'u-pvc': 'uPVC',
    'hdpe': 'HDPE',
    'ppr': 'PPR',
    # synonyms (lowercase keys)
    'ống pvc': 'PVC', 'ống ppr': 'PPR', 'ống hdpe': 'HDPE'
}

TYPE_KEYWORDS = ['90', '45', 'angle', 'ren', 'threaded', 'socket', 'socketed']

def clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("/", " ").replace(",", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_pipe_size(text: str):
    if text is None:
        return None
    m = SIZE_RE.search(str(text).lower())
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

def extract_material_tokens(text: str):
    t = str(text).lower()
    found = set()
    for token, norm in MATERIAL_TOKENS.items():
        if token in t:
            found.add(norm)
    return list(found)

def has_category_keyword(text: str) -> bool:
    t = str(text).lower()
    for k in PIPE_CATEGORY + FITTING_KEYWORDS:
        if k in t:
            return True
    return False

def is_fitting_item(text: str) -> bool:
    t = str(text).lower()
    for k in FITTING_KEYWORDS:
        if k in t:
            return True
    # also detect words like "elbow", "tee", "reducer", etc.
    return False

def extract_type_keywords(text: str):
    t = str(text).lower()
    found = []
    for k in TYPE_KEYWORDS:
        if k in t:
            found.append(k)
    return found

def weighted_material_score_pipe(query_tokens, target_tokens) -> float:
    weights = {'PVC':1.0, 'uPVC':1.0, 'HDPE':1.0, 'PPR':0.95}
    all_keys = set(query_tokens) | set(target_tokens)
    if not all_keys:
        return 0.0
    max_score = sum(weights.get(k, 0.5) for k in all_keys)
    score = 0.0
    for k in all_keys:
        if k in query_tokens and k in target_tokens:
            score += weights.get(k, 0.5)
    return (score / max_score) * 100.0

def weighted_material_score_fitting(query_tokens, target_tokens) -> float:
    # Similar weighting for fittings (PVC/uPVC/HDPE/PPR)
    return weighted_material_score_pipe(query_tokens, target_tokens)

# ------------------------------
# Streamlit: Login & session state
# ------------------------------
st.set_page_config(page_title="BuildWise Piping Estimation Tool", page_icon="📐", layout="wide")
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""

def do_login(user: str, pwd: str):
    user = user.strip()
    if user in USERS and USERS[user]["password"] == pwd:
        st.session_state["logged_in"] = True
        st.session_state["username"] = user
        st.session_state["role"] = USERS[user].get("role", "user")
        return True
    return False

def do_logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""

# Login screen
if not st.session_state["logged_in"]:
    st.title("📐 BuildWise - Sign in")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
        if submitted:
            ok = do_login(u, p)
            if ok:
                st.success(f"Logged in as {st.session_state['username']} ({st.session_state['role']})")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password. Edit users.json to add users if needed.")
    st.stop()

# After login: variables
username = st.session_state["username"]
role = st.session_state["role"]

# Header
col1, col2 = st.columns([8,1])
with col1:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=120)
    st.markdown("## :triangular_ruler: BuildWise - Smart Piping Estimation Tool")
with col2:
    if st.button("🔒 Logout"):
        do_logout()
        st.experimental_rerun()

# Sidebar controls
match_threshold = st.sidebar.slider("Match threshold", 0, 100, 70,
                                    help="Minimum acceptance score for a 'good' match. Increase to be stricter.")
st.sidebar.markdown("---")
st.sidebar.write(f"Signed in as **{username}** ({role})")

# Ensure folders exist
user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)
os.makedirs(FORM_FOLDER, exist_ok=True)

# ------------------------------
# Admin: manage users UI
# ------------------------------
if role == "admin":
    st.sidebar.markdown("### 🔧 Admin - User management")
    users = load_users()
    with st.sidebar.expander("Manage users (admin)"):
        st.write("Add a new user (stored in users.json):")
        new_user = st.text_input("New username", key="new_user")
        new_pwd = st.text_input("New password", key="new_pwd")
        new_role = st.selectbox("Role", ["user", "admin"], key="new_role")
        if st.button("Add user"):
            if not new_user:
                st.sidebar.error("Please provide a username.")
            elif new_user in users:
                st.sidebar.error("User already exists.")
            else:
                users[new_user] = {"password": new_pwd, "role": new_role}
                save_users(users)
                st.sidebar.success(f"User {new_user} added.")
        st.markdown("---")
        st.write("Delete a user:")
        deletable = [u for u in users.keys() if u != "Admin123"]
        user_to_delete = st.selectbox("Select user to delete", [""] + deletable, key="del_user")
        if st.button("Delete user"):
            if user_to_delete and user_to_delete in users:
                if user_to_delete == username:
                    st.sidebar.error("You cannot delete your own account while logged in.")
                else:
                    users.pop(user_to_delete, None)
                    save_users(users)
                    st.sidebar.success(f"Deleted user {user_to_delete}.")
        st.markdown("---")
        st.write("Current users:")
        for u, v in users.items():
            st.write(f"- **{u}** ({v.get('role','user')})")
    st.sidebar.markdown("---")

# ------------------------------
# Shared Forms (Admin uploads / deletes ; users download)
# ------------------------------
st.subheader(":scroll: Piping Price List and Estimation Request Form (Mẫu Bảng Giá Ống và Mẫu Yêu Cầu Vào Giá)")
form_files = sorted(os.listdir(FORM_FOLDER))
if role == "admin":
    form_uploads = st.file_uploader("Admin: Upload form files (xlsx/xls)", type=["xlsx", "xls"], accept_multiple_files=True, key="forms_up")
    if form_uploads:
        for f in form_uploads:
            with open(os.path.join(FORM_FOLDER, f.name), "wb") as out_f:
                out_f.write(f.read())
        st.success("Form file(s) uploaded.")

    if form_files:
        chosen = st.selectbox("Select a form to delete", [""] + form_files, key="form_del")
        if chosen and st.button("Delete selected form"):
            try:
                os.remove(os.path.join(FORM_FOLDER, chosen))
                st.success(f"Deleted {chosen}")
            except Exception as e:
                st.error(f"Error deleting form: {e}")
else:
    if form_files:
        for f in form_files:
            path = os.path.join(FORM_FOLDER, f)
            with open(path, "rb") as fh:
                st.download_button(f"📄 Download {f}", fh.read(), file_name=f)
    else:
        st.info("No shared forms available.")

# ------------------------------
# Price list upload & manage (per user)
# ------------------------------
st.subheader(":file_folder: Upload Price List Files (Tải lên Bảng Giá)")
uploaded_files = st.file_uploader("Upload one or more price list Excel files (.xlsx) — Upload Price List (Tải lên Bảng Giá)", type=["xlsx"], accept_multiple_files=True, key="pl_up")
if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(user_folder, f.name), "wb") as out_f:
            out_f.write(f.read())
    st.success("Price list(s) uploaded.")

st.subheader(":open_file_folder: Manage Price Lists (Quản lý Tệp Bảng Giá)")
price_list_files = sorted(os.listdir(user_folder))
selected_file = st.radio("Choose one file to match or use all", ["All files"] + price_list_files)

if price_list_files:
    cola, colb = st.columns([3,1])
    with cola:
        to_del = st.selectbox("Select a price list to delete", [""] + price_list_files, key="del_pl")
    with colb:
        if st.button("Delete selected price list"):
            if to_del:
                try:
                    os.remove(os.path.join(user_folder, to_del))
                    st.success(f"Deleted {to_del}")
                except Exception as e:
                    st.error(f"Error deleting file: {e}")

# ------------------------------
# Upload Estimation File + Matching
# ------------------------------
st.subheader(":page_facing_up: Upload Estimation File (Tải lên File Yêu Cầu Báo Giá)")
estimation_file = st.file_uploader("Upload estimation request (.xlsx) — Upload Estimation File (Tải lên File Yêu Cầu Báo Giá)", type=["xlsx"], key="est_file")
run_matching = st.button("🔎 Match now")

if run_matching:
    if estimation_file is None:
        st.error("Please upload an estimation file first.")
        st.stop()
    if not price_list_files:
        st.error("Please upload at least one price list for your account first.")
        st.stop()

    # Read estimation
    est = pd.read_excel(estimation_file).dropna(how='all')
    est_cols = est.columns.tolist()
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns (Model, Description, Specification, Unit, Quantity).")
        st.stop()

    base_est = est[est_cols[0]].fillna('').astype(str) + " " + est[est_cols[1]].fillna('').astype(str) + " " + est[est_cols[2]].fillna('').astype(str)
    est["combined"] = base_est.apply(clean)
    # extract for piping and fittings
    est["pipe_size"] = base_est.apply(extract_pipe_size)
    est["materials"] = base_est.apply(extract_material_tokens)
    est["has_category"] = base_est.apply(has_category_keyword)
    est["type_tokens"] = base_est.apply(extract_type_keywords)
    est["is_fitting"] = base_est.apply(is_fitting_item)

    # Read DB(s)
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
        st.error("Price list file must have at least 6 columns (Model, Description, Spec, ..., MaterialCost, LabourCost).")
        st.stop()

    # prepare DB text fields
    base_db = db[db_cols[0]].fillna('').astype(str) + " " + db[db_cols[1]].fillna('').astype(str) + " " + db[db_cols[2]].fillna('').astype(str)
    db["combined"]  = base_db.apply(clean)
    db["pipe_size"]  = base_db.apply(extract_pipe_size)
    db["materials"] = base_db.apply(extract_material_tokens)
    db["has_category"] = base_db.apply(has_category_keyword)
    db["type_tokens"] = base_db.apply(extract_type_keywords)
    db["is_fitting"] = db["combined"].apply(is_fitting_item)

    # ------------------------------
    # Matching loop (choose pipes vs fittings automatically)
    # ------------------------------
    output_data = []

    for _, row in est.iterrows():
        query = row["combined"]
        q_size = row["pipe_size"]
        q_mats = row["materials"]
        q_has_cat = row["has_category"]
        q_type = row["type_tokens"]
        q_is_fitting = row["is_fitting"]
        unit = row[est_cols[3]]
        qty  = row[est_cols[4]]

        best = None
        best_score = -1.0

        # prefer DB rows that have category keyword (fitting or pipe)
        candidates = db.copy()
        if not db[db["has_category"] == True].empty:
            candidates = db[db["has_category"] == True].copy()

        # If it's a fitting in request: use fitting scoring
        if q_is_fitting:
            # Stage 0: strict size & type & material
            c0 = candidates.copy()
            if q_size:
                c0 = c0[c0["pipe_size"] == q_size]
            # prefer only fitting rows if exist
            if not c0[c0["is_fitting"] == True].empty:
                c0 = c0[c0["is_fitting"] == True].copy()

            def score_fitting(r):
                s = 0.0
                # size exact match high priority
                if q_size and r["pipe_size"] == q_size and r["pipe_size"] is not None:
                    s += 50.0
                # material weighted (0-100 scaled to 25)
                mat = weighted_material_score_fitting(q_mats, r["materials"])
                s += 0.25 * mat
                # type keyword (elbow/tee/reducer/coupling) medium
                # reward if the DB description contains any fitting keywords
                if any(k in r["combined"] for k in FITTING_KEYWORDS):
                    s += 15.0
                # angle/ren/socket tokens (small)
                type_match_count = 0
                for tkn in q_type:
                    if tkn and tkn in " ".join(r.get("type_tokens", [])):
                        type_match_count += 1
                if type_match_count:
                    s += min(10.0, 5.0 * type_match_count)
                # fuzzy small contributor (0-100 scaled to 10)
                s += 0.10 * fuzz.token_set_ratio(query, r["combined"])
                return s

            if not c0.empty:
                c0 = c0.copy()
                c0["score"] = c0.apply(score_fitting, axis=1)
                top = c0.sort_values("score", ascending=False).head(1)
                if not top.empty and top.iloc[0]["score"] >= match_threshold:
                    best = top.iloc[0]
                    best_score = best["score"]

            # Stage 1: relaxed over candidates (still prioritize fittings)
            if best is None:
                c1 = candidates.copy()
                # prefer fitting rows if there are any in candidates
                if not c1[c1["is_fitting"] == True].empty:
                    c1 = c1[c1["is_fitting"] == True].copy()

                c1["score"] = c1.apply(score_fitting, axis=1)
                top2 = c1.sort_values("score", ascending=False).head(1)
                if not top2.empty and top2.iloc[0]["score"] >= match_threshold:
                    best = top2.iloc[0]
                    best_score = best["score"]

            # Stage 2: fallback to fuzzy DB if still no match
            if best is None:
                c2 = db.copy()
                c2["fuzz"] = c2["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
                top3 = c2.sort_values("fuzz", ascending=False).head(1)
                if not top3.empty and top3.iloc[0]["fuzz"] >= 60:
                    best = top3.iloc[0]
                    best_score = top3.iloc[0]["fuzz"]

        else:
            # It's a pipe (not fitting) → use pipe scoring (strict size)
            # Stage 0: exact size match among candidates
            c0 = candidates.copy()
            if q_size:
                c0 = c0[c0["pipe_size"] == q_size]
            # prefer DB rows that are not fittings
            if not c0[c0["is_fitting"] == False].empty:
                c0 = c0[c0["is_fitting"] == False].copy()

            def score_pipe(r):
                s = 0.0
                if q_size and r["pipe_size"] == q_size and r["pipe_size"] is not None:
                    s += 50.0
                mat = weighted_material_score_pipe(q_mats, r["materials"])
                s += 0.30 * mat
                s += 0.15 * fuzz.token_set_ratio(query, r["combined"])
                return s

            if not c0.empty:
                c0 = c0.copy()
                c0["score"] = c0.apply(score_pipe, axis=1)
                top = c0.sort_values("score", ascending=False).head(1)
                if not top.empty and top.iloc[0]["score"] >= match_threshold:
                    best = top.iloc[0]
                    best_score = best["score"]

            # Stage 1: relaxed search among candidates
            if best is None:
                c1 = candidates.copy()
                c1["score"] = c1.apply(score_pipe, axis=1)
                top2 = c1.sort_values("score", ascending=False).head(1)
                if not top2.empty and top2.iloc[0]["score"] >= match_threshold:
                    best = top2.iloc[0]
                    best_score = best["score"]

            # Stage 2: final fuzzy fallback
            if best is None:
                c2 = db.copy()
                c2["fuzz"] = c2["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
                top3 = c2.sort_values("fuzz", ascending=False).head(1)
                if not top3.empty and top3.iloc[0]["fuzz"] >= 60:
                    best = top3.iloc[0]
                    best_score = top3.iloc[0]["fuzz"]

        # Pull costs and build output row
        if best is not None and best_score >= 0:
            desc_proposed = best[db_cols[1]]
            m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce")
            if pd.isna(m_cost): m_cost = 0
            if pd.isna(l_cost): l_cost = 0
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
            row[est_cols[0]], row[est_cols[1]], desc_proposed, row[est_cols[2]], unit, qty,
            m_cost, l_cost, amt_mat, amt_lab, total
        ])

    # Build result dataframe
    result_df = pd.DataFrame(output_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Unit", "Quantity",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
    result_df.loc[len(result_df.index)] = [""] * 10 + [grand_total]

    # Display formatting
    st.subheader(":mag: Matched Estimation")
    display_df = result_df.copy()
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

    # export
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Matched Results")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")
    st.download_button("📥 Download Cleaned Estimation File (Tải File Báo Giá Đã Làm Sạch)", buffer.getvalue(), file_name="Estimation_Result_BuildWise_Piping.xlsx")
else:
    st.info("Upload your estimation file and price list(s), then click **Match now**.")
