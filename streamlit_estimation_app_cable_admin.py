# buildwise_with_user_management.py
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
FORM_FOLDER = "shared_forms"

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

# Load or create users db
USERS = load_users()

# ------------------------------
# Utility / Parsing / Scoring
# ------------------------------
MAIN_SIZE_RE = re.compile(r'\b(\d{1,2})\s*[cC]?\s*[x×]\s*(\d{1,3}(?:\.\d+)?)\b')
AUX_RE = re.compile(r'\+\s*(?:([1-9]\d*)\s*[cC]?\s*[x×]\s*)?((?:pe|e|n))?(\d{1,3}(?:\.\d+)?)', flags=re.IGNORECASE)

def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"0[,.]?6kv|1[,.]?0kv", "", text)
    text = text.replace("mm2", "").replace("mm²", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("/", " ").replace(",", "")
    text = text.replace("-", " ")
    text = text.replace("cáp", "").replace("cable", "").replace("dây", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_cable_spec(text: str) -> dict:
    text = str(text).lower().replace("mm2", "").replace("mm²", "")
    text = re.sub(r"\s+", " ", text)

    main_match = MAIN_SIZE_RE.search(text)
    main_cores, main_size = None, None
    if main_match:
        try:
            main_cores = int(main_match.group(1))
        except:
            main_cores = None
        try:
            main_size = float(main_match.group(2))
        except:
            main_size = None

    aux_match = AUX_RE.search(text)
    aux_type = ""
    aux_cores = None
    aux_size = None
    if aux_match:
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

    main_key = f"{int(main_cores)}x{int(main_size) if main_size and float(main_size).is_integer() else main_size}" if main_cores and main_size else ""
    if aux_type and aux_size:
        aux_key = f"{aux_type}{int(aux_size) if aux_size and float(aux_size).is_integer() else aux_size}"
    elif aux_cores and aux_size:
        aux_key = f"{aux_cores}x{int(aux_size) if aux_size and float(aux_size).is_integer() else aux_size}"
    else:
        aux_key = ""

    full_key = f"{main_key}+{aux_key}" if main_key and aux_key else main_key
    return {
        "main_cores": main_cores,
        "main_size": main_size,
        "aux_type": aux_type,
        "aux_cores": aux_cores,
        "aux_size": aux_size,
        "main_key": main_key,
        "aux_key": aux_key,
        "full_key": full_key
    }

def extract_material_structure_tokens(text: str):
    text = str(text).lower()
    tokens = re.findall(r'(cu|al|aluminium|xlpe|pvc|pe|lszh|hdpe)', text)
    norm = []
    for t in tokens:
        if t == "aluminium":
            norm.append("al")
        else:
            norm.append(t)
    return norm

def weighted_material_score(query_tokens, target_tokens) -> float:
    weights = {
        'cu': 1.0, 'al': 1.0,
        'xlpe': 0.8, 'pvc': 0.6,
        'lszh': 0.6, 'pe': 0.5, 'hdpe': 0.5
    }
    all_keys = set(query_tokens) | set(target_tokens)
    if not all_keys:
        return 0.0
    max_score = sum(weights.get(k, 0.3) for k in all_keys)
    score = 0.0
    for k in all_keys:
        if (k in query_tokens) and (k in target_tokens):
            score += weights.get(k, 0.3)
    return (score / max_score) * 100.0

# ------------------------------
# Streamlit: Login & session state
# ------------------------------
st.set_page_config(page_title="BuildWise", page_icon="📐", layout="wide")
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

# After login: main app
username = st.session_state["username"]
role = st.session_state["role"]

# Layout header
col1, col2 = st.columns([8,1])
with col1:
    # logo (keep your assets/logo.png)
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=120)
    st.markdown("## :triangular_ruler: BuildWise - Smart Estimation Tool")
with col2:
    if st.button("🔒 Logout"):
        do_logout()
        st.experimental_rerun()

# Sidebar controls
match_threshold = st.sidebar.slider("Match threshold", 0, 100, 70,
                                    help="Minimum acceptance score for a 'good' match. Increase to be stricter.")
st.sidebar.markdown("---")
st.sidebar.write(f"Signed in as *{username}* ({role})")

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
            st.write(f"- *{u}* ({v.get('role','user')})")
    st.sidebar.markdown("---")

# ------------------------------
# Shared Forms (Admin uploads / deletes ; users download)
# ------------------------------
st.subheader(":scroll: Price List and Estimation Request Form (Mẫu Bảng Giá và Mẫu Yêu Cầu Vào Giá)")
form_files = sorted(os.listdir(FORM_FOLDER))
if role == "admin":
    form_uploads = st.file_uploader("Admin: Upload form files (xlsx/xls)", type=["xlsx", "xls"], accept_multiple_files=True, key="forms_up")
    if form_uploads:
        for f in form_uploads:
            with open(os.path.join(FORM_FOLDER, f.name), "wb") as out_f:
                out_f.write(f.read())
        st.success("Form file(s) uploaded.")
        # no explicit rerun; the page will reflect new files on next interaction

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
st.subheader(":file_folder: Upload Price List Files")
uploaded_files = st.file_uploader("Upload one or more price list Excel files (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="pl_up")
if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(user_folder, f.name), "wb") as out_f:
            out_f.write(f.read())
    st.success("Price list(s) uploaded.")

st.subheader(":open_file_folder: Manage Price Lists")
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
st.subheader(":page_facing_up: Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est_file")
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

    base_est = est[est_cols[0]].fillna('') + " " + est[est_cols[1]].fillna('') + " " + est[est_cols[2]].fillna('')
    est["combined"] = base_est.apply(clean)
    parsed_est = base_est.apply(parse_cable_spec)
    est["main_key"] = parsed_est.apply(lambda d: d["main_key"])
    est["aux_key"]  = parsed_est.apply(lambda d: d["aux_key"])
    est["materials"] = base_est.apply(extract_material_structure_tokens)

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

    base_db = db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna('')
    db["combined"]  = base_db.apply(clean)
    parsed_db = base_db.apply(parse_cable_spec)
    db["main_key"]  = parsed_db.apply(lambda d: d["main_key"])
    db["aux_key"]   = parsed_db.apply(lambda d: d["aux_key"])
    db["materials"] = base_db.apply(extract_material_structure_tokens)

    # matching
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

        # Stage 0: filter on main size if available
        c0 = db.copy()
        if q_main:
            c0 = c0[c0["main_key"] == q_main]

        def score_row(r):
            s = 0.0
            if q_aux:
                if r["aux_key"] == q_aux and r["aux_key"]:
                    s += 35.0
                else:
                    if r["aux_key"]:
                        s += 10.0
            mat = weighted_material_score(q_mats, r["materials"])
            s += 0.6 * mat
            s += 0.4 * fuzz.token_set_ratio(query, r["combined"])
            return s

        if not c0.empty:
            c0 = c0.copy()
            c0["score"] = c0.apply(score_row, axis=1)
            top = c0.sort_values("score", ascending=False).head(1)
            if not top.empty and top.iloc[0]["score"] >= match_threshold:
                best = top.iloc[0]
                best_score = best["score"]

        if best is None:
            c1 = db.copy()
            c1["score"] = c1.apply(score_row, axis=1)
            top2 = c1.sort_values("score", ascending=False).head(1)
            if not top2.empty and top2.iloc[0]["score"] >= match_threshold:
                best = top2.iloc[0]
                best_score = best["score"]

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

    result_df = pd.DataFrame(output_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Unit", "Quantity",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
    result_df.loc[len(result_df.index)] = [""] * 10 + [grand_total]

    # display + formatting
    st.subheader(":mag: Matched Estimation")
    display_df = result_df.copy()
    if "Quantity" in display_df.columns:
        display_df["Quantity"] = pd.to_numeric(display_df["Quantity"], errors="coerce").fillna(0).astype(int).map("{:,}".format)
    for col in ["Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0).astype(int).map("{:,}".format)
    st.dataframe(display_df, use_container_width=True)

    st.subheader(":x: Unmatched Rows")
    unmatched_df = result_df[result_df["Description (proposed)"] == ""
                             ]
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
    st.download_button("📥 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")
else:
    st.info("Upload your estimation file and price list(s), then click *Match now*.")
