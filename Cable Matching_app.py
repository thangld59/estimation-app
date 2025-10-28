# streamlit_estimation_app_cable_admin.py
# BuildWise Estimation Tool - Combined single-file app
# Includes login/users, admin form handling, per-user price lists, and improved cable matching

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

USERS = load_users()

# ------------------------------
# Parsing & scoring helpers
# ------------------------------
# Regexes for cable sizes and auxiliary (neutral/earth)
MAIN_SIZE_RE = re.compile(r'\b(\d{1,2})\s*[cC]?\s*[x×]?\s*(\d{1,3}(?:\.\d+)?)\b')
# matches + 1C x 50 or + E50 or + e50 or + PE50
AUX_RE = re.compile(r'\+\s*(?:(\d{1,2})\s*[cC]?\s*[x×]?\s*(\d{1,3}(?:\.\d+)?)|([eEnNpP]{1,2})(\d{1,3}(?:\.\d+)?))', flags=re.IGNORECASE)

def clean(text: str) -> str:
    text = str(text or "")
    text = text.lower()
    text = re.sub(r"0[,.]?6kv|1[,.]?0kv", "", text)
    text = text.replace("mm2", "").replace("mm²", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("/", " ").replace(",", " ")
    text = text.replace("-", " ")
    text = text.replace("cáp", "").replace("cable", "").replace("dây", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_cable_spec(text: str) -> dict:
    """
    Parse main cores x size and auxiliary (neutral/earth) if present.
    Returns dict with main_cores (int), main_size (float), main_key (like '3x70'),
    aux_type ('E' or 'N' or ''), aux_cores, aux_size, aux_key, full_key.
    """
    s = str(text or "").lower()
    s = s.replace("mm2", "").replace("mm²", "")
    s = re.sub(r"\s+", " ", s)

    main_match = MAIN_SIZE_RE.search(s)
    main_cores = None
    main_size = None
    if main_match:
        try:
            main_cores = int(main_match.group(1))
        except:
            main_cores = None
        try:
            main_size = float(main_match.group(2))
        except:
            main_size = None

    aux_match = AUX_RE.search(s)
    aux_type = ""
    aux_cores = None
    aux_size = None
    if aux_match:
        # group 1/2 for "1C x 50" style, group 3/4 for "E50" style
        if aux_match.group(1) and aux_match.group(2):
            try:
                aux_cores = int(aux_match.group(1))
            except:
                aux_cores = None
            try:
                aux_size = float(aux_match.group(2))
            except:
                aux_size = None
        elif aux_match.group(3) and aux_match.group(4):
            t = aux_match.group(3).strip().upper()
            if t in ("E", "PE"):
                aux_type = "E"
            elif t == "N":
                aux_type = "N"
            try:
                aux_size = float(aux_match.group(4))
            except:
                aux_size = None

    main_key = ""
    if main_cores and main_size:
        main_key = f"{int(main_cores)}x{int(main_size) if float(main_size).is_integer() else main_size}"

    aux_key = ""
    if aux_type and aux_size:
        aux_key = f"{aux_type}{int(aux_size) if aux_size and float(aux_size).is_integer() else aux_size}"
    elif aux_cores and aux_size:
        aux_key = f"{aux_cores}x{int(aux_size) if aux_size and float(aux_size).is_integer() else aux_size}"

    full_key = f"{main_key}+{aux_key}" if main_key and aux_key else main_key or aux_key

    return {
        "main_cores": main_cores,
        "main_size": main_size,
        "main_key": main_key,
        "aux_type": aux_type,
        "aux_cores": aux_cores,
        "aux_size": aux_size,
        "aux_key": aux_key,
        "full_key": full_key
    }

def extract_material_tokens(text: str):
    """
    Extract normalized material/insulation tokens from text.
    Normalizes 'aluminium'->'al'
    """
    s = str(text or "").lower()
    tokens = re.findall(r'\b(cu|aluminium|al|xlpe|pvc|pe|lszh|hdpe|dsta|tape|shield|armour|armored|swa)\b', s)
    norm = []
    for t in tokens:
        if t == "aluminium":
            norm.append("al")
        elif t in ("armour", "armored", "swa"):
            norm.append("armored")
        elif t == "dsta":
            norm.append("dsta")
        else:
            norm.append(t)
    return list(dict.fromkeys(norm))  # unique preserve order

def extract_voltage(text: str):
    s = str(text or "").lower()
    # detect 0.6/1kv or 0.6kv / 1kv etc.
    if re.search(r'\b0[.,]?6[ /-]?1[.,]?0?k?[v]?\b', s):
        return "0.6/1kV"
    if re.search(r'\b0[.,]?6k?v\b', s):
        return "0.6kV"
    if re.search(r'\b1[.,]?0k?v\b', s):
        return "1.0kV"
    return ""

def weighted_material_score(query_tokens, target_tokens) -> float:
    """
    Return 0-100 score comparing material/insulation layers using tuned weights.
    """
    weights = {
        'cu': 1.0, 'al': 1.0,
        'xlpe': 0.9, 'pvc': 0.7,
        'lszh': 0.6, 'pe': 0.5, 'hdpe': 0.5,
        'armored': 0.8, 'dsta': 0.5, 'tape': 0.3, 'shield': 0.4
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

def contains_category_keyword(text: str):
    s = str(text or "").lower()
    keys = ["cáp", "cable", "dây", "dây điện", "cable", "cáp điện"]
    return any(k in s for k in keys)

# ------------------------------
# Streamlit login & session
# ------------------------------
st.set_page_config(page_title="BuildWise", page_icon="📐", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""

def do_login(user: str, pwd: str):
    user = (user or "").strip()
    users = load_users()
    if user in users and users[user]["password"] == pwd:
        st.session_state["logged_in"] = True
        st.session_state["username"] = user
        st.session_state["role"] = users[user].get("role", "user")
        return True
    return False

def do_logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""

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

username = st.session_state["username"]
role = st.session_state["role"]

# Header and logout
col1, col2 = st.columns([8,1])
with col1:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=120)
    st.markdown("## :triangular_ruler: BuildWise - Smart Estimation Tool")
with col2:
    if st.button("🔒 Logout"):
        do_logout()
        st.experimental_rerun()

# Sidebar: match threshold and admin controls
match_threshold = st.sidebar.slider("Match threshold (0-100)", 0, 100, 75,
                                    help="Minimum acceptance score for a 'good' match. Increase to be stricter.")
st.sidebar.markdown("---")
st.sidebar.write(f"Signed in as *{username}* ({role})")

# Ensure folders
user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)
os.makedirs(FORM_FOLDER, exist_ok=True)

# ------------------------------
# Admin: user management UI
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
    est["materials"] = base_est.apply(extract_material_tokens)
    est["voltage"] = base_est.apply(extract_voltage)
    est["has_keyword"] = base_est.apply(contains_category_keyword)

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
    db["materials"] = base_db.apply(extract_material_tokens)
    db["voltage"] = base_db.apply(extract_voltage)
    db["has_keyword"] = base_db.apply(contains_category_keyword)

    # Matching procedure
    output_data = []
    for _, row in est.iterrows():
        query = row["combined"]
        q_main = row["main_key"]
        q_aux  = row["aux_key"]
        q_mats = row["materials"]
        q_volt = row["voltage"]
        q_keyword = row["has_keyword"]
        unit = row[est_cols[3]]
        qty  = row[est_cols[4]]

        best = None
        best_score = -1.0

        # scoring helpers
        def compute_score(r):
            # material 40%, size 40%, voltage 10%, category keyword 10%
            score = 0.0
            # material contribution
            mat_score = weighted_material_score(q_mats or [], r["materials"] or [])
            score += 0.40 * mat_score  # contributes up to 40
            # size contribution
            if q_main and r.get("main_key"):
                if q_main == r.get("main_key"):
                    score += 40.0  # full 40
                else:
                    # partial size similarity — compare numeric parts (cores or size)
                    # fallback: if main_cores equal OR main_size equal within small tolerance
                    try:
                        qcores, qsize = (int(q_main.split("x")[0]), float(q_main.split("x")[1]))
                        rcores, rsize = (int(r["main_key"].split("x")[0]), float(r["main_key"].split("x")[1]))
                        cores_match = 1 if qcores == rcores else 0
                        size_match = 1 if abs(qsize - rsize) <= 0.0001 else 0
                        partial = (cores_match + size_match) / 2.0
                        score += 40.0 * partial
                    except Exception:
                        # no numeric comparison possible -> no size credit
                        pass
            else:
                # no main size: give 0 size credit here
                pass
            # voltage
            if q_volt and r.get("voltage"):
                if q_volt == r.get("voltage"):
                    score += 10.0
            # category keyword
            if q_keyword and r.get("has_keyword"):
                score += 10.0
            return score

        # Stage A: if q_main exists, filter DB by same main_key first for speed
        c0 = db.copy()
        if q_main:
            c0 = c0[c0["main_key"] == q_main]
        if not c0.empty:
            c0 = c0.copy()
            c0["score"] = c0.apply(lambda r: compute_score(r), axis=1)
            top = c0.sort_values("score", ascending=False).head(1)
            if not top.empty and float(top.iloc[0]["score"]) >= match_threshold:
                best = top.iloc[0]
                best_score = best["score"]

        # Stage B: if no good match found, compute score across whole DB using scoring function
        if best is None:
            c1 = db.copy()
            c1 = c1.copy()
            c1["score"] = c1.apply(lambda r: compute_score(r), axis=1)
            top2 = c1.sort_values("score", ascending=False).head(3)
            if not top2.empty:
                if float(top2.iloc[0]["score"]) >= match_threshold:
                    best = top2.iloc[0]
                    best_score = best["score"]

        # Stage C: final fallback to fuzzy token match (only if still no match)
        if best is None:
            c2 = db.copy()
            c2["fuzz"] = c2["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
            top3 = c2.sort_values("fuzz", ascending=False).head(1)
            if not top3.empty and float(top3.iloc[0]["fuzz"]) >= max(50, match_threshold - 10):  # looser fallback
                best = top3.iloc[0]
                # create a synthetic score mapping fuzz 0-100 to 0-100
                best_score = float(top3.iloc[0]["fuzz"])

        if best is not None and best_score >= 0:
            desc_proposed = best[db_cols[1]]
            try:
                m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce")
            except Exception:
                m_cost = 0
            try:
                l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce")
            except Exception:
                l_cost = 0
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

    # grand total row
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
