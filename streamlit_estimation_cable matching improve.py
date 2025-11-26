# streamlit_estimation_app_with_customers.py
# BuildWise - Streamlit app with Estimation + Customer Management (Table layout)
# NOTE: Matching logic preserved from your working version.

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
# (UNCHANGED — keep your working matching logic)
# ------------------------------
MAIN_SIZE_RE = re.compile(r'\b(\d{1,2})\s*[cC]?\s*[x×]\s*(\d{1,3}(?:\.\d+)?)\b')
AUX_RE = re.compile(r'\+\s*(?:([1-9]\d*)\s*[cC]?\s*[x×]\s*)?((?:pe|e|n))?(\d{1,3}(?:\.\d+)?)', flags=re.IGNORECASE)
MATERIAL_TOKEN_RE = re.compile(r'(cu|aluminium|al|xlpe|pvc|pe|lszh|hdpe|dsta|sta|swa)', flags=re.IGNORECASE)


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
    """Extract main cores & size and auxiliary (E/N or extra cores) from a cable description."""
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

    main_key = (
        f"{int(main_cores)}x{int(main_size) if main_size and float(main_size).is_integer() else main_size}"
        if main_cores and main_size else ""
    )

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
    tokens = MATERIAL_TOKEN_RE.findall(text)
    norm = []
    for t in tokens:
        tt = t.lower()
        if tt == 'aluminium':
            norm.append('al')
        else:
            norm.append(tt)
    return norm


def material_structure_score(query_tokens, target_tokens):
    """Compute similarity score for material layers.
    Penalise extra layers in target not present in query and reward exact layer matches.
    Returns 0..100
    """
    if not query_tokens and not target_tokens:
        return 100.0
    if not query_tokens or not target_tokens:
        return 0.0

    # weights for tokens
    weights = {
        'cu': 1.0, 'al': 1.0,
        'xlpe': 0.9, 'pvc': 0.7,
        'lszh': 0.6, 'pe': 0.5, 'hdpe': 0.5,
        'dsta': 0.4, 'sta': 0.4, 'swa': 0.4
    }

    q_set = list(dict.fromkeys(query_tokens))
    t_set = list(dict.fromkeys(target_tokens))

    match_score = 0.0
    possible_score = 0.0

    # Consider order-insensitive matching but penalize extras
    all_keys = list(dict.fromkeys(q_set + t_set))
    for k in all_keys:
        w = weights.get(k, 0.3)
        possible_score += w
        if k in q_set and k in t_set:
            match_score += w
        # if present only in target or only in query, no addition to match_score

    base = (match_score / possible_score) * 100.0 if possible_score > 0 else 0.0

    # Penalty for extra layers in target compared to query
    extra_in_target = len([k for k in t_set if k not in q_set])
    extra_in_query = len([k for k in q_set if k not in t_set])
    penalty = (extra_in_target * 5.0) + (extra_in_query * 2.0)  # target extras heavier penalty

    score = max(0.0, base - penalty)
    return score


def combined_match_score(query, q_main_key, q_aux_key, q_mats, row_combined, r_main_key, r_aux_key, r_mats, threshold, weights):
    """Compute final combined score using weights dict:
    weights = {'size':0.45, 'cores':0.25, 'material':0.30}
    """
    size_score = 0.0
    cores_score = 0.0
    mat_score = 0.0

    # Size (main_key) exact match -> full, else partial by fuzz on main size text
    if q_main_key and r_main_key:
        if q_main_key == r_main_key:
            size_score = 100.0
        else:
            # fuzz on the core-size substring
            size_score = fuzz.token_set_ratio(q_main_key, r_main_key)
    else:
        # fallback to fuzzy on whole description
        size_score = fuzz.partial_ratio(query, row_combined)

    # Auxiliary / cores
    if q_aux_key and r_aux_key:
        if q_aux_key == r_aux_key:
            cores_score = 100.0
        else:
            cores_score = fuzz.token_set_ratio(str(q_aux_key), str(r_aux_key))
    else:
        # if none provided, neutral score
        cores_score = 100.0 if not q_aux_key and not r_aux_key else 0.0

    # Material structure
    mat_score = material_structure_score(q_mats, r_mats)

    final = (weights['size'] * size_score) + (weights['cores'] * cores_score) + (weights['material'] * mat_score)
    # Normalize to 0..100
    return final

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
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=120)
    st.markdown("## :triangular_ruler: BuildWise - Smart Estimation Tool")
with col2:
    if st.button("🔒 Logout"):
        do_logout()
        st.experimental_rerun()

# Add simple page navigation
page = st.sidebar.selectbox("Page", ["Estimation", "Customers", "Company Profile", "Quotations"] + (["Admin"] if role=="admin" else []))

# Ensure folders exist
user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)
os.makedirs(FORM_FOLDER, exist_ok=True)
os.makedirs(f"{user_folder}/quotations", exist_ok=True)

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
            st.write(f"- {u} ({v.get('role','user')})")
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
            try:
                with open(os.path.join(FORM_FOLDER, f.name), "wb") as out_f:
                    out_f.write(f.read())
            except Exception as e:
                st.error(f"Error saving form file {f.name}: {e}")
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
            try:
                with open(path, "rb") as fh:
                    st.download_button(f"📄 Download {f}", fh.read(), file_name=f, key=f"down_form_{f}")
            except Exception:
                continue
    else:
        st.info("No shared forms available.")

# ------------------------------
# Customers utilities (per-user storage)
# ------------------------------
def user_customers_file(user):
    folder = f"user_data/{user}"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "customers.json")

def load_customers_for(user):
    path = user_customers_file(user)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_customers_for(user, customers):
    path = user_customers_file(user)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(customers, f, indent=2, ensure_ascii=False)

# ------------------------------
# Price list upload & manage (per user)
# ------------------------------
def page_price_list_and_matching():
    st.subheader(":file_folder: Upload Price List Files")
    uploaded_files = st.file_uploader("Upload one or more price list Excel files (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="pl_up_main")
    if uploaded_files:
        for f in uploaded_files:
            try:
                with open(os.path.join(user_folder, f.name), "wb") as out_f:
                    out_f.write(f.read())
            except Exception as e:
                st.error(f"Error saving price list {f.name}: {e}")
        st.success("Price list(s) uploaded.")

    st.subheader(":open_file_folder: Manage Price Lists")
    price_list_files = sorted(os.listdir(user_folder))
    # Show files even if empty list
    if price_list_files:
        st.write("Your uploaded price lists:")
        for f in price_list_files:
            st.write(f"- {f}")
    else:
        st.info("No price lists uploaded yet. Use the upload box above.")

    # choice for matching: select file or all
    selected_file = st.radio("Choose one file to match or use all", ["All files"] + price_list_files, index=0, key="select_pl_radio_main")

    if price_list_files:
        cola, colb = st.columns([3,1])
        with cola:
            to_del = st.selectbox("Select a price list to delete", [""] + price_list_files, key="del_pl_main")
        with colb:
            if st.button("Delete selected price list", key="del_pl_btn_main"):
                if to_del:
                    try:
                        os.remove(os.path.join(user_folder, to_del))
                        st.success(f"Deleted {to_del}")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {e}")

    # ------------------------------
    # Upload Estimation File + Matching
    # ------------------------------
    st.subheader(":page_facing_up: Upload Estimation File")
    estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est_file_main")

    # Sidebar matching controls (exposed globally earlier) - replicate to local variables
    match_threshold = st.sidebar.slider("Match threshold", 0, 100, 70,
                                        help="Minimum acceptance score for a 'good' match. Increase to be stricter.")
    # weights editable
    st.sidebar.markdown("### Matching weights")
    w_size = st.sidebar.slider("Size weight", 0.0, 1.0, 0.45, step=0.05)
    w_cores = st.sidebar.slider("Cores weight", 0.0, 1.0, 0.25, step=0.05)
    w_material = st.sidebar.slider("Material weight", 0.0, 1.0, 0.30, step=0.05)
    _total_w = w_size + w_cores + w_material
    if _total_w <= 0:
        weights = {'size': 0.45, 'cores': 0.25, 'material': 0.30}
    else:
        weights = {'size': w_size/_total_w, 'cores': w_cores/_total_w, 'material': w_material/_total_w}

    # Match button - run only when user explicitly clicks it
    run_matching = st.button("🔎 Match now", key="run_match_main")

    if run_matching:
        # Validate inputs only when matching is requested
        if estimation_file is None:
            st.error("Please upload an estimation file first.")
        else:
            price_list_files = sorted(os.listdir(user_folder))
            if not price_list_files:
                st.error("Please upload at least one price list for your account first.")
            else:
                try:
                    # Read estimation
                    est = pd.read_excel(estimation_file).dropna(how='all')
                except Exception as e:
                    st.error(f"Cannot read estimation file: {e}")
                    est = None

                if est is None or est.empty:
                    st.error("Estimation file appears empty or unreadable.")
                else:
                    est_cols = est.columns.tolist()
                    if len(est_cols) < 5:
                        st.error("Estimation file must have at least 5 columns (Model, Description, Specification, Unit, Quantity).")
                    else:
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
                                try:
                                    df = pd.read_excel(os.path.join(user_folder, f)).dropna(how='all')
                                    df["source"] = f
                                    db_frames.append(df)
                                except Exception:
                                    continue
                            db = pd.concat(db_frames, ignore_index=True) if db_frames else pd.DataFrame()
                        else:
                            try:
                                db = pd.read_excel(os.path.join(user_folder, selected_file)).dropna(how='all')
                                db["source"] = selected_file
                            except Exception as e:
                                st.error(f"Cannot read selected price list: {e}")
                                db = pd.DataFrame()

                        if db.empty:
                            st.error("No rows found in selected price list file(s).")
                        else:
                            db_cols = db.columns.tolist()
                            if len(db_cols) < 6:
                                st.error("Price list file must have at least 6 columns (Model, Description, Spec, ..., MaterialCost, LabourCost).")
                            else:
                                base_db = db[db_cols[0]].fillna('') + " " + db[db_cols[1]].fillna('') + " " + db[db_cols[2]].fillna('')
                                db["combined"]  = base_db.apply(clean)
                                parsed_db = base_db.apply(parse_cable_spec)
                                db["main_key"]  = parsed_db.apply(lambda d: d["main_key"])
                                db["aux_key"]   = parsed_db.apply(lambda d: d["aux_key"])
                                db["materials"] = base_db.apply(extract_material_structure_tokens)

                                # Matching loop (preserved)
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

                                    # Stage 1: filter on main size if available
                                    c0 = db.copy()
                                    if q_main:
                                        c0 = c0[c0["main_key"] == q_main]

                                    def score_row(r):
                                        try:
                                            r_main = r.get("main_key", "")
                                            r_aux = r.get("aux_key", "")
                                            r_mats = r.get("materials", [])
                                            score = combined_match_score(query, q_main, q_aux, q_mats, r.get("combined", ""), r_main, r_aux, r_mats, match_threshold, weights)
                                            return score
                                        except Exception:
                                            return 0.0

                                    if not c0.empty:
                                        c0 = c0.copy()
                                        c0["score"] = c0.apply(score_row, axis=1)
                                        top = c0.sort_values("score", ascending=False).head(1)
                                        if not top.empty and float(top.iloc[0]["score"]) >= match_threshold:
                                            best = top.iloc[0]
                                            best_score = float(best["score"])

                                    if best is None:
                                        c1 = db.copy()
                                        c1["score"] = c1.apply(score_row, axis=1)
                                        top2 = c1.sort_values("score", ascending=False).head(1)
                                        if not top2.empty and float(top2.iloc[0]["score"]) >= match_threshold:
                                            best = top2.iloc[0]
                                            best_score = float(best["score"])

                                    if best is None:
                                        c2 = db.copy()
                                        c2["score"] = c2["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
                                        top3 = c2.sort_values("score", ascending=False).head(1)
                                        if not top3.empty:
                                            best = top3.iloc[0]
                                            best_score = float(best["score"])

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
                                st.download_button("📥 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")

# ------------------------------
# Customers page (Table format)
# ------------------------------
def page_customers():
    st.subheader("👥 Customer Management (Table view)")
    # Admin can view all users; normal user sees only their own
    if role == "admin":
        st.info("Admin: choose a user to view their customers.")
        # list user folders under user_data
        base = "user_data"
        os.makedirs(base, exist_ok=True)
        user_dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
        chosen_user = st.selectbox("Select user", options=["--Select user--"] + user_dirs, index=0, key="admin_choose_user")
        if chosen_user and chosen_user != "--Select user--":
            customers = load_customers_for(chosen_user)
            owner = chosen_user
        else:
            customers = []
            owner = None
    else:
        customers = load_customers_for(username)
        owner = username

    # Add customer form
    with st.expander("➕ Add new customer", expanded=False):
        c_name = st.text_input("Customer Name", key="add_c_name")
        c_company = st.text_input("Company", key="add_c_company")
        c_address = st.text_input("Address", key="add_c_address")
        c_phone = st.text_input("Phone", key="add_c_phone")
        c_email = st.text_input("Email", key="add_c_email")
        c_notes = st.text_area("Notes", key="add_c_notes")
        if st.button("Add Customer", key="add_c_btn"):
            if not c_name.strip():
                st.error("Customer name required")
            else:
                new = {
                    "id": f"C{int(pd.Timestamp.now().timestamp())}",
                    "name": c_name.strip(),
                    "company": c_company.strip(),
                    "address": c_address.strip(),
                    "phone": c_phone.strip(),
                    "email": c_email.strip(),
                    "notes": c_notes.strip(),
                    "created_at": pd.Timestamp.now().isoformat()
                }
                # save to appropriate owner (admin may be adding for selected_user)
                target = owner if owner else username
                custs = load_customers_for(target)
                custs.append(new)
                save_customers_for(target, custs)
                st.success(f"Added customer {new['name']} for user {target}")
                st.experimental_rerun()

    # Show table
    if not customers:
        st.info("No customers found for this user.")
        return

    # Prepare DataFrame for table
    df = pd.DataFrame(customers)
    # keep columns in nice order
    cols_order = ["id", "name", "company", "phone", "email", "address", "notes", "created_at"]
    cols = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
    df = df[cols]

    st.markdown("### Customer list")
    st.dataframe(df.reset_index(drop=True), use_container_width=True)

    # Management actions: select ID to edit/delete
    st.markdown("### Manage a customer")
    ids = df["id"].astype(str).tolist()
    selected_id = st.selectbox("Select customer ID", options=[""] + ids, key="sel_cust_id")
    if selected_id:
        row = df[df["id"].astype(str) == selected_id].iloc[0].to_dict()
        st.write("Selected customer:")
        st.write(row)
        col1, col2 = st.columns(2)
        if col1.button("Edit customer", key="edit_cust_btn"):
            with st.form(f"edit_form_{selected_id}"):
                e_name = st.text_input("Customer Name", value=row.get("name",""))
                e_company = st.text_input("Company", value=row.get("company",""))
                e_address = st.text_input("Address", value=row.get("address",""))
                e_phone = st.text_input("Phone", value=row.get("phone",""))
                e_email = st.text_input("Email", value=row.get("email",""))
                e_notes = st.text_area("Notes", value=row.get("notes",""))
                submitted = st.form_submit_button("Save changes")
                if submitted:
                    target = owner if owner else username
                    custs = load_customers_for(target)
                    for i,c in enumerate(custs):
                        if str(c.get("id")) == selected_id:
                            custs[i].update({
                                "name": e_name.strip(),
                                "company": e_company.strip(),
                                "address": e_address.strip(),
                                "phone": e_phone.strip(),
                                "email": e_email.strip(),
                                "notes": e_notes.strip(),
                                "updated_at": pd.Timestamp.now().isoformat()
                            })
                            save_customers_for(target, custs)
                            st.success("Customer updated.")
                            st.experimental_rerun()
        if col2.button("Delete customer", key="del_cust_btn"):
            target = owner if owner else username
            custs = load_customers_for(target)
            new_custs = [c for c in custs if str(c.get("id")) != selected_id]
            save_customers_for(target, new_custs)
            st.success("Customer deleted.")
            st.experimental_rerun()

# ------------------------------
# Company Profile (simple)
# ------------------------------
def page_company_profile():
    st.subheader("🏢 Company Profile")
    comp_file = os.path.join(user_folder, "company.json")
    profile = {}
    if os.path.exists(comp_file):
        try:
            with open(comp_file, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception:
            profile = {}
    name = st.text_input("Company name", value=profile.get("name",""))
    address = st.text_input("Address", value=profile.get("address",""))
    phone = st.text_input("Phone", value=profile.get("phone",""))
    email = st.text_input("Email", value=profile.get("email",""))
    if st.button("Save company profile", key="save_comp_btn"):
        to_save = {"name": name.strip(), "address": address.strip(), "phone": phone.strip(), "email": email.strip()}
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2, ensure_ascii=False)
        st.success("Company profile saved.")

# ------------------------------
# Quotations page (list only)
# ------------------------------
def page_quotations():
    st.subheader("📄 Quotations")
    q_folder = os.path.join(user_folder, "quotations")
    os.makedirs(q_folder, exist_ok=True)
    files = sorted(os.listdir(q_folder))
    if files:
        for f in files:
            path = os.path.join(q_folder, f)
            c1, c2 = st.columns([4,1])
            with c1:
                st.write(f)
            with c2:
                if st.button(f"Download {f}", key=f"dl_q_{f}"):
                    with open(path,"rb") as fh:
                        st.download_button(f"Download {f}", fh.read(), file_name=f, key=f"down_q_file_{f}")
                if role == "admin" or True:
                    if st.button(f"Delete {f}", key=f"del_q_{f}"):
                        os.remove(path)
                        st.success("Deleted.")
                        st.experimental_rerun()
    else:
        st.info("No quotations yet.")

# ------------------------------
# Admin page (view all users customers & quotations)
# ------------------------------
def page_admin():
    st.subheader("🛠️ Admin - All users' customers & quotations")
    base = "user_data"
    os.makedirs(base, exist_ok=True)
    users = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))])
    if not users:
        st.info("No user folders found yet.")
        return
    for u in users:
        st.markdown(f"### User: {u}")
        custs = load_customers_for(u)
        if custs:
            st.write(pd.DataFrame(custs))
        else:
            st.write("No customers.")
        qf = os.path.join(base, u, "quotations")
        if os.path.exists(qf):
            st.write("Quotations:", sorted(os.listdir(qf)))
        else:
            st.write("Quotations: (none)")

# ------------------------------
# Page routing
# ------------------------------
if page == "Estimation":
    page_price_list_and_matching()
elif page == "Customers":
    page_customers()
elif page == "Company Profile":
    page_company_profile()
elif page == "Quotations":
    page_quotations()
elif page == "Admin" and role == "admin":
    page_admin()
else:
    st.info("Select a page from the sidebar.")

# small footer/help
st.markdown("---")
st.caption("BuildWise — Estimation tool. Matching logic unchanged. Customer data stored per user in user_data/<username>/customers.json")
