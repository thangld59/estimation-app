# buildwise_piping_estimation.py
import streamlit as st
import pandas as pd
import os
import re
import json
from io import BytesIO
from rapidfuzz import fuzz

# ------------------------------
# Config / Files
# ------------------------------
USERS_FILE = "users.json"
FORM_FOLDER = "piping_shared_forms"
REFERENCE_FILE = "piping_reference.xlsx"   # <-- place your reference file here

DEFAULT_USERS = {
    "Admin123": {"password": "BuildWise2025", "role": "admin"},
    "User123": {"password": "User2025", "role": "user"}
}

# ------------------------------
# Users persistence
# ------------------------------
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
# Reference loader (dynamic)
# ------------------------------
def load_reference_table(path):
    """
    Load piping_reference.xlsx and build normalization maps:
    - return None if file not found or invalid
    """
    if not os.path.exists(path):
        return None

    try:
        ref = pd.read_excel(path).dropna(how="all")
    except Exception:
        return None

    ref_cols = ref.columns.tolist()
    if len(ref_cols) < 4:
        return None

    ref = ref[[ref_cols[0], ref_cols[1], ref_cols[2], ref_cols[3]]].copy()
    ref.columns = ["name", "material", "size", "pressure"]

    for c in ["name", "material", "size", "pressure"]:
        ref[c] = ref[c].fillna("").astype(str)

    name_lookup = {}
    materials_set = set()
    sizes_set = set()
    pressures_set = set()
    for _, r in ref.iterrows():
        key = clean(r["name"])
        name_lookup[key] = {
            "name": r["name"],
            "material": r["material"].strip(),
            "size_raw": r["size"].strip(),
            "pressure_raw": r["pressure"].strip()
        }
        if r["material"].strip():
            materials_set.add(r["material"].strip().upper())
        if r["size"].strip():
            sizes_set.add(r["size"].strip().lower())
        if r["pressure"].strip():
            pressures_set.add(r["pressure"].strip().upper())

    return {
        "df": ref,
        "name_lookup": name_lookup,
        "materials": materials_set,
        "sizes": sizes_set,
        "pressures": pressures_set
    }

# ------------------------------
# Text cleaning and token extraction
# ------------------------------
SIZE_RE = re.compile(r'\b(?:d|dn|ø|phi|φ)?\s*(\d{1,3})(?:\s*mm)?\b', flags=re.IGNORECASE)
PRESSURE_RE = re.compile(r'\bpn\s*[-:]?\s*(\d{1,3})\b', flags=re.IGNORECASE)

FITTING_KEYWORDS = [
    "cút", "cut", "elbow", "góc", "goc", "tee", "tê", "te", "t", "nối",
    "coupling", "thu", "reducer", "giảm", "giam", "bịt", "bit", "cap", "măng sông", "mang song",
    "union", "rắc co", "rac co", "flange", "bích", "bich", "adapter", "socket", "thread", "ren"
]
PIPE_KEYWORDS = ["ống", "ong", "pipe", "ống nước", "ống dẫn"]

MATERIAL_KEYWORDS = ["pvc", "upvc", "u-pvc", "hdpe", "ppr", "cpvc"]

TYPE_KEYWORDS = ["90", "45", "angle", "ren", "thread", "socket", "socketed", "male", "female"]

def clean(text: str) -> str:
    if text is None:
        return ""
    s = str(text).lower()
    s = s.replace("(", " ").replace(")", " ").replace("/", " ")
    s = s.replace("-", " ").replace(",", " ").replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_size_val(text: str):
    if text is None:
        return None
    m = SIZE_RE.search(str(text).lower())
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

def extract_pressure_val(text: str):
    if text is None:
        return ""
    m = PRESSURE_RE.search(str(text).lower())
    if m:
        try:
            return "PN" + str(int(m.group(1)))
        except:
            return ""
    t = str(text).upper()
    if "PN" in t:
        m2 = re.search(r'PN\D*(\d{1,3})', t)
        if m2:
            return "PN" + str(int(m2.group(1)))
    return ""

def detect_material_from_text(text: str, ref_materials=None):
    t = str(text).lower()
    for m in MATERIAL_KEYWORDS:
        if m in t:
            if m in ["upvc", "u-pvc"]:
                return "UPVC"
            return m.upper()
    if ref_materials:
        tl = t.upper()
        for rm in ref_materials:
            if rm.lower() in t:
                return rm.upper()
    return ""

def extract_type_keywords(text: str):
    t = str(text).lower()
    found = []
    for k in TYPE_KEYWORDS:
        if k in t:
            found.append(k)
    return found

def is_fitting_text(text: str):
    t = str(text).lower()
    for k in FITTING_KEYWORDS:
        if k in t:
            return True
    return False

# ------------------------------
# Weighted scoring (user weights)
# Size 30, Material 20, Pressure 15, Type 20, Fuzzy 15
# ------------------------------
WEIGHTS = {
    "size": 30.0,
    "material": 20.0,
    "pressure": 15.0,
    "type": 20.0,
    "fuzzy": 15.0
}
MIN_MATCH_SCORE = 70.0
FALLBACK_FUZZY_MIN = 60.0

def compute_match_score(req_attrs, cand_attrs):
    score = 0.0
    # Size
    if req_attrs.get("size") and cand_attrs.get("size"):
        if req_attrs["size"] == cand_attrs["size"]:
            score += WEIGHTS["size"]
    # Material
    rq_mat = (req_attrs.get("material") or "").upper()
    cd_mat = (cand_attrs.get("material") or "").upper()
    if rq_mat and cd_mat and rq_mat == cd_mat:
        score += WEIGHTS["material"]
    # Pressure
    rq_pr = (req_attrs.get("pressure") or "").upper()
    cd_pr = (cand_attrs.get("pressure") or "").upper()
    if rq_pr and cd_pr and rq_pr == cd_pr:
        score += WEIGHTS["pressure"]
    # Type tokens overlap
    rq_types = set([t for t in req_attrs.get("type_tokens", []) if t])
    cd_types = set([t for t in cand_attrs.get("type_tokens", []) if t])
    if rq_types and cd_types and len(rq_types & cd_types) > 0:
        score += WEIGHTS["type"]
    # Fuzzy similarity scaled
    fuzz_score = fuzz.token_set_ratio(req_attrs.get("combined", ""), cand_attrs.get("combined", ""))
    score += (fuzz_score / 100.0) * WEIGHTS["fuzzy"]
    return float(score)

# ------------------------------
# Streamlit UI & matching process
# ------------------------------
st.set_page_config(page_title="BuildWise Piping Estimation Tool", page_icon="📐", layout="wide")

# session login state
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

# Login
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
match_threshold = st.sidebar.slider("Match threshold", 0, 100, int(MIN_MATCH_SCORE),
                                    help="Minimum acceptance score for a 'good' match. Increase to be stricter.")
st.sidebar.markdown("---")
st.sidebar.write(f"Signed in as **{username}** ({role})")

# Ensure folders exist
user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)
os.makedirs(FORM_FOLDER, exist_ok=True)

# Admin user management
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

# Shared forms (admin uploads/downloads)
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

# Price list upload & manage
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

# Upload Estimation File + Matching
st.subheader(":page_facing_up: Upload Estimation File (Tải lên File Yêu Cầu Báo Giá)")
estimation_file = st.file_uploader("Upload estimation request (.xlsx) — Upload Estimation File (Tải lên File Yêu Cầu Báo Giá)", type=["xlsx"], key="est_file")
run_matching = st.button("🔎 Match now")

# Load reference (if available)
reference = load_reference_table(REFERENCE_FILE)
if reference is None:
    st.info(f"Reference file '{REFERENCE_FILE}' not found or invalid. App will run with rule-based extraction only. To enable dynamic matching, upload {REFERENCE_FILE} to the app folder.")
else:
    st.write(f"Loaded reference dictionary: {len(reference['df'])} rows.")

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
    est["size"] = base_est.apply(extract_size_val)
    est["pressure"] = base_est.apply(extract_pressure_val)
    est["material"] = base_est.apply(lambda x: detect_material_from_text(x, ref_materials=reference["materials"] if reference else None))
    est["type_tokens"] = base_est.apply(extract_type_keywords)
    est["is_fitting"] = base_est.apply(is_fitting_text)

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

    base_db = db[db_cols[0]].fillna('').astype(str) + " " + db[db_cols[1]].fillna('').astype(str) + " " + db[db_cols[2]].fillna('').astype(str)
    db["combined"]  = base_db.apply(clean)
    db["size"]  = base_db.apply(extract_size_val)
    db["pressure"] = base_db.apply(extract_pressure_val)
    db["material"] = base_db.apply(lambda x: detect_material_from_text(x, ref_materials=reference["materials"] if reference else None))
    db["type_tokens"] = base_db.apply(extract_type_keywords)
    db["is_fitting"] = db["combined"].apply(is_fitting_text)

    # If reference exists, prefer reference attributes for DB when name matches
    if reference:
        name_lookup = reference["name_lookup"]
        for idx, r in db.iterrows():
            k = r["combined"]
            if k in name_lookup:
                info = name_lookup[k]
                if info.get("material"):
                    db.at[idx, "material"] = info["material"].upper()
                try:
                    s = extract_size_val(info.get("size_raw", ""))
                    if s:
                        db.at[idx, "size"] = s
                except:
                    pass
                p = extract_pressure_val(info.get("pressure_raw", ""))
                if p:
                    db.at[idx, "pressure"] = p

    # ------ Matching loop using weighted scoring ------
    output_data = []
    for _, row in est.iterrows():
        query = row["combined"]
        req_attrs = {
            "combined": query,
            "size": row.get("size"),
            "material": row.get("material") or "",
            "pressure": row.get("pressure") or "",
            "type_tokens": row.get("type_tokens") or []
        }
        unit = row[est_cols[3]]
        qty  = row[est_cols[4]]

        best = None
        best_score = -1.0

        # Candidate set: prefer rows that include pipe/fitting keywords
        pref_mask = db["combined"].apply(lambda x: any(k in x for k in (PIPE_KEYWORDS + FITTING_KEYWORDS)))
        if not db[pref_mask].empty:
            candidates = db[pref_mask].copy()
        else:
            candidates = db.copy()

        cand_scores = []
        for idx, cand in candidates.iterrows():
            c_attrs = {
                "combined": cand["combined"],
                "size": cand.get("size"),
                "material": cand.get("material") or "",
                "pressure": cand.get("pressure") or "",
                "type_tokens": cand.get("type_tokens") or []
            }
            s = compute_match_score(req_attrs, c_attrs)
            cand_scores.append((s, idx, cand))

        if cand_scores:
            cand_scores.sort(key=lambda x: x[0], reverse=True)
            top_score, top_idx, top_row = cand_scores[0]
            if top_score >= match_threshold:
                best = top_row
                best_score = top_score
            else:
                if top_score >= FALLBACK_FUZZY_MIN:
                    best = top_row
                    best_score = top_score

        if best is None:
            db_f = db.copy()
            db_f["fuzz"] = db_f["combined"].apply(lambda x: fuzz.token_set_ratio(query, x))
            top = db_f.sort_values("fuzz", ascending=False).head(1)
            if not top.empty and top.iloc[0]["fuzz"] >= FALLBACK_FUZZY_MIN:
                best = top.iloc[0]
                best_score = float(top.iloc[0]["fuzz"])

        # Pull costs
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

    # Build result DataFrame
    result_df = pd.DataFrame(output_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Unit", "Quantity",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
    result_df.loc[len(result_df.index)] = [""] * 10 + [grand_total]

    # Display + format
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

    # Export
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Matched Results")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")
    st.download_button("📥 Download Cleaned Estimation File (Tải File Báo Giá Đã Làm Sạch)", buffer.getvalue(), file_name="Estimation_Result_BuildWise_Piping.xlsx")
else:
    st.info("Upload your estimation file and price list(s), then click **Match now**.")
