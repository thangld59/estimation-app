# streamlit_estimation_app_final_quotation.py
# BuildWise - Estimation & Quotation app (Excel template version)
# - Login / users (users.json)
# - Per-user price lists, customers, company profile
# - Cable matching (same logic as previous working version)
# - Trading terms
# - Quotation generation using quotation_template.xlsx (no PDF)
# - Quotation history + download + preview

import streamlit as st
import pandas as pd
import os
import re
import json
from io import BytesIO
from datetime import datetime
from rapidfuzz import fuzz
from openpyxl import load_workbook

# ------------------------------
# Constants / folders
# ------------------------------
USERS_FILE = "users.json"
FORM_FOLDER = "shared_forms"
ASSETS_FOLDER = "assets"
TEMPLATE_FILENAME = "quotation_template.xlsx"

DEFAULT_USERS = {
    "Admin123": {"password": "BuildWise2025", "role": "admin"},
    "User123": {"password": "User2025", "role": "user"}
}

# ------------------------------
# Users
# ------------------------------
def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


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


USERS = load_users()

# ------------------------------
# Matching utilities (same logic)
# ------------------------------
MAIN_SIZE_RE = re.compile(r'\b(\d{1,2})\s*[cC]?\s*[x×]\s*(\d{1,3}(?:\.\d+)?)\b')
AUX_RE = re.compile(
    r'\+\s*(?:([1-9]\d*)\s*[cC]?\s*[x×]\s*)?((?:pe|e|n))?(\d{1,3}(?:\.\d+)?)',
    flags=re.IGNORECASE
)
MATERIAL_TOKEN_RE = re.compile(
    r'(cu|aluminium|al|xlpe|pvc|pe|lszh|hdpe|dsta|sta|swa)',
    flags=re.IGNORECASE
)


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
        except Exception:
            main_cores = None
        try:
            main_size = float(main_match.group(2))
        except Exception:
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
            except Exception:
                aux_cores = None

        if type_str:
            t = type_str.strip().upper()
            if t in ["E", "PE"]:
                aux_type = "E"
            elif t == "N":
                aux_type = "N"

        try:
            aux_size = float(size_str)
        except Exception:
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
        if tt == "aluminium":
            norm.append("al")
        else:
            norm.append(tt)
    return norm


def material_structure_score(query_tokens, target_tokens):
    if not query_tokens and not target_tokens:
        return 100.0
    if not query_tokens or not target_tokens:
        return 0.0

    weights_map = {
        "cu": 1.0, "al": 1.0,
        "xlpe": 0.9, "pvc": 0.7,
        "lszh": 0.6, "pe": 0.5, "hdpe": 0.5,
        "dsta": 0.4, "sta": 0.4, "swa": 0.4
    }

    q_set = list(dict.fromkeys(query_tokens))
    t_set = list(dict.fromkeys(target_tokens))

    match_score = 0.0
    possible_score = 0.0

    all_keys = list(dict.fromkeys(q_set + t_set))
    for k in all_keys:
        w = weights_map.get(k, 0.3)
        possible_score += w
        if k in q_set and k in t_set:
            match_score += w

    base = (match_score / possible_score) * 100.0 if possible_score > 0 else 0.0
    extra_in_target = len([k for k in t_set if k not in q_set])
    extra_in_query = len([k for k in q_set if k not in t_set])
    penalty = extra_in_target * 5.0 + extra_in_query * 2.0
    score = max(0.0, base - penalty)
    return score


def combined_match_score(
    query, q_main_key, q_aux_key, q_mats,
    row_combined, r_main_key, r_aux_key, r_mats,
    threshold, weights
):
    size_score = 0.0
    cores_score = 0.0
    mat_score = 0.0

    if q_main_key and r_main_key:
        if q_main_key == r_main_key:
            size_score = 100.0
        else:
            size_score = fuzz.token_set_ratio(q_main_key, r_main_key)
    else:
        size_score = fuzz.partial_ratio(query, row_combined)

    if q_aux_key and r_aux_key:
        if q_aux_key == r_aux_key:
            cores_score = 100.0
        else:
            cores_score = fuzz.token_set_ratio(str(q_aux_key), str(r_aux_key))
    else:
        cores_score = 100.0 if (not q_aux_key and not r_aux_key) else 0.0

    mat_score = material_structure_score(q_mats, r_mats)

    final = (
        weights["size"] * size_score
        + weights["cores"] * cores_score
        + weights["material"] * mat_score
    )
    return final


# ------------------------------
# Streamlit setup & login
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


if not st.session_state["logged_in"]:
    st.title("📐 BuildWise - Sign in")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
        if submitted:
            ok = do_login(u, p)
            if ok:
                st.success(
                    f"Logged in as {st.session_state['username']} ({st.session_state['role']})"
                )
                st.experimental_rerun()
            else:
                st.error("Invalid username or password. Edit users.json to add users if needed.")
    st.stop()

username = st.session_state["username"]
role = st.session_state["role"]

# Header
col1, col2 = st.columns([8, 1])
with col1:
    logo_path = os.path.join(ASSETS_FOLDER, "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
    st.markdown("## :triangular_ruler: BuildWise - Smart Estimation Tool")
with col2:
    if st.button("Logout"):
        do_logout()
        st.experimental_rerun()

# Folders
user_folder = os.path.join("user_data", username)
os.makedirs(user_folder, exist_ok=True)
os.makedirs(FORM_FOLDER, exist_ok=True)
os.makedirs(os.path.join(user_folder, "quotations"), exist_ok=True)

# ------------------------------
# Matching settings (per-user)
# ------------------------------
def weights_file_for(user):
    folder = os.path.join("user_data", user)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "weights.json")


def load_weights_for(user):
    path = weights_file_for(user)
    defaults = {"threshold": 70, "size": 0.45, "cores": 0.25, "material": 0.30}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "threshold": int(data.get("threshold", defaults["threshold"])),
                    "size": float(data.get("size", defaults["size"])),
                    "cores": float(data.get("cores", defaults["cores"])),
                    "material": float(data.get("material", defaults["material"]))
                }
        except Exception:
            return defaults
    return defaults


def save_weights_for(user, wdict):
    path = weights_file_for(user)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wdict, f, indent=2, ensure_ascii=False)


if "match_settings_loaded" not in st.session_state:
    st.session_state["match_settings_loaded"] = True
    ws = load_weights_for(username)
    st.session_state["match_threshold"] = ws["threshold"]
    st.session_state["weight_size"] = ws["size"]
    st.session_state["weight_cores"] = ws["cores"]
    st.session_state["weight_material"] = ws["material"]

# ------------------------------
# Customers & trading terms utils
# ------------------------------
def user_customers_file(user):
    folder = os.path.join("user_data", user)
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


def trading_terms_file(user):
    folder = os.path.join("user_data", user)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "trading_terms.json")


def load_trading_terms(user):
    path = trading_terms_file(user)
    defaults = {
        "payment": "",
        "delivery": "",
        "transportation_fee": "",
        "validity": ""
    }
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
                for k in defaults:
                    defaults[k] = d.get(k, defaults[k])
                return defaults
        except Exception:
            return defaults
    return defaults


def save_trading_terms(user, data):
    path = trading_terms_file(user)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_price_list_files(folder_path):
    try:
        return sorted(
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith((".xlsx", ".xls"))
        )
    except Exception:
        return []


def make_quotation_filename(prefix="Quotation", ext="xlsx"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


# ------------------------------
# Quotation template handling
# ------------------------------
def create_quotation_from_template(user, result_df, company_info, customer_info, trading_terms):
    """
    Fill quotation_template.xlsx with company, customer, terms, and matched items,
    then save to user history and return a BytesIO for download.
    """
    template_path = os.path.join(ASSETS_FOLDER, TEMPLATE_FILENAME)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found at {template_path}")

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()

    wb = load_workbook(template_path)
    sheet_names = wb.sheetnames
    if len(sheet_names) < 2:
        raise ValueError("Template must have at least two sheets (Quotation, Matched items).")
    sh_quote = wb[sheet_names[0]]
    sh_items = wb[sheet_names[1]]

    # Company info
    sh_quote["A1"] = company_info.get("name", "")
    sh_quote["A2"] = company_info.get("address", "")
    sh_quote["A3"] = company_info.get("phone", "")
    sh_quote["A4"] = company_info.get("email", "")

    # Customer info
    sh_quote["D3"] = customer_info.get("name", "")
    sh_quote["D4"] = customer_info.get("company", "")
    sh_quote["D5"] = customer_info.get("address", "")
    sh_quote["D6"] = customer_info.get("phone", "")
    sh_quote["D7"] = customer_info.get("email", "")

    # Trading terms
    sh_quote["A21"] = trading_terms.get("payment", "")
    sh_quote["A22"] = trading_terms.get("delivery", "")
    sh_quote["A23"] = trading_terms.get("transportation_fee", "")
    sh_quote["A24"] = trading_terms.get("validity", "")

    # Grand total on Quotation sheet
    try:
        sh_quote["K33"] = float(grand_total)
    except Exception:
        sh_quote["K33"] = grand_total

    # Matched items on second sheet
    # Skip last row of result_df (grand total row)
    items_df = result_df.iloc[:-1].copy() if len(result_df) > 0 else result_df.copy()

    # Clear old data in rows 2..40 for columns A..K (keep header/format in row 1)
    max_row_to_clear = 40
    for r in range(2, max_row_to_clear + 1):
        for c in range(1, 12):
            sh_items.cell(row=r, column=c, value=None)

    # Write rows starting from row 2
    for idx, (_, row) in enumerate(items_df.iterrows(), start=2):
        sh_items.cell(row=idx, column=1, value=row.get("Model", ""))
        sh_items.cell(row=idx, column=2, value=row.get("Description (requested)", ""))
        sh_items.cell(row=idx, column=3, value=row.get("Description (proposed)", ""))
        sh_items.cell(row=idx, column=4, value=row.get("Specification", ""))
        sh_items.cell(row=idx, column=5, value=row.get("Unit", ""))
        sh_items.cell(row=idx, column=6, value=float(row.get("Quantity", 0) or 0))
        sh_items.cell(row=idx, column=7, value=float(row.get("Material Cost", 0) or 0))
        sh_items.cell(row=idx, column=8, value=float(row.get("Labour Cost", 0) or 0))
        sh_items.cell(row=idx, column=9, value=float(row.get("Amount Material", 0) or 0))
        sh_items.cell(row=idx, column=10, value=float(row.get("Amount Labour", 0) or 0))
        sh_items.cell(row=idx, column=11, value=float(row.get("Total", 0) or 0))

    # Save to buffer
    output = BytesIO()
    wb.save(output)
    output.seek(0)

    # Also save to user's quotation history
    q_folder = os.path.join("user_data", user, "quotations")
    os.makedirs(q_folder, exist_ok=True)
    filename = make_quotation_filename("Quotation", "xlsx")
    path = os.path.join(q_folder, filename)
    with open(path, "wb") as f:
        f.write(output.getvalue())

    return output, path


def preview_quotation_from_bytes(xlsx_bytes):
    """
    Load Quotation sheet from xlsx bytes and show as a pandas DataFrame (for preview).
    """
    buffer = BytesIO(xlsx_bytes)
    wb = load_workbook(buffer, data_only=True)
    sh_quote = wb[wb.sheetnames[0]]
    data = []
    for row in sh_quote.iter_rows(values_only=True):
        data.append(list(row))
    df = pd.DataFrame(data)
    return df


# ------------------------------
# Pages
# ------------------------------
def page_company_profile(user_folder):
    st.subheader("🏢 Company Profile")
    comp_file = os.path.join(user_folder, "company.json")
    profile = {}
    if os.path.exists(comp_file):
        try:
            with open(comp_file, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception:
            profile = {}

    name = st.text_input("Company name", value=profile.get("name", ""))
    address = st.text_input("Address", value=profile.get("address", ""))
    phone = st.text_input("Phone", value=profile.get("phone", ""))
    email = st.text_input("Email", value=profile.get("email", ""))

    if st.button("Save company profile"):
        data = {
            "name": name.strip(),
            "address": address.strip(),
            "phone": phone.strip(),
            "email": email.strip(),
        }
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        st.success("Company profile saved.")


def page_customers():
    st.subheader("👥 Customers")

    if role == "admin":
        st.info("Admin: choose a user to view their customers.")
        base = "user_data"
        os.makedirs(base, exist_ok=True)
        user_dirs = sorted(
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        )
        chosen_user = st.selectbox("Select user", ["--Select--"] + user_dirs, index=0)
        if chosen_user != "--Select--":
            customers = load_customers_for(chosen_user)
            owner = chosen_user
        else:
            customers = []
            owner = None
    else:
        customers = load_customers_for(username)
        owner = username

    # Add new customer
    with st.expander("Add new customer", expanded=False):
        c_name = st.text_input("Customer name")
        c_company = st.text_input("Company")
        c_address = st.text_input("Address")
        c_phone = st.text_input("Phone")
        c_email = st.text_input("Email")
        c_notes = st.text_area("Notes")

        if st.button("Add customer"):
            if not c_name.strip():
                st.error("Customer name required.")
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
                target = owner if owner else username
                lst = load_customers_for(target)
                lst.append(new)
                save_customers_for(target, lst)
                st.success(f"Added customer for user {target}")
                st.experimental_rerun()

    if not customers:
        st.info("No customers.")
        return

    df = pd.DataFrame(customers)
    cols_order = ["id", "name", "company", "phone", "email", "address", "notes", "created_at"]
    cols = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
    df = df[cols]

    st.markdown("### Customer list")
    st.dataframe(df.reset_index(drop=True), use_container_width=True)

    st.markdown("### Manage customer")
    ids = df["id"].astype(str).tolist()
    sel_id = st.selectbox("Select customer ID", [""] + ids)
    if not sel_id:
        return

    row = df[df["id"].astype(str) == sel_id].iloc[0].to_dict()
    st.write(row)

    col1, col2 = st.columns(2)
    if col1.button("Edit customer"):
        with st.form(f"edit_customer_{sel_id}"):
            e_name = st.text_input("Customer name", value=row.get("name", ""))
            e_company = st.text_input("Company", value=row.get("company", ""))
            e_address = st.text_input("Address", value=row.get("address", ""))
            e_phone = st.text_input("Phone", value=row.get("phone", ""))
            e_email = st.text_input("Email", value=row.get("email", ""))
            e_notes = st.text_area("Notes", value=row.get("notes", ""))
            submitted = st.form_submit_button("Save changes")
            if submitted:
                target = owner if owner else username
                lst = load_customers_for(target)
                for i, c in enumerate(lst):
                    if str(c.get("id")) == sel_id:
                        lst[i].update({
                            "name": e_name.strip(),
                            "company": e_company.strip(),
                            "address": e_address.strip(),
                            "phone": e_phone.strip(),
                            "email": e_email.strip(),
                            "notes": e_notes.strip(),
                            "updated_at": pd.Timestamp.now().isoformat()
                        })
                        break
                save_customers_for(target, lst)
                st.success("Customer updated.")
                st.experimental_rerun()

    if col2.button("Delete customer"):
        target = owner if owner else username
        lst = load_customers_for(target)
        new_lst = [c for c in lst if str(c.get("id")) != sel_id]
        save_customers_for(target, new_lst)
        st.success("Customer deleted.")
        st.experimental_rerun()


def page_forms_and_instructions():
    st.subheader("📂 Forms and Instructions")
    st.write("Shared templates and forms. Admin can upload; all users can download.")

    form_files = sorted(os.listdir(FORM_FOLDER))

    if role == "admin":
        uploads = st.file_uploader(
            "Admin: Upload forms (xlsx/xls)",
            type=["xlsx", "xls"],
            accept_multiple_files=True
        )
        if uploads:
            for f in uploads:
                path = os.path.join(FORM_FOLDER, f.name)
                try:
                    with open(path, "wb") as out_f:
                        out_f.write(f.read())
                except Exception as e:
                    st.error(f"Error saving {f.name}: {e}")
            st.success("Forms uploaded.")

        if form_files:
            to_del = st.selectbox("Select form to delete", [""] + form_files)
            if to_del and st.button("Delete selected form"):
                try:
                    os.remove(os.path.join(FORM_FOLDER, to_del))
                    st.success("Form deleted.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error deleting form: {e}")
    else:
        if form_files:
            for f in form_files:
                path = os.path.join(FORM_FOLDER, f)
                try:
                    with open(path, "rb") as fh:
                        data = fh.read()
                    st.download_button(f"Download {f}", data, file_name=f, key=f"down_form_{f}")
                except Exception:
                    continue
        else:
            st.info("No forms available.")


def page_quotations():
    st.subheader("📄 Quotations")
    q_folder = os.path.join(user_folder, "quotations")
    os.makedirs(q_folder, exist_ok=True)
    files = sorted(os.listdir(q_folder))
    if not files:
        st.info("No quotations yet.")
        return

    for f in files:
        path = os.path.join(q_folder, f)
        c1, c2, c3 = st.columns([4, 1, 1])
        with c1:
            st.write(f)
        with c2:
            with open(path, "rb") as fh:
                data = fh.read()
            st.download_button("Download", data, file_name=f, key=f"down_q_{f}")
        with c3:
            if st.button("Delete", key=f"del_q_{f}"):
                os.remove(path)
                st.success("Deleted.")
                st.experimental_rerun()


def page_estimation():
    # 1. Upload price lists
    st.subheader("1. Upload Price List Files")
    uploads = st.file_uploader(
        "Upload one or more price list Excel files (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        key="pl_up_main"
    )
    if uploads:
        for f in uploads:
            try:
                with open(os.path.join(user_folder, f.name), "wb") as out_f:
                    out_f.write(f.read())
            except Exception as e:
                st.error(f"Error saving {f.name}: {e}")
        st.success("Price lists uploaded.")

    # 2. Manage price lists
    st.subheader("2. Manage Price Lists")
    price_list_files = list_price_list_files(user_folder)
    if price_list_files:
        st.write("Your price lists:")
        for f in price_list_files:
            st.write(f"- {f}")
    else:
        st.info("No price lists uploaded yet.")

    selected_file = st.radio(
        "Choose one price list or use all",
        ["All files"] + price_list_files,
        index=0
    )

    if price_list_files:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            to_del = st.selectbox(
                "Select a price list to delete",
                [""] + price_list_files,
                key="del_pl_main"
            )
        with col_b:
            if st.button("Delete selected price list"):
                if to_del:
                    try:
                        os.remove(os.path.join(user_folder, to_del))
                        st.success(f"Deleted {to_del}")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {e}")

    # 3. Matching estimation request
    st.subheader("3. Matching Estimation Request File")
    estimation_file = st.file_uploader(
        "Upload estimation request (.xlsx)",
        type=["xlsx"],
        key="estimation_file_main"
    )

    run_matching = st.button("Match now")

    match_threshold = st.session_state.get("match_threshold", 70)
    w_size = st.session_state.get("weight_size", 0.45)
    w_cores = st.session_state.get("weight_cores", 0.25)
    w_material = st.session_state.get("weight_material", 0.30)
    total_w = w_size + w_cores + w_material
    if total_w <= 0:
        total_w = 1.0
    weights = {
        "size": w_size / total_w,
        "cores": w_cores / total_w,
        "material": w_material / total_w
    }

    if run_matching:
        if estimation_file is None:
            st.error("Please upload an estimation file first.")
        elif not price_list_files:
            st.error("Please upload at least one price list first.")
        else:
            try:
                est = pd.read_excel(estimation_file).dropna(how="all")
            except Exception as e:
                st.error(f"Cannot read estimation file: {e}")
                est = None

            if est is not None:
                est_cols = est.columns.tolist()
                if len(est_cols) < 5:
                    st.error("Estimation file must have at least 5 columns (Model, Description, Spec, Unit, Quantity).")
                else:
                    base_est = (
                        est[est_cols[0]].fillna("") + " "
                        + est[est_cols[1]].fillna("") + " "
                        + est[est_cols[2]].fillna("")
                    )
                    est["combined"] = base_est.apply(clean)
                    parsed_est = base_est.apply(parse_cable_spec)
                    est["main_key"] = parsed_est.apply(lambda d: d["main_key"])
                    est["aux_key"] = parsed_est.apply(lambda d: d["aux_key"])
                    est["materials"] = base_est.apply(extract_material_structure_tokens)

                    # build DB
                    if selected_file == "All files":
                        frames = []
                        for f in price_list_files:
                            try:
                                df_pl = pd.read_excel(os.path.join(user_folder, f)).dropna(how="all")
                                df_pl["source"] = f
                                frames.append(df_pl)
                            except Exception:
                                continue
                        db = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                    else:
                        try:
                            db = pd.read_excel(os.path.join(user_folder, selected_file)).dropna(how="all")
                            db["source"] = selected_file
                        except Exception as e:
                            st.error(f"Cannot read price list: {e}")
                            db = pd.DataFrame()

                    if db.empty:
                        st.error("No rows found in price list file(s).")
                    else:
                        db_cols = db.columns.tolist()
                        if len(db_cols) < 6:
                            st.error("Price list requires at least 6 columns (Model, Description, Spec, ..., MaterialCost, LabourCost).")
                        else:
                            base_db = (
                                db[db_cols[0]].fillna("") + " "
                                + db[db_cols[1]].fillna("") + " "
                                + db[db_cols[2]].fillna("")
                            )
                            db["combined"] = base_db.apply(clean)
                            parsed_db = base_db.apply(parse_cable_spec)
                            db["main_key"] = parsed_db.apply(lambda d: d["main_key"])
                            db["aux_key"] = parsed_db.apply(lambda d: d["aux_key"])
                            db["materials"] = base_db.apply(extract_material_structure_tokens)

                            results = []
                            for _, row in est.iterrows():
                                query = row["combined"]
                                q_main = row["main_key"]
                                q_aux = row["aux_key"]
                                q_mats = row["materials"]
                                unit = row[est_cols[3]]
                                qty_value = row[est_cols[4]]

                                best = None
                                best_score = -1.0

                                c0 = db.copy()
                                if q_main:
                                    c0 = c0[c0["main_key"] == q_main]

                                def score_row(r):
                                    try:
                                        r_main = r.get("main_key", "")
                                        r_aux = r.get("aux_key", "")
                                        r_mats = r.get("materials", [])
                                        return combined_match_score(
                                            query, q_main, q_aux, q_mats,
                                            r.get("combined", ""),
                                            r_main, r_aux, r_mats,
                                            match_threshold, weights
                                        )
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
                                    c2["score"] = c2["combined"].apply(
                                        lambda x: fuzz.token_set_ratio(query, x)
                                    )
                                    top3 = c2.sort_values("score", ascending=False).head(1)
                                    if not top3.empty:
                                        best = top3.iloc[0]
                                        best_score = float(best["score"])

                                if best is not None and best_score >= 0:
                                    matched_desc = best[db_cols[1]]
                                    matched_model = best[db_cols[0]]
                                    matched_spec = best[db_cols[2]]
                                    m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce")
                                    l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce")
                                    if pd.isna(m_cost):
                                        m_cost = 0
                                    if pd.isna(l_cost):
                                        l_cost = 0
                                else:
                                    matched_desc = ""
                                    matched_model = ""
                                    matched_spec = ""
                                    m_cost = 0
                                    l_cost = 0

                                qty_num = pd.to_numeric(qty_value, errors="coerce")
                                if pd.isna(qty_num):
                                    qty_num = 0
                                amt_mat = qty_num * m_cost
                                amt_lab = qty_num * l_cost
                                total = amt_mat + amt_lab

                                results.append([
                                    matched_model,
                                    row[est_cols[1]],
                                    matched_desc,
                                    matched_spec,
                                    unit,
                                    qty_num,
                                    m_cost,
                                    l_cost,
                                    amt_mat,
                                    amt_lab,
                                    total
                                ])

                            result_df = pd.DataFrame(
                                results,
                                columns=[
                                    "Model",
                                    "Description (requested)",
                                    "Description (proposed)",
                                    "Specification",
                                    "Unit",
                                    "Quantity",
                                    "Material Cost",
                                    "Labour Cost",
                                    "Amount Material",
                                    "Amount Labour",
                                    "Total"
                                ]
                            )

                            grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
                            result_df.loc[len(result_df.index)] = [""] * 10 + [grand_total]

                            unmatched_df = result_df[result_df["Description (proposed)"] == ""]

                            st.session_state["result_df"] = result_df
                            st.session_state["unmatched_df"] = unmatched_df

    # Show result matching table (persistent)
    st.markdown("### Result matching table")
    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]
        unmatched_df = st.session_state.get("unmatched_df", pd.DataFrame())
        display_df = result_df.copy()
        display_df["Quantity"] = pd.to_numeric(
            display_df["Quantity"], errors="coerce"
        ).fillna(0).astype(int)
        for col in ["Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"]:
            display_df[col] = pd.to_numeric(
                display_df[col], errors="coerce"
            ).fillna(0).map("{:,.0f}".format)
        st.dataframe(display_df, use_container_width=True)

        st.subheader("Download matching file (.xlsx)")
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            result_df.to_excel(writer, index=False, sheet_name="Matched Results")
            if not unmatched_df.empty:
                unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")
        st.download_button(
            "Download cleaned estimation file",
            buffer.getvalue(),
            file_name="Estimation_Result_BuildWise.xlsx"
        )
    else:
        st.info("Run matching to see results.")

    # 4. Quotation generation
    st.subheader("4. Quotation generation")

    # Select customer
    customers = load_customers_for(username)
    cust_labels = ["--Select customer--"] + [
        f"{c.get('name', '')} ({c.get('company', '')})" for c in customers
    ]
    c1, c2 = st.columns([3, 1])
    with c1:
        selected_cust_label = st.selectbox("Select customer", cust_labels, index=0)
    active_customer = None
    if selected_cust_label != "--Select customer--":
        idx = cust_labels.index(selected_cust_label) - 1
        active_customer = customers[idx]

    # Edit customer inline
    with c2:
        if active_customer is not None:
            if st.button("Edit selected customer"):
                with st.form("edit_selected_customer_form_main"):
                    e_name = st.text_input("Customer name", value=active_customer.get("name", ""))
                    e_company = st.text_input("Company", value=active_customer.get("company", ""))
                    e_address = st.text_input("Address", value=active_customer.get("address", ""))
                    e_phone = st.text_input("Phone", value=active_customer.get("phone", ""))
                    e_email = st.text_input("Email", value=active_customer.get("email", ""))
                    e_notes = st.text_area("Notes", value=active_customer.get("notes", ""))
                    submitted = st.form_submit_button("Save customer")
                    if submitted:
                        customers[idx].update({
                            "name": e_name.strip(),
                            "company": e_company.strip(),
                            "address": e_address.strip(),
                            "phone": e_phone.strip(),
                            "email": e_email.strip(),
                            "notes": e_notes.strip(),
                            "updated_at": datetime.now().isoformat()
                        })
                        save_customers_for(username, customers)
                        st.success("Customer updated.")
                        st.experimental_rerun()

    if active_customer is not None:
        st.markdown("**Selected customer details:**")
        st.write(active_customer)
    else:
        st.info("Please select a customer for quotation.")

    # Trading terms (persistent inputs)
    st.markdown("### Trading terms / Điều khoản thương mại")
    terms = load_trading_terms(username)
    payment = st.text_area(
        "Payment / Thanh toán",
        key="payment_text",
        value=terms.get("payment", ""),
        height=80
    )
    delivery = st.text_input(
        "Delivery schedule / Tiến độ",
        key="delivery_text",
        value=terms.get("delivery", "")
    )
    trans_fee = st.text_input(
        "Transportation fee / Phí vận chuyển",
        key="trans_fee_text",
        value=terms.get("transportation_fee", "")
    )
    validity = st.text_input(
        "Quotation validity / Hiệu lực báo giá",
        key="validity_text",
        value=terms.get("validity", "")
    )

    if st.button("Save trading terms"):
        new_terms = {
            "payment": st.session_state.get("payment_text", ""),
            "delivery": st.session_state.get("delivery_text", ""),
            "transportation_fee": st.session_state.get("trans_fee_text", ""),
            "validity": st.session_state.get("validity_text", "")
        }
        save_trading_terms(username, new_terms)
        st.success("Trading terms saved.")

    # Generate quotation using template
    st.markdown("### Generate quotation")
    if st.button("Generate quotation (.xlsx)"):
        if "result_df" not in st.session_state:
            st.error("Please run matching first.")
        elif active_customer is None:
            st.error("Please select a customer.")
        else:
            # company info
            comp_file = os.path.join(user_folder, "company.json")
            company_info = {}
            if os.path.exists(comp_file):
                try:
                    with open(comp_file, "r", encoding="utf-8") as f:
                        company_info = json.load(f)
                except Exception:
                    company_info = {}

            # latest terms
            current_terms = {
                "payment": st.session_state.get("payment_text", ""),
                "delivery": st.session_state.get("delivery_text", ""),
                "transportation_fee": st.session_state.get("trans_fee_text", ""),
                "validity": st.session_state.get("validity_text", "")
            }
            save_trading_terms(username, current_terms)

            result_df = st.session_state["result_df"]

            try:
                xlsx_buffer, saved_path = create_quotation_from_template(
                    username,
                    result_df,
                    company_info,
                    active_customer,
                    current_terms
                )
            except Exception as e:
                st.error(f"Error generating quotation from template: {e}")
            else:
                st.success("Quotation generated, saved to history, and ready to download.")
                xlsx_bytes = xlsx_buffer.getvalue()
                st.download_button(
                    "Download quotation (.xlsx)",
                    xlsx_bytes,
                    file_name=os.path.basename(saved_path)
                )

                st.markdown("#### Quotation preview (Quotation sheet)")
                try:
                    df_preview = preview_quotation_from_bytes(xlsx_bytes)
                    st.dataframe(df_preview, use_container_width=True)
                except Exception:
                    st.info("Preview not available, but the file was generated successfully.")


# ------------------------------
# Sidebar navigation + match settings
# ------------------------------
st.sidebar.title("Navigation")
nav_items = ["Estimation", "Customers", "Company Profile", "Quotation", "Forms and Instructions"]
page = st.sidebar.radio("Go to", nav_items, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Matching settings")
th = st.sidebar.slider(
    "Match threshold",
    0, 100,
    st.session_state.get("match_threshold", 70)
)
w_s = st.sidebar.slider(
    "Size weight",
    0.0, 1.0,
    st.session_state.get("weight_size", 0.45),
    step=0.05
)
w_c = st.sidebar.slider(
    "Cores weight",
    0.0, 1.0,
    st.session_state.get("weight_cores", 0.25),
    step=0.05
)
w_m = st.sidebar.slider(
    "Material weight",
    0.0, 1.0,
    st.session_state.get("weight_material", 0.30),
    step=0.05
)

if st.sidebar.button("Save matching settings"):
    settings = {
        "threshold": int(th),
        "size": float(w_s),
        "cores": float(w_c),
        "material": float(w_m)
    }
    save_weights_for(username, settings)
    st.session_state["match_threshold"] = settings["threshold"]
    st.session_state["weight_size"] = settings["size"]
    st.session_state["weight_cores"] = settings["cores"]
    st.session_state["weight_material"] = settings["material"]
    st.sidebar.success("Settings saved.")

# Routing
if page == "Estimation":
    page_estimation()
elif page == "Customers":
    page_customers()
elif page == "Company Profile":
    page_company_profile(user_folder)
elif page == "Quotation":
    page_quotations()
elif page == "Forms and Instructions":
    page_forms_and_instructions()

st.markdown("---")
st.caption("BuildWise — Estimation & Quotation tool (Excel template version)")
