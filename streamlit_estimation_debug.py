# BuildWise Estimation App - Final Version with Cable + Conduit Matching Logic
import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

# ------------------------------ UTILITY FUNCTIONS ------------------------------
def clean(text):
    text = str(text).lower()
    text = re.sub(r"[(),/-]", " ", text)
    text = text.replace("mm2", "").replace("mm²", "").replace("mm", "")
    text = text.replace("cáp", "cable").replace("dây", "cable")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_cable_attributes(text):
    text = clean(text)
    size_match = re.search(r"\b\d{1,2}\s*[cC×x]\s*\d{1,3}(\.\d+)?", text)
    voltage_match = re.search(r"\b0[.,]?6[ /-]?1[.,]?0?k?[vV]?", text)
    material_match = re.search(r"\b(cu|nhôm|aluminium|al)\b", text)
    insulation_match = re.search(r"\b(xlpe|pvc|pe|lszh)\b", text)
    shielding_match = re.search(r"\b(screen|tape|shield|armored|swa|sta|a)\b", text)
    has_keyword = any(x in text for x in ["cáp", "cable", "dây"])

    return {
        "size": size_match.group(0).replace(" ", "") if size_match else "",
        "voltage": "0.6/1kV" if voltage_match else "",
        "material": material_match.group(1).lower() if material_match else "",
        "insulation": insulation_match.group(1).upper() if insulation_match else "",
        "shielding": shielding_match.group(1).lower() if shielding_match else "",
        "has_keyword": has_keyword
    }

def extract_conduit_attributes(text):
    text = clean(text)
    size_match = re.search(r"\b(d|ø|phi)?\s*\d{1,3}(mm)?\b", text)
    type_match = re.search(r"\b(pvc|hdpe|imc|emt|rsc|flexible|corrugated|ống mềm|ống cứng)\b", text)
    material_match = re.search(r"\b(thép|inox|nhựa|aluminum|galvanized)\b", text)
    has_keyword = any(x in text for x in ["ống", "conduit", "ống luồn", "ống dây", "ống mềm", "flexible"])

    return {
        "size": size_match.group(0).replace(" ", "") if size_match else "",
        "type": type_match.group(1).lower() if type_match else "",
        "material": material_match.group(1).lower() if material_match else "",
        "has_keyword": has_keyword
    }

# ------------------------------ STREAMLIT CONFIG ------------------------------
st.set_page_config(page_title="BuildWise", page_icon="📀", layout="wide")
st.image("assets/logo.png", width=120)
st.title(":triangular_ruler: BuildWise - Smart Estimation Tool")

username = st.sidebar.text_input("Username")
if not username:
    st.warning("Please enter your username to continue.")
    st.stop()

user_folder = f"user_data/{username}"
os.makedirs(user_folder, exist_ok=True)

# ------------------------------ FILE UPLOADS ------------------------------
st.subheader(":file_folder: Upload Price List Files")
uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(user_folder, file.name), "wb") as f:
            f.write(file.read())
    st.success("✅ Price list uploaded.")

st.subheader(":open_file_folder: Manage Price Lists")
price_list_files = os.listdir(user_folder)
selected_file = st.radio("Choose one file to match or use all", ["All files"] + price_list_files)

st.subheader(":page_facing_up: Upload Estimation File")
estimation_file = st.file_uploader("Upload estimation request (.xlsx)", type=["xlsx"], key="est")

if estimation_file and price_list_files:
    est = pd.read_excel(estimation_file).dropna(how="all")
    est_cols = est.columns.tolist()
    if len(est_cols) < 5:
        st.error("Estimation file must have at least 5 columns.")
        st.stop()

    est["combined"] = (est[est_cols[0]].fillna("") + " " + est[est_cols[1]].fillna("") + " " + est[est_cols[2]].fillna(""))
    est["cleaned"] = est["combined"].apply(clean)

    db_frames = []
    if selected_file == "All files":
        for f in price_list_files:
            df = pd.read_excel(os.path.join(user_folder, f)).dropna(how="all")
            df["source"] = f
            db_frames.append(df)
        db = pd.concat(db_frames, ignore_index=True)
    else:
        db = pd.read_excel(os.path.join(user_folder, selected_file)).dropna(how="all")

    db_cols = db.columns.tolist()
    if len(db_cols) < 6:
        st.error("Price list must have at least 6 columns.")
        st.stop()

    db["combined"] = (db[db_cols[0]].fillna("") + " " + db[db_cols[1]].fillna("") + " " + db[db_cols[2]].fillna(""))
    db["cleaned"] = db["combined"].apply(clean)

    output_data = []
    for i, row in est.iterrows():
        qtext = row["cleaned"]
        cable_attr = extract_cable_attributes(qtext)
        conduit_attr = extract_conduit_attributes(qtext)
        unit = row[est_cols[3]]
        qty = row[est_cols[4]]
        best = None

        candidates = db.copy()
        candidates["score"] = 0

        if cable_attr["size"] and cable_attr["has_keyword"]:
            candidates["attr"] = candidates["cleaned"].apply(extract_cable_attributes)
            filtered = candidates[candidates["attr"].apply(lambda a: a["size"] == cable_attr["size"] and a["has_keyword"])]
            filtered["score"] = filtered["attr"].apply(lambda a: fuzz.token_set_ratio(qtext, a["size"]))
            best = filtered.loc[filtered["score"].idxmax()] if not filtered.empty and filtered["score"].max() >= 70 else None

        elif conduit_attr["size"] and conduit_attr["has_keyword"]:
            candidates["attr"] = candidates["cleaned"].apply(extract_conduit_attributes)
            filtered = candidates[candidates["attr"].apply(lambda a: a["size"] == conduit_attr["size"] and a["has_keyword"])]
            filtered["score"] = filtered["attr"].apply(lambda a: fuzz.token_set_ratio(qtext, a["size"]))
            best = filtered.loc[filtered["score"].idxmax()] if not filtered.empty and filtered["score"].max() >= 70 else None

        if best is not None:
            desc_proposed = best[db_cols[1]]
            m_cost = pd.to_numeric(best[db_cols[4]], errors="coerce")
            l_cost = pd.to_numeric(best[db_cols[5]], errors="coerce")
        else:
            desc_proposed = ""
            m_cost = l_cost = 0

        qty_val = pd.to_numeric(qty, errors="coerce")
        if pd.isna(qty_val): qty_val = 0
        amt_mat = qty_val * m_cost
        amt_lab = qty_val * l_cost
        total = amt_mat + amt_lab

        output_data.append([
            row[est_cols[0]], row[est_cols[1]], desc_proposed, row[est_cols[2]],
            unit, qty, m_cost, l_cost, amt_mat, amt_lab, total
        ])

    result_df = pd.DataFrame(output_data, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Unit", "Quantity",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])

    grand_total = pd.to_numeric(result_df["Total"], errors="coerce").sum()
    result_df.loc[len(result_df.index)] = [""]*10 + [grand_total]

    st.subheader(":mag: Matched Estimation")
    display_df = result_df.copy()
    display_df["Quantity"] = pd.to_numeric(display_df["Quantity"], errors="coerce").fillna(0).astype(int).map("{:,}".format)
    for col in ["Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0).astype(int).map("{:,}".format)
    st.dataframe(display_df)

    st.subheader(":x: Unmatched Rows")
    unmatched_df = result_df[result_df["Description (proposed)"] == ""]
    if not unmatched_df.empty:
        st.dataframe(unmatched_df)
    else:
        st.info("✅ All rows matched successfully!")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Matched Results")
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched Items")

    st.download_button("📥 Download Cleaned Estimation File", buffer.getvalue(), file_name="Estimation_Result_BuildWise.xlsx")
