
import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

def clean(text):
    text = str(text).lower()
    text = re.sub(r"0[,.]?6kv|1[,.]?0kv", "", text)
    text = text.replace("mm2", "").replace("mm²", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("/", " ").replace(",", "")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_conduit_size(text):
    text = str(text).lower()
    match = re.search(r"(d|ø|phi)?\s*(\d{1,3})(mm)?", text)
    return match.group(2) if match else ""

def extract_dimensions(text):
    text = str(text).lower()
    dims = {}
    width = re.search(r"w[=\s]*([0-9]{2,4})", text)
    height = re.search(r"h[=\s]*([0-9]{2,4})", text)
    thickness = re.search(r"(t|dày)[=\s]*([0-9.]+)", text)
    if width:
        dims["w"] = int(width.group(1))
    if height:
        dims["h"] = int(height.group(1))
    if thickness:
        dims["t"] = float(thickness.group(2))
    return dims

def get_category(description):
    desc = str(description).lower()
    if "cable" in desc or "dây" in desc:
        return "cable"
    elif "conduit" in desc or "ống luồn" in desc or "pipe" in desc:
        return "conduit"
    elif "tray" in desc or "máng cáp" in desc or "duct" in desc:
        return "cable_tray"
    elif "rack" in desc or "thang cáp" in desc:
        return "cable_rack"
    return "other"

# dummy placeholder for full Streamlit logic
def main():
    st.title("BuildWise Estimation Tool")
    st.write("This is a placeholder to test category assignment.")
    sample_data = pd.DataFrame({
        "Description": [
            "Dây điện Cadivi 3x2.5mm²",
            "Ống luồn Conduit D25",
            "Máng cáp sơn tĩnh điện W300xH100xT1.5",
            "Thang cáp mạ kẽm W500xH100xT2.0",
            "Khác"
        ]
    })
    sample_data["Category"] = sample_data["Description"].apply(get_category)
    st.dataframe(sample_data)

if __name__ == "__main__":
    main()
