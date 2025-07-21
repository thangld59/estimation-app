import streamlit as st
import pandas as pd
import os
import re
from io import BytesIO
from rapidfuzz import fuzz

# ------------------------------
# Utility Functions
# ------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"0[,.]?6kv|1[,.]?0kv", "", text)
    text = re.sub(r"mm2|mm²|\(.*?\)", "", text)
    text = re.sub(r"[/,-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_cable_size(text):
    text = text.lower()
    match = re.search(r'\b\d{1,2}\s*[x×cC]\s*\d{1,3}(\.\d+)?\b', text)
    return match.group(0).replace(" ", "") if match else ""

def extract_voltage(text):
    match = re.search(r'\b0[.,]?6[ /-]?1[.,]?0?k?[vV]?\b', text)
    return "0.6/1kV" if match else ""

def extract_cable_material(text):
    if "nhôm" in text or "al" in text or "aluminium" in text:
        return "al"
    if "cu" in text or "đồng" in text:
        return "cu"
    return ""

def extract_insulation(text):
    for ins in ["xlpe", "pvc", "pe", "lszh"]:
        if ins in text:
            return ins
    return ""

def extract_shield(text):
    for kw in ["swa", "sta", "armored", "tape", "screen", "shield"]:
        if kw in text:
            return kw
    return ""

def extract_conduit_size(text):
    match = re.search(r'\b(d\s*\d{1,3}|(ø|phi)?\s*\d{1,3}(mm)?)\b', text.lower())
    return match.group(0).replace(" ", "") if match else ""

def extract_conduit_material(text):
    for mat in ["thép", "inox", "nhựa", "aluminum", "galvanized"]:
        if mat in text:
            return mat
    return ""

def extract_conduit_type(text):
    for typ in ["pvc", "hdpe", "imc", "emt", "rsc", "flexible", "corrugated", "ống mềm", "ống cứng"]:
        if typ in text:
            return typ
    return ""

def contains_keywords(text, keywords):
    return any(kw in text for kw in keywords)

# ------------------------------
# Matching Function
# ------------------------------
def match_items(est_df, db_df):
    output_rows = []
    for _, row in est_df.iterrows():
        query = row["combined"]
        match_type = "none"
        best_match = None
        best_score = 0

        # Detect if it's a cable
        is_cable = contains_keywords(query, ["cáp", "cable", "dây điện", "wire"])
        is_conduit = contains_keywords(query, ["ống", "conduit", "ống luồn", "ống dây", "ống mềm", "flexible"])

        if is_cable:
            size = extract_cable_size(query)
            voltage = extract_voltage(query)
            material = extract_cable_material(query)
            insulation = extract_insulation(query)
            shield = extract_shield(query)

            candidates = db_df.copy()
            candidates["score"] = 0
            for i, db_row in candidates.iterrows():
                db_text = db_row["combined"]
                score = 0
                if size and size in db_text:
                    score += 40
                if material and material in db_text:
                    score += 15
                if insulation and insulation in db_text:
                    score += 15
                if shield and shield in db_text:
                    score += 10
                if voltage and voltage in db_text:
                    score += 10
                score += fuzz.token_set_ratio(query, db_text) * 0.1
                candidates.at[i, "score"] = score

            best = candidates.loc[candidates["score"].idxmax()]
            if best["score"] >= 70:
                best_match = best
                match_type = "cable"

        elif is_conduit:
            size = extract_conduit_size(query)
            material = extract_conduit_material(query)
            typ = extract_conduit_type(query)

            candidates = db_df.copy()
            candidates["score"] = 0
            for i, db_row in candidates.iterrows():
                db_text = db_row["combined"]
                score = 0
                if size and size in db_text:
                    score += 40
                if material and material in db_text:
                    score += 25
                if typ and typ in db_text:
                    score += 15
                score += fuzz.token_set_ratio(query, db_text) * 0.1
                candidates.at[i, "score"] = score

            best = candidates.loc[candidates["score"].idxmax()]
            if best["score"] >= 70:
                best_match = best
                match_type = "conduit"

        if not best_match:
            best_match = pd.Series([None] * len(db_df.columns), index=db_df.columns)

        qty = pd.to_numeric(row["Quantity"], errors="coerce") or 0
        m_cost = pd.to_numeric(best_match.iloc[4], errors="coerce") or 0
        l_cost = pd.to_numeric(best_match.iloc[5], errors="coerce") or 0

        output_rows.append([
            row["Model"], row["Description"], best_match.iloc[1] or "", row["Specification"],
            row["Unit"], row["Quantity"], m_cost, l_cost,
            qty * m_cost, qty * l_cost, qty * (m_cost + l_cost)
        ])
    return pd.DataFrame(output_rows, columns=[
        "Model", "Description (requested)", "Description (proposed)", "Specification", "Unit", "Quantity",
        "Material Cost", "Labour Cost", "Amount Material", "Amount Labour", "Total"
    ])
