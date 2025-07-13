
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Estimation Tool", layout="wide")

st.title("🧮 Estimation Matching Tool")
st.markdown("Upload your **Database Excel** and **Estimation Request Excel** to generate results.")

db_file = st.file_uploader("Upload Database Excel File", type=["xlsx"], key="db")
est_file = st.file_uploader("Upload Estimation Request Excel File", type=["xlsx"], key="est")

if db_file and est_file:
    with st.spinner("Processing files..."):

        # Read the Excel files
        db = pd.read_excel(db_file)
        est = pd.read_excel(est_file)

        # Normalize column names
        db = db.rename(columns={"Desciption": "Description"})
        est = est.rename(columns={"Desciption": "Description"})

        result = est.copy()
        result["Material cost"] = None
        result["Labour cost"] = None

        # Matching logic
        def find_match(row):
            for col in ["Model", "Description", "Specification"]:
                value = row[col]
                if pd.isna(value):
                    continue
                match = db[db[col].astype(str).str.strip().str.lower() == str(value).strip().lower()]
                if not match.empty:
                    return match.iloc[0]["Material cost"], match.iloc[0]["Labour cost"]
            return None, None

        result[["Material cost", "Labour cost"]] = result.apply(find_match, axis=1, result_type="expand")

        # Calculations
        result["Amount Material"] = result["Quantity"] * result["Material cost"]
        result["Amount Labour"] = result["Quantity"] * result["Labour cost"]
        result["Total"] = result["Amount Material"].fillna(0) + result["Amount Labour"].fillna(0)

        # Add grand total row
        grand_total = result["Total"].sum()
        grand_total_row = pd.Series([""] * len(result.columns), index=result.columns)
        grand_total_row["Total"] = grand_total
        result = pd.concat([result, pd.DataFrame([grand_total_row])], ignore_index=True)

        st.success("✅ Estimation completed!")
        st.dataframe(result)

        # Download link
        from io import BytesIO
        output = BytesIO()
        result.to_excel(output, index=False)
        st.download_button(
            label="📥 Download Result Excel",
            data=output.getvalue(),
            file_name="Estimation_Result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
