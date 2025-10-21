# =====================================================
# app.py — Smart Personal Expense Categorizer (Stable)
# =====================================================

import io, os, chardet, pdfplumber, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from categories import Classifier
from typing import Any


# ---------- 1. Setup ----------
load_dotenv("pass.env")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(page_title="Expense Categorizer", layout="wide")
st.markdown("<h1 style='text-align:center;'>💰 Smart Expense Categorizer</h1>", unsafe_allow_html=True)
st.caption("Upload your transaction file — we'll clean, categorize, and summarize it for you.")

# ---------- 2. Helper Functions ----------
def read_any_file(uploaded_file: Any) -> pd.DataFrame:
    """Reads CSV, Excel, PDF, or TXT into a cleaned DataFrame."""
    name = uploaded_file.name.lower()

    # --- CSV / Excel / PDF / TXT readers ---
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif name.endswith(".csv"):
        raw = uploaded_file.read()
        enc = chardet.detect(raw).get("encoding", "utf-8")
        uploaded_file.seek(0)
        df = pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
    elif name.endswith(".pdf"):
        rows = []
        with pdfplumber.open(uploaded_file) as pdf:
            for p in pdf.pages:
                table = p.extract_table()
                if table:
                    rows += table
        if not rows:
            raise ValueError("PDF has no readable text tables (maybe scanned).")
        df = pd.DataFrame(rows[1:], columns=rows[0])
    elif name.endswith(".txt"):
        raw = uploaded_file.read()
        enc = chardet.detect(raw).get("encoding", "utf-8")
        uploaded_file.seek(0)
        df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
    else:
        raise ValueError("Unsupported file type. Upload CSV, Excel, PDF, or TXT.")

    # --- Basic cleaning ---
    df.dropna(how="all", inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Save standardized version (optional)
    df.to_csv("standardized_transactions.csv", index=False)
    return df


def detect_column(df: pd.DataFrame, target: str):
    """Finds a likely match for Description or Amount."""
    if df is None or not isinstance(df,pd.DataFrame):
        return None
    target = target.lower()
    for col in df.columns:
        name = col.lower()
        if target == "description" and any(x in name for x in ["desc", "narrat", "detail", "remark", "particular"]):
            return col
        if target == "amount" and any(x in name for x in ["amount", "debit", "credit", "value", "amt"]):
            return col
    return None


# app.py

def prepare_for_summary(df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:

    """Cleans and merges categorized data for plotting (bullet-proof)."""

    # Input validation
    if df is None or result_df is None:
        raise ValueError("Input DataFrames cannot be None")
    
    if len(df) == 0 or len(result_df) == 0:
        raise ValueError("Input DataFrames are empty")

    # Check if result_df already has the required columns
    if "Category" in result_df.columns and "Amount" in result_df.columns:
        # Use result_df directly if it has both columns
        merged = result_df.copy()
        # Clean the data
        merged["Category"] = pd.Series(merged["Category"]).astype(str)
        merged = merged[merged["Category"].str.strip() != ""]
        merged.dropna(subset=["Category"], inplace=True)
        return merged[["Category", "Amount"]]

    # --- 1. Detect and clean amount column ---

    amt_col = next(

        (c for c in df.columns if any(w in c.lower() for w in ["amount","debit","credit","value","amt"])),

        None,

    )

    if amt_col is None:

        raise ValueError("Couldn't find amount column.")

    amt_series = df[amt_col]

    if isinstance(amt_series, pd.DataFrame):

        amt_series = amt_series.iloc[:, 0]

    df["Amount"] = pd.to_numeric(

        amt_series.astype(str).replace(r"[^\d\.\-]", "", regex=True),

        errors="coerce"

    )



    # --- 2. Flatten Category column from classifier ---

    if "Category" not in result_df.columns:

        raise ValueError("No 'Category' column found in result_df.")



    cat_col = result_df["Category"]



    # If it's a DataFrame, squeeze it down

    if isinstance(cat_col, pd.DataFrame):

        cat_col = cat_col.iloc[:, 0]

    # If it contains lists or tuples, take the first element
    if len(cat_col) > 0 and isinstance(cat_col.iloc[0], (list, tuple)):
        cat_col = cat_col.apply(lambda x: x[0] if x else None)

    cat_col = pd.Series(cat_col, name="Category").astype(str)

    # --- 3. Align and merge safely ---

    min_len = min(len(df), len(cat_col))

    df = df.iloc[:min_len].reset_index(drop=True)

    cat_col = cat_col.iloc[:min_len].reset_index(drop=True)

    merged = pd.concat([df, cat_col], axis=1)



    # --- 4. Final cleaning ---

    if isinstance(merged["Category"], pd.DataFrame):

        merged["Category"] = merged["Category"].iloc[:, 0]

    # Ensure Category is a Series before using .str
    merged["Category"] = pd.Series(merged["Category"]).astype(str)

    merged = merged[merged["Category"].str.strip() != ""]

    merged.dropna(subset=["Category"], inplace=True)



    return merged[["Category", "Amount"]]



# ---------- 3. Main Streamlit Flow ----------

st.divider()
st.subheader("Step 1 — Upload your file")

uploaded = st.file_uploader(
    "We accept CSV, Excel, PDF, or TXT files",
    type=["csv", "xlsx", "xls", "pdf", "txt"]
)

if not uploaded:
    st.info("⬆ Please upload a file to start.")
    st.stop()
df = None
# --- Read file ---
try:
    df = read_any_file(uploaded)
    st.session_state["df"]=df
    st.success("✅ File loaded successfully.")
    st.dataframe(df.head(10), use_container_width=True)
except Exception as e:
    st.error(f"⚠️ Couldn't read file: {e}")
    st.stop()

# --- Detect columns ---
desc_col = detect_column(df, "description")
amt_col = detect_column(df, "amount")

if desc_col:
    df.rename(columns={desc_col: "Description"}, inplace=True)
    st.success(f"🧠 Found Description column: {desc_col}")
else:
    st.warning("⚠ No clear description column detected.")

if amt_col:
    df["Amount"] = pd.to_numeric(df[amt_col], errors="coerce")
    st.info(f"💰 Found Amount column: {amt_col}")
else:
    st.warning("⚠ No clear amount column detected.")

# --- Ensure required columns exist for classification ---
if df is None or not isinstance(df, pd.DataFrame):
    st.error("Dataframe not loaded correctly. Please re-upload the file.")
    st.stop()

if "Description" not in df.columns:
    # Try to fall back to the first textual column
    object_cols = [c for c in df.columns if df[c].dtype == object]
    if object_cols:
        df["Description"] = df[object_cols[0]].astype(str)
        st.info(f"Using `{object_cols[0]}` as Description.")
    else:
        # Fallback to the first column, coerced to string
        first_col = df.columns[0]
        df["Description"] = df[first_col].astype(str)
        st.info(f"Using `{first_col}` as Description (fallback).")

# Normalize Description to string
df["Description"] = df["Description"].astype(str).fillna("").str.strip()

# --- Categorization ---
st.divider()
st.subheader("Step 2 — Categorize your transactions")

categories = [
    "RENT", "UTILITIES", "TRANSPORT", "FOOD & BEVERAGES", "GROCERIES",
    "BILLS & SUBSCRIPTIONS", "HEALTHCARE", "SHOPPING", "TRAVEL",
    "INCOME/REFUND", "FEES/CHARGES", "MISCELLANEOUS"
]

if "df" in st.session_state:
    df=st.session_state["df"]
else:
    st.error("No data loaded. Please upload a file first")
    st.stop()

# LLM Configuration
st.markdown("#### 🤖 AI Configuration")
col1, col2 = st.columns(2)
with col1:
    enable_llm = st.checkbox("Enable AI-powered categorization", value=True, help="Uses AI to understand transaction context")
with col2:
    llm_threshold = st.slider("AI Confidence Threshold", 0.0, 1.0, 0.8, 0.1, help="Lower = AI used more often, Higher = Rules used more often")

clf = Classifier(categories=categories, enable_llm=enable_llm, llm_threshold=llm_threshold)
with st.spinner("Analyzing transactions..."):
    result_df = clf.classify_dataframe(df)

st.success("✅ Categorization complete.")
st.dataframe(result_df, use_container_width=True)

# --- AI Analysis Section ---
if enable_llm:
    st.divider()
    st.subheader("🤖 AI Analysis Results")
    
    # Show AI vs Rule breakdown
    ai_transactions = result_df[result_df["Confidence"] < llm_threshold] if "Confidence" in result_df.columns else pd.DataFrame()
    rule_transactions = result_df[result_df["Confidence"] >= llm_threshold] if "Confidence" in result_df.columns else result_df
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(result_df))
    with col2:
        st.metric("AI Classified", len(ai_transactions), delta=f"{len(ai_transactions)/len(result_df)*100:.1f}%")
    with col3:
        st.metric("Rule Classified", len(rule_transactions), delta=f"{len(rule_transactions)/len(result_df)*100:.1f}%")
    
    if len(ai_transactions) > 0:
        st.info(f"🤖 AI analyzed {len(ai_transactions)} transactions that rules couldn't handle confidently")
        with st.expander("View AI-classified transactions"):
            st.dataframe(ai_transactions[["Description", "Category", "Confidence"]], use_container_width=True)
    
    # Show MISCELLANEOUS transactions
    misc_transactions = result_df[result_df["Category"] == "MISCELLANEOUS"] if "Category" in result_df.columns else pd.DataFrame()
    if len(misc_transactions) > 0:
        st.warning(f"⚠️ {len(misc_transactions)} transactions classified as MISCELLANEOUS")
        with st.expander("View MISCELLANEOUS transactions"):
            st.dataframe(misc_transactions[["Description", "Category", "Confidence"]], use_container_width=True)
            st.info("💡 These transactions didn't match any specific patterns. Consider adding new rules or using AI for better classification.")

# --- Visualization ---
st.divider()
st.subheader("Step 3 — Explore your spending")

try:
    merged = prepare_for_summary(df, result_df)
except Exception as e:
    st.error(f"⚠️ Couldn't prepare summary: {e}")
    st.write("Debug info:")
    st.write(f"df shape: {df.shape if df is not None else 'None'}")
    st.write(f"result_df shape: {result_df.shape if result_df is not None else 'None'}")
    st.write(f"df columns: {df.columns.tolist() if df is not None else 'None'}")
    st.write(f"result_df columns: {result_df.columns.tolist() if result_df is not None else 'None'}")
    st.stop()

spend = (
    merged.groupby("Category")["Amount"]
    .sum()
    .abs()
    .sort_values(ascending=False)
)

if spend.empty:
    st.info("No spend data found to display.")
    st.stop()

top = spend.idxmax()
total = spend.sum()

c1, c2 = st.columns(2)
c1.metric("Highest spend category", top)
c2.metric("Total spend", f"₹{total:,.0f}")

# Chart type selection
chart_type = st.radio("Choose classification method:", ["Rule-based (Current)", "AI-powered (Experimental)"], horizontal=True)

# Visualization options
viz_options = st.multiselect(
    "📊 Choose visualizations:", 
    ["Bar Chart", "Pie Chart", "Line Chart", "Area Chart", "Scatter Plot", "Heatmap", "Sunburst"],
    default=["Bar Chart", "Pie Chart"]
)

if chart_type == "Rule-based (Current)":
    st.markdown("#### 📊 Rule-based Category Analysis")
    st.info("💡 **Rule-based**: Uses pattern matching for fast, predictable categorization")
    
    # Create comprehensive visualizations
    if "Bar Chart" in viz_options:
        st.markdown("##### 📊 Spending by Category (Bar Chart)")
        st.bar_chart(spend)
    
    if "Pie Chart" in viz_options:
        st.markdown("##### 🥧 Category Distribution (Pie Chart)")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(range(len(spend)))
        wedges, texts, autotexts = ax.pie(spend.values, labels=spend.index, autopct="%1.1f%%", 
                                         startangle=90, colors=colors)
        ax.axis("equal")
        st.pyplot(fig)
    
    if "Line Chart" in viz_options:
        st.markdown("##### 📈 Spending Trends (Line Chart)")
        # Create monthly spending trends
        if "Date" in merged.columns:
            merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
            monthly_spend = merged.groupby([merged["Date"].dt.to_period("M"), "Category"])["Amount"].sum().abs()
            monthly_pivot = monthly_spend.unstack(fill_value=0)
            st.line_chart(monthly_pivot)
        else:
            st.info("Date column not found for trend analysis")
    
    if "Area Chart" in viz_options:
        st.markdown("##### 📊 Cumulative Spending (Area Chart)")
        cumulative_spend = spend.cumsum()
        st.area_chart(cumulative_spend)
    
    if "Scatter Plot" in viz_options:
        st.markdown("##### 🎯 Transaction Analysis (Scatter Plot)")
        if "Confidence" in merged.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(merged["Amount"].abs(), merged["Confidence"], 
                               c=merged["Category"].astype("category").cat.codes, 
                               alpha=0.6, s=50)
            ax.set_xlabel("Transaction Amount")
            ax.set_ylabel("Confidence Score")
            ax.set_title("Amount vs Confidence by Category")
            st.pyplot(fig)
        else:
            st.info("Confidence data not available for scatter plot")
    
    if "Heatmap" in viz_options:
        st.markdown("##### 🔥 Category Heatmap")
        # Create a correlation matrix if we have numeric data
        numeric_data = merged.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45)
            ax.set_yticklabels(corr_matrix.columns)
            plt.colorbar(im)
            st.pyplot(fig)
        else:
            st.info("Insufficient numeric data for heatmap")
    
    if "Sunburst" in viz_options:
        st.markdown("##### ☀️ Hierarchical Spending (Sunburst)")
        try:
            import plotly.express as px
            # Create a simple hierarchy: Category -> Subcategory
            sunburst_data = merged.copy()
            sunburst_data["Subcategory"] = sunburst_data["Category"]  # Simplified for demo
            fig = px.sunburst(sunburst_data, path=['Category', 'Subcategory'], values='Amount',
                            title="Spending Hierarchy")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install plotly for sunburst chart: pip install plotly")
    
    # Show rule-based confidence
    if "Confidence" in merged.columns:
        avg_confidence = merged["Confidence"].mean()
        st.metric("Average Rule Confidence", f"{avg_confidence:.2f}")

else:
    st.markdown("#### 🤖 AI-powered Category Analysis")
    st.info("💡 **AI-powered**: Uses LLM to understand context and meaning for smarter categorization")
    
    # Generate AI-powered categorization
    with st.spinner("🤖 AI is analyzing your transactions..."):
        try:
            # Create AI classifier with different model
            ai_clf = Classifier(categories=categories, enable_llm=True, model_name="openrouter/auto")
            ai_result_df = ai_clf.classify_dataframe(df)
            
            # Prepare AI results
            ai_merged = prepare_for_summary(df, ai_result_df)
            ai_spend = (
                ai_merged.groupby("Category")["Amount"]
                .sum()
                .abs()
                .sort_values(ascending=False)
            )
            
            # Display AI results with same visualization options
            if "Bar Chart" in viz_options:
                st.markdown("##### 📊 AI Spending by Category (Bar Chart)")
                st.bar_chart(ai_spend)
            
            if "Pie Chart" in viz_options:
                st.markdown("##### 🥧 AI Category Distribution (Pie Chart)")
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.Set3(range(len(ai_spend)))
                wedges, texts, autotexts = ax.pie(ai_spend.values, labels=ai_spend.index, 
                                                autopct="%1.1f%%", startangle=90, colors=colors)
                ax.axis("equal")
                st.pyplot(fig)
            
            # Show AI confidence
            if "Confidence" in ai_merged.columns:
                avg_ai_confidence = ai_merged["Confidence"].mean()
                st.metric("Average AI Confidence", f"{avg_ai_confidence:.2f}")
                
            # Compare with rule-based
            st.markdown("#### 🔄 Comparison: Rule-based vs AI")
            comparison_data = pd.DataFrame({
                "Rule-based": spend,
                "AI-powered": ai_spend
            }).fillna(0)
            
            st.bar_chart(comparison_data)
            
        except Exception as e:
            st.error(f"AI analysis failed: {e}")
            st.info("Falling back to rule-based analysis...")
            st.bar_chart(spend)

# --- Optional AI insight ---
st.divider()
st.subheader("Step 4 — Optional: Get a short AI summary")

if st.checkbox("Generate a short summary using AI"):
    if not OPENROUTER_KEY:
        st.warning("Please add your OpenRouter key in the .env file.")
    else:
        summary_text = (
            spend.head(5)
            .to_frame()
            .reset_index()
            .rename(columns={"index": "Category", "Amount": "Spend"})
            .to_string(index=False)
        )
        system_prompt = (
            "You're a friendly financial analyst. Write a 2–3 sentence note on this spending summary."
        )
        payload = {
            "model": "openrouter/auto",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summary_text},
            ],
            "temperature": 0.4,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            # Recommended by OpenRouter for browser or web-app usage
            "HTTP-Referer": "http://localhost",
            "X-Title": "Expense Categorizer",
        }
        with st.spinner("Writing summary..."):
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=25,
                )
                # Handle non-200 responses first
                if not r.ok:
                    st.error(f"OpenRouter error {r.status_code}: {r.text}")
                else:
                    data = r.json()
                    # Standard OpenAI-compatible shape
                    if isinstance(data, dict) and "choices" in data and data["choices"]:
                        insight = data["choices"][0]["message"]["content"].strip()
                        st.info(insight)
                    # Error payload shape
                    elif isinstance(data, dict) and "error" in data:
                        err = data.get("error")
                        st.error(f"OpenRouter returned an error: {err}")
                    else:
                        st.error(f"Unexpected response from OpenRouter: {data}")
            except Exception as e:
                st.error(f"Couldn't generate insight: {e}")

# --- Export ---
st.divider()
csv_buf = io.StringIO()
result_df.to_csv(csv_buf, index=False)
st.download_button(
    "📥 Download categorized CSV",
    csv_buf.getvalue(),
    file_name="categorized_transactions.csv",
    mime="text/csv",
)