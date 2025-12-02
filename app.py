# streamlit_m_posts_dashboard.py
import os
#import db_dtypes
from typing import Tuple, Dict, List
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
from datetime import datetime
from google.oauth2 import service_account

st.set_page_config(page_title="M-posts Single-Table Dashboard", layout="wide")

# --- CONFIG ---
PROJECT_TABLE = "pilot-run-turn-bq-integration.923141851055.contacts"
CACHE_TTL_SECONDS = 300
NUM_M_POSTS = 6

# --- SQL: extract one representative row per whatsapp_id with raw values ---
QUERY = f"""
WITH raw AS (
  SELECT
    JSON_EXTRACT_SCALAR(details, '$.whatsapp_id') AS whatsapp_id,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.m1post'), '') AS m1_raw,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.m2post'), '') AS m2_raw,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.m3post'), '') AS m3_raw,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.m4post'), '') AS m4_raw,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.m5post'), '') AS m5_raw,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.m6post'), '') AS m6_raw,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.cohort_no'), '') AS cohort_no,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.indrive_module_1_complete'), '') AS indrive_module_1_complete,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.indrive_module_2_complete'), '') AS indrive_module_2_complete,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.indrive_module_3_complete'), '') AS indrive_module_3_complete,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.indrive_module_4_complete'), '') AS indrive_module_4_complete,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.indrive_module_5_complete'), '') AS indrive_module_5_complete,
    COALESCE(JSON_EXTRACT_SCALAR(details, '$.indrive_module_6_complete'), '') AS indrive_module_6_complete
  FROM `{PROJECT_TABLE}`
  WHERE JSON_EXTRACT_SCALAR(details, '$.whatsapp_id') IS NOT NULL
    AND JSON_EXTRACT_SCALAR(details, '$.whatsapp_id') != ''
),
agg AS (
  SELECT
    whatsapp_id,
    MAX(m1_raw) AS m1_value,
    MAX(m2_raw) AS m2_value,
    MAX(m3_raw) AS m3_value,
    MAX(m4_raw) AS m4_value,
    MAX(m5_raw) AS m5_value,
    MAX(m6_raw) AS m6_value,
    MAX(cohort_no) AS cohort_no,
    MAX(indrive_module_1_complete) AS indrive_module_1_value,
    MAX(indrive_module_2_complete) AS indrive_module_2_value,
    MAX(indrive_module_3_complete) AS indrive_module_3_value,
    MAX(indrive_module_4_complete) AS indrive_module_4_value,
    MAX(indrive_module_5_complete) AS indrive_module_5_value,
    MAX(indrive_module_6_complete) AS indrive_module_6_value
  FROM raw
  GROUP BY whatsapp_id
)
SELECT * FROM agg
ORDER BY whatsapp_id;
"""

# --- BigQuery loader (cached) ---
from google.oauth2 import service_account

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_data_from_bq(query: str) -> Tuple[pd.DataFrame, Dict[str, any]]:
    # Try Streamlit secrets first (for Streamlit Cloud)
    if "gcp_service_account" in st.secrets:
        creds_info = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(creds_info)
        client = bigquery.Client(
            credentials=credentials,
            project=creds_info.get("project_id"),
        )
    else:
        # Local dev fallback (uses GOOGLE_APPLICATION_CREDENTIALS)
        cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not cred:
            st.error(
                "Configure 'gcp_service_account' in Streamlit secrets "
                "or set GOOGLE_APPLICATION_CREDENTIALS locally."
            )
            st.stop()
        client = bigquery.Client()

    job = client.query(query)
    metadata = {
        "job_id": None,
        "total_bytes_processed": None,
        "location": None,
        "queried_at": None,
    }

    try:
        df = job.result().to_dataframe(create_bqstorage_client=True)
    except GoogleAPIError as e:
        st.warning(f"BigQuery Storage API unavailable, falling back: {e}")
        df = job.result().to_dataframe(create_bqstorage_client=False)
    except Exception as e:
        st.error(f"Failed to fetch results from BigQuery: {e}")
        st.stop()

    try:
        metadata["job_id"] = getattr(job, "job_id", None) or job.job_id
        total_bytes = getattr(job, "total_bytes_processed", None)
        if total_bytes is None:
            total_bytes = job._properties.get("statistics", {}).get("totalBytesProcessed")
        metadata["total_bytes_processed"] = total_bytes
        metadata["location"] = getattr(job, "location", None) or job._properties.get("jobReference", {}).get("location")
    except Exception:
        pass

    metadata["queried_at"] = datetime.utcnow().isoformat() + "Z"
    return df, metadata

# --- Helper: coerce strings to numeric for mpost values (for sum) ---
def parse_numeric_value(s: str) -> float:
    """Try to convert string to float. Treat '', 'null', 'None' as 0. Non-numeric -> 0."""
    if s is None:
        return 0.0
    s2 = str(s).strip()
    if s2.lower() in ("", "null", "none"):
        return 0.0
    try:
        return float(s2)
    except Exception:
        # try common boolean -> numeric mapping
        if s2.lower() in ("true", "t", "yes", "y"):
            return 1.0
        if s2.lower() in ("false", "f", "no", "n"):
            return 0.0
        return 0.0

# --- Business rule B: completed if mX_value IN {'0','1','2'} (strings) ---
COMPLETED_VALUES = {"0", "1", "2"}

# --- Prepare dataframe from raw BQ output ---
def prepare_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(
            columns=["whatsapp_id", "cohort_no"] +
                    [f"m{i}_value" for i in range(1, NUM_M_POSTS + 1)] +
                    ["completed_count", "sum_mposts_value"] +
                    [f"indrive_module_{i}_value" for i in range(1, NUM_M_POSTS + 1)] +
                    ["sum_indrive"]
        )

    df = df_raw.copy()
    # normalize columns existence
    for i in range(1, NUM_M_POSTS + 1):
        col = f"m{i}_value"
        if col not in df.columns:
            df[col] = ""
        ind_col = f"indrive_module_{i}_value"
        if ind_col not in df.columns:
            df[ind_col] = ""

    df["whatsapp_id"] = df["whatsapp_id"].astype(str).str.strip()
    df["cohort_no"] = df.get("cohort_no", "").astype(str).fillna("").str.strip()

    # Completed flag per module according to rule B
    for i in range(1, NUM_M_POSTS + 1):
        col = f"m{i}_value"
        df[f"m{i}_completed_flag"] = df[col].astype(str).str.strip().apply(lambda x: 1 if x in COMPLETED_VALUES else 0)

    # completed_count = count of modules with completed_flag == 1
    completed_cols = [f"m{i}_completed_flag" for i in range(1, NUM_M_POSTS + 1)]
    df["completed_count"] = df[completed_cols].sum(axis=1).astype(int)

    # sum_mposts_value = numeric sum of m1..m6 values (coerced)
    numeric_cols_values = []
    for i in range(1, NUM_M_POSTS + 1):
        col = f"m{i}_value"
        num_col = f"m{i}_num_value"
        df[num_col] = df[col].apply(parse_numeric_value).astype(float)
        numeric_cols_values.append(num_col)
    df["sum_mposts_value"] = df[numeric_cols_values].sum(axis=1).astype(float)

    # Parse indrive columns to numeric (handle '1','0','true','false')
    indrive_num_cols = []
    for i in range(1, NUM_M_POSTS + 1):
        ind_col = f"indrive_module_{i}_value"
        ind_num_col = f"indrive_module_{i}_num"
        # normalize null-ish
        df[ind_col] = df[ind_col].fillna("").astype(str).str.strip()
        def parse_indrive(x: str) -> int:
            if x.lower() in ("", "null", "none"):
                return 0
            if x.lower() in ("1", "true", "t", "yes", "y"):
                return 1
            if x.lower() in ("0", "false", "f", "no", "n"):
                return 0
            # try numeric
            try:
                return int(float(x))
            except Exception:
                return 0
        df[ind_num_col] = df[ind_col].apply(parse_indrive).astype(int)
        indrive_num_cols.append(ind_num_col)

    df["sum_indrive"] = df[indrive_num_cols].sum(axis=1).astype(int)

    # Final tidy columns
    keep_cols = ["whatsapp_id", "cohort_no"] \
                + [f"m{i}_value" for i in range(1, NUM_M_POSTS + 1)] \
                + ["completed_count", "sum_mposts_value"] \
                + [f"indrive_module_{i}_value" for i in range(1, NUM_M_POSTS + 1)] \
                + ["sum_indrive"]
    # Ensure all exist
    for c in keep_cols:
        if c not in df.columns:
            df[c] = "" if c.endswith("_value") or c == "cohort_no" else 0
    df_final = df[keep_cols].drop_duplicates(subset=["whatsapp_id"]).reset_index(drop=True)
    return df_final

# --- UI: Sidebar (all filters) ---
st.sidebar.title("Filters & Controls")
st.sidebar.markdown("**Filter Out Those Who Have Completed Modules Post Quizez**")

# exact completed count filter
exact_choices = [str(i) for i in range(1, NUM_M_POSTS + 1)]
selected_exact = st.sidebar.multiselect("Show users who completed exactly:", options=exact_choices, default=[])

# toggle show only completed all modules
only_all_completed = st.sidebar.checkbox("Show only users who completed all 6 modules", value=False)

# cohort filter placeholder (populated after data load)
cohort_select = st.sidebar.multiselect("Filter by cohort_no (leave empty for all)", options=[], default=[])

st.sidebar.markdown("---")
st.sidebar.markdown("**Top Leader Board Ranking Based On Modules**")
st.sidebar.write("You can use sum_mposts_value (numeric sum of m1..m6) and sum_indrive to rank/sort in the table.")

# numeric filters for ranking ranges
min_sum_mposts_val = st.sidebar.number_input("Min sum of m-post values", value=0.0, format="%.2f")
max_sum_mposts_val = st.sidebar.number_input("Max sum of m-post values", value=1_000_000.0, format="%.2f")
min_sum_indrive = st.sidebar.number_input("Min sum_indrive", value=0, step=1)
max_sum_indrive = st.sidebar.number_input("Max sum_indrive", value=NUM_M_POSTS, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filter by indrive module status (require selected modules = chosen value)**")
indrive_options = [f"indrive_module_{i}_value" for i in range(1, NUM_M_POSTS + 1)]
selected_indrive_modules = st.sidebar.multiselect("Select indrive modules (AND semantics):", options=indrive_options, default=[])
indrive_value_choice = st.sidebar.selectbox("Require selected indrive modules to be:", options=["1", "0"], index=0)

st.sidebar.markdown("---")
if st.sidebar.button("Refresh data (clear cache)"):
    st.cache_data.clear()
    st.experimental_rerun()

# --- Load and prepare data ---
with st.spinner("Loading data from BigQuery..."):
    raw_df, meta = load_data_from_bq(QUERY)

df = prepare_dataframe(raw_df)

# populate cohort select options now that df is ready
unique_cohorts = sorted([c for c in df["cohort_no"].unique() if str(c).strip() not in ("", "None", "null")])
if unique_cohorts:
    # if user hasn't set cohort_select yet, keep it empty by default
    cohort_select = st.sidebar.multiselect("Filter by cohort_no (leave empty for all)", options=unique_cohorts, default=cohort_select)

# --- Apply filters (in this order) ---
mask = pd.Series(True, index=df.index)

# cohort filter
if cohort_select:
    mask &= df["cohort_no"].isin(cohort_select)

# exact completed count filter
if selected_exact:
    desired = set(int(x) for x in selected_exact)
    mask &= df["completed_count"].isin(desired)

# only all completed toggle (overrides/combines with above, keep AND semantics)
if only_all_completed:
    mask &= df["completed_count"] == NUM_M_POSTS

# sum_mposts_value numeric range
mask &= df["sum_mposts_value"].between(float(min_sum_mposts_val), float(max_sum_mposts_val))

# sum_indrive numeric range
mask &= df["sum_indrive"].between(int(min_sum_indrive), int(max_sum_indrive))

# indrive module filters (AND): require selected indrive modules to equal chosen value
if selected_indrive_modules:
    # map indrive_value_choice to int (0 or 1)
    try:
        required_val = int(indrive_value_choice)
    except Exception:
        required_val = 1 if str(indrive_value_choice).lower() in ("1", "true", "yes") else 0
    for col in selected_indrive_modules:
        # col is like indrive_module_1_value -> we compare to parsed numeric column created earlier?
        num_col = col.replace("_value", "_num")  # indrive_module_1_num
        if num_col in df.columns:
            mask &= df[num_col] == required_val
        else:
            # For safety, try to compute on-the-fly:
            def parse_indrive_val(x):
                if str(x).strip().lower() in ("", "null", "none"):
                    return 0
                if str(x).strip().lower() in ("1", "true", "t", "yes", "y"):
                    return 1
                if str(x).strip().lower() in ("0", "false", "f", "no", "n"):
                    return 0
                try:
                    return int(float(x))
                except Exception:
                    return 0
            mask &= df[col].apply(parse_indrive_val) == required_val

filtered = df.loc[mask].copy().reset_index(drop=True)

# --- Add indrive_num columns into filtered for display sorting if not present ---
for i in range(1, NUM_M_POSTS + 1):
    num_col = f"indrive_module_{i}_num"
    val_col = f"indrive_module_{i}_value"
    if num_col not in filtered.columns and val_col in filtered.columns:
        # create numeric view column
        filtered[num_col] = filtered[val_col].apply(lambda x: 1 if str(x).strip().lower() in ("1", "true", "yes", "y") else 0)

# --- Build display columns and show one table only ---
display_cols = [
    "whatsapp_id",
    "cohort_no",
] + [f"m{i}_value" for i in range(1, NUM_M_POSTS + 1)] + [
    "completed_count",
    "sum_mposts_value",
] + [f"indrive_module_{i}_value" for i in range(1, NUM_M_POSTS + 1)] + [
    "sum_indrive"
]

display_cols = [c for c in display_cols if c in filtered.columns]

st.title("M-posts — Single Table Dashboard")
st.markdown("**Table shows unique WhatsApp IDs and supports all requested filters.**")
st.write(f"Query run at: {meta.get('queried_at','n/a')} — rows after filter: {len(filtered):,}")

# Show table (single view). Users can sort columns via the Streamlit dataframe UI.
st.dataframe(filtered[display_cols], use_container_width=True, height=700)

# CSV download
@st.cache_data
def to_csv_bytes(df_: pd.DataFrame) -> bytes:
    return df_.to_csv(index=False).encode("utf-8")

csv_bytes = to_csv_bytes(filtered[display_cols])
st.download_button("Download filtered table as CSV", data=csv_bytes, file_name="m_posts_single_table_filtered.csv", mime="text/csv")
