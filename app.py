# app_patched.py
"""
Patched Streamlit Dashboard (full file)

Implements:
 - Contacts / Modules page with module fields m1v1..m6i1 extracted
 - Leaderboard (Top 100) computed from contacts aggregated data
 - list_users_page returns leaderboard users
 - get_user_messages returns only timestamps
 - Single-User View removed
 - CSV export reflects current sorted view captured from st.data_editor
 - Robust cache clearing + rerun
 - Login modal using st.secrets["auth"]
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from google.cloud import bigquery
from google.oauth2 import service_account

# ---------------------------
# CONFIG
# ---------------------------
PROJECT_ID = "pilot-run-turn-bq-integration"
DATASET = "923141851055"
MESSAGES_TABLE = "messages"
CONTACTS_TABLE = "contacts"  # new table for contacts page

# Leaderboard cache TTL (seconds)
LEADERBOARD_TTL = 3600  # 1 hour

# Page size for user listing in sidebar (if needed)
USERS_PAGE_SIZE = 100

# Top N users to fetch for leaderboard
LEADERBOARD_LIMIT = 100

# Number of module groups: 6
NUM_MODULES = 6

# ---------------------------
# UTIL: BigQuery client with proper auth
# ---------------------------
@st.cache_resource
def get_bq_client() -> Optional[bigquery.Client]:
    """
    Get BigQuery client with proper authentication.
    Tries multiple auth methods:
    1. Streamlit secrets (gcp_service_account)
    2. GOOGLE_APPLICATION_CREDENTIALS env var
    3. Default credentials
    """
    try:
        # Method 1: Try to use Streamlit secrets
        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
            try:
                creds_dict = dict(st.secrets["gcp_service_account"])
                creds = service_account.Credentials.from_service_account_info(creds_dict)
                client = bigquery.Client(credentials=creds, project=creds.project_id)
                return client
            except Exception as e:
                st.warning(f"Failed to authenticate using Streamlit secrets: {e}")

        # Method 2: Check for GOOGLE_APPLICATION_CREDENTIALS env var
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if os.path.exists(creds_path):
                try:
                    creds = service_account.Credentials.from_service_account_file(creds_path)
                    client = bigquery.Client(credentials=creds, project=creds.project_id)
                    return client
                except Exception as e:
                    st.warning(f"Failed to authenticate using env var: {e}")

        # Method 3: Try default application credentials
        try:
            client = bigquery.Client(project=PROJECT_ID)
            return client
        except Exception as e:
            raise e

    except Exception as e:
        st.error(f"❌ BigQuery Authentication Failed: {str(e)}")
        st.info(
            "Please set up Google Cloud authentication: Streamlit secrets -> GOOGLE_APPLICATION_CREDENTIALS -> gcloud auth application-default"
        )
        return None


def table_ref(table_name: str = MESSAGES_TABLE) -> str:
    return f"`{PROJECT_ID}.{DATASET}.{table_name}`"

# ---------------------------
# SQL / Data functions
# ---------------------------

@st.cache_data(ttl=300)
def fetch_contacts_progress() -> pd.DataFrame:
    """
    Fetch contacts table and extract per-whatsapp_id aggregated fields.
    Extract module fields as requested: m1v1,m1v2,m1i1,...,m6v1,m6v2,m6i1 using SAFE_CAST(...) AS INT64.
    Also extract timestamps from details JSON where available (first_message_received_at, last_message_received_at, last_seen_at).
    Aggregation semantics:
      - MAX(...) for flags / module fields (safe representative)
      - SUM computed in SQL for m-posts if needed
    """
    client = get_bq_client()
    if client is None:
        return pd.DataFrame()

    # Build the SELECT list for module fields m1v1..m6i1
    module_selects = []
    for m in range(1, NUM_MODULES + 1):
        # v1, v2, i1 pattern
        for suffix in ("v1", "v2", "i1"):
            field = f"m{m}{suffix}"
            module_selects.append(
                f"SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.{field}') AS INT64) AS {field}"
            )

    # Also extract m#post fields and indrive_module_X_complete
    mpost_selects = []
    indrive_selects = []
    for m in range(1, NUM_MODULES + 1):
        mpost_selects.append(f"SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.m{m}post') AS INT64) AS m{m}post")
        indrive_selects.append(f"SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.indrive_module_{m}_complete') AS INT64) AS indrive_module_{m}_complete")

    # Attempt to extract timestamp fields from details JSON
    # We'll SAFE_CAST them to TIMESTAMP (if stored as RFC3339), else fallback to NULL
    ts_selects = [
        "JSON_EXTRACT_SCALAR(details, '$.first_message_received_at') AS first_message_received_at_str",
        "JSON_EXTRACT_SCALAR(details, '$.last_message_received_at') AS last_message_received_at_str",
        "JSON_EXTRACT_SCALAR(details, '$.last_seen_at') AS last_seen_at_str",
        "JSON_EXTRACT_SCALAR(details, '$.last_message_sent_at') AS last_message_sent_at_str"
    ]

    # Build full query
    q = f"""
    WITH raw AS (
      SELECT
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.whatsapp_id') AS STRING) AS whatsapp_id,
        JSON_EXTRACT_SCALAR(details, '$.cohort_no') AS cohort_no,
        {', '.join(module_selects)},
        {', '.join(mpost_selects)},
        {', '.join(indrive_selects)},
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.total_indrive_modules_completed') AS INT64) AS total_indrive_modules_completed,
        {', '.join(ts_selects)}
      FROM {table_ref(CONTACTS_TABLE)}
      WHERE details IS NOT NULL
    )

    -- Aggregate per whatsapp_id (MAX for module fields to represent "has value")
    SELECT
      whatsapp_id,
      ANY_VALUE(cohort_no) AS cohort_no,
      {', '.join([f"MAX(COALESCE({c.split()[-1].strip()},{0})) AS {c.split()[-1].strip()}" for c in module_selects])},
      {', '.join([f"MAX(COALESCE({c.split()[-1].strip()},{0})) AS {c.split()[-1].strip()}" for c in mpost_selects])},
      {', '.join([f"MAX(COALESCE({c.split()[-1].strip()},{0})) AS {c.split()[-1].strip()}" for c in indrive_selects])},
      MAX(COALESCE(total_indrive_modules_completed, 0)) AS total_indrive_modules_completed,
      -- timestamps as strings aggregated (take any non-null)
      ANY_VALUE(first_message_received_at_str) AS first_message_received_at_str,
      ANY_VALUE(last_message_received_at_str) AS last_message_received_at_str,
      ANY_VALUE(last_seen_at_str) AS last_seen_at_str,
      ANY_VALUE(last_message_sent_at_str) AS last_message_sent_at_str
    FROM raw
    WHERE whatsapp_id IS NOT NULL AND TRIM(whatsapp_id) != ''
    GROUP BY whatsapp_id
    """

    try:
        job = client.query(q)
        df = job.result().to_dataframe(create_bqstorage_client=False)
    except Exception as e:
        st.error(f"Error fetching contacts data from BigQuery: {e}")
        return pd.DataFrame()

    # Normalize and coerce types in pandas
    if df.empty:
        return df

    # Ensure whatsapp_id and cohort_no types
    df["whatsapp_id"] = df["whatsapp_id"].astype(str).str.strip()
    if "cohort_no" not in df.columns:
        df["cohort_no"] = None
    else:
        df["cohort_no"] = df["cohort_no"].astype(str).fillna("").replace("None","").replace("null","").str.strip()
        df.loc[df["cohort_no"] == "", "cohort_no"] = "Unknown"

    # Module fields list
    module_fields = []
    for m in range(1, NUM_MODULES + 1):
        for suffix in ("v1", "v2", "i1"):
            module_fields.append(f"m{m}{suffix}")

    # mpost fields
    mpost_fields = [f"m{m}post" for m in range(1, NUM_MODULES + 1)]
    # indrive fields
    indrive_fields = [f"indrive_module_{m}_complete" for m in range(1, NUM_MODULES + 1)]

    # Coerce module fields to int (fill NaN with 0)
    for c in module_fields + mpost_fields + indrive_fields + ["total_indrive_modules_completed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0

    # Parse timestamps (strings from details). We'll coerce to pandas datetime (UTC-aware if possible)
    def _parse_ts_col(s):
        try:
            if pd.isna(s):
                return pd.NaT
            # bigquery JSON strings might sometimes be like "2025-12-09T02:47:07Z" or with microseconds
            return pd.to_datetime(s, utc=True, errors="coerce")
        except Exception:
            return pd.NaT

    df["first_message_received_at"] = df.get("first_message_received_at_str", None).apply(_parse_ts_col) if "first_message_received_at_str" in df.columns else pd.NaT
    df["last_message_received_at"] = df.get("last_message_received_at_str", None).apply(_parse_ts_col) if "last_message_received_at_str" in df.columns else pd.NaT
    df["last_seen_at"] = df.get("last_seen_at_str", None).apply(_parse_ts_col) if "last_seen_at_str" in df.columns else pd.NaT
    df["last_message_sent_at"] = df.get("last_message_sent_at_str", None).apply(_parse_ts_col) if "last_message_sent_at_str" in df.columns else pd.NaT

    # Compute derived columns server-side (in pandas)
    # sum_posts_values = sum of m#post numeric values
    df["sum_posts_values"] = df[mpost_fields].sum(axis=1).astype(int)

    # count_posts_completed = count of non-zero m#post fields
    df["count_posts_completed"] = df[mpost_fields].apply(lambda row: int((row != 0).sum()), axis=1)

    # sum_indrive_completes = sum of indrive_module_X_complete (cap at NUM_MODULES)
    df["sum_indrive_completes"] = df[indrive_fields].sum(axis=1).astype(int)
    df["sum_indrive_completes"] = df["sum_indrive_completes"].clip(upper=NUM_MODULES)

    # Ensure total_indrive_modules_completed is consistent: if missing or 0, use sum_indrive_completes
    df["total_indrive_modules_completed"] = df.get("total_indrive_modules_completed", df["sum_indrive_completes"])
    df["total_indrive_modules_completed"] = pd.to_numeric(df["total_indrive_modules_completed"], errors="coerce").fillna(0).astype(int)
    df["total_indrive_modules_completed"] = df[["total_indrive_modules_completed", "sum_indrive_completes"]].max(axis=1).astype(int)
    df["total_indrive_modules_completed"] = df["total_indrive_modules_completed"].clip(upper=NUM_MODULES)

    # Compute last_interaction_timestamp for each user row:
    # Preferred precedence (per your confirmation):
    # 1) details JSON last_message_received_at (we parsed to last_message_received_at)
    # 2) details JSON last_message_sent_at
    # 3) details JSON last_seen_at
    # If none available, we leave NaT (messages table fallback could be implemented if required)
    df["last_interaction_timestamp"] = df[["last_message_received_at", "last_message_sent_at", "last_seen_at"]].bfill(axis=1).iloc[:, 0]
    # If still NaT, leave as NaT

    # Keep only desired columns and order for convenience
    # core columns
    core_cols = [
        "whatsapp_id", "cohort_no",
        # timestamps
        "first_message_received_at", "last_message_received_at", "last_seen_at", "last_interaction_timestamp",
        # mpost & indrive & totals
        *mpost_fields,
        *indrive_fields,
        "total_indrive_modules_completed",
        "sum_posts_values", "count_posts_completed", "sum_indrive_completes"
    ]
    # include module fields
    core_cols = core_cols + module_fields

    # Some columns might be missing in df; keep those present
    present_cols = [c for c in core_cols if c in df.columns]
    df = df[present_cols].copy()

    # Reset index and return
    df = df.reset_index(drop=True)
    return df

# ---------------------------
# Leaderboard & helper functions
# ---------------------------

@st.cache_data(ttl=LEADERBOARD_TTL)
def precompute_leaderboard(cohort_filter: Optional[List[str]] = None, limit: int = LEADERBOARD_LIMIT) -> pd.DataFrame:
    """
    Compute Top leaderboard (Top 'limit') from contacts aggregated data.
    Filter-first (apply cohort_filter), then compute top N by:
      - sum_indrive_completes DESC
      - sum_posts_values DESC
    Returns a DataFrame with requested columns including module fields.
    """
    # We'll reuse fetch_contacts_progress() to get contacts aggregated data, then filter / sort in pandas
    contacts = fetch_contacts_progress()
    if contacts is None or contacts.empty:
        return pd.DataFrame()

    df = contacts.copy()

    # Apply cohort filter first (per your confirmation)
    if cohort_filter and len(cohort_filter) > 0 and "All" not in cohort_filter:
        df = df[df["cohort_no"].isin(cohort_filter)]

    # Ensure columns present
    if "sum_indrive_completes" not in df.columns:
        df["sum_indrive_completes"] = df[[c for c in df.columns if c.startswith("indrive_module_")]].sum(axis=1).fillna(0).astype(int)

    if "sum_posts_values" not in df.columns:
        # sum m1post..mNpost if present
        mpost_cols = [c for c in df.columns if c.endswith("post") and c.startswith("m")]
        if mpost_cols:
            df["sum_posts_values"] = df[mpost_cols].sum(axis=1).fillna(0).astype(int)
        else:
            df["sum_posts_values"] = 0

    # last_interaction_timestamp ensure datetime dtype
    if "last_interaction_timestamp" in df.columns:
        df["last_interaction_timestamp"] = pd.to_datetime(df["last_interaction_timestamp"], utc=True, errors="coerce")
    else:
        df["last_interaction_timestamp"] = pd.NaT

    # Sort by sum_indrive_completes desc, then sum_posts_values desc
    df = df.sort_values(by=["sum_indrive_completes", "sum_posts_values"], ascending=[False, False])

    # Limit to top N
    df_top = df.head(limit).reset_index(drop=True)

    # Add rank column
    df_top.insert(0, "rank", range(1, len(df_top) + 1))

    return df_top

@st.cache_data(ttl=300)
def list_users_page_from_leaderboard(cohort_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Return a small DataFrame listing leaderboard users (whatsapp_id) — used for any user-selection UI.
    """
    leaderboard = precompute_leaderboard(cohort_filter=cohort_filter, limit=LEADERBOARD_LIMIT)
    if leaderboard is None or leaderboard.empty:
        return pd.DataFrame(columns=["whatsapp_id"])
    return leaderboard[["whatsapp_id", "cohort_no"]].copy()

@st.cache_data(ttl=300)
def get_user_messages_timestamps(whatsapp_id: str) -> dict:
    """
    Return only timestamps for a user: first_message_received_at, last_message_received_at, last_seen_at
    Data is returned from contacts aggregation (fast). No chat logs returned.
    """
    contacts = fetch_contacts_progress()
    if contacts is None or contacts.empty:
        return {}
    row = contacts[contacts["whatsapp_id"] == str(whatsapp_id)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "first_message_received_at": r.get("first_message_received_at", pd.NaT),
        "last_message_received_at": r.get("last_message_received_at", pd.NaT),
        "last_seen_at": r.get("last_seen_at", pd.NaT),
        "last_interaction_timestamp": r.get("last_interaction_timestamp", pd.NaT),
    }

# ---------------------------
# UI helpers
# ---------------------------

def _minutes_to_hm(minutes: int) -> str:
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def show_login_modal():
    """Display login form (modal-like center column). Returns (user, pwd, submit_pressed)"""
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("### 🔐 Login to Dashboard")
        st.markdown("---")
        user_in = st.text_input("Username", value="", key="login_user")
        pass_in = st.text_input("Password", type="password", key="login_pass")
        col1, col2 = st.columns(2)
        with col1:
            submit = st.button("Login", use_container_width=True)
        with col2:
            st.button("Cancel", use_container_width=True, disabled=True)
        return user_in, pass_in, submit

# ---------------------------
# Contacts / Modules Page Renderer
# ---------------------------

def render_contacts_modules_page():
    st.header("Contacts — Module & Post-Quiz Progress")
    st.markdown("Aggregated per **whatsapp_id**. Extracted from `details` JSON in the `contacts` table.")

    with st.spinner("Fetching contacts data..."):
        contacts_df = fetch_contacts_progress()

    if contacts_df is None or contacts_df.empty:
        st.info("No contacts data available or BigQuery auth missing.")
        return

    # compute derived columns already done in fetch_contacts_progress
    df = contacts_df.copy()

    # Prepare module and indrive column lists for use
    module_fields = []
    for m in range(1, NUM_MODULES + 1):
        for suffix in ("v1", "v2", "i1"):
            module_fields.append(f"m{m}{suffix}")
    mpost_fields = [f"m{m}post" for m in range(1, NUM_MODULES + 1)]
    indrive_fields = [f"indrive_module_{m}_complete" for m in range(1, NUM_MODULES + 1)]

    # Sidebar filters specific to this page (must keep unchanged)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Contacts / Modules Filters")
    cohorts = sorted(df["cohort_no"].dropna().unique().tolist())
    cohort_sel = st.sidebar.multiselect("Cohort (multi)", options=["All"] + cohorts, default=["All"])
    # Module completed filter
    module_filter_mode = st.sidebar.radio("Filter modules completed by", options=["All", "Exact", "At least"], index=0)
    module_filter_n = st.sidebar.slider("Modules completed (N)", min_value=0, max_value=NUM_MODULES, value=0)
    # Post-quiz filter
    post_filter_mode = st.sidebar.radio("Filter post-quiz completed by", options=["All", "Exact", "At least"], index=0, key="post_mode")
    post_filter_n = st.sidebar.slider("Post quizzes completed (N)", min_value=0, max_value=NUM_MODULES, value=0, key="post_n")

    # NEW: Indrive module completion filter (AND semantics: require all selected modules to be completed)
    # This is the filter you requested: let user pick which indrive modules to require completion for.
    indrive_options = indrive_fields  # names like 'indrive_module_1_complete'
    selected_indrive_modules = st.sidebar.multiselect(
        "Require completion of these indrive modules (AND semantics):",
        options=indrive_options,
        default=[]
    )

    # Apply filters (cohort, modules, post-quiz) — keep logic unchanged
    df_filtered = df.copy()
    if cohort_sel and "All" not in cohort_sel:
        df_filtered = df_filtered[df_filtered["cohort_no"].isin(cohort_sel)]

    if module_filter_mode == "Exact":
        df_filtered = df_filtered[df_filtered["sum_indrive_completes"] == int(module_filter_n)]
    elif module_filter_mode == "At least":
        df_filtered = df_filtered[df_filtered["sum_indrive_completes"] >= int(module_filter_n)]

    if post_filter_mode == "Exact":
        df_filtered = df_filtered[df_filtered["count_posts_completed"] == int(post_filter_n)]
    elif post_filter_mode == "At least":
        df_filtered = df_filtered[df_filtered["count_posts_completed"] >= int(post_filter_n)]

    # APPLY NEW Indrive module filter (AND): require selected indrive modules to equal 1
    if selected_indrive_modules:
        # ensure numeric coercion already done; treat missing as 0
        for col in selected_indrive_modules:
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == 1]
            else:
                # if column missing, treat as none completed (filter will yield empty)
                df_filtered = df_filtered[[]]

    st.markdown(f"**Showing {len(df_filtered):,} unique whatsapp ids**")

    # Bar chart: users who completed exactly 1..6 modules (counts)
    col1, col2 = st.columns(2)
    with col1:
        distr = df_filtered["sum_indrive_completes"].value_counts().sort_index()
        distr_df = distr.reset_index()
        distr_df.columns = ["modules_completed", "count"]
        # Ensure rows for 0..6 present
        all_idx = pd.Series(range(0, NUM_MODULES + 1), name="modules_completed")
        distr_full = pd.merge(all_idx.to_frame(), distr_df, on="modules_completed", how="left").fillna(0)
        distr_full["count"] = distr_full["count"].astype(int)
        if distr_full["count"].sum() == 0:
            st.info("No data for modules-completed distribution.")
        else:
            fig = px.bar(distr_full[distr_full["modules_completed"] > 0], x="modules_completed", y="count",
                         title="Users by Modules Completed (exactly N modules) — exclude 0")
            fig.update_layout(xaxis_title="Modules Completed (N)", yaxis_title="User Count")
            st.plotly_chart(fig, use_container_width=True)

    # Bar chart: post-quizzes completed distribution
    with col2:
        distr2 = df_filtered["count_posts_completed"].value_counts().sort_index()
        distr2_df = distr2.reset_index()
        distr2_df.columns = ["posts_completed", "count"]
        all_idx2 = pd.Series(range(0, NUM_MODULES + 1), name="posts_completed")
        distr2_full = pd.merge(all_idx2.to_frame(), distr2_df, on="posts_completed", how="left").fillna(0)
        distr2_full["count"] = distr2_full["count"].astype(int)
        if distr2_full["count"].sum() == 0:
            st.info("No data for post-quiz distribution.")
        else:
            fig2 = px.bar(distr2_full, x="posts_completed", y="count", title="Users by Post-Quizzes Completed (N)")
            fig2.update_layout(xaxis_title="Post-Quizzes Completed (N)", yaxis_title="User Count")
            st.plotly_chart(fig2, use_container_width=True)

    # Prepare display DataFrame and ordering
    display_cols = [
        "whatsapp_id", "cohort_no",
        "first_message_received_at", "last_message_received_at", "last_seen_at", "last_interaction_timestamp",
        *mpost_fields,
        *indrive_fields,
        "total_indrive_modules_completed",
        "sum_posts_values", "count_posts_completed", "sum_indrive_completes",
        *module_fields
    ]
    display_cols = [c for c in display_cols if c in df_filtered.columns]
    display_df = df_filtered[display_cols].copy().reset_index(drop=True)

    st.markdown("**Contacts Table (sortable)**")
    st.markdown("Tip: Click column headers to sort. Use the export button to download the currently displayed (sorted) CSV.")

    # Use st.data_editor and capture returned DataFrame (this preserves sort order)
    # disabled=True prevents edits but still allows sorting in current Streamlit versions
    edited = st.data_editor(display_df, disabled=True, use_container_width=True)

    # Export CSV reflects the edited DataFrame (which maintains sort order)
    csv_bytes = edited.to_csv(index=False).encode("utf-8")
    st.download_button("Download contacts CSV (sorted view)", data=csv_bytes, file_name="contacts_modules_progress.csv", mime="text/csv")

    # Cohort summary (optional)
    if len(df_filtered) > 0:
        cohort_summary = df_filtered.groupby("cohort_no").agg(
            unique_whatsapp_ids=("whatsapp_id", "nunique"),
            avg_modules_completed=("sum_indrive_completes", "mean"),
            avg_posts_completed=("count_posts_completed", "mean"),
        ).reset_index()
        cohort_summary["avg_modules_completed"] = cohort_summary["avg_modules_completed"].round(2)
        cohort_summary["avg_posts_completed"] = cohort_summary["avg_posts_completed"].round(2)
        st.markdown("**Cohort Summary**")
        st.dataframe(cohort_summary, use_container_width=True)


# ---------------------------
# Leaderboard Page Renderer
# ---------------------------

def render_leaderboard_page():
    st.header("Leaderboard — Top Users by Modules & Posts")
    st.markdown("Top users are computed from aggregated contacts data. Filter by cohort on the sidebar to compute Top N within selected cohort(s).")

    # Sidebar cohort filter for leaderboard (kept on sidebar)
    contacts_df = fetch_contacts_progress()
    if contacts_df is None or contacts_df.empty:
        st.info("No contacts data available to compute leaderboard.")
        return

    cohorts = sorted(contacts_df["cohort_no"].dropna().unique().tolist())
    cohort_sel = st.sidebar.multiselect("Leaderboard cohort filter (All default)", options=["All"] + cohorts, default=["All"])

    # Compute leaderboard using cohort filter (apply filter first then Top N)
    cohort_filter = None
    if cohort_sel and "All" not in cohort_sel:
        cohort_filter = cohort_sel

    leaderboard = precompute_leaderboard(cohort_filter=cohort_filter, limit=LEADERBOARD_LIMIT)
    if leaderboard is None or leaderboard.empty:
        st.info("Leaderboard is empty for the selected cohort(s).")
        return

    # Optionally allow user to change Top N shown (limit to 100)
    top_n = st.slider("Top N users to show", min_value=5, max_value=LEADERBOARD_LIMIT, value=min(25, LEADERBOARD_LIMIT))
    lb_display = leaderboard.head(top_n).copy()

    # Format timestamps for display
    if "last_interaction_timestamp" in lb_display.columns:
        lb_display["last_interaction_timestamp"] = pd.to_datetime(lb_display["last_interaction_timestamp"], utc=True, errors="coerce")
        lb_display["last_interaction_timestamp"] = lb_display["last_interaction_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    # Select columns to display: whatsapp_id, cohort_no, sum_indrive_completes, sum_posts_values, last_interaction_timestamp, module fields
    module_fields = [f"m{m}{s}" for m in range(1, NUM_MODULES + 1) for s in ("v1", "v2", "i1")]
    display_cols = ["rank", "whatsapp_id", "cohort_no", "sum_indrive_completes", "sum_posts_values", "last_interaction_timestamp"] + [c for c in module_fields if c in lb_display.columns]
    display_cols = [c for c in display_cols if c in lb_display.columns]

    st.markdown(f"**Showing top {len(lb_display):,} users (after cohort filter)**")
    # Use st.dataframe (sorting available). The exported CSV button will use the dataframe created below.
    st.dataframe(lb_display[display_cols], use_container_width=True)

    # Provide CSV export of leaderboard currently shown
    csv_bytes = lb_display[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download leaderboard CSV (current view)", data=csv_bytes, file_name="leaderboard_top_users.csv", mime="text/csv")


# ---------------------------
# Streamlit Main
# ---------------------------

def main():
    st.set_page_config(page_title="Messages Analytics Dashboard", layout="wide")
    st.title("Messages Analytics Dashboard")
    st.markdown("Analyze user engagement patterns from BigQuery tables.")

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.auth_method = None

    # --- Authentication (secrets approach)
    auth_secrets = st.secrets.get("auth", {}) if hasattr(st, "secrets") else {}
    AUTH_USERNAME = auth_secrets.get("username", "admin")
    AUTH_PASSWORD = auth_secrets.get("password", "admin123@#")

    # Show login if not authenticated
    if not st.session_state.authenticated:
        user_in, pass_in, submit = show_login_modal()
        if submit:
            if str(user_in) == str(AUTH_USERNAME) and str(pass_in) == str(AUTH_PASSWORD):
                st.session_state.authenticated = True
                st.session_state.username = user_in
                # robust rerun
                try:
                    st.experimental_rerun()
                except Exception:
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_set_query_params(_login=int(time.time()))
                        st.stop()
            else:
                st.error("❌ Invalid username or password")
        return

    # Sidebar: account and actions
    st.sidebar.header("Account")
    st.sidebar.write(f"Logged in as **{st.session_state.username}**")
    client = get_bq_client()
    if client is None:
        st.sidebar.error("BigQuery: Not Authenticated")
    else:
        st.sidebar.success("BigQuery: Authenticated")

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        try:
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                st.experimental_set_query_params(_logout=int(time.time()))
                st.stop()

    # Sidebar: Refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh data (clear cache)"):
        # clear both cache_data and cache_resource where supported
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        # robust rerun
        try:
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                st.experimental_set_query_params(_refresh=int(time.time()))
                st.stop()

    # Page navigation - single-user view removed
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox("Page", ["Leaderboard View", "Contacts / Modules"])

    if page == "Contacts / Modules":
        render_contacts_modules_page()
        return
    elif page == "Leaderboard View":
        render_leaderboard_page()
        return

    # Shouldn't reach here
    st.info("Select a page from the sidebar.")

if __name__ == '__main__':
    main()
