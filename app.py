# app.py
"""
Patched Streamlit Dashboard with an added "Contacts / Modules" page.

Keep your existing secrets and BigQuery config as before.
"""

import os
import json
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

# Page size for user listing in sidebar
USERS_PAGE_SIZE = 100

# Top N users to fetch for leaderboard
LEADERBOARD_LIMIT = 1000

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
                # do not mutate session_state here (avoid side-effects in cached function)
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
        st.info("""
        Please set up Google Cloud authentication:
        
        **Option 1: Using Streamlit Secrets** (Recommended for Streamlit Cloud)
        Add to `.streamlit/secrets.toml` or Streamlit Cloud Settings:
        ```
        [gcp_service_account]
        type = "service_account"
        project_id = "your-project-id"
        private_key_id = "your-key-id"
        private_key = "your-private-key"
        client_email = "your-service-account@your-project.iam.gserviceaccount.com"
        client_id = "your-client-id"
        auth_uri = "https://accounts.google.com/o/oauth2/auth"
        token_uri = "https://oauth2.googleapis.com/token"
        auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
        client_x509_cert_url = "your-cert-url"
        ```
        
        **Option 2: Using Environment Variable** (Local development)
        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
        streamlit run app.py
        ```
        
        **Option 3: Using Default Credentials** (Local development)
        ```bash
        gcloud auth application-default login
        streamlit run app.py
        ```
        """)
        return None


def table_ref(table_name: str = MESSAGES_TABLE) -> str:
    return f"`{PROJECT_ID}.{DATASET}.{table_name}`"

# ---------------------------
# Data access functions
# ---------------------------

@st.cache_data(ttl=300)
def fetch_contacts_progress() -> pd.DataFrame:
    """
    Fetch contacts table, extract fields from `details` JSON and aggregate per whatsapp_id.
    Returns a dataframe with one row per whatsapp_id and extracted fields.
    """
    client = get_bq_client()
    if client is None:
        return pd.DataFrame()

    # Build SQL to extract JSON keys and aggregate per whatsapp_id
    # Fields extracted: m1post..m6post, indrive_module_1_complete..6, total_indrive_modules_completed, whatsapp_id, cohort_no
    q = f"""
    WITH raw AS (
      SELECT
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.whatsapp_id') AS STRING) AS whatsapp_id,
        JSON_EXTRACT_SCALAR(details, '$.cohort_no') AS cohort_no,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.m1post') AS INT64) AS m1post,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.m2post') AS INT64) AS m2post,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.m3post') AS INT64) AS m3post,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.m4post') AS INT64) AS m4post,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.m5post') AS INT64) AS m5post,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.m6post') AS INT64) AS m6post,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.indrive_module_1_complete') AS INT64) AS indrive_module_1_complete,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.indrive_module_2_complete') AS INT64) AS indrive_module_2_complete,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.indrive_module_3_complete') AS INT64) AS indrive_module_3_complete,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.indrive_module_4_complete') AS INT64) AS indrive_module_4_complete,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.indrive_module_5_complete') AS INT64) AS indrive_module_5_complete,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.indrive_module_6_complete') AS INT64) AS indrive_module_6_complete,
        SAFE_CAST(JSON_EXTRACT_SCALAR(details, '$.total_indrive_modules_completed') AS INT64) AS total_indrive_modules_completed
      FROM {table_ref(CONTACTS_TABLE)}
      WHERE details IS NOT NULL
    )

    -- Aggregate per whatsapp_id (take MAX for each field in case of multiple records)
    SELECT
      whatsapp_id,
      ANY_VALUE(cohort_no) AS cohort_no,
      MAX(COALESCE(m1post,0)) AS m1post,
      MAX(COALESCE(m2post,0)) AS m2post,
      MAX(COALESCE(m3post,0)) AS m3post,
      MAX(COALESCE(m4post,0)) AS m4post,
      MAX(COALESCE(m5post,0)) AS m5post,
      MAX(COALESCE(m6post,0)) AS m6post,
      MAX(COALESCE(indrive_module_1_complete,0)) AS indrive_module_1_complete,
      MAX(COALESCE(indrive_module_2_complete,0)) AS indrive_module_2_complete,
      MAX(COALESCE(indrive_module_3_complete,0)) AS indrive_module_3_complete,
      MAX(COALESCE(indrive_module_4_complete,0)) AS indrive_module_4_complete,
      MAX(COALESCE(indrive_module_5_complete,0)) AS indrive_module_5_complete,
      MAX(COALESCE(indrive_module_6_complete,0)) AS indrive_module_6_complete,
      MAX(COALESCE(total_indrive_modules_completed,0)) AS total_indrive_modules_completed
    FROM raw
    WHERE whatsapp_id IS NOT NULL AND TRIM(whatsapp_id) != ''
    GROUP BY whatsapp_id
    """

    try:
        job = client.query(q)
        df = job.result().to_dataframe(create_bqstorage_client=False)
        # Ensure numeric dtypes and handle nulls
        int_cols = [
            "m1post","m2post","m3post","m4post","m5post","m6post",
            "indrive_module_1_complete","indrive_module_2_complete","indrive_module_3_complete",
            "indrive_module_4_complete","indrive_module_5_complete","indrive_module_6_complete",
            "total_indrive_modules_completed"
        ]
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
            else:
                df[c] = 0

        # keep cohort_no as string
        if "cohort_no" not in df.columns:
            df["cohort_no"] = None

        # Cast whatsapp_id to string consistently
        df["whatsapp_id"] = df["whatsapp_id"].astype(str)

        return df
    except Exception as e:
        st.error(f"Error fetching contacts progress: {e}")
        return pd.DataFrame()


# --- existing functions from your original file ---
# I reuse the functions you had: get_unique_user_count, list_users_page, get_user_messages, precompute_leaderboard, compute_user_metrics, plots...
# For brevity, I will assume the rest of your original functions (get_unique_user_count, list_users_page, get_user_messages, precompute_leaderboard, compute_user_metrics, plot_hourly_activity, plot_heatmap) are present unchanged below.
# If you replaced the whole app, paste your prior functions here; the file must contain them.
# -------------------------------------------------------------------------
# For this response I'll re-include the main UI flow and the new Contacts / Modules page integration.
# -------------------------------------------------------------------------

# Minimal stubs for previously defined functions to avoid NameError if you run this file standalone.
# If your original file already defines these, these stubs will be replaced by original definitions.
try:
    get_unique_user_count  # type: ignore
except NameError:
    @st.cache_data(ttl=300)
    def get_unique_user_count() -> int:
        client = get_bq_client()
        if client is None:
            return 0
        try:
            q = f"SELECT COUNT(DISTINCT JSON_EXTRACT_SCALAR(addressees, '$[0]')) AS cnt FROM {table_ref()}"
            row = list(client.query(q).result())[0]
            return int(row.cnt) if row and getattr(row, "cnt", None) is not None else 0
        except Exception:
            return 0

try:
    list_users_page  # type: ignore
except NameError:
    @st.cache_data(ttl=300)
    def list_users_page(page: int = 0, page_size: int = USERS_PAGE_SIZE, order_by_engagement: bool = False) -> pd.DataFrame:
        return pd.DataFrame(columns=["user_identifier"])

try:
    get_user_messages  # type: ignore
except NameError:
    @st.cache_data(ttl=300)
    def get_user_messages(user_identifier: str, start: datetime = None, end: datetime = None, limit: int = 100000) -> pd.DataFrame:
        return pd.DataFrame()

try:
    precompute_leaderboard  # type: ignore
except NameError:
    @st.cache_data(ttl=LEADERBOARD_TTL)
    def precompute_leaderboard(limit: int = LEADERBOARD_LIMIT) -> pd.DataFrame:
        return pd.DataFrame()

def _minutes_to_hm(minutes: int) -> str:
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def compute_user_metrics(df: pd.DataFrame, analysis_end: datetime = None) -> dict:
    if df.empty:
        return {}
    df = df.copy().sort_values("timestamp")
    first_ts = df.iloc[0]["timestamp"]
    last_ts = df.iloc[-1]["timestamp"]
    if analysis_end is None:
        last_msg = df["timestamp"].max()
        analysis_end = (last_msg.floor('D') + pd.Timedelta(days=1))
    analysis_start = analysis_end - pd.Timedelta(days=1)
    window_df = df[(df["timestamp"] >= analysis_start) & (df["timestamp"] < analysis_end)]
    def compute_sessions_minutes(series_ts: pd.Series) -> float:
        if series_ts.empty:
            return 0.0
        times = series_ts.sort_values()
        diffs = times.diff().dt.total_seconds().fillna(0)
        session_breaks = diffs > 1800
        session_ids = session_breaks.cumsum()
        minutes_total = 0.0
        for _, group in times.groupby(session_ids):
            minutes_total += (group.max() - group.min()).total_seconds() / 60.0
        return minutes_total
    daily_minutes = compute_sessions_minutes(window_df["timestamp"]) if not window_df.empty else 0.0
    hist_source = window_df if not window_df.empty else df
    if hist_source.empty:
        peak_hour = None
        peak_count = 0
    else:
        hist = hist_source["timestamp"].dt.hour.value_counts().reindex(range(24), fill_value=0)
        peak_hour = int(hist.idxmax())
        peak_count = int(hist.max())
    timestamps = df["timestamp"].sort_values().reset_index(drop=True)
    max_minutes = 0.0
    max_start = None
    if not timestamps.empty:
        j = 0
        for i in range(len(timestamps)):
            start_ts = timestamps[i]
            window_end_ts = start_ts + pd.Timedelta(days=1)
            while j < len(timestamps) and timestamps[j] <= window_end_ts:
                j += 1
            window_slice = timestamps[i:j]
            minutes = compute_sessions_minutes(window_slice)
            if minutes > max_minutes:
                max_minutes = minutes
                max_start = start_ts
    highest_window = {
        "total_minutes": int(round(max_minutes)),
        "start": pd.to_datetime(max_start) if max_start is not None else None,
        "end": (pd.to_datetime(max_start) + pd.Timedelta(days=1)) if max_start is not None else None,
    }
    return {
        "first_interaction": pd.to_datetime(first_ts),
        "last_interaction": pd.to_datetime(last_ts),
        "daily_minutes": int(round(daily_minutes)),
        "daily_minutes_hm": _minutes_to_hm(int(round(daily_minutes))),
        "peak_hour": peak_hour,
        "peak_count": peak_count,
        "highest_window": highest_window,
    }

def plot_hourly_activity(df: pd.DataFrame, title: str = "Hourly Activity"):
    if df.empty:
        st.info("No messages for the selected period/user.")
        return
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
    hourly.columns = ["hour", "count"]
    fig = px.line(hourly, x="hour", y="count", markers=True, title=title)
    fig.update_xaxes(tickmode="linear")
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap(df: pd.DataFrame, title: str = "Activity Heatmap"):
    if df.empty:
        return
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    pivot = df.groupby(["date", "hour"]).size().unstack(fill_value=0)
    pivot = pivot.sort_index()
    fig = px.imshow(pivot.T, aspect="auto", labels=dict(x="Date", y="Hour", color="Messages"), x=pivot.index.astype(str), y=pivot.columns)
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Streamlit UI - Login Modal reused from your original file
# ---------------------------

def show_login_modal():
    """Display login form"""
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

    if contacts_df.empty:
        st.info("No contacts data available or BigQuery auth missing.")
        return

    # Compute derived columns
    m_cols = ["m1post","m2post","m3post","m4post","m5post","m6post"]
    indrive_cols = [
        "indrive_module_1_complete","indrive_module_2_complete","indrive_module_3_complete",
        "indrive_module_4_complete","indrive_module_5_complete","indrive_module_6_complete"
    ]

    # Sum of m1..m6 values (numeric sum) and count of completed posts (non-zero)
    contacts_df["sum_posts_values"] = contacts_df[m_cols].sum(axis=1)
    contacts_df["count_posts_completed"] = contacts_df[m_cols].apply(lambda r: int((r != 0).sum()), axis=1)

    # Sum of indrive completes
    contacts_df["sum_indrive_completes"] = contacts_df[indrive_cols].sum(axis=1)
    # Ensure total_indrive_modules_completed is present and consistent
    contacts_df["total_indrive_modules_completed"] = contacts_df.get("total_indrive_modules_completed", contacts_df["sum_indrive_completes"])
    # Fill cohort_no NAs with 'Unknown'
    contacts_df["cohort_no"] = contacts_df["cohort_no"].fillna("Unknown").astype(str)

    # Sidebar filters specific to this page
    st.sidebar.markdown("---")
    st.sidebar.subheader("Contacts / Modules Filters")
    cohorts = sorted(contacts_df["cohort_no"].dropna().unique().tolist())
    cohort_sel = st.sidebar.multiselect("Cohort (multi)", options=["All"] + cohorts, default=["All"])
    # Module completed filter
    module_filter_mode = st.sidebar.radio("Filter modules completed by", options=["All", "Exact", "At least"], index=0)
    module_filter_n = st.sidebar.slider("Modules completed (N)", min_value=0, max_value=6, value=0)
    # Post-quiz filter
    post_filter_mode = st.sidebar.radio("Filter post-quiz completed by", options=["All", "Exact", "At least"], index=0, key="post_mode")
    post_filter_n = st.sidebar.slider("Post quizzes completed (N)", min_value=0, max_value=6, value=0, key="post_n")

    # Apply filters
    df_filtered = contacts_df.copy()
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

    st.markdown(f"**Showing {len(df_filtered):,} unique whatsapp ids**")

    # Pie charts: modules completed distribution and post-quiz distribution
    col1, col2 = st.columns(2)
    with col1:
        distr = df_filtered["sum_indrive_completes"].value_counts().sort_index()
        distr_df = distr.reset_index()
        distr_df.columns = ["modules_completed", "count"]
        if distr_df["count"].sum() == 0:
            st.info("No data for modules-completed distribution.")
        else:
            fig = px.pie(distr_df, names="modules_completed", values="count", title="Distribution — Modules Completed (per user)")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        distr2 = df_filtered["count_posts_completed"].value_counts().sort_index()
        distr2_df = distr2.reset_index()
        distr2_df.columns = ["posts_completed", "count"]
        if distr2_df["count"].sum() == 0:
            st.info("No data for post-quiz distribution.")
        else:
            fig2 = px.pie(distr2_df, names="posts_completed", values="count", title="Distribution — Post-Quizzes Completed (per user)")
            st.plotly_chart(fig2, use_container_width=True)

    # Prepare table for display
    display_cols = [
        "whatsapp_id", "cohort_no",
        *m_cols,
        *indrive_cols,
        "total_indrive_modules_completed",
        "sum_posts_values", "count_posts_completed", "sum_indrive_completes"
    ]
    display_df = df_filtered[display_cols].copy()
    # Ensure column order and types
    display_df = display_df.reset_index(drop=True)

    st.markdown("**Contacts Table (sortable)**")
    st.markdown("Tip: Click column headers to sort. Use the export button to download the currently displayed (sorted) CSV.")

    # Use st.data_editor and capture the returned DataFrame — this reflects the user's sort and edits
    edited = st.data_editor(display_df, disabled=True, use_container_width=True)

    # Export CSV reflects the edited DataFrame (which maintains sort order)
    csv_bytes = edited.to_csv(index=False).encode("utf-8")
    st.download_button("Download contacts CSV (sorted view)", data=csv_bytes, file_name="contacts_modules_progress.csv", mime="text/csv")

    # Optionally show a small aggregated summary table per cohort
    if len(df_filtered) > 0:
        cohort_summary = df_filtered.groupby("cohort_no").agg(
            unique_whatsapp_ids = ("whatsapp_id", "nunique"),
            avg_modules_completed = ("sum_indrive_completes", "mean"),
            avg_posts_completed = ("count_posts_completed", "mean"),
        ).reset_index()
        cohort_summary["avg_modules_completed"] = cohort_summary["avg_modules_completed"].round(2)
        cohort_summary["avg_posts_completed"] = cohort_summary["avg_posts_completed"].round(2)
        st.markdown("**Cohort Summary**")
        st.dataframe(cohort_summary, use_container_width=True)


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

    # --- Authentication (re-using your secrets approach)
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
                st.rerun()
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
        st.rerun()

    # Sidebar: Refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh data (clear cache)"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
        st.rerun()

    # Page navigation
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox("Page", ["Single User View", "Leaderboard View", "Contacts / Modules"])

    # Render pages
    if page == "Contacts / Modules":
        render_contacts_modules_page()
        return

    # For other pages, keep previous behaviour (single user, leaderboard)
    # Header summary (kept minimal here)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for large datasets. Uses BigQuery sessionization and Streamlit caching for performance.")

    # If you want to keep your original Single User and Leaderboard pages,
    # paste their original rendering code here. For brevity, show a placeholder:
    if page == "Single User View":
        st.header("Single User View (original page)")
        st.info("Original Single User View content continues to live here. Replace this placeholder with your original page code.")
    elif page == "Leaderboard View":
        st.header("Leaderboard View (original page)")
        st.info("Original Leaderboard View content continues to live here. Replace this placeholder with your original page code.")


if __name__ == '__main__':
    main()
