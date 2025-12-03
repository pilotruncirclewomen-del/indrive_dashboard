"""
Corrected Streamlit dashboard for BigQuery messages table.
- Moved auth UI to main()
- Fixed get_unique_user_count() (no UI, no syntax errors)
- Added time import and robust rerun fallback
- Replaced deprecated use_container_width with width="stretch"
- Detects BigQuery Storage availability safely
- Avoids caching functions that contain UI
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPIError

# ---------------------------
# CONFIG
# ---------------------------
PROJECT_ID = "pilot-run-turn-bq-integration"
DATASET = "923141851055"
MESSAGES_TABLE = "messages"

# Path to JSON credential file provided by user (use env or st.secrets in production)
CREDENTIALS_PATH = "/Users/maysam/Desktop/AUSTIN-BIKEHSHARE-main/pilot-run-turn-bq-integration-6aef543e26a5.json"

# Leaderboard cache TTL (seconds)
LEADERBOARD_TTL = 3600  # 1 hour
USERS_PAGE_SIZE = 100
LEADERBOARD_LIMIT = 1000

# ---------------------------
# UTIL: BigQuery client
# ---------------------------
@st.cache_resource
def get_bq_client() -> bigquery.Client:
    """
    Create and cache a BigQuery client.

    Priority:
    1) Use st.secrets['gcp_service_account'] (Streamlit Cloud)
    2) Fallback to Application Default Credentials (ADC)
    """
    # 1) Streamlit Cloud secrets (primary)
    try:
        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(creds_info)
            project = creds_info.get("project_id") or creds.project_id or PROJECT_ID
            return bigquery.Client(credentials=creds, project=project)
    except Exception as e:
        st.warning(f"Could not initialize BigQuery with st.secrets credentials: {e}")

    # 2) Fallback to ADC (local environment or system credentials)
    try:
        env_project = (
            os.environ.get("GOOGLE_CLOUD_PROJECT") or
            os.environ.get("GCLOUD_PROJECT") or
            PROJECT_ID
        )
        return bigquery.Client(project=env_project)
    except Exception as e:
        raise RuntimeError(
            "Unable to create a BigQuery client.\n"
            "Please ensure st.secrets['gcp_service_account'] exists OR "
            "set GOOGLE_CLOUD_PROJECT and ADC credentials locally.\n"
            f"Underlying error: {e}"
        )

# ---------------------------
# Data access functions (no UI inside cached functions)
# ---------------------------
@st.cache_data(ttl=300)
def get_unique_user_count_bq() -> int:
    """Query distinct users using addressees array (fallback to raw_body if needed)."""
    client = get_bq_client()
    # Primary: use addressees array
    try:
        q = f"""
        SELECT COUNT(DISTINCT JSON_EXTRACT_SCALAR(addressees, '$[0]')) AS cnt
        FROM {table_ref()}
        WHERE addressees IS NOT NULL
        """
        row = list(client.query(q).result())[0]
        return int(row.cnt) if row and row.cnt is not None else 0
    except Exception:
        # Fallback: try JSON in raw_body
        try:
            q2 = f"SELECT COUNT(DISTINCT JSON_EXTRACT_SCALAR(raw_body, '$.wa_id')) AS cnt FROM {table_ref()}"
            row2 = list(client.query(q2).result())[0]
            return int(row2.cnt) if row2 and row2.cnt is not None else 0
        except Exception:
            return 0


@st.cache_data(ttl=300)
def list_users_page(page: int = 0, page_size: int = USERS_PAGE_SIZE, order_by_engagement: bool = False) -> pd.DataFrame:
    client = get_bq_client()
    offset = page * page_size
    if order_by_engagement:
        q = f"""
        SELECT user_phone, message_count FROM (
          SELECT
            JSON_EXTRACT_SCALAR(addressees, '$[0]') AS user_phone,
            COUNT(1) AS message_count
          FROM {table_ref()}
          WHERE JSON_EXTRACT_SCALAR(addressees, '$[0]') IS NOT NULL
          GROUP BY user_phone
        )
        ORDER BY message_count DESC
        LIMIT {page_size} OFFSET {offset}
        """
    else:
        q = f"""
        SELECT user_phone FROM (
          SELECT DISTINCT JSON_EXTRACT_SCALAR(addressees, '$[0]') AS user_phone
          FROM {table_ref()}
        )
        WHERE user_phone IS NOT NULL
        ORDER BY user_phone
        LIMIT {page_size} OFFSET {offset}
        """
    df = client.query(q).result().to_dataframe(create_bqstorage_client=_bq_storage_available())
    if 'user_phone' in df.columns:
        df = df.rename(columns={'user_phone': 'user_identifier'})
    return df


@st.cache_data(ttl=300)
def get_user_messages(user_identifier: str, start: datetime = None, end: datetime = None, limit: int = 100000) -> pd.DataFrame:
    client = get_bq_client()
    time_filter = ""
    if start and end:
        time_filter = (
            f"AND COALESCE(SAFE_CAST(inserted_at AS TIMESTAMP), SAFE_CAST(JSON_EXTRACT_SCALAR(raw_body, '$.timestamp') AS TIMESTAMP)) "
            f"BETWEEN TIMESTAMP('{start.isoformat()}') AND TIMESTAMP('{end.isoformat()}')"
        )
    q = f"""
    SELECT
      JSON_EXTRACT_SCALAR(addressees, '$[0]') AS user_identifier,
      uuid AS message_uuid,
      message_type,
      COALESCE(rendered_content, JSON_EXTRACT_SCALAR(raw_body, '$.text.body'), raw_body) AS message_text,
      COALESCE(SAFE_CAST(inserted_at AS TIMESTAMP), SAFE_CAST(JSON_EXTRACT_SCALAR(raw_body, '$.timestamp') AS TIMESTAMP)) AS timestamp,
      direction
    FROM {table_ref()}
    WHERE JSON_EXTRACT_SCALAR(addressees, '$[0]') = @user_identifier
      AND COALESCE(SAFE_CAST(inserted_at AS TIMESTAMP), SAFE_CAST(JSON_EXTRACT_SCALAR(raw_body, '$.timestamp') AS TIMESTAMP)) IS NOT NULL
      {time_filter}
    ORDER BY timestamp ASC
    LIMIT {limit}
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("user_identifier", "STRING", user_identifier)]
    )
    job = client.query(q, job_config=job_config)
    df = job.result().to_dataframe(create_bqstorage_client=_bq_storage_available())
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=LEADERBOARD_TTL)
def precompute_leaderboard(limit: int = LEADERBOARD_LIMIT) -> pd.DataFrame:
    client = get_bq_client()
    q = f"""
    WITH messages_parsed AS (
      SELECT
        JSON_EXTRACT_SCALAR(addressees, '$[0]') AS user_identifier,
        COALESCE(SAFE_CAST(inserted_at AS TIMESTAMP), SAFE_CAST(JSON_EXTRACT_SCALAR(raw_body, '$.timestamp') AS TIMESTAMP)) AS ts
      FROM {table_ref()}
      WHERE JSON_EXTRACT_SCALAR(addressees, '$[0]') IS NOT NULL
        AND COALESCE(SAFE_CAST(inserted_at AS TIMESTAMP), SAFE_CAST(JSON_EXTRACT_SCALAR(raw_body, '$.timestamp') AS TIMESTAMP)) IS NOT NULL
    ), numbered AS (
      SELECT user_identifier, ts, LAG(ts) OVER (PARTITION BY user_identifier ORDER BY ts) AS prev_ts
      FROM messages_parsed
    ), sessions AS (
      SELECT user_identifier, ts, prev_ts,
             IF(prev_ts IS NULL OR TIMESTAMP_DIFF(ts, prev_ts, MINUTE) > 30, 1, 0) AS is_new_session
      FROM numbered
    ), grouped AS (
      SELECT user_identifier, ts,
             SUM(is_new_session) OVER (PARTITION BY user_identifier ORDER BY ts) AS session_id
      FROM sessions
    ), bounds AS (
      SELECT user_identifier, session_id, MIN(ts) AS session_start, MAX(ts) AS session_end, COUNT(1) AS messages_in_session
      FROM grouped
      GROUP BY user_identifier, session_id
    )
    SELECT
      user_identifier,
      SUM(TIMESTAMP_DIFF(session_end, session_start, SECOND) / 60.0) AS total_minutes,
      SUM(messages_in_session) AS message_count,
      MIN(session_start) AS first_message,
      MAX(session_end) AS last_message,
      AVG(TIMESTAMP_DIFF(session_end, session_start, SECOND) / 60.0) AS avg_session_minutes
    FROM bounds
    WHERE user_identifier IS NOT NULL
    GROUP BY user_identifier
    ORDER BY total_minutes DESC
    LIMIT {limit}
    """
    df = client.query(q).result().to_dataframe(create_bqstorage_client=_bq_storage_available())
    if not df.empty:
        df["percentage_of_total"] = df["total_minutes"] / df["total_minutes"].sum() * 100
        df["first_message"] = pd.to_datetime(df["first_message"])
        df["last_message"] = pd.to_datetime(df["last_message"])
    return df

# ---------------------------
# Metrics calculations (pandas)
# ---------------------------

def _minutes_to_hm(minutes: int) -> str:
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def compute_user_metrics(df: pd.DataFrame, analysis_end: datetime = None) -> dict:
    if df is None or df.empty:
        return {}
    df = df.copy().sort_values("timestamp")

    first_ts = df.iloc[0]["timestamp"]
    last_ts = df.iloc[-1]["timestamp"]

    # Default analysis window: most recent full day (24 hours before last message)
    if analysis_end is None:
        last_msg = pd.to_datetime(df["timestamp"].max())
        analysis_end = (last_msg.normalize() + pd.Timedelta(days=1))
    analysis_start = analysis_end - pd.Timedelta(days=1)

    window_df = df[(df["timestamp"] >= analysis_start) & (df["timestamp"] < analysis_end)]

    def compute_sessions_minutes(series_ts: pd.Series) -> float:
        if series_ts.empty:
            return 0.0
        times = pd.to_datetime(series_ts).sort_values()
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

    # Highest individual engagement sliding 24-hour window
    timestamps = pd.to_datetime(df["timestamp"]).sort_values().reset_index(drop=True)
    max_minutes = 0.0
    max_start = None
    if not timestamps.empty:
        for i in range(len(timestamps)):
            start_ts = timestamps[i]
            window_end_ts = start_ts + pd.Timedelta(days=1)
            window_slice = timestamps[(timestamps >= start_ts) & (timestamps <= window_end_ts)]
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

# ---------------------------
# Visualizations
# ---------------------------

def plot_hourly_activity(df: pd.DataFrame, title: str = "Hourly Activity"):
    if df is None or df.empty:
        st.info("No messages for the selected period/user.")
        return
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
    hourly.columns = ["hour", "count"]
    fig = px.line(hourly, x="hour", y="count", markers=True, title=title)
    fig.update_xaxes(tickmode="linear")
    st.plotly_chart(fig, width="stretch")


def plot_heatmap(df: pd.DataFrame, title: str = "Activity Heatmap"):
    if df is None or df.empty:
        return
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    pivot = df.groupby(["date", "hour"]).size().unstack(fill_value=0)
    pivot = pivot.sort_index()
    fig = px.imshow(pivot.T.values, aspect="auto", labels=dict(x="Date", y="Hour", color="Messages"), x=pivot.index.astype(str), y=pivot.columns)
    fig.update_layout(title=title)
    st.plotly_chart(fig, width="stretch")

# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    # single set_page_config at top-level already applied; don't reapply here
    st.title("Messages Analytics Dashboard")
    st.markdown("Analyze user engagement patterns from the BigQuery `messages` table.")

    # --- Auth UI (moved out of cached functions) ---
    auth_secrets = st.secrets.get("auth", {}) if hasattr(st, "secrets") else {}
    AUTH_USERNAME = auth_secrets.get("username", "Admin")
    AUTH_PASSWORD = auth_secrets.get("password", "admin123@#")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    with st.sidebar:
        st.header("Account")
        if st.session_state.get("authenticated"):
            st.write(f"Logged in as **{st.session_state.get('username')}**")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                # trigger rerun
                try:
                    if hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()
                    else:
                        st.experimental_set_query_params(_logout=int(time.time()))
                        st.stop()
                except Exception:
                    st.experimental_set_query_params(_logout=int(time.time()))
                    st.stop()
        else:
            st.subheader("Login to access dashboard")
            _user_in = st.text_input("Username", value="")
            _pass_in = st.text_input("Password", type="password")
            if st.button("Login"):
                if str(_user_in) == str(AUTH_USERNAME) and str(_pass_in) == str(AUTH_PASSWORD):
                    st.session_state.authenticated = True
                    st.session_state.username = _user_in
                    st.success("Logged in successfully")
                    try:
                        if hasattr(st, "experimental_rerun"):
                            st.experimental_rerun()
                        else:
                            st.experimental_set_query_params(_login=int(time.time()))
                            st.stop()
                    except Exception:
                        st.experimental_set_query_params(_login=int(time.time()))
                        st.stop()
                else:
                    st.error("Invalid username or password")

    if not st.session_state.get("authenticated"):
        st.sidebar.info("Please login with your credentials to view the dashboard.")
        st.sidebar.caption("Provide auth credentials via st.secrets under [auth] for production.")
        return

    client = get_bq_client()

    # Header summary
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        try:
            total_users = get_unique_user_count_bq()
        except Exception as e:
            total_users = "Unknown"
            st.error(f"Error fetching unique user count: {e}")

        col1.metric("Unique Users", f"{total_users:,}" if isinstance(total_users, int) else total_users)
        try:
            q = f"SELECT COUNT(1) as cnt FROM {table_ref()}"
            cnt = list(client.query(q).result())[0].cnt
            col2.metric("Total Records", f"{int(cnt):,}")
        except Exception:
            col2.metric("Total Records", "Unknown")
        col3.metric("Data Source", f"BigQuery: {DATASET}.{MESSAGES_TABLE}")
        col4.metric("Last Refresh", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

    # Sidebar controls
    st.sidebar.header("Filters & Controls")
    mode = st.sidebar.selectbox("Display Mode", ["Single User View", "Leaderboard View"]) 
    start_date = st.sidebar.date_input("Start date (optional)", value=None)
    end_date = st.sidebar.date_input("End date (optional)", value=None)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Select User / Phone Number")
    page = st.sidebar.number_input("Users page", min_value=0, value=0, step=1)
    order_by_eng = st.sidebar.checkbox("Order users by engagement", value=False)
    users_df = list_users_page(page=page, page_size=USERS_PAGE_SIZE, order_by_engagement=order_by_eng)
    users_list = users_df.get("user_identifier", pd.Series(dtype=str)).dropna().astype(str).tolist()
    users_list_display = ["View All"] + users_list
    selected_users = st.sidebar.multiselect("Choose user(s)", options=users_list_display, default=[users_list_display[0]])

    # Refresh control
    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh data (clear cache)"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        # trigger rerun
        try:
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            else:
                st.experimental_set_query_params(_refresh=int(time.time()))
                st.stop()
        except Exception:
            st.experimental_set_query_params(_refresh=int(time.time()))
            st.stop()

    # Main layout
    if mode == "Single User View":
        st.header("Single User View")
        if not selected_users or "View All" in selected_users:
            st.info("Please select a specific user from the sidebar (unselect 'View All') to see user metrics.")
        else:
            for user in selected_users:
                st.subheader(f"User: {user}")
                s = None
                e = None
                if start_date and end_date:
                    s = datetime.combine(start_date, datetime.min.time())
                    e = datetime.combine(end_date, datetime.max.time())
                elif end_date and not start_date:
                    e = datetime.combine(end_date, datetime.max.time())
                    s = e - timedelta(days=1)
                elif start_date and not end_date:
                    s = datetime.combine(start_date, datetime.min.time())
                    e = s + timedelta(days=1)

                user_msgs = get_user_messages(user, start=s, end=e)
                metrics = compute_user_metrics(user_msgs, analysis_end=e)

                col1, col2, col3, col4 = st.columns(4)
                if metrics:
                    col1.metric("First Chat Date", metrics["first_interaction"].strftime("%Y-%m-%d %H:%M:%S"))
                    col2.metric("24-hr Minutes", f"{metrics['daily_minutes']} min", metrics['daily_minutes_hm'])
                    if metrics['peak_hour'] is not None:
                        peak_label = f"{metrics['peak_hour']:02d}:00 - {metrics['peak_hour']:02d}:59"
                        col3.metric("Peak Hour", peak_label, f"{metrics['peak_count']} msgs")
                    else:
                        col3.metric("Peak Hour", "N/A")
                    if metrics['highest_window']['start'] is not None:
                        hw = metrics['highest_window']
                        col4.metric("Top 24-hr Window", f"{hw['total_minutes']} min", f"{hw['start'].date()} to {hw['end'].date()}")
                else:
                    col1.info("No data for user")

                st.markdown("**Hourly distribution (selected period)**")
                plot_hourly_activity(user_msgs, title=f"Hourly Activity for {user}")
                with st.expander("Show raw messages (first 500)"):
                    st.dataframe(user_msgs.head(500), width="stretch")

    else:
        st.header("Leaderboard View — Top Users by Total Interaction Minutes")
        leaderboard = precompute_leaderboard(limit=LEADERBOARD_LIMIT)
        if leaderboard.empty:
            st.info("Leaderboard currently empty — no data available")
        else:
            leaderboard_display = leaderboard.copy()
            leaderboard_display["total_minutes"] = leaderboard_display["total_minutes"].round().astype(int)
            leaderboard_display["percentage_of_total"] = leaderboard_display["percentage_of_total"].round(2)
            leaderboard_display["first_message"] = pd.to_datetime(leaderboard_display["first_message"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            leaderboard_display["last_message"] = pd.to_datetime(leaderboard_display["last_message"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            leaderboard_display["rank"] = range(1, len(leaderboard_display) + 1)
            cols = ["rank", "user_identifier", "total_minutes", "percentage_of_total", "message_count", "avg_session_minutes", "first_message", "last_message"]
            st.dataframe(leaderboard_display[cols], width="stretch")

            top_n = st.slider("Top N users to chart", min_value=5, max_value=100, value=10)
            top_df = leaderboard_display.head(top_n)
            fig = px.bar(top_df, x="user_identifier", y="total_minutes", title=f"Top {top_n} Users by Total Minutes")
            st.plotly_chart(fig, width="stretch")

            csv = leaderboard_display.to_csv(index=False)
            st.download_button("Export Leaderboard CSV", csv, file_name="leaderboard_top_users.csv")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for large datasets. Uses BigQuery sessionization and Streamlit caching for performance.")


if __name__ == '__main__':
    main()
