"""
Pull all JPL charging sessions from Caltech's ACN-Data API and save to CSV.

v2 changes:
    - Retries automatically on transient errors (502/503/504/connection errors)
      with exponential backoff, instead of crashing.
    - Saves progress incrementally to a .jsonl file as it goes, so if it DOES
      crash for good, you can resume from the last saved page instead of
      starting over from page 1.

Auth:
    Set ACN_API_TOKEN as an environment variable (PyCharm Run Configuration
    -> Environment variables, as you already set up). Never hardcode it here.

Before running:
    pip install requests pandas
"""

import os
import json
import time
import requests
import pandas as pd

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
SITE_ID = "jpl"
BASE_URL = f"https://ev.caltech.edu/api/v1/sessions/{SITE_ID}"
OUTPUT_CSV = "jpl_sessions_raw.csv"
PROGRESS_FILE = "jpl_sessions_progress.jsonl"  # raw sessions saved as we go

MAX_RETRIES = 6
INITIAL_BACKOFF = 2  # seconds, doubles each retry

TOKEN = os.environ.get("ACN_API_TOKEN")
if not TOKEN:
    raise EnvironmentError(
        "ACN_API_TOKEN environment variable not set. "
        "Set it in your Run Configuration's Environment variables field."
    )

AUTH = (TOKEN, "")


def fetch_with_retry(url):
    """GET a URL, retrying on transient server errors with exponential backoff."""
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, auth=AUTH, timeout=30)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (502, 503, 504):
                print(f"  Server error {response.status_code}, "
                      f"retry {attempt}/{MAX_RETRIES} in {backoff}s...")
            else:
                # Non-transient error (e.g. 401 auth failure) -- don't retry, fail fast
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            print(f"  Connection error ({e}), retry {attempt}/{MAX_RETRIES} in {backoff}s...")

        time.sleep(backoff)
        backoff *= 2

    raise Exception(f"Failed after {MAX_RETRIES} retries: {url}")


# ---------------------------------------------------------------
# STEP 1 -- figure out where to resume from (if progress file exists)
# ---------------------------------------------------------------
already_have = []
if os.path.exists(PROGRESS_FILE):
    print(f"Found existing progress file, resuming...")
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                already_have.append(json.loads(line))
    print(f"  -> {len(already_have)} sessions already saved.")

all_sessions = already_have[:]

# ---------------------------------------------------------------
# STEP 2 -- pull remaining pages, appending to progress file as we go
# ---------------------------------------------------------------
# We resume by using 'where' to skip sessions we already have, filtering on
# connectionTime being before the earliest one we've already collected
# (since default sort is descending by connectionTime).
if already_have:
    earliest_time = min(s["connectionTime"] for s in already_have if s.get("connectionTime"))
    url = f"{BASE_URL}?pretty&where=connectionTime<\"{earliest_time}\""
else:
    url = f"{BASE_URL}?pretty"

page_num = 1
print(f"\nStarting pull for site '{SITE_ID}'...")

with open(PROGRESS_FILE, "a", encoding="utf-8") as progress_f:
    while url:
        print(f"Fetching page {page_num}...")
        data = fetch_with_retry(url)

        sessions = data.get("_items", [])
        for s in sessions:
            progress_f.write(json.dumps(s) + "\n")
        progress_f.flush()

        all_sessions.extend(sessions)

        meta = data.get("_meta", {})
        print(f"  -> {len(sessions)} sessions on this page "
              f"(total so far: {len(all_sessions)} / {meta.get('total', '?')})")

        next_link = data.get("_links", {}).get("next", {}).get("href")
        if next_link:
            url = f"https://ev.caltech.edu/api/v1/{next_link}"
            page_num += 1
            time.sleep(0.3)
        else:
            url = None

print(f"\nPull complete. Total sessions retrieved: {len(all_sessions)}")

# ---------------------------------------------------------------
# STEP 3 -- flatten into a DataFrame
# ---------------------------------------------------------------
flat_rows = []

for s in all_sessions:
    row = {
        "sessionID": s.get("sessionID"),
        "siteID": s.get("siteID"),
        "spaceID": s.get("spaceID"),
        "stationID": s.get("stationID"),
        "clusterID": s.get("clusterID"),
        "userID": s.get("userID"),
        "connectionTime": s.get("connectionTime"),
        "disconnectTime": s.get("disconnectTime"),
        "doneChargingTime": s.get("doneChargingTime"),
        "kWhDelivered": s.get("kWhDelivered"),
        "timezone": s.get("timezone"),
    }

    user_inputs = s.get("userInputs") or []
    if len(user_inputs) > 0:
        first_input = user_inputs[0]
        row["kWhRequested"] = first_input.get("kWhRequested")
        row["milesRequested"] = first_input.get("milesRequested")
        row["minutesAvailable"] = first_input.get("minutesAvailable")
        row["requestedDeparture"] = first_input.get("requestedDeparture")
        row["paymentRequired"] = first_input.get("paymentRequired")
        row["WhPerMile"] = first_input.get("WhPerMile")
        row["num_user_inputs"] = len(user_inputs)
    else:
        row["kWhRequested"] = None
        row["milesRequested"] = None
        row["minutesAvailable"] = None
        row["requestedDeparture"] = None
        row["paymentRequired"] = None
        row["WhPerMile"] = None
        row["num_user_inputs"] = 0

    flat_rows.append(row)

df = pd.DataFrame(flat_rows)

# de-duplicate in case resume overlapped by a session or two at the boundary
before = len(df)
df = df.drop_duplicates(subset="sessionID")
after = len(df)
if before != after:
    print(f"Removed {before - after} duplicate sessions from resume boundary.")

# ---------------------------------------------------------------
# STEP 4 -- save to CSV
# ---------------------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df):,} rows x {len(df.columns)} columns to {OUTPUT_CSV}")
print("\nColumn summary:")
print(df.dtypes)
print("\nNull counts:")
print(df.isnull().sum())

print(f"\nYou can now delete {PROGRESS_FILE} if you want -- it was only needed for resuming.")