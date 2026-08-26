"""
Load JPL EV charging sessions (jpl_sessions_raw.csv) into SQL Server.

What this does:
1. Connects to your local SQL Server instance (master) and creates a new
   database called EVChargingDB if it doesn't already exist.
2. Reconnects into EVChargingDB.
3. Reads jpl_sessions_raw.csv with pandas.
4. Pushes it into SQL Server as a staging table: stg_jpl_sessions.

Same pattern as your last two projects: load raw first, then do all
cleaning (dropping the partial April 2019 month, deciding what to do with
nulls, etc.) as a second pass using SQL -- CTEs, CASE WHEN, window functions.

Before running:
    pip install pandas sqlalchemy pyodbc
"""

import os
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------
# CONFIG -- change these if needed
# ---------------------------------------------------------------
SERVER_NAME = r"Bibhanshu\SQLEXPRESS01"
CSV_PATH = r"C:\Users\ACER\PycharmProjects\ACN\jpl_sessions_raw.csv"  # update if it's elsewhere
DB_NAME = "EVChargingDB"
TABLE_NAME = "stg_jpl_sessions"

# ---------------------------------------------------------------
# STEP 1 -- connect to master and create the database if missing
# ---------------------------------------------------------------
master_params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER_NAME};"
    f"DATABASE=master;"
    f"Trusted_Connection=yes;"
)
master_engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={master_params}",
    isolation_level="AUTOCOMMIT",
)

with master_engine.connect() as conn:
    result = conn.execute(
        text("SELECT database_id FROM sys.databases WHERE name = :name"),
        {"name": DB_NAME},
    ).fetchone()
    if result is None:
        print(f"Creating database '{DB_NAME}'...")
        conn.execute(text(f"CREATE DATABASE [{DB_NAME}]"))
        print("Database created.")
    else:
        print(f"Database '{DB_NAME}' already exists, skipping creation.")

master_engine.dispose()

# ---------------------------------------------------------------
# STEP 2 -- connect into EVChargingDB
# ---------------------------------------------------------------
db_params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER_NAME};"
    f"DATABASE={DB_NAME};"
    f"Trusted_Connection=yes;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={db_params}")

# ---------------------------------------------------------------
# STEP 3 -- load the CSV
# ---------------------------------------------------------------
print(f"\nReading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
print(f"  -> {len(df):,} rows, {len(df.columns)} columns")

# Parse timestamp columns properly so SQL Server gets real datetime types,
# not text strings
timestamp_cols = ["connectionTime", "disconnectTime", "doneChargingTime", "requestedDeparture"]
for col in timestamp_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

print(f"\nLoading into {DB_NAME}.dbo.{TABLE_NAME} ...")
df.to_sql(
    TABLE_NAME,
    engine,
    if_exists="replace",
    index=False,
    chunksize=2000,
)

print(f"\nDone. {len(df):,} rows loaded into {DB_NAME}.dbo.{TABLE_NAME}.")
print("Open SSMS, connect to your instance, and you'll find it under:")
print(f"  {DB_NAME} > Tables > dbo.{TABLE_NAME}")